#!/usr/bin/env python3
"""Build a persistent signed-W4/group-16 PLE sidecar from FP8 shards."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

from qwen4_ple_w4_sidecar import (
    FORMAT_NAME,
    FORMAT_VERSION,
    MANIFEST_DIGEST_NAME,
    MANIFEST_NAME,
    SCALE_FILE,
    WEIGHT_FILE,
    canonical_json_bytes,
)


SHARD_RE = re.compile(r"\.ngram_embedding\.shard_(\d+)\.weight$")
EXPECTED_SHARDS = 128
GROUP_SIZE = 16


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as reader:
        if hasattr(os, "posix_fadvise"):
            try:
                os.posix_fadvise(reader.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
            except OSError:
                pass
        while True:
            chunk = reader.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
        if hasattr(os, "posix_fadvise"):
            try:
                os.posix_fadvise(reader.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
    return digest.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as writer:
        write_all(writer, data)
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(temporary, path)


def write_all(writer, data: bytes) -> None:
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        written = writer.write(view[offset:])
        if written is None or written <= 0:
            raise OSError(f"short write after {offset} of {len(view)} bytes")
        offset += written


@torch.no_grad()
def quantize_fp8_rows(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if source.dtype != torch.float8_e4m3fn or source.ndim != 2:
        raise TypeError("source must be a rank-2 float8_e4m3fn tensor")
    rows, embedding_dim = source.shape
    if embedding_dim % GROUP_SIZE or embedding_dim % 2:
        raise ValueError("embedding dimension must be divisible by 16 and 2")
    groups_per_row = embedding_dim // GROUP_SIZE
    values = source.float()
    blocks = values.reshape(rows, groups_per_row, GROUP_SIZE)
    max_positive = blocks.clamp_min(0).amax(dim=-1) / 7.0
    max_negative = (-blocks.clamp_max(0)).amax(dim=-1) / 8.0
    scale = torch.maximum(max_positive, max_negative).clamp_min(2**-9)
    scale_fp8 = scale.to(torch.float8_e4m3fn)
    quantized = torch.round(blocks / scale_fp8.float().unsqueeze(-1))
    quantized = quantized.clamp(-8, 7).to(torch.int16).reshape(rows, embedding_dim)
    low = quantized[:, 0::2] & 0xF
    high = (quantized[:, 1::2] & 0xF) << 4
    packed = (low | high).to(torch.uint8).contiguous()
    return packed, scale_fp8.contiguous()


def inspect_checkpoint(model_dir: Path) -> dict:
    index_path = model_dir / "model.safetensors.index.json"
    config_path = model_dir / "config.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("model index is missing weight_map")

    shards: dict[int, tuple[str, Path]] = {}
    scale_entries = []
    for key, relative_file in weight_map.items():
        match = SHARD_RE.search(key)
        if match:
            shard_index = int(match.group(1))
            if shard_index in shards:
                raise RuntimeError(f"duplicate PLE shard index {shard_index}")
            shards[shard_index] = (key, model_dir / relative_file)
        elif key.endswith(".ngram_embedding.weight_scale"):
            scale_entries.append((key, model_dir / relative_file))
    expected = set(range(EXPECTED_SHARDS))
    if set(shards) != expected:
        raise RuntimeError(
            f"checkpoint PLE shard set mismatch: missing={sorted(expected-set(shards))} "
            f"extra={sorted(set(shards)-expected)}"
        )
    if len(scale_entries) != 1:
        raise RuntimeError(f"expected one PLE weight_scale, found {len(scale_entries)}")

    scale_key, scale_path = scale_entries[0]
    checkpoint_module_prefix = scale_key[: -len(".ngram_embedding.weight_scale")]
    if not checkpoint_module_prefix.endswith(".ple.ple_embedding"):
        raise RuntimeError(
            f"unexpected PLE module prefix: {checkpoint_module_prefix}"
        )
    for shard_index, (key, _) in shards.items():
        if not key.startswith(
            f"{checkpoint_module_prefix}.ngram_embedding.shard_"
        ):
            raise RuntimeError(
                f"PLE shard {shard_index} belongs to a different module: {key}"
            )
    if checkpoint_module_prefix.startswith("model.language_model."):
        module_prefix = "model." + checkpoint_module_prefix[
            len("model.language_model.") :
        ]
    else:
        module_prefix = checkpoint_module_prefix
    with safe_open(scale_path, framework="pt", device="cpu") as handle:
        outer_weight_scale = float(handle.get_tensor(scale_key).float().item())

    first_key, first_path = shards[0]
    with safe_open(first_path, framework="pt", device="cpu") as handle:
        first = handle.get_slice(first_key)
        first_shape = tuple(first.get_shape())
        first_dtype = str(first.get_dtype())
    if len(first_shape) != 2 or first_shape[1] % GROUP_SIZE:
        raise RuntimeError(f"unsupported first PLE shard shape {first_shape}")
    if first_dtype not in {"F8_E4M3", "F8_E4M3FN"}:
        raise RuntimeError(f"unsupported first PLE shard dtype {first_dtype}")

    rows_per_shard, embedding_dim = first_shape
    for shard_index, (key, path) in shards.items():
        if not path.is_file():
            raise RuntimeError(f"missing checkpoint file for shard {shard_index}: {path}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            tensor_slice = handle.get_slice(key)
            shape = tuple(tensor_slice.get_shape())
            dtype = str(tensor_slice.get_dtype())
        if shape != first_shape or dtype != first_dtype:
            raise RuntimeError(
                f"PLE shard {shard_index} metadata mismatch: {shape}/{dtype} "
                f"!= {first_shape}/{first_dtype}"
            )
    return {
        "index_path": index_path,
        "config_path": config_path,
        "shards": shards,
        "rows_per_shard": rows_per_shard,
        "rows": rows_per_shard * EXPECTED_SHARDS,
        "embedding_dim": embedding_dim,
        "outer_weight_scale": outer_weight_scale,
        "module_prefix": module_prefix,
        "checkpoint_module_prefix": checkpoint_module_prefix,
        "index_sha256": sha256_file(index_path),
        "config_sha256": sha256_file(config_path),
    }


def verify_lfs_contract(
    model_dir: Path, source_revision: str, checkpoint: dict
) -> tuple[str, list[dict], dict[str, tuple[int, int, int, int, int]]]:
    tree_path = (
        model_dir
        / ".cache"
        / "huggingface"
        / "trees"
        / f"{source_revision}.json"
    )
    tree_sha256 = sha256_file(tree_path)
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    files = tree.get("files")
    if not isinstance(files, dict):
        raise RuntimeError(f"invalid Hugging Face tree receipt: {tree_path}")
    relative_files = sorted(
        {
            path.relative_to(model_dir).as_posix()
            for _, path in checkpoint["shards"].values()
        }
    )
    contract = []
    fingerprints = {}
    for relative_file in relative_files:
        entry = files.get(relative_file)
        if not isinstance(entry, dict):
            raise RuntimeError(f"tree receipt lacks {relative_file}")
        expected_sha256 = entry.get("lfs_sha256")
        expected_size = entry.get("lfs_size")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise RuntimeError(f"tree receipt lacks LFS SHA-256 for {relative_file}")
        if not isinstance(expected_size, int) or expected_size <= 0:
            raise RuntimeError(f"tree receipt lacks LFS size for {relative_file}")
        path = model_dir / relative_file
        stat = path.stat()
        if stat.st_size != expected_size:
            raise RuntimeError(
                f"LFS size mismatch for {relative_file}: "
                f"{stat.st_size} != {expected_size}"
            )
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"LFS digest mismatch for {relative_file}: "
                f"{actual_sha256} != {expected_sha256}"
            )
        fingerprints[relative_file] = (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        contract.append(
            {
                "file": relative_file,
                "bytes": expected_size,
                "lfs_sha256": expected_sha256,
            }
        )
        print(f"verified LFS source {relative_file} {actual_sha256}", flush=True)
    return tree_sha256, contract, fingerprints


def build(args: argparse.Namespace) -> None:
    model_dir = args.model_dir.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = inspect_checkpoint(model_dir)
    tree_sha256, lfs_contract, source_fingerprints = verify_lfs_contract(
        model_dir, args.source_revision, checkpoint
    )
    rows = checkpoint["rows"]
    embedding_dim = checkpoint["embedding_dim"]
    groups_per_row = embedding_dim // GROUP_SIZE
    weight_bytes = rows * (embedding_dim // 2)
    scale_bytes = rows * groups_per_row
    required = weight_bytes + scale_bytes + 5 * 1024**3
    free = shutil.disk_usage(output_dir).free
    if free < required:
        raise RuntimeError(
            f"insufficient free disk: need {required}, have {free} bytes"
        )

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    partial_weight = output_dir / f".{WEIGHT_FILE}.partial-{os.getpid()}"
    partial_scale = output_dir / f".{SCALE_FILE}.partial-{os.getpid()}"
    final_weight = output_dir / WEIGHT_FILE
    final_scale = output_dir / SCALE_FILE
    weight_digest = hashlib.sha256()
    scale_digest = hashlib.sha256()
    source_digest = hashlib.sha256()
    source_shard_digests = []
    weight_written = 0
    scale_written = 0
    started = time.monotonic()

    print(
        f"Building {FORMAT_NAME} rows={rows} dim={embedding_dim} "
        f"weight={weight_bytes} scales={scale_bytes}",
        flush=True,
    )
    try:
        with partial_weight.open("xb", buffering=0) as weight_out, partial_scale.open(
            "xb", buffering=0
        ) as scale_out:
            for shard_index in range(EXPECTED_SHARDS):
                key, path = checkpoint["shards"][shard_index]
                shard_started = time.monotonic()
                source_shard_digest = hashlib.sha256()
                with safe_open(path, framework="pt", device="cpu") as handle:
                    source = handle.get_tensor(key)
                    if source.dtype != torch.float8_e4m3fn:
                        raise RuntimeError(
                            f"shard {shard_index} loaded as {source.dtype}, expected fp8 e4m3fn"
                        )
                    for offset in range(0, source.shape[0], args.chunk_rows):
                        chunk = source[offset : offset + args.chunk_rows]
                        source_data = (
                            chunk.contiguous()
                            .view(torch.uint8)
                            .numpy()
                            .tobytes(order="C")
                        )
                        source_digest.update(source_data)
                        source_shard_digest.update(source_data)
                        packed, scales = quantize_fp8_rows(chunk)
                        weight_data = packed.numpy().tobytes(order="C")
                        scale_data = scales.view(torch.uint8).numpy().tobytes(order="C")
                        write_all(weight_out, weight_data)
                        write_all(scale_out, scale_data)
                        weight_digest.update(weight_data)
                        scale_digest.update(scale_data)
                        weight_written += len(weight_data)
                        scale_written += len(scale_data)
                        del packed, scales, source_data, weight_data, scale_data
                    del source
                source_shard_digests.append(source_shard_digest.hexdigest())
                elapsed = time.monotonic() - shard_started
                total_elapsed = time.monotonic() - started
                print(
                    f"shard {shard_index + 1:03d}/{EXPECTED_SHARDS} "
                    f"{elapsed:.2f}s total={total_elapsed:.1f}s",
                    flush=True,
                )
                gc.collect()
            weight_out.flush()
            scale_out.flush()
            os.fsync(weight_out.fileno())
            os.fsync(scale_out.fileno())
        if weight_written != weight_bytes or scale_written != scale_bytes:
            raise RuntimeError(
                f"output byte count mismatch: weight {weight_written}/{weight_bytes}, "
                f"scales {scale_written}/{scale_bytes}"
            )
        if partial_weight.stat().st_size != weight_bytes:
            raise RuntimeError("packed weight file size does not match byte count")
        if partial_scale.stat().st_size != scale_bytes:
            raise RuntimeError("block scale file size does not match byte count")
        os.replace(partial_weight, final_weight)
        os.replace(partial_scale, final_scale)
    except BaseException:
        print(
            f"Build interrupted; partial files retained for diagnosis: "
            f"{partial_weight.name}, {partial_scale.name}",
            file=sys.stderr,
        )
        raise

    source_revision = args.source_revision
    for relative_file, before in source_fingerprints.items():
        stat = (model_dir / relative_file).stat()
        after = (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        if after != before:
            raise RuntimeError(f"source file changed during build: {relative_file}")
    manifest = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "builder": {
            "script": Path(__file__).name,
            "torch_version": torch.__version__,
            "chunk_rows": args.chunk_rows,
            "threads": args.threads,
        },
        "source": {
            "model_path": str(model_dir),
            "model_revision": source_revision,
            "hf_tree_receipt_sha256": tree_sha256,
            "ple_files": lfs_contract,
            "model_index_sha256": checkpoint["index_sha256"],
            "config_sha256": checkpoint["config_sha256"],
            "ple_raw_sha256": source_digest.hexdigest(),
            "ple_shard_sha256": source_shard_digests,
            "checkpoint_shards": EXPECTED_SHARDS,
            "rows_per_shard": checkpoint["rows_per_shard"],
            "outer_weight_scale": checkpoint["outer_weight_scale"],
            "module_prefix": checkpoint["module_prefix"],
            "checkpoint_module_prefix": checkpoint["checkpoint_module_prefix"],
        },
        "quantization": {
            "algorithm": "signed_uniform_int4",
            "group_size": GROUP_SIZE,
            "scale_dtype": "float8_e4m3fn",
            "nibble_order": "even_low_odd_high_twos_complement",
            "rounding": "torch_round_nearest_even",
            "scale_rule": "max(max_positive/7,max_negative/8).clamp_min(2**-9)",
        },
        "layout": {
            "rows": rows,
            "embedding_dim": embedding_dim,
            "group_size": GROUP_SIZE,
            "packed_dim": embedding_dim // 2,
            "groups_per_row": groups_per_row,
        },
        "tensors": {
            "weight": {
                "file": WEIGHT_FILE,
                "dtype": "uint8",
                "shape": [rows, embedding_dim // 2],
                "bytes": weight_bytes,
                "sha256": weight_digest.hexdigest(),
            },
            "block_scales": {
                "file": SCALE_FILE,
                "dtype": "float8_e4m3fn",
                "shape": [rows, groups_per_row],
                "bytes": scale_bytes,
                "sha256": scale_digest.hexdigest(),
            },
        },
        "build": {
            "elapsed_s": round(time.monotonic() - started, 3),
            "weight_bytes": weight_written,
            "scale_bytes": scale_written,
        },
    }
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    atomic_write(output_dir / MANIFEST_NAME, manifest_bytes)
    atomic_write(
        output_dir / MANIFEST_DIGEST_NAME,
        f"{manifest_sha256}  {MANIFEST_NAME}\n".encode("ascii"),
    )
    directory_fd = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(output_dir),
                "manifest_sha256": manifest_sha256,
                "weight_sha256": weight_digest.hexdigest(),
                "scale_sha256": scale_digest.hexdigest(),
                "ple_raw_sha256": source_digest.hexdigest(),
                "elapsed_s": manifest["build"]["elapsed_s"],
            },
            indent=2,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--chunk-rows", type=int, default=65_536)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.chunk_rows < 1 or args.threads < 1:
        parser.error("chunk-rows and threads must be positive")
    return args


if __name__ == "__main__":
    build(parse_args())
