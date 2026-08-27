"""Persistent sidecar support for Qwen4-Exp signed-W4 PLE tables.

The runtime owns the pinned tensors. This module validates an immutable
manifest, streams each binary directly into the destination tensor, and hashes
the bytes during that one read. It deliberately has no online-quantization
fallback; the caller must opt into fallback explicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import torch


FORMAT_NAME = "sglang-qwen4-ple-signed-w4-g16"
FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
MANIFEST_DIGEST_NAME = "manifest.sha256"
WEIGHT_FILE = "qweight.u8"
SCALE_FILE = "block_scales.f8"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SidecarError(RuntimeError):
    """Raised when a PLE sidecar is incomplete, incompatible, or corrupt."""


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _require_int(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SidecarError(f"{label} must be an integer >= {minimum}, got {value!r}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise SidecarError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _safe_child(root: Path, name: Any, expected_name: str) -> Path:
    if name != expected_name:
        raise SidecarError(
            f"sidecar tensor file must be {expected_name!r}, got {name!r}"
        )
    child = root / expected_name
    if child.is_symlink():
        raise SidecarError(f"sidecar tensor file may not be a symlink: {child}")
    try:
        resolved = child.resolve(strict=True)
    except FileNotFoundError as exc:
        raise SidecarError(f"missing sidecar tensor file: {child}") from exc
    if resolved.parent != root.resolve(strict=True):
        raise SidecarError(f"sidecar tensor escaped its root: {child}")
    if not resolved.is_file():
        raise SidecarError(f"sidecar tensor is not a regular file: {child}")
    return resolved


def load_and_validate_manifest(
    sidecar_dir: str | os.PathLike[str],
    *,
    rows: int,
    embedding_dim: int,
    group_size: int = 16,
) -> tuple[Path, dict[str, Any], dict[str, Path], str]:
    root = Path(sidecar_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise SidecarError(f"sidecar path is not a directory: {root}")

    manifest_path = root / MANIFEST_NAME
    digest_path = root / MANIFEST_DIGEST_NAME
    if manifest_path.is_symlink() or digest_path.is_symlink():
        raise SidecarError("sidecar manifest files may not be symlinks")
    try:
        manifest_bytes = manifest_path.read_bytes()
        digest_text = digest_path.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise SidecarError(f"could not read sidecar manifest in {root}: {exc}") from exc
    expected_manifest_sha = digest_text.split()[0] if digest_text else ""
    _require_sha256(expected_manifest_sha, "manifest.sha256")
    actual_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_manifest_sha != expected_manifest_sha:
        raise SidecarError(
            "sidecar manifest digest mismatch: "
            f"expected {expected_manifest_sha}, got {actual_manifest_sha}"
        )
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise SidecarError(f"invalid sidecar manifest JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise SidecarError("sidecar manifest root must be an object")

    if manifest.get("format") != FORMAT_NAME:
        raise SidecarError(f"unsupported sidecar format: {manifest.get('format')!r}")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise SidecarError(
            f"unsupported sidecar format_version: {manifest.get('format_version')!r}"
        )
    quant = manifest.get("quantization")
    if not isinstance(quant, dict):
        raise SidecarError("sidecar quantization entry must be an object")
    expected_quant = {
        "algorithm": "signed_uniform_int4",
        "group_size": group_size,
        "scale_dtype": "float8_e4m3fn",
        "nibble_order": "even_low_odd_high_twos_complement",
        "rounding": "torch_round_nearest_even",
        "scale_rule": "max(max_positive/7,max_negative/8).clamp_min(2**-9)",
    }
    for key, expected in expected_quant.items():
        if quant.get(key) != expected:
            raise SidecarError(
                f"sidecar quantization.{key} mismatch: "
                f"expected {expected!r}, got {quant.get(key)!r}"
            )

    layout = manifest.get("layout")
    if not isinstance(layout, dict):
        raise SidecarError("sidecar layout entry must be an object")
    if _require_int(layout.get("rows"), "layout.rows", 1) != rows:
        raise SidecarError(
            f"sidecar row mismatch: expected {rows}, got {layout.get('rows')!r}"
        )
    if _require_int(layout.get("embedding_dim"), "layout.embedding_dim", 1) != embedding_dim:
        raise SidecarError(
            "sidecar embedding_dim mismatch: "
            f"expected {embedding_dim}, got {layout.get('embedding_dim')!r}"
        )
    if _require_int(layout.get("group_size"), "layout.group_size", 1) != group_size:
        raise SidecarError("sidecar group_size mismatch")
    if embedding_dim % group_size or embedding_dim % 2:
        raise SidecarError("runtime embedding_dim is not compatible with W4 group layout")

    tensors = manifest.get("tensors")
    if not isinstance(tensors, dict) or set(tensors) != {"weight", "block_scales"}:
        raise SidecarError("sidecar tensors must contain exactly weight and block_scales")
    expected_specs = {
        "weight": {
            "file": WEIGHT_FILE,
            "dtype": "uint8",
            "shape": [rows, embedding_dim // 2],
            "bytes": rows * (embedding_dim // 2),
        },
        "block_scales": {
            "file": SCALE_FILE,
            "dtype": "float8_e4m3fn",
            "shape": [rows, embedding_dim // group_size],
            "bytes": rows * (embedding_dim // group_size),
        },
    }
    resolved_tensors = {}
    for tensor_name, expected in expected_specs.items():
        spec = tensors.get(tensor_name)
        if not isinstance(spec, dict):
            raise SidecarError(f"sidecar tensor {tensor_name} must be an object")
        for key, value in expected.items():
            if spec.get(key) != value:
                raise SidecarError(
                    f"sidecar tensors.{tensor_name}.{key} mismatch: "
                    f"expected {value!r}, got {spec.get(key)!r}"
                )
        _require_sha256(spec.get("sha256"), f"tensors.{tensor_name}.sha256")
        path = _safe_child(root, spec.get("file"), expected["file"])
        resolved_tensors[tensor_name] = path
        actual_size = path.stat().st_size
        if actual_size != expected["bytes"]:
            raise SidecarError(
                f"sidecar {tensor_name} size mismatch: "
                f"expected {expected['bytes']}, got {actual_size}"
            )

    source = manifest.get("source")
    if not isinstance(source, dict):
        raise SidecarError("sidecar source entry must be an object")
    if _require_int(source.get("checkpoint_shards"), "source.checkpoint_shards", 1) != 128:
        raise SidecarError("sidecar must cover all 128 synchronized PLE shards")
    rows_per_shard = _require_int(
        source.get("rows_per_shard"), "source.rows_per_shard", 1
    )
    if rows_per_shard * 128 != rows:
        raise SidecarError(
            f"source.rows_per_shard does not cover layout.rows: {rows_per_shard}"
        )
    for key in ("model_index_sha256", "config_sha256"):
        _require_sha256(source.get(key), f"source.{key}")
    _require_sha256(
        source.get("hf_tree_receipt_sha256"),
        "source.hf_tree_receipt_sha256",
    )
    ple_files = source.get("ple_files")
    if not isinstance(ple_files, list) or len(ple_files) != 10:
        raise SidecarError("source.ple_files must contain 10 LFS file records")
    seen_files = set()
    for file_index, file_record in enumerate(ple_files):
        if not isinstance(file_record, dict):
            raise SidecarError(f"source.ple_files[{file_index}] must be an object")
        file_name = file_record.get("file")
        if not isinstance(file_name, str) or not re.fullmatch(
            r"model-plefp8-\d{5}\.safetensors", file_name
        ):
            raise SidecarError(f"invalid PLE LFS filename: {file_name!r}")
        if file_name in seen_files:
            raise SidecarError(f"duplicate PLE LFS filename: {file_name}")
        seen_files.add(file_name)
        _require_int(file_record.get("bytes"), f"source.ple_files[{file_index}].bytes", 1)
        _require_sha256(
            file_record.get("lfs_sha256"),
            f"source.ple_files[{file_index}].lfs_sha256",
        )
    expected_files = {
        f"model-plefp8-{file_index:05d}.safetensors" for file_index in range(10)
    }
    if seen_files != expected_files:
        raise SidecarError(
            "source.ple_files does not cover model-plefp8-00000..00009"
        )
    _require_sha256(source.get("ple_raw_sha256"), "source.ple_raw_sha256")
    shard_digests = source.get("ple_shard_sha256")
    if not isinstance(shard_digests, list) or len(shard_digests) != 128:
        raise SidecarError("source.ple_shard_sha256 must contain 128 digests")
    for shard_index, digest in enumerate(shard_digests):
        _require_sha256(digest, f"source.ple_shard_sha256[{shard_index}]")
    revision = source.get("model_revision")
    if not isinstance(revision, str) or not revision.strip():
        raise SidecarError("source.model_revision must be a nonempty string")
    module_prefix = source.get("module_prefix")
    if not isinstance(module_prefix, str) or not module_prefix.endswith(
        ".ple.ple_embedding"
    ):
        raise SidecarError("source.module_prefix is invalid")
    weight_scale = source.get("outer_weight_scale")
    if (
        not isinstance(weight_scale, (int, float))
        or isinstance(weight_scale, bool)
        or not math.isfinite(float(weight_scale))
        or float(weight_scale) <= 0
    ):
        raise SidecarError("source.outer_weight_scale must be finite and positive")
    return root, manifest, resolved_tensors, actual_manifest_sha


def _byte_view(tensor: torch.Tensor, label: str) -> memoryview:
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise SidecarError(f"destination {label} must be a contiguous CPU tensor")
    if tensor.requires_grad:
        raise SidecarError(f"destination {label} must not require gradients")
    # NumPy does not expose float8 directly, but a uint8 view shares storage.
    array = tensor.detach().view(torch.uint8).reshape(-1).numpy()
    return memoryview(array).cast("B")


def _stream_file_into_tensor(
    path: Path,
    tensor: torch.Tensor,
    *,
    expected_bytes: int,
    expected_sha256: str,
    label: str,
    chunk_bytes: int,
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    view = _byte_view(tensor, label)
    if len(view) != expected_bytes:
        raise SidecarError(
            f"destination {label} byte size mismatch: {len(view)} != {expected_bytes}"
        )
    digest = hashlib.sha256()
    offset = 0
    started = time.monotonic()
    with path.open("rb", buffering=0) as reader:
        if hasattr(os, "posix_fadvise"):
            try:
                os.posix_fadvise(reader.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
            except OSError:
                pass
        while offset < expected_bytes:
            end = min(offset + chunk_bytes, expected_bytes)
            count = reader.readinto(view[offset:end])
            if count is None or count <= 0:
                raise SidecarError(
                    f"unexpected EOF in {label}: read {offset} of {expected_bytes} bytes"
                )
            digest.update(view[offset : offset + count])
            offset += count
        if reader.read(1):
            raise SidecarError(f"unexpected trailing data in {label}")
        if hasattr(os, "posix_fadvise"):
            try:
                os.posix_fadvise(reader.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise SidecarError(
            f"sidecar {label} digest mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    elapsed = time.monotonic() - started
    if progress is not None:
        progress(
            f"Loaded PLE sidecar {label}: {expected_bytes / (1024**3):.3f} GiB "
            f"in {elapsed:.2f}s ({expected_bytes / max(elapsed, 1e-9) / (1024**3):.2f} GiB/s)"
        )
    return {
        "bytes": expected_bytes,
        "sha256": actual_sha256,
        "elapsed_s": round(elapsed, 3),
    }


@torch.no_grad()
def load_sidecar_into_tensors(
    sidecar_dir: str | os.PathLike[str],
    weight: torch.Tensor,
    block_scales: torch.Tensor,
    *,
    group_size: int = 16,
    chunk_bytes: int = 64 * 1024 * 1024,
    progress: Callable[[str], None] | None = None,
    expected_source: Mapping[str, str] | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    if weight.dtype != torch.uint8 or weight.ndim != 2:
        raise SidecarError("destination weight must be a rank-2 uint8 tensor")
    if block_scales.dtype != torch.float8_e4m3fn or block_scales.ndim != 2:
        raise SidecarError(
            "destination block_scales must be a rank-2 float8_e4m3fn tensor"
        )
    if weight.shape[0] != block_scales.shape[0]:
        raise SidecarError("destination sidecar tensors have different row counts")
    embedding_dim = weight.shape[1] * 2
    if block_scales.shape[1] != embedding_dim // group_size:
        raise SidecarError("destination block scale shape is incompatible with weight")
    chunk_bytes = _require_int(chunk_bytes, "chunk_bytes", 4096)
    root, manifest, resolved_tensors, manifest_sha256 = load_and_validate_manifest(
        sidecar_dir,
        rows=weight.shape[0],
        embedding_dim=embedding_dim,
        group_size=group_size,
    )
    if (
        expected_manifest_sha256 is not None
        and manifest_sha256 != expected_manifest_sha256
    ):
        raise SidecarError(
            "sidecar manifest does not match the externally sealed digest: "
            f"expected {expected_manifest_sha256}, got {manifest_sha256}"
        )
    if expected_source is not None:
        for key, expected in expected_source.items():
            actual = manifest["source"].get(key)
            if actual != expected:
                raise SidecarError(
                    f"sidecar source.{key} mismatch: expected {expected!r}, "
                    f"got {actual!r}"
                )
    tensors = manifest["tensors"]
    started = time.monotonic()
    loaded = {}
    for name, tensor in (("weight", weight), ("block_scales", block_scales)):
        spec = tensors[name]
        loaded[name] = _stream_file_into_tensor(
            resolved_tensors[name],
            tensor,
            expected_bytes=spec["bytes"],
            expected_sha256=spec["sha256"],
            label=name,
            chunk_bytes=chunk_bytes,
            progress=progress,
        )
    return {
        "format": manifest["format"],
        "format_version": manifest["format_version"],
        "manifest_sha256": manifest_sha256,
        "outer_weight_scale": float(manifest["source"]["outer_weight_scale"]),
        "source": manifest["source"],
        "tensors": loaded,
        "elapsed_s": round(time.monotonic() - started, 3),
    }
