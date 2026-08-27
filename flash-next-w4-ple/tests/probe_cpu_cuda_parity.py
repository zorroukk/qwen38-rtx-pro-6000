#!/usr/bin/env python3
"""Stratified real-checkpoint CPU builder versus online CUDA parity probe."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from safetensors import safe_open

from build_ple_w4_sidecar import GROUP_SIZE, inspect_checkpoint, quantize_fp8_rows


@torch.no_grad()
def quantize_cuda(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, embedding_dim = source.shape
    groups_per_row = embedding_dim // GROUP_SIZE
    values = source.to(device="cuda", dtype=torch.float32)
    blocks = values.reshape(rows, groups_per_row, GROUP_SIZE)
    max_positive = blocks.clamp_min(0).amax(dim=-1) / 7.0
    max_negative = (-blocks.clamp_max(0)).amax(dim=-1) / 8.0
    scale = torch.maximum(max_positive, max_negative).clamp_min(2**-9)
    scale_fp8 = scale.to(torch.float8_e4m3fn)
    quantized = torch.round(blocks / scale_fp8.float().unsqueeze(-1))
    quantized = quantized.clamp(-8, 7).to(torch.int16).reshape(rows, embedding_dim)
    low = quantized[:, 0::2] & 0xF
    high = (quantized[:, 1::2] & 0xF) << 4
    packed = (low | high).to(torch.uint8)
    return packed.cpu(), scale_fp8.cpu()


def digest(tensor: torch.Tensor) -> str:
    raw = tensor.contiguous().view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the parity probe")
    torch.set_num_threads(2)
    checkpoint = inspect_checkpoint(Path("/models/flash-next"))
    sample_rows = 32_768
    results = []
    for shard_index in (0, 37, 63, 127):
        key, path = checkpoint["shards"][shard_index]
        with safe_open(path, framework="pt", device="cpu") as handle:
            source = handle.get_tensor(key)
            indices = (
                torch.arange(sample_rows, dtype=torch.int64) * source.shape[0]
            ) // sample_rows
            sample = source[indices].contiguous()
        cpu_weight, cpu_scales = quantize_fp8_rows(sample)
        cuda_weight, cuda_scales = quantize_cuda(sample)
        weights_equal = torch.equal(cpu_weight, cuda_weight)
        scales_equal = torch.equal(
            cpu_scales.view(torch.uint8), cuda_scales.view(torch.uint8)
        )
        result = {
            "shard": shard_index,
            "rows": sample_rows,
            "weights_equal": weights_equal,
            "scales_equal": scales_equal,
            "weight_sha256": digest(cpu_weight),
            "scale_sha256": digest(cpu_scales),
        }
        print(result, flush=True)
        results.append(result)
        if not weights_equal or not scales_equal:
            weight_diff = int((cpu_weight != cuda_weight).sum().item())
            scale_diff = int(
                (
                    cpu_scales.view(torch.uint8)
                    != cuda_scales.view(torch.uint8)
                ).sum().item()
            )
            raise AssertionError(
                f"CPU/CUDA W4 mismatch shard={shard_index}: "
                f"weight_bytes={weight_diff} scale_bytes={scale_diff}"
            )
    print(
        "PLE_W4_CPU_CUDA_PARITY_OK",
        f"shards={len(results)}",
        f"rows={len(results) * sample_rows}",
    )


if __name__ == "__main__":
    main()
