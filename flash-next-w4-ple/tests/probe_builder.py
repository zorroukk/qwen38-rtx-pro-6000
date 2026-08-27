#!/usr/bin/env python3
"""Read-only real-checkpoint probe for the sidecar builder."""

from pathlib import Path

import hashlib
import torch
from safetensors import safe_open

from build_ple_w4_sidecar import inspect_checkpoint, quantize_fp8_rows


model = Path("/models/flash-next")
checkpoint = inspect_checkpoint(model)
key, path = checkpoint["shards"][0]
with safe_open(path, framework="pt", device="cpu") as handle:
    source = handle.get_tensor(key)[:4096]
packed, scales = quantize_fp8_rows(source)
packed_bytes = packed.numpy().tobytes(order="C")
scale_bytes = scales.view(torch.uint8).numpy().tobytes(order="C")
print(
    "PLE_W4_REAL_PROBE_OK",
    f"rows={checkpoint['rows']}",
    f"dim={checkpoint['embedding_dim']}",
    f"outer_scale={checkpoint['outer_weight_scale']!r}",
    f"packed_sha256={hashlib.sha256(packed_bytes).hexdigest()}",
    f"scales_sha256={hashlib.sha256(scale_bytes).hexdigest()}",
)
