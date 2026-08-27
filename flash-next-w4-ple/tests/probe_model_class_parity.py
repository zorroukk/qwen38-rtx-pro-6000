#!/usr/bin/env python3
"""Compare the CPU sidecar builder with the actual online W4 class path."""

from types import SimpleNamespace

import torch
from safetensors import safe_open

from build_ple_w4_sidecar import quantize_fp8_rows
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.models.qwen4_exp import Qwen4ExpPackedPinnedHostEmbedding


ROWS = 4096
DIM = 160
fake = SimpleNamespace(
    quant_method=UnquantizedEmbeddingMethod(),
    weight=torch.nn.Parameter(
        torch.empty((ROWS, DIM), device="cuda", dtype=torch.float8_e4m3fn),
        requires_grad=False,
    ),
    weight_scale=torch.tensor(
        [0.00019931793212890625], device="cuda", dtype=torch.bfloat16
    ),
    quant_config=None,
    enable_tp=False,
    use_attn_tp_group=False,
    tp_size=1,
    num_embeddings=ROWS,
    num_embeddings_padded=ROWS,
    org_vocab_size=ROWS,
    padding_size=0,
    num_added_embeddings=0,
    use_presharded_weights=False,
    org_vocab_size_padded=ROWS,
    shard_indices=SimpleNamespace(
        org_vocab_start_index=0, org_vocab_end_index=ROWS
    ),
    embedding_dim=DIM,
    num_embeddings_per_partition=ROWS,
    num_org_embeddings_per_partition=ROWS,
    num_added_embeddings_per_partition=0,
)

with safe_open(
    "/models/flash-next/model-plefp8-00000.safetensors",
    framework="pt",
    device="cpu",
) as handle:
    key = next(k for k in handle.keys() if k.endswith("shard_0.weight"))
    source = handle.get_tensor(key)[:ROWS].contiguous()

expected_weight, expected_scales = quantize_fp8_rows(source)
embedding = Qwen4ExpPackedPinnedHostEmbedding(fake)
embedding.load_fp8_rows(source, 0)
assert torch.equal(embedding.weight.data, expected_weight)
assert torch.equal(
    embedding.block_scales.view(torch.uint8), expected_scales.view(torch.uint8)
)

ids = torch.tensor([0, 1, 7, 31, 127, 1023, 2047, 4095], device="cuda")
actual = embedding.gather(ids).cpu()
packed = expected_weight[ids.cpu()]
nibbles = torch.empty((len(ids), DIM), dtype=torch.int16)
nibbles[:, 0::2] = packed & 0xF
nibbles[:, 1::2] = (packed >> 4) & 0xF
signed = torch.where(nibbles >= 8, nibbles - 16, nibbles).float()
scales = expected_scales[ids.cpu()].float().repeat_interleave(16, dim=1)
reference = (signed * scales).to(torch.bfloat16)
assert torch.equal(actual, reference)

embedding._sidecar_loaded = True
try:
    embedding.load_fp8_rows(source, 0)
except RuntimeError:
    pass
else:
    raise AssertionError("verified sidecar did not reject online overwrite")

print(
    "PLE_W4_ACTUAL_CLASS_PARITY_OK",
    f"rows={ROWS}",
    f"gather_rows={len(ids)}",
    f"pinned={embedding.weight.is_pinned() and embedding.block_scales.is_pinned()}",
)
