---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Lewfkrad/Qwen3.8-Flash-Next-NVFP4-W4-PLE/blob/main/LICENSE
base_model: RadixArk/Qwen3.8-Flash-Next-NVFP4
library_name: sglang
tags:
- qwen
- qwen3.8
- sglang
- modelopt
- int4
- blackwell
---

# Qwen3.8-Flash-Next NVFP4 signed-W4 PLE sidecar

This repository contains a **companion PLE sidecar**, not a standalone model.
It replaces the 51.200 GB / 47.684 GiB FP8 n-gram embedding table in
[`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
with a 28.800 GB / 26.822 GiB signed-W4/group-16 representation for use with a
pinned downstream SGLang runtime.

You must separately download the exact base checkpoint at revision
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`.

> [!WARNING]
> This is experimental downstream code, not a released SGLang quantization
> format. It was qualified only with TP=1 on an NVIDIA RTX PRO 6000 Blackwell
> (SM120), BF16 KV cache, and the exact code/image pins below. Do not enable
> fallback or reuse this sidecar with a changed or fine-tuned checkpoint.

## Contents

| File | Shape | Dtype | Bytes | SHA-256 |
| --- | --- | --- | ---: | --- |
| `qweight.u8` | `[320001536, 80]` | `uint8` | 25,600,122,880 | `7e18b8dac400bda73b24a2dd135ca009163972031c7b83095ef1816650df3297` |
| `block_scales.f8` | `[320001536, 10]` | FP8 E4M3 | 3,200,015,360 | `667736278723db00dc70a7afc37aaf9131deab6e9345047f467035add7e09a1c` |
| `manifest.json` | source/layout contract | JSON | 13,451 | `a11028a945bac40c7a2d5f41f21c829a08f5d531c39184dda2a7c8731d7e1691` |

The table has 320,001,536 rows and 160 source values per row. Each group of 16
FP8 values becomes eight signed-INT4 packed bytes plus one FP8 E4M3 scale:

```text
scale = max(max_positive / 7, max_negative / 8).clamp_min(2**-9)
scale = cast_to_float8_e4m3fn(scale)
q = round_to_nearest_even(value / scale).clamp(-8, 7)
```

Even dimensions occupy the low nibble and odd dimensions the high nibble,
using four-bit two's-complement values. The checkpoint's outer BF16 scale is
preserved exactly as `0.00019931793212890625`.

This is a lossy quantization of the FP8 source. CPU/CUDA parity tests are
bit-exact for the resulting W4 representation; that does not make it lossless.

The W4 PLE artifacts are derivative works of
`Qwen/Qwen3.8-Flash-Next`, obtained from
`RadixArk/Qwen3.8-Flash-Next-NVFP4` revision
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`. They were modified by
quantizing the PLE from FP8 E4M3 to signed INT4 with group size 16 and E4M3
scales. They are distributed under the Qwen Community License 1.0 and are not
official Qwen or RadixArk releases.

## Strict source binding

The manifest binds this artifact to:

- model revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594`;
- model index SHA-256 `da5ca9c3b65e48e151329e64e141c2fa700bf2f99aec53cc014e4b52a6ff7a84`;
- config SHA-256 `e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624`;
- Hugging Face tree receipt SHA-256 `f84acd65b08e4de8f9f1698b85136655f24ef04f9d8b2e739f102ff47c9fa572`;
- raw FP8 PLE SHA-256 `b070f9644adf93794d8a1030584ab705809387e64396a9327a68fa3a3a6666b3`;
- all ten source PLE LFS objects and all 128 logical PLE shards.

The runtime streams both binaries directly into pinned CPU tensors while
hashing them, rejects any digest/source/layout mismatch, gathers selected rows
through CUDA UVA, and dequantizes them to BF16 on the GPU.

## Runtime and reproduction

Source, patches, tests, sanitized evidence, serving example, and the field
report are published at:

[`lEWFkRAD/qwen38-pro6000-report/flash-next-w4-ple`](https://github.com/lEWFkRAD/qwen38-pro6000-report/tree/Cloud1/flash-next-w4-ple)

Important pins:

- SGLang Qwen4-Exp PR #36497 head:
  `73a255206f916366c8d26d4022f82ddfb0ab558d`
- Runtime image:
  `lmsysorg/sglang@sha256:59f06adce6f91401adf443bd168d45fdb2044d77671fd591c7c57a29d851cbae`
- Sidecar manifest:
  `a11028a945bac40c7a2d5f41f21c829a08f5d531c39184dda2a7c8731d7e1691`
- TP=1 and BF16 KV cache are required by the qualified profile.

Verify `SHA256SUMS` before mounting the artifact. Mount the checkpoint,
sidecar, and patched runtime files read-only, and set
`SGLANG_QWEN4_PLE_W4_SIDECAR_FALLBACK=0`.

## License

`qweight.u8`, `block_scales.f8`, `manifest.json`, and `manifest.sha256` are
distributed subject to the included Qwen Community License 1.0. Preserve its
copyright and permission notice. Certain commercial Model-as-a-Service and AI
Work Assistant uses require a separate Qwen license; its large-product display
requirements also continue to apply.

The accompanying SGLang-derived patches are separately distributed under
Apache License 2.0 in the linked GitHub repository.

This is an independent community implementation and is not an official release
of Alibaba Qwen, SGLang, NVIDIA, RadixArk, or FlashInfer.
