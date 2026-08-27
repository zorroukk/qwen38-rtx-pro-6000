# Notices

This directory is an independent community reproduction package for a
downstream Qwen3.8-Flash-Next serving experiment.

## Qwen model-derived artifact

The `qweight.u8`, `block_scales.f8`, `manifest.json`, and `manifest.sha256`
companion artifacts are transformed from or bind the Qwen3.8-Flash-Next PLE
weights. Their use and redistribution are subject to the Qwen Community License 1.0 in
[`LICENSES/QWEN-COMMUNITY-1.0.txt`](LICENSES/QWEN-COMMUNITY-1.0.txt). Preserve
the Qwen copyright and license notice with redistributed copies. The license
contains additional terms for certain commercial Model-as-a-Service and AI
Work Assistant uses.

The W4 PLE artifacts are derivative works of `Qwen/Qwen3.8-Flash-Next`,
obtained from `RadixArk/Qwen3.8-Flash-Next-NVFP4` revision
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`. They were modified by quantizing
the PLE from FP8 E4M3 to signed INT4 with group size 16 and E4M3 scales. They
are not official Qwen or RadixArk releases.

## SGLang-derived patches

The patches under `patches/` modify files from SGLang, which is distributed
under the Apache License 2.0. A copy is included at
[`LICENSES/APACHE-2.0.txt`](LICENSES/APACHE-2.0.txt). The patches are provided
against the exact upstream revision identified in the README rather than as
complete copies of upstream source files.

## Original downstream files

The sidecar loader, builder, examples, tests, documentation, and receipts in
this directory are licensed under Apache License 2.0 as stated in
[`LICENSE.md`](LICENSE.md). They are provided without warranty. Where a file
incorporates or patches upstream code, the upstream license continues to apply.

No endorsement or affiliation with Alibaba Qwen, SGLang, NVIDIA, RadixArk, or
FlashInfer is claimed.
