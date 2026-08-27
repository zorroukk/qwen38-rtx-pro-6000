# Source provenance

`hf-tree-7b719225242aacd3dbd3f9407468c2ee9a9d2594.json` is the exact
Hugging Face tree receipt captured with the qualified checkpoint revision. It
contains public repository filenames, sizes, blob identities, and LFS SHA-256
digests; it contains no credential or private host path.

The receipt is required by the sealed builder because a normal
`hf download --local-dir` may create per-file metadata without this aggregate
JSON. After downloading the exact base checkpoint, install the receipt at:

```text
<MODEL_DIR>/.cache/huggingface/trees/7b719225242aacd3dbd3f9407468c2ee9a9d2594.json
```

Its required SHA-256 is:

```text
f84acd65b08e4de8f9f1698b85136655f24ef04f9d8b2e739f102ff47c9fa572
```

The builder then re-hashes all ten PLE LFS files against this receipt, hashes
all 128 raw logical shards while converting them, and detects source changes
during the build.

