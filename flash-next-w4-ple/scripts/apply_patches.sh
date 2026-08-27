#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/sglang-checkout" >&2
  exit 64
fi

sglang_dir=$(realpath "$1")
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
release_dir=$(cd -- "$script_dir/.." && pwd)
qwen_file="$sglang_dir/python/sglang/srt/models/qwen4_exp.py"
qsa_file="$sglang_dir/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py"
loader_file="$sglang_dir/python/sglang/srt/models/qwen4_ple_w4_sidecar.py"

expected_commit=73a255206f916366c8d26d4022f82ddfb0ab558d
expected_qwen_base=f406977eb2373937393241f453477867f7dc943bd4839216db8fe66fa9f921d8
expected_qwen_w4=95bb2059669f9a66abe1be8e271037eeaa11ea534ace808f1c171ca78fee101f
expected_qwen_final=b54de0a07a16a7a3070aabead3c53b80f108d89528c30763ca8e032f330d97ac
expected_qsa_base=c959835d05d0f395ad7eae4330cf264af9f6f7c1bff3d45a39bb953d2536f5f2
expected_qsa_final=584e2acdce11c6e1e6dc50b9f61ca8018a3634e1bafb3d079b367fc30d1f7634
expected_loader=9dbace396f69ca2a319c7ae9cf74380549b62b6cdcc9f88135776552c245bb68
expected_sidecar_patch=938c43e9d283883ffc285c4e5e3a03967ebc0167480767606131fe361f1d8644

require_sha() {
  local path=$1
  local expected=$2
  local label=$3
  local actual
  actual=$(sha256sum -- "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "REFUSED: $label SHA-256 $actual != $expected" >&2
    exit 78
  fi
}

for command_name in git install patch python3 realpath sha256sum; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "REFUSED: required command is unavailable: $command_name" >&2
    exit 69
  fi
done
if ! python3 -c 'import sys' >/dev/null 2>&1; then
  echo "REFUSED: python3 is present but cannot execute" >&2
  exit 69
fi

if [[ "$(git -C "$sglang_dir" rev-parse --is-inside-work-tree 2>/dev/null)" != "true" ]]; then
  echo "REFUSED: not a Git checkout: $sglang_dir" >&2
  exit 78
fi
actual_commit=$(git -C "$sglang_dir" rev-parse HEAD)
if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "REFUSED: SGLang HEAD $actual_commit != $expected_commit" >&2
  exit 78
fi
if [[ -e "$loader_file" ]]; then
  echo "REFUSED: sidecar loader already exists: $loader_file" >&2
  exit 78
fi

require_sha "$qwen_file" "$expected_qwen_base" "Qwen4-Exp base"
require_sha "$qsa_file" "$expected_qsa_base" "QSA base"
require_sha "$release_dir/src/qwen4_ple_w4_sidecar.py" "$expected_loader" \
  "sidecar loader source"
require_sha "$release_dir/patches/0002-qwen4-exp-sidecar.patch" \
  "$expected_sidecar_patch" "sidecar patch"

patch --dry-run --batch --fuzz=0 --posix -p1 -d "$sglang_dir" \
  < "$release_dir/patches/0001-qwen4-exp-w4-ple.patch"
patch --dry-run --batch --fuzz=0 --posix -p1 -d "$sglang_dir" \
  < "$release_dir/patches/0003-qsa-sm120-xqa.patch"

patch --batch --fuzz=0 --posix -p1 -d "$sglang_dir" \
  < "$release_dir/patches/0001-qwen4-exp-w4-ple.patch"
require_sha "$qwen_file" "$expected_qwen_w4" "Qwen4-Exp W4 intermediate"

temporary_qwen="${qwen_file}.sidecar.$$"
trap 'rm -f -- "$temporary_qwen"' EXIT
python3 "$script_dir/build_sidecar_overlay.py" \
  --base "$qwen_file" \
  --patch "$release_dir/patches/0002-qwen4-exp-sidecar.patch" \
  --output "$temporary_qwen"
mv -- "$temporary_qwen" "$qwen_file"

patch --batch --fuzz=0 --posix -p1 -d "$sglang_dir" \
  < "$release_dir/patches/0003-qsa-sm120-xqa.patch"
install -m 0644 -- "$release_dir/src/qwen4_ple_w4_sidecar.py" "$loader_file"

require_sha "$qwen_file" "$expected_qwen_final" "Qwen4-Exp final"
require_sha "$qsa_file" "$expected_qsa_final" "QSA final"
require_sha "$loader_file" "$expected_loader" "sidecar loader"

echo "PATCH_SET_VERIFIED $expected_commit"
