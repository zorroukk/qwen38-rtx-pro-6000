#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?set MODEL_DIR to the pinned RadixArk checkpoint directory}"
: "${SIDECAR_DIR:?set SIDECAR_DIR to the qualified sidecar directory}"
: "${SGLANG_CHECKOUT:?set SGLANG_CHECKOUT to the patched SGLang checkout}"

image='lmsysorg/sglang@sha256:59f06adce6f91401adf443bd168d45fdb2044d77671fd591c7c57a29d851cbae'
model_revision='7b719225242aacd3dbd3f9407468c2ee9a9d2594'
manifest_sha='a11028a945bac40c7a2d5f41f21c829a08f5d531c39184dda2a7c8731d7e1691'
index_sha='da5ca9c3b65e48e151329e64e141c2fa700bf2f99aec53cc014e4b52a6ff7a84'
config_sha='e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624'
tree_sha='f84acd65b08e4de8f9f1698b85136655f24ef04f9d8b2e739f102ff47c9fa572'
raw_ple_sha='b070f9644adf93794d8a1030584ab705809387e64396a9327a68fa3a3a6666b3'

qwen_file="$SGLANG_CHECKOUT/python/sglang/srt/models/qwen4_exp.py"
loader_file="$SGLANG_CHECKOUT/python/sglang/srt/models/qwen4_ple_w4_sidecar.py"
qsa_file="$SGLANG_CHECKOUT/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py"

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

require_sha "$MODEL_DIR/model.safetensors.index.json" "$index_sha" "model index"
require_sha "$MODEL_DIR/config.json" "$config_sha" "model config"
require_sha \
  "$MODEL_DIR/.cache/huggingface/trees/$model_revision.json" \
  "$tree_sha" \
  "Hugging Face tree receipt"

# These ten immutable LFS objects contain all 128 source PLE shards. Verifying
# them adds a one-time 51.2 GB read, but prevents a same-shape or stale source
# checkpoint from being paired with the sealed sidecar.
ple_files=(
  'dc2e845b7edd35bda92834fba3626bf7d199e28d6aceac99fee654aade390cfd model-plefp8-00000.safetensors'
  '899eaa0716e28594468a1389ee58cb065c23907f1270de3831f5ecb0a4f82d56 model-plefp8-00001.safetensors'
  '06fd8a11abf0419a669f89397b8d70dd6ff42d401e6b2a037c65e49704faaf71 model-plefp8-00002.safetensors'
  'c6aaa1fc08e84eced3c8151ac8679ed943888eba0fcef2556963693430f95bd9 model-plefp8-00003.safetensors'
  'd94e97c96d3ea09208614da016960f3f4b429f47a044c898a51b493f42f74ba2 model-plefp8-00004.safetensors'
  '03d5d4792e14a4ab55bae50bb459624b01786f818bbcfcaed1a5d0235af484c6 model-plefp8-00005.safetensors'
  '586cccbc12383021bc9bc02f206d9b19fbd1672373ed9a5d91e3b0ce34c2418f model-plefp8-00006.safetensors'
  'c4a23bcc10f3cde6b633e82c282a9a518e3a64d433628283e1bc592c94cf3d6c model-plefp8-00007.safetensors'
  '2dc8098c0d020bff277c9cf499a6b908e17836b70a5f949dfa24793371c9a87e model-plefp8-00008.safetensors'
  '61de98b89bb79f386795787d7a76827a26f1e292c26edbb0a1b613da455f5a9c model-plefp8-00009.safetensors'
)
for record in "${ple_files[@]}"; do
  expected=${record%% *}
  filename=${record#* }
  require_sha "$MODEL_DIR/$filename" "$expected" "source $filename"
done

require_sha "$SIDECAR_DIR/manifest.json" "$manifest_sha" "sidecar manifest"
require_sha "$SIDECAR_DIR/qweight.u8" \
  7e18b8dac400bda73b24a2dd135ca009163972031c7b83095ef1816650df3297 \
  "sidecar qweight"
require_sha "$SIDECAR_DIR/block_scales.f8" \
  667736278723db00dc70a7afc37aaf9131deab6e9345047f467035add7e09a1c \
  "sidecar block scales"
require_sha "$qwen_file" \
  b54de0a07a16a7a3070aabead3c53b80f108d89528c30763ca8e032f330d97ac \
  "Qwen4-Exp overlay"
require_sha "$loader_file" \
  9dbace396f69ca2a319c7ae9cf74380549b62b6cdcc9f88135776552c245bb68 \
  "sidecar loader"
require_sha "$qsa_file" \
  584e2acdce11c6e1e6dc50b9f61ca8018a3634e1bafb3d079b367fc30d1f7634 \
  "SM120 QSA overlay"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "REFUSED: nvidia-smi is required" >&2
  exit 78
fi
mapfile -t gpu_rows < <(
  nvidia-smi --query-gpu=index,name,compute_cap,memory.free \
    --format=csv,noheader,nounits
)
if [[ ${#gpu_rows[@]} -ne 1 ]]; then
  echo "REFUSED: the qualified profile expects exactly one visible GPU" >&2
  exit 78
fi
IFS=',' read -r gpu_index gpu_name gpu_compute_cap gpu_free_mib <<< "${gpu_rows[0]}"
gpu_index=${gpu_index//[[:space:]]/}
gpu_name=${gpu_name#${gpu_name%%[![:space:]]*}}
gpu_name=${gpu_name%${gpu_name##*[![:space:]]}}
gpu_compute_cap=${gpu_compute_cap//[[:space:]]/}
gpu_free_mib=${gpu_free_mib//[[:space:]]/}
if [[ "$gpu_compute_cap" != "12.0" ]]; then
  echo "REFUSED: GPU $gpu_index ($gpu_name) has compute capability $gpu_compute_cap; the qualified profile requires SM120 / 12.0" >&2
  exit 78
fi
if (( gpu_free_mib < 90000 )); then
  echo "REFUSED: GPU free memory ${gpu_free_mib} MiB is below 90000 MiB" >&2
  exit 78
fi
mapfile -t compute_pids < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | sed '/^[[:space:]]*$/d'
)
if [[ ${#compute_pids[@]} -ne 0 ]]; then
  echo "REFUSED: another GPU compute process is active" >&2
  exit 78
fi

mem_available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
swap_free_kib=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)
if (( mem_available_kib < 40 * 1024 * 1024 )); then
  echo "REFUSED: MemAvailable is below 40 GiB" >&2
  exit 78
fi
if (( swap_free_kib < 2 * 1024 * 1024 )); then
  echo "REFUSED: SwapFree is below 2 GiB" >&2
  exit 78
fi
if (( mem_available_kib + swap_free_kib < 56 * 1024 * 1024 )); then
  echo "REFUSED: combined MemAvailable plus SwapFree is below 56 GiB" >&2
  exit 78
fi
if command -v ss >/dev/null 2>&1 && ss -H -ltn 'sport = :8002' | grep -q .; then
  echo "REFUSED: TCP port 8002 is already listening" >&2
  exit 78
fi

docker run \
  --name qwen38-flash-next-w4-ple-example \
  --log-driver local \
  --log-opt max-size=50m \
  --log-opt max-file=3 \
  --gpus all \
  --ipc=host \
  --network=host \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SGLANG_QWEN4_PLE_W4=1 \
  -e SGLANG_QWEN4_PLE_W4_CHUNK_ROWS=262144 \
  -e SGLANG_QWEN4_PLE_W4_SIDECAR=/sidecars/ple-w4 \
  -e SGLANG_QWEN4_PLE_W4_SIDECAR_CHUNK_BYTES=67108864 \
  -e SGLANG_QWEN4_PLE_W4_SIDECAR_FALLBACK=0 \
  -e SGLANG_QWEN4_PLE_W4_SIDECAR_MANIFEST_SHA256="$manifest_sha" \
  -e SGLANG_QWEN4_PLE_W4_SOURCE_INDEX_SHA256="$index_sha" \
  -e SGLANG_QWEN4_PLE_W4_SOURCE_CONFIG_SHA256="$config_sha" \
  -e SGLANG_QWEN4_PLE_W4_SOURCE_TREE_SHA256="$tree_sha" \
  -e SGLANG_QWEN4_PLE_W4_SOURCE_PLE_SHA256="$raw_ple_sha" \
  -e SGLANG_QWEN4_PLE_W4_SOURCE_REVISION="$model_revision" \
  -v "$MODEL_DIR:/models/flash-next:ro" \
  -v "$SIDECAR_DIR:/sidecars/ple-w4:ro" \
  -v "$qwen_file:/sgl-workspace/sglang/python/sglang/srt/models/qwen4_exp.py:ro" \
  -v "$loader_file:/sgl-workspace/sglang/python/sglang/srt/models/qwen4_ple_w4_sidecar.py:ro" \
  -v "$qsa_file:/sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py:ro" \
  "$image" \
  sglang serve \
    --model-path /models/flash-next \
    --served-model-name qwen3.8-flash-next \
    --host 127.0.0.1 \
    --port 8002 \
    --tp 1 \
    --load-format safetensors \
    --quantization modelopt_fp4 \
    --ple-offload-embedding \
    --context-length 131072 \
    --kv-cache-dtype bfloat16 \
    --max-total-tokens 163072 \
    --max-running-requests 4 \
    --max-queued-requests 16 \
    --chunked-prefill-size 1024 \
    --linear-attn-prefill-backend flashinfer \
    --linear-attn-decode-backend triton \
    --mamba-ssm-dtype float32 \
    --reasoning-parser auto \
    --tool-call-parser auto \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --mem-fraction-static 0.94 \
    --cuda-graph-config '{"decode":{"backend":"full","max_bs":4,"bs":[1,2,4]},"prefill":{"backend":"disabled"}}' \
    --disable-radix-cache
