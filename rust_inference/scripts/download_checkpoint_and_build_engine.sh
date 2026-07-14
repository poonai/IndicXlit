#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RELEASE_TAG="${RELEASE_TAG:-trtllm-checkpoint-en-hi-fp16-v1}"
ASSET_NAME="${ASSET_NAME:-indicxlit-trtllm-checkpoint-en-hi-fp16.tar.gz}"
RELEASE_URL="${RELEASE_URL:-https://github.com/poonai/IndicXlit/releases/download/${RELEASE_TAG}/${ASSET_NAME}}"

DOWNLOAD_DIR="${DOWNLOAD_DIR:-${PROJECT_DIR}/artifacts/downloads}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${DOWNLOAD_DIR}/${ASSET_NAME}}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_DIR}/checkpoints}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${CHECKPOINT_ROOT}/trtllm_checkpoint_en_hi_fp16}"
ENGINE_DIR="${ENGINE_DIR:-${PROJECT_DIR}/engines/en_hi_beam5_fp16_b256_continuous_decoder_kv}"
ENGINE_NAME="$(basename "${ENGINE_DIR}")"
BUILD_IMAGE="${BUILD_IMAGE:-nvcr.io/nvidia/tritonserver:26.02-trtllm-python-py3}"

if [[ -n "${DOCKER_CMD:-}" ]]; then
  read -r -a DOCKER_BIN <<< "${DOCKER_CMD}"
else
  DOCKER_BIN=(docker)
fi

if ! "${DOCKER_BIN[@]}" info >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1; then
  DOCKER_BIN=(sudo docker)
fi

MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-256}"
MAX_BEAM_WIDTH="${MAX_BEAM_WIDTH:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-128}"
MAX_ENCODER_INPUT_LEN="${MAX_ENCODER_INPUT_LEN:-4096}"
OPT_NUM_TOKENS_ENCODER="${OPT_NUM_TOKENS_ENCODER:-$((MAX_BATCH_SIZE * MAX_INPUT_LEN))}"
OPT_NUM_TOKENS_DECODER="${OPT_NUM_TOKENS_DECODER:-$((MAX_BATCH_SIZE * MAX_NEW_TOKENS))}"

mkdir -p "${DOWNLOAD_DIR}" "${CHECKPOINT_ROOT}" "${ENGINE_DIR}"

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  echo "Downloading ${RELEASE_URL}"
  if command -v curl >/dev/null 2>&1; then
    curl -fL "${RELEASE_URL}" -o "${ARCHIVE_PATH}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${ARCHIVE_PATH}" "${RELEASE_URL}"
  else
    echo "Missing downloader: install curl or wget." >&2
    exit 1
  fi
else
  echo "Using cached archive ${ARCHIVE_PATH}"
fi

if [[ ! -f "${CHECKPOINT_DIR}/encoder/rank0.safetensors" ]] || [[ ! -f "${CHECKPOINT_DIR}/decoder/rank0.safetensors" ]]; then
  echo "Extracting checkpoint to ${CHECKPOINT_ROOT}"
  tar -xzf "${ARCHIVE_PATH}" -C "${CHECKPOINT_ROOT}"
fi

if [[ ! -f "${CHECKPOINT_DIR}/encoder/rank0.safetensors" ]] || [[ ! -f "${CHECKPOINT_DIR}/decoder/rank0.safetensors" ]]; then
  echo "Checkpoint is incomplete after extraction: ${CHECKPOINT_DIR}" >&2
  exit 1
fi

cat <<INFO
Building portable TensorRT-LLM engine:
  checkpoint: ${CHECKPOINT_DIR}
  output:     ${ENGINE_DIR}
  image:      ${BUILD_IMAGE}
INFO

"${DOCKER_BIN[@]}" run --rm --gpus all \
  --entrypoint bash \
  -v "${PROJECT_DIR}:/workspace/rust_inference" \
  -w /workspace/rust_inference \
  "${BUILD_IMAGE}" \
  -lc "
set -euo pipefail

CHECKPOINT='/workspace/rust_inference/checkpoints/trtllm_checkpoint_en_hi_fp16'
ENGINE='/workspace/rust_inference/engines/${ENGINE_NAME}'

python3 - <<'PY'
import site
print('Python site-packages:', site.getsitepackages()[0])
PY

echo 'Building encoder engine'
trtllm-build \
  --checkpoint_dir \"\${CHECKPOINT}/encoder\" \
  --output_dir \"\${ENGINE}/encoder\" \
  --max_batch_size '${MAX_BATCH_SIZE}' \
  --max_input_len '${MAX_INPUT_LEN}' \
  --max_seq_len '${MAX_INPUT_LEN}' \
  --max_beam_width '${MAX_BEAM_WIDTH}' \
  --max_num_tokens '$((MAX_BATCH_SIZE * MAX_INPUT_LEN))' \
  --opt_num_tokens '${OPT_NUM_TOKENS_ENCODER}' \
  --kv_cache_type paged \
  --remove_input_padding enable \
  --bert_attention_plugin auto

echo 'Building decoder engine'
trtllm-build \
  --checkpoint_dir \"\${CHECKPOINT}/decoder\" \
  --output_dir \"\${ENGINE}/decoder\" \
  --max_batch_size '${MAX_BATCH_SIZE}' \
  --max_input_len 1 \
  --max_seq_len '$((MAX_NEW_TOKENS * 2))' \
  --max_encoder_input_len '${MAX_ENCODER_INPUT_LEN}' \
  --max_beam_width '${MAX_BEAM_WIDTH}' \
  --max_num_tokens 8192 \
  --opt_num_tokens '${OPT_NUM_TOKENS_DECODER}' \
  --kv_cache_type continuous \
  --remove_input_padding enable \
  --gpt_attention_plugin auto
"

echo "Engine ready: ${ENGINE_DIR}"
