#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PORT_DIR}/../.." && pwd)"

KV_CACHE_TYPE="continuous"
OUT=""
MAX_BATCH_SIZE=256
MAX_BEAM_WIDTH=5
MAX_NEW_TOKENS=32
MAX_INPUT_LEN=128
MAX_ENCODER_INPUT_LEN=4096
OPT_NUM_TOKENS_ENCODER=$((MAX_BATCH_SIZE * MAX_INPUT_LEN))
OPT_NUM_TOKENS_DECODER=$((MAX_BATCH_SIZE * MAX_NEW_TOKENS))
BUILD_IN_DOCKER="${INDICXLIT_BUILD_IN_DOCKER:-1}"
BUILD_IMAGE="${INDICXLIT_ENGINE_BUILD_IMAGE:-indicxlit-trtllm:b256-fp16-continuous-kv}"
BUILD_WITH_MPS="${INDICXLIT_BUILD_WITH_MPS:-0}"
MPS_PARTITION_INDEX="${INDICXLIT_MPS_PARTITION_INDEX:-1}"

usage() {
  cat <<USAGE
Usage:
  $0 [--kv-cache-type continuous|paged] [--output-dir PATH]

Build Rust-runtime-compatible FP16 beam-5 b256 TensorRT-LLM engines.

Defaults:
  --kv-cache-type ${KV_CACHE_TYPE}
  --output-dir inference_experiment/trtllm_port/artifacts/trtllm_engines_en_hi_beam5_host_fp16_b256_continuous_kv

Notes:
  - By default this runs trtllm-build inside ${BUILD_IMAGE}.
  - That image must contain the patched TensorRT-LLM runtime/plugin build.
  - continuous KV is faster for the direct fixed-batch runner in this workspace.
  - paged KV is still the expected choice for Triton inflight batching.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kv-cache-type)
      KV_CACHE_TYPE="$2"
      shift 2
      ;;
    --output-dir)
      OUT="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${BUILD_IN_DOCKER}" == "1" && -z "${INDICXLIT_BUILD_IN_DOCKER_REENTRY:-}" ]]; then
  docker_args=(--kv-cache-type "${KV_CACHE_TYPE}")
  docker_volumes=()
  if [[ "${BUILD_WITH_MPS}" == "1" ]]; then
    mps_source_container="${INDICXLIT_MPS_SOURCE_CONTAINER:-indicxlit-b256-inference-1-1}"
    docker_volumes+=(--volumes-from "${mps_source_container}")
  fi
  if [[ -n "${OUT}" ]]; then
    container_out="${OUT}"
    if [[ "${OUT}" == "${REPO_ROOT}"* ]]; then
      container_out="/workspace${OUT#${REPO_ROOT}}"
    fi
    docker_args+=(--output-dir "${container_out}")
  fi
  exec sudo docker run --rm --gpus all \
    --entrypoint bash \
    -e INDICXLIT_BUILD_IN_CONTAINER=1 \
    -e INDICXLIT_BUILD_IN_DOCKER_REENTRY=1 \
    -e INDICXLIT_BUILD_IN_DOCKER=1 \
    -e INDICXLIT_BUILD_WITH_MPS="${BUILD_WITH_MPS}" \
    -e INDICXLIT_MPS_PARTITION_INDEX="${MPS_PARTITION_INDEX}" \
    -e INDICXLIT_ENGINE_BUILD_IMAGE="${BUILD_IMAGE}" \
    -v "${REPO_ROOT}:/workspace" \
    "${docker_volumes[@]}" \
    -w /workspace \
    "${BUILD_IMAGE}" \
    -lc "bash 'inference_experiment/trtllm_port/scripts/build_host_b256_engine.sh' ${docker_args[*]@Q}"
fi

if [[ "${KV_CACHE_TYPE}" != "continuous" && "${KV_CACHE_TYPE}" != "paged" ]]; then
  echo "--kv-cache-type must be 'continuous' or 'paged'" >&2
  exit 2
fi

if [[ -z "${OUT}" ]]; then
  suffix=""
  if [[ "${KV_CACHE_TYPE}" == "continuous" ]]; then
    suffix="_continuous_kv"
  fi
  OUT="${PORT_DIR}/artifacts/trtllm_engines_en_hi_beam5_host_fp16_b256${suffix}"
fi

CHECKPOINT="${PORT_DIR}/artifacts/trtllm_checkpoint_en_hi_fp16"
if [[ ! -d "${CHECKPOINT}/encoder" || ! -d "${CHECKPOINT}/decoder" ]]; then
  echo "Missing FP16 checkpoint at ${CHECKPOINT}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${PORT_DIR}/env_trtllm.sh"

if [[ "${BUILD_WITH_MPS}" == "1" ]]; then
  export CUDA_MPS_PARTITION_INDEX="${MPS_PARTITION_INDEX}"
  # shellcheck source=/dev/null
  source "${PORT_DIR}/docker_b256/static-mps-client-env.sh"
fi

mkdir -p "${OUT}"

echo "Building encoder -> ${OUT}/encoder"
trtllm-build \
  --checkpoint_dir "${CHECKPOINT}/encoder" \
  --output_dir "${OUT}/encoder" \
  --max_batch_size "${MAX_BATCH_SIZE}" \
  --max_input_len "${MAX_INPUT_LEN}" \
  --max_seq_len "${MAX_INPUT_LEN}" \
  --max_beam_width "${MAX_BEAM_WIDTH}" \
  --max_num_tokens $((MAX_BATCH_SIZE * MAX_INPUT_LEN)) \
  --opt_num_tokens "${OPT_NUM_TOKENS_ENCODER}" \
  --kv_cache_type paged \
  --remove_input_padding enable \
  --bert_attention_plugin auto

echo "Building decoder (${KV_CACHE_TYPE} KV) -> ${OUT}/decoder"
trtllm-build \
  --checkpoint_dir "${CHECKPOINT}/decoder" \
  --output_dir "${OUT}/decoder" \
  --max_batch_size "${MAX_BATCH_SIZE}" \
  --max_input_len 1 \
  --max_seq_len $((MAX_NEW_TOKENS * 2)) \
  --max_encoder_input_len "${MAX_ENCODER_INPUT_LEN}" \
  --max_beam_width "${MAX_BEAM_WIDTH}" \
  --max_num_tokens 8192 \
  --opt_num_tokens "${OPT_NUM_TOKENS_DECODER}" \
  --kv_cache_type "${KV_CACHE_TYPE}" \
  --remove_input_padding enable \
  --gpt_attention_plugin auto

printf 'Built host engines:\n  %s\n' "${OUT}"
