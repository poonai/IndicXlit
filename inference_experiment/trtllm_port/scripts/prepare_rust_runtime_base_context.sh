#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PORT_DIR}/../.." && pwd)"
DOCKER_DIR="${PORT_DIR}/docker_b256"
SOURCE_CTX="${DOCKER_DIR}/context_cont"
CTX="${INDICXLIT_RUNTIME_BASE_CONTEXT:-${DOCKER_DIR}/context_runtime}"

ENGINE_SRC="${INDICXLIT_ENGINE_DIR:-${PORT_DIR}/artifacts/trtllm_engines_en_hi_beam5_runtime_fp16_b256_continuous_kv_fmha_static_mps_36sm_runtime_patchedbuild_36sm}"
MODEL_ASSETS_SRC="${INDICXLIT_MODEL_ASSETS:-${REPO_ROOT}/app/ai4bharat/transliteration/transformer/models/en2indic}"
PATCHED_LIB_DIR="${INDICXLIT_PATCHED_TRTLLM_LIB_DIR:-${SOURCE_CTX}/patched_libs_flat}"

required_libs=(
  libtensorrt_llm.so
  libnvinfer_plugin_tensorrt_llm.so
  libth_common.so
  libdecoder_attention_0.so
  libdecoder_attention_1.so
)

usage() {
  cat <<USAGE
Usage:
  $0 [--engine-dir PATH] [--assets-dir PATH] [--patched-lib-dir PATH]

Prepare the Docker context for indicxlit-trtllm:b256-fp16-continuous-kv.

Defaults:
  --engine-dir      ${ENGINE_SRC}
  --assets-dir      ${MODEL_ASSETS_SRC}
  --patched-lib-dir ${PATCHED_LIB_DIR}

The patched lib directory must contain all TensorRT-LLM runtime libraries that
will be overlaid into the base image.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine-dir)
      ENGINE_SRC="$2"
      shift 2
      ;;
    --assets-dir)
      MODEL_ASSETS_SRC="$2"
      shift 2
      ;;
    --patched-lib-dir)
      PATCHED_LIB_DIR="$2"
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

if [[ ! -d "${ENGINE_SRC}/encoder" || ! -d "${ENGINE_SRC}/decoder" ]]; then
  echo "Missing engine artifact at ${ENGINE_SRC}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_ASSETS_SRC}/lang_list.txt" ]]; then
  echo "Missing model assets at ${MODEL_ASSETS_SRC}" >&2
  exit 1
fi

for lib in "${required_libs[@]}"; do
  if [[ ! -f "${PATCHED_LIB_DIR}/${lib}" ]]; then
    echo "Missing patched runtime library: ${PATCHED_LIB_DIR}/${lib}" >&2
    exit 1
  fi
done

rm -rf "${CTX}"
mkdir -p "${CTX}/engines" "${CTX}/assets" "${CTX}/patched_libs_flat" "${CTX}/scripts"

cp "${SOURCE_CTX}/Dockerfile" "${CTX}/Dockerfile"
cp "${SOURCE_CTX}/scripts/server.py" "${CTX}/scripts/server.py"
cp -a "${ENGINE_SRC}/." "${CTX}/engines/"
cp -a "${MODEL_ASSETS_SRC}/." "${CTX}/assets/"
for lib in "${required_libs[@]}"; do
  cp "${PATCHED_LIB_DIR}/${lib}" "${CTX}/patched_libs_flat/${lib}"
done

printf 'Prepared runtime base context:\n  %s\n\nBuild:\n  docker build -t indicxlit-trtllm:b256-fp16-continuous-kv "%s"\n' "${CTX}" "${CTX}"
