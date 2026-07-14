#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-indicxlit-rust-inference:continuous-decoder-kv}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/tritonserver:26.02-trtllm-python-py3}"
ENGINE_DIR="${ENGINE_DIR:-${PROJECT_DIR}/engines/en_hi_beam5_fp16_b256_continuous_decoder_kv}"
MODEL_ASSET_DIR="${MODEL_ASSET_DIR:-${REPO_ROOT}/app/ai4bharat/transliteration/transformer/models/en2indic}"
RESCORE_DICTS_URL="${RESCORE_DICTS_URL:-https://github.com/AI4Bharat/IndicXlit/releases/download/v1.0/word_prob_dicts.zip}"
RESCORE_DICTS_ARCHIVE="${RESCORE_DICTS_ARCHIVE:-${PROJECT_DIR}/artifacts/downloads/word_prob_dicts.zip}"
RESCORE_DICTS_DIR="${RESCORE_DICTS_DIR:-${PROJECT_DIR}/artifacts/word_prob_dicts}"
RESCORE_LANGS="${RESCORE_LANGS:-hi}"
CONTEXT_DIR="${CONTEXT_DIR:-$(mktemp -d /tmp/indicxlit-rust-executor-context.XXXXXX)}"
TRTLLM_VERSION="${TRTLLM_VERSION:-v1.1.0}"
TRTLLM_SOURCE_URL="${TRTLLM_SOURCE_URL:-https://github.com/NVIDIA/TensorRT-LLM/archive/refs/tags/${TRTLLM_VERSION}.tar.gz}"
TRTLLM_SOURCE_ARCHIVE="${TRTLLM_SOURCE_ARCHIVE:-${PROJECT_DIR}/artifacts/downloads/TensorRT-LLM-${TRTLLM_VERSION}.tar.gz}"
TRTLLM_SOURCE_ROOT="${TRTLLM_SOURCE_ROOT:-${PROJECT_DIR}/artifacts/tensorrt_llm_source}"
TRTLLM_SOURCE_DIR="${TRTLLM_SOURCE_DIR:-${TRTLLM_SOURCE_ROOT}/TensorRT-LLM-${TRTLLM_VERSION#v}}"
CLEAN_CONTEXT=0
if [[ "${CONTEXT_DIR}" == /tmp/indicxlit-rust-executor-context.* ]]; then
  CLEAN_CONTEXT=1
fi

if [[ -n "${DOCKER_CMD:-}" ]]; then
  read -r -a DOCKER_BIN <<< "${DOCKER_CMD}"
else
  DOCKER_BIN=(docker)
fi

if ! "${DOCKER_BIN[@]}" info >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1; then
  DOCKER_BIN=(sudo docker)
fi

cleanup() {
  if [[ "${CLEAN_CONTEXT}" == "1" ]]; then
    rm -rf "${CONTEXT_DIR}"
  fi
}
trap cleanup EXIT

if [[ ! -f "${ENGINE_DIR}/encoder/rank0.engine" ]] || [[ ! -f "${ENGINE_DIR}/decoder/rank0.engine" ]]; then
  echo "Missing engine files in ${ENGINE_DIR}" >&2
  echo "Run: bash scripts/download_checkpoint_and_build_engine.sh" >&2
  exit 1
fi

if [[ ! -f "${MODEL_ASSET_DIR}/lang_list.txt" ]]; then
  echo "Missing model assets at ${MODEL_ASSET_DIR}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_ASSET_DIR}/v1.0/corpus-bin" ]]; then
  echo "Missing fairseq dictionaries at ${MODEL_ASSET_DIR}/v1.0/corpus-bin" >&2
  exit 1
fi

if [[ ! -d "${RESCORE_DICTS_DIR}" ]]; then
  mkdir -p "$(dirname "${RESCORE_DICTS_ARCHIVE}")" "${PROJECT_DIR}/artifacts"
  if [[ ! -f "${RESCORE_DICTS_ARCHIVE}" ]]; then
    echo "Downloading rescoring dictionaries from ${RESCORE_DICTS_URL}"
    if command -v curl >/dev/null 2>&1; then
      curl -fL "${RESCORE_DICTS_URL}" -o "${RESCORE_DICTS_ARCHIVE}"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "${RESCORE_DICTS_ARCHIVE}" "${RESCORE_DICTS_URL}"
    else
      echo "Missing downloader: install curl or wget." >&2
      exit 1
    fi
  fi
  python3 -m zipfile -e "${RESCORE_DICTS_ARCHIVE}" "${PROJECT_DIR}/artifacts"
fi

if [[ ! -d "${RESCORE_DICTS_DIR}" ]]; then
  echo "Missing rescoring dictionaries at ${RESCORE_DICTS_DIR}" >&2
  exit 1
fi

if [[ ! -d "${TRTLLM_SOURCE_DIR}/cpp/include" ]]; then
  mkdir -p "$(dirname "${TRTLLM_SOURCE_ARCHIVE}")" "${TRTLLM_SOURCE_ROOT}"
  if [[ ! -f "${TRTLLM_SOURCE_ARCHIVE}" ]]; then
    echo "Downloading TensorRT-LLM source headers from ${TRTLLM_SOURCE_URL}"
    if command -v curl >/dev/null 2>&1; then
      curl -fL "${TRTLLM_SOURCE_URL}" -o "${TRTLLM_SOURCE_ARCHIVE}"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "${TRTLLM_SOURCE_ARCHIVE}" "${TRTLLM_SOURCE_URL}"
    else
      echo "Missing downloader: install curl or wget." >&2
      exit 1
    fi
  fi
  tar -xzf "${TRTLLM_SOURCE_ARCHIVE}" -C "${TRTLLM_SOURCE_ROOT}"
fi

if [[ ! -d "${TRTLLM_SOURCE_DIR}/cpp/include" ]]; then
  echo "Missing TensorRT-LLM source headers at ${TRTLLM_SOURCE_DIR}/cpp/include" >&2
  exit 1
fi

if [[ -e "${CONTEXT_DIR}" && "${CLEAN_CONTEXT}" != "1" ]]; then
  rm -rf "${CONTEXT_DIR}"
fi
mkdir -p "${CONTEXT_DIR}"
cp -a "${PROJECT_DIR}/rust_executor_server/." "${CONTEXT_DIR}/"
mkdir -p "${CONTEXT_DIR}/engine" "${CONTEXT_DIR}/assets"
cp -a "${ENGINE_DIR}/." "${CONTEXT_DIR}/engine/"
cp -a "${MODEL_ASSET_DIR}/lang_list.txt" "${CONTEXT_DIR}/assets/"
cp -a "${MODEL_ASSET_DIR}/v1.0/." "${CONTEXT_DIR}/assets/"
mkdir -p "${CONTEXT_DIR}/assets/word_prob_dicts"
IFS=',' read -r -a RESCORE_LANG_ARRAY <<< "${RESCORE_LANGS}"
for lang in "${RESCORE_LANG_ARRAY[@]}"; do
  lang="$(echo "${lang}" | xargs)"
  if [[ -z "${lang}" ]]; then
    continue
  fi
  dict_path="${RESCORE_DICTS_DIR}/${lang}_word_prob_dict.json"
  if [[ ! -f "${dict_path}" ]]; then
    echo "Missing rescoring dictionary for '${lang}' at ${dict_path}" >&2
    exit 1
  fi
  cp -a "${dict_path}" "${CONTEXT_DIR}/assets/word_prob_dicts/"
done
mkdir -p "${CONTEXT_DIR}/trtllm_src"
cp -a "${TRTLLM_SOURCE_DIR}/cpp" "${CONTEXT_DIR}/trtllm_src/cpp"

"${DOCKER_BIN[@]}" build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -t "${IMAGE_TAG}" \
  "${CONTEXT_DIR}"

echo "Built image: ${IMAGE_TAG}"
