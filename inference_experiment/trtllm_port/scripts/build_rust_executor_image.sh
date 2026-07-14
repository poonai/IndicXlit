#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PORT_DIR}/../.." && pwd)"
SERVER_DIR="${PORT_DIR}/docker_b256/rust_executor_server"

IMAGE_TAG="${INDICXLIT_RUST_EXECUTOR_IMAGE:-indicxlit-trtllm:rust-executor}"
BASE_IMAGE="${INDICXLIT_RUST_EXECUTOR_BASE_IMAGE:-indicxlit-trtllm:b256-fp16-continuous-kv}"
TRTLLM_ROOT="${INDICXLIT_TRTLLM_ROOT:-/tmp/TRTLLM-v1.1.0}"

usage() {
  cat <<USAGE
Usage:
  $0 [--image-tag TAG] [--base-image TAG] [--trtllm-root PATH]

Build the production Rust executor image from source.

Defaults:
  --image-tag   ${IMAGE_TAG}
  --base-image  ${BASE_IMAGE}
  --trtllm-root ${TRTLLM_ROOT}

The TensorRT-LLM root must match the base image runtime libraries at the C++
ABI level. Do not point this at a different patched checkout unless the base
image was built from that same checkout.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image-tag)
      IMAGE_TAG="$2"
      shift 2
      ;;
    --base-image)
      BASE_IMAGE="$2"
      shift 2
      ;;
    --trtllm-root)
      TRTLLM_ROOT="$2"
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

if [[ ! -d "${TRTLLM_ROOT}/cpp/include/tensorrt_llm" ]]; then
  echo "Invalid TensorRT-LLM source root: ${TRTLLM_ROOT}" >&2
  echo "Expected: ${TRTLLM_ROOT}/cpp/include/tensorrt_llm" >&2
  exit 1
fi

if git -C "${TRTLLM_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "TensorRT-LLM source: ${TRTLLM_ROOT}"
  git -C "${TRTLLM_ROOT}" log -1 --oneline
fi

export DOCKER_BUILDKIT=1

docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then
  docker_cmd=(sudo docker)
fi

"${docker_cmd[@]}" buildx build \
  --load \
  --build-context "trtllm_src=${TRTLLM_ROOT}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -t "${IMAGE_TAG}" \
  -f "${SERVER_DIR}/Dockerfile" \
  "${SERVER_DIR}"

echo "Built Rust executor image: ${IMAGE_TAG}"
