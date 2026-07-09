#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
MODEL_REPOSITORY="${REPO_ROOT}/inference experiment/trtllm_port/triton_indicxlit/model_repository"

TRITON_HOME="${TRITON_HOME:-/tmp/tritonserver-2.70.0/tritonserver}"
TRITON_BIN="${TRITON_BIN:-${TRITON_HOME}/bin/tritonserver}"
TRTLLM_VENV="${TRTLLM_VENV:-${REPO_ROOT}/.venv-trtllm}"
TRTLLM_SITE="${TRTLLM_SITE:-${TRTLLM_VENV}/lib/python3.10/site-packages}"

if [[ ! -x "${TRITON_BIN}" ]]; then
  TRITON_BIN="$(command -v tritonserver)"
fi

TRITON_ROOT="$(cd "$(dirname "${TRITON_BIN}")/.." && pwd)"

export PYTHONHOME="${PYTHONHOME:-/usr}"
export PYTHONPATH="${PYTHONPATH:-/usr/local/lib/python3.12/dist-packages:/usr/lib/python3.12:/usr/lib/python3.12/lib-dynload:/usr/lib/python3/dist-packages}"
export LD_LIBRARY_PATH="${TRTLLM_SITE}/tensorrt_llm/libs:${TRTLLM_SITE}/torch/lib:${TRTLLM_SITE}/nvidia/cuda_runtime/lib:${TRTLLM_SITE}/nvidia/cu13/lib:${TRTLLM_SITE}/nvidia/cudnn/lib:${TRTLLM_SITE}/nvidia/cublas/lib:${TRTLLM_SITE}/nvidia/nccl/lib:/usr/lib/x86_64-linux-gnu:/tmp/tritonserver-2.70.0/cuda13/nvidia/cu13/lib:/tmp/tritonserver-2.70.0/compat/root/usr/lib/x86_64-linux-gnu:/tmp/tritonserver-2.70.0/dcgm/root/usr/lib/x86_64-linux-gnu:/tmp/tritonserver-2.70.0/libarchive/root/usr/lib/x86_64-linux-gnu:${TRITON_ROOT}/backends/python:${TRITON_ROOT}/lib64:${LD_LIBRARY_PATH:-}"

ARGS=(
  --model-repository "${MODEL_REPOSITORY}"
  --backend-directory "${TRITON_ROOT}/backends"
  --repoagent-directory "${TRITON_ROOT}/repoagents"
  --cache-directory "${TRITON_ROOT}/caches"
  --http-port "${HTTP_PORT:-8010}"
  --metrics-port "${METRICS_PORT:-8012}"
  --allow-grpc "${ALLOW_GRPC:-false}"
)

if [[ -n "${LOAD_MODEL:-}" ]]; then
  ARGS+=(--model-control-mode explicit --load-model "${LOAD_MODEL}")
fi

exec "${TRITON_BIN}" "${ARGS[@]}"
