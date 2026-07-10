#!/usr/bin/env bash
set -euo pipefail

TRITON_BIN="${TRITON_BIN:-/opt/tritonserver/bin/tritonserver}"
MODEL_REPOSITORY="${MODEL_REPOSITORY:-/workspace/IndicXlit/inference experiment/trtllm_port/triton_indicxlit/model_repository}"
HTTP_PORT="${HTTP_PORT:-8000}"
METRICS_PORT="${METRICS_PORT:-8002}"
LOAD_MODEL="${LOAD_MODEL:-indicxlit_tensorrt_llm}"

if [[ ! -x "${TRITON_BIN}" ]]; then
  TRITON_BIN="$(command -v tritonserver)"
fi

exec "${TRITON_BIN}" \
  --model-repository "${MODEL_REPOSITORY}" \
  --http-port "${HTTP_PORT}" \
  --metrics-port "${METRICS_PORT}" \
  --allow-grpc=false \
  --model-control-mode explicit \
  --load-model "${LOAD_MODEL}"

