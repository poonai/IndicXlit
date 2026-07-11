#!/usr/bin/env bash
set -euo pipefail
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
rm -f "${CUDA_MPS_PIPE_DIRECTORY}/control"
nvidia-cuda-mps-control -d
trap 'echo quit | nvidia-cuda-mps-control || true' EXIT INT TERM
while kill -0 "$(cat "${CUDA_MPS_PIPE_DIRECTORY}/nvidia-cuda-mps-control.pid")" 2>/dev/null; do
  sleep 5
done
