#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_MPS_PIPE_DIRECTORY:=/tmp/nvidia-mps}"
: "${CUDA_MPS_PARTITION_INDEX:?CUDA_MPS_PARTITION_INDEX must be set}"

partition_file="${CUDA_MPS_PIPE_DIRECTORY}/partition-${CUDA_MPS_PARTITION_INDEX}"

for _ in $(seq 1 180); do
  if [ -s "${partition_file}" ]; then
    break
  fi
  sleep 1
done

if [ ! -s "${partition_file}" ]; then
  echo "MPS static partition file was not created: ${partition_file}" >&2
  exit 1
fi

export CUDA_MPS_SM_PARTITION="$(cat "${partition_file}")"
unset CUDA_MPS_ACTIVE_THREAD_PERCENTAGE

echo "Using MPS static partition ${CUDA_MPS_PARTITION_INDEX}: ${CUDA_MPS_SM_PARTITION}"
