#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_MPS_PIPE_DIRECTORY:=/tmp/nvidia-mps}"
: "${CUDA_MPS_LOG_DIRECTORY:=/tmp/nvidia-log}"
: "${CUDA_MPS_STATIC_PARTITION_CHUNKS:=9}"
: "${CUDA_MPS_STATIC_PARTITION_COUNT:=2}"

mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
rm -f \
  "${CUDA_MPS_PIPE_DIRECTORY}/control" \
  "${CUDA_MPS_PIPE_DIRECTORY}/nvidia-cuda-mps-control.pid" \
  "${CUDA_MPS_PIPE_DIRECTORY}"/partition-*

stop_mps() {
  echo quit | nvidia-cuda-mps-control || true
}

nvidia-cuda-mps-control -d -S
trap stop_mps EXIT INT TERM

for _ in $(seq 1 30); do
  if [ -S "${CUDA_MPS_PIPE_DIRECTORY}/control" ]; then
    break
  fi
  sleep 1
done

if [ ! -S "${CUDA_MPS_PIPE_DIRECTORY}/control" ]; then
  echo "MPS control socket did not become ready" >&2
  exit 1
fi

gpu_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n1)"
if [ -z "${gpu_uuid}" ]; then
  echo "Could not determine GPU UUID for static MPS partitioning" >&2
  exit 1
fi

echo "Starting static MPS partitions on ${gpu_uuid}: count=${CUDA_MPS_STATIC_PARTITION_COUNT}, chunks=${CUDA_MPS_STATIC_PARTITION_CHUNKS}"

for index in $(seq 1 "${CUDA_MPS_STATIC_PARTITION_COUNT}"); do
  created="$(
    printf 'sm_partition add %s %s\n' "${gpu_uuid}" "${CUDA_MPS_STATIC_PARTITION_CHUNKS}" \
      | nvidia-cuda-mps-control
  )"
  partition="$(
    printf '%s\n' "${created}" \
      | awk '/^Partition / { sub(/^Partition /, ""); sub(/ created$/, ""); print; exit }'
  )"
  if [ -z "${partition}" ]; then
    echo "Failed to create MPS static partition ${index}: ${created}" >&2
    exit 1
  fi
  printf '%s\n' "${partition}" > "${CUDA_MPS_PIPE_DIRECTORY}/partition-${index}"
  chmod 0666 "${CUDA_MPS_PIPE_DIRECTORY}/partition-${index}"
  echo "Created MPS static partition ${index}: ${partition}"
done

printf 'lspart\n' | nvidia-cuda-mps-control || true

while [ -S "${CUDA_MPS_PIPE_DIRECTORY}/control" ]; do
  sleep 5
done
