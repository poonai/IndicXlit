# IndicXlit b256 production stack

This deployment serves the FP16 beam-5, max-batch-256 encoder/decoder through
the Rust HTTP batch server backed by the direct TensorRT C++ runtime:

```text
client -> NGINX on :8000
       -> inference-1 Rust/direct-C++ server -> one TensorRT-LLM worker -> MPS partition 1
       -> inference-2 Rust/direct-C++ server -> one TensorRT-LLM worker -> MPS partition 2
       -> NVIDIA static-partition MPS daemon on the shared GPU
```

The stack runs two Rust inference containers under NVIDIA MPS static SM
partitioning. Each container sets `INDICXLIT_WORKERS=1`, consumes from its own
local queue, and pins its single worker thread to one CPU core. The MPS service
starts `nvidia-cuda-mps-control -d -S`, creates two 9-chunk partitions on A10
(36 SM each), and writes the generated `CUDA_MPS_SM_PARTITION` values into the
shared MPS volume before the inference services start. The mounted engine
artifact was rebuilt inside the patched runtime image under the same static
36-SM MPS partition:

```text
artifacts/trtllm_engines_en_hi_beam5_runtime_fp16_b256_continuous_kv_fmha_static_mps_36sm_runtime_patchedbuild_36sm
```

Prometheus scrapes both
Rust `/metrics` endpoints and DCGM. Grafana provisions the
`IndicXlit b256 Rust Runtime` dashboard. The public inference port is `8000`,
Prometheus is `9090`, and Grafana is `3000`.

## Prerequisites

- Linux NVIDIA driver with MPS static partition support (`nvidia-cuda-mps-control -S`);
  this host was validated with NVIDIA server driver `595.71.05`
- NVIDIA Container Toolkit configured for Docker
- Docker Compose with `gpus` support
- Built local image `indicxlit-trtllm:rust-executor`
- TensorRT-LLM source tree matching the runtime image ABI at `/tmp/TRTLLM-v1.1.0`
- complete `en2indic` assets, including dictionaries and `lang_list.txt`

The engines and model payloads are generated/downloaded artifacts and remain
outside Git by design.

## Build

Prepare the runtime base image context. This copies the selected engine,
assets, and patched TensorRT-LLM runtime libraries into a generated Docker
context under `docker_b256/context_runtime`.

```bash
"inference_experiment/trtllm_port/scripts/prepare_rust_runtime_base_context.sh"
```

Build the runtime base image:

```bash
docker build \
  -t indicxlit-trtllm:b256-fp16-continuous-kv \
  "inference_experiment/trtllm_port/docker_b256/context_runtime"
```

Build the Rust executor image from source. This compiles the C++ bridge inside
Docker against the patched TensorRT-LLM source tree and the patched runtime
base image.

```bash
"inference_experiment/trtllm_port/scripts/build_rust_executor_image.sh" \
  --trtllm-root /tmp/TRTLLM-v1.1.0 \
  --base-image indicxlit-trtllm:b256-fp16-continuous-kv \
  --image-tag indicxlit-trtllm:rust-executor
```

Do not substitute `/home/ubuntu/TensorRT-LLM` with the current base image. That
checkout has the contiguous-KV patch commit, but it does not match the C++ ABI
exported by `indicxlit-trtllm:b256-fp16-continuous-kv`. To use it in
production, rebuild the base image from that same checkout first, then rebuild
the Rust executor image and engines from the resulting image.

Build or rebuild the b256 36-SM engine under static MPS:

```bash
INDICXLIT_BUILD_WITH_MPS=1 \
INDICXLIT_MPS_PARTITION_INDEX=1 \
"inference_experiment/trtllm_port/scripts/build_host_b256_engine.sh" \
  --output-dir "inference_experiment/trtllm_port/artifacts/trtllm_engines_en_hi_beam5_runtime_fp16_b256_continuous_kv_fmha_static_mps_36sm_runtime_patchedbuild_36sm"
```

## Run

From the repository root:

```bash
docker compose \
  -f "inference_experiment/trtllm_port/docker_b256/docker-compose.yml" \
  up -d
```

Select another GPU or override the static partition chunk count if required.

```bash
CUDA_VISIBLE_DEVICES=1 CUDA_MPS_STATIC_PARTITION_CHUNKS=9 docker compose \
  -f "inference_experiment/trtllm_port/docker_b256/docker-compose.yml" up -d
```

## Verify

```bash
curl -f http://localhost:8000/v2/health/ready
curl -f http://localhost:8000/nginx_status
curl -f http://localhost:9090/-/ready
curl -f http://localhost:3000/api/health
```

The public model endpoint is `/v2/models/indicxlit/infer`. It accepts both
single-item and client-side batched payloads in the Triton HTTP JSON shape.
The Rust executor exports request, queue, batching, and engine-call metrics via
`metrics-rs`/`metrics-exporter-prometheus`.

## Operations

```bash
docker compose -f "inference_experiment/trtllm_port/docker_b256/docker-compose.yml" ps
docker compose -f "inference_experiment/trtllm_port/docker_b256/docker-compose.yml" logs -f inference-1 inference-2 nginx mps
docker compose -f "inference_experiment/trtllm_port/docker_b256/docker-compose.yml" down
```

Change `GRAFANA_ADMIN_PASSWORD` before exposing Grafana outside localhost.
The DCGM exporter needs `SYS_ADMIN`; remove the service and its Prometheus job
if GPU telemetry is not required or that capability is prohibited.
