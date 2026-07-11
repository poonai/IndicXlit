# IndicXlit b256 production stack

This deployment serves the FP16 beam-5, max-batch-256 encoder/decoder through a
complete Triton ensemble:

```text
client -> NGINX (least_conn) -> Triton replica 1/2
                                C++ preprocess
                                TensorRT-LLM
                                C++ postprocess
```

Both inference containers use NVIDIA MPS on one selected GPU. Prometheus
scrapes both Triton replicas, NGINX, and DCGM; Grafana provisions the
`IndicXlit b256 Production` dashboard. The public inference port is `8000`,
Prometheus is `9090`, and Grafana is `3000`.

## Prerequisites

- Linux NVIDIA driver compatible with the CUDA 13/Triton 25.06 images
- NVIDIA Container Toolkit configured for Docker
- Docker Compose with `gpus` support
- FP16 b256 engines at
  `artifacts/trtllm_engines_en_hi_beam5_triton_fp16_b256/{encoder,decoder}`
- complete `en2indic` assets, including dictionaries and `lang_list.txt`

The engines and model payloads are generated/downloaded artifacts and remain
outside Git by design.

## Build and run

From the repository root:

```bash
bash "inference experiment/trtllm_port/docker_b256/prepare_context.sh"
docker compose \
  -f "inference experiment/trtllm_port/docker_b256/docker-compose.yml" \
  build
docker compose \
  -f "inference experiment/trtllm_port/docker_b256/docker-compose.yml" \
  up -d
```

Select another GPU or override the per-client MPS thread cap if required. The
packaged engines are built with a 50% cap and must run with the same value so
TensorRT sees the same 36-SM A10 topology at build and runtime.

```bash
CUDA_VISIBLE_DEVICES=1 MPS_ACTIVE_THREAD_PERCENTAGE=50 docker compose \
  -f "inference experiment/trtllm_port/docker_b256/docker-compose.yml" up -d
```

MPS requires the inference and MPS containers to share the host PID/IPC
namespaces and the MPS pipe volume. Do not deploy this Compose file on an
untrusted shared Docker host.

## Verify

```bash
curl -f http://localhost:8000/v2/health/ready
python3 "inference experiment/trtllm_port/docker_b256/scripts/smoke_ensemble.py"
curl -f http://localhost:9090/-/ready
curl -f http://localhost:3000/api/health
```

The public model is `indicxlit_ensemble`. It accepts concurrent single-item
requests as documented in `CHAT_HANDOFF.md`; true client-side batches remain
limited by the non-decoupled TensorRT-LLM/static-ensemble contract.

The native postprocessor decodes beam candidates. Dictionary probability
rescoring is intentionally not performed in the native backend; keep
`rescore=false` (the production default). This avoids bringing JSON dictionaries
and per-request CPU sorting into the latency-sensitive path.

## Operations

```bash
docker compose -f "inference experiment/trtllm_port/docker_b256/docker-compose.yml" ps
docker compose -f "inference experiment/trtllm_port/docker_b256/docker-compose.yml" logs -f inference-1 inference-2
docker compose -f "inference experiment/trtllm_port/docker_b256/docker-compose.yml" down
```

Change `GRAFANA_ADMIN_PASSWORD` before exposing Grafana outside localhost.
The DCGM exporter needs `SYS_ADMIN`; remove the service and its Prometheus job
if GPU telemetry is not required or that capability is prohibited.
