# IndicXlit Rust Executor Server

Rust owns HTTP, batching, metrics, and worker pinning. C++ owns the direct
TensorRT engine runner behind a small C ABI. The C++ bridge is built from
source as part of the Docker image; production builds must not copy a stale
`cpp_build_*` artifact from the worktree.

## Required Inputs

- TensorRT-LLM source tree matching the runtime image ABI:
  `/tmp/TRTLLM-v1.1.0`
- Patched TensorRT-LLM runtime/base image:
  `indicxlit-trtllm:b256-fp16-continuous-kv`
- Engine artifact mounted at runtime:
  `artifacts/trtllm_engines_en_hi_beam5_runtime_fp16_b256_continuous_kv_fmha_static_mps_36sm_runtime_patchedbuild_36sm`

The source tree and base image must match at the C++ ABI level. A test build
against `/home/ubuntu/TensorRT-LLM` failed at process start with an unresolved
`tensorrt_llm::common::TllmException` symbol, so that checkout cannot be used
with the current `indicxlit-trtllm:b256-fp16-continuous-kv` runtime image until
the base image is rebuilt from the same source tree.

## Build

From the repository root:

```bash
"inference_experiment/trtllm_port/scripts/build_rust_executor_image.sh" \
  --trtllm-root /tmp/TRTLLM-v1.1.0 \
  --base-image indicxlit-trtllm:b256-fp16-continuous-kv \
  --image-tag indicxlit-trtllm:rust-executor
```

The script uses Docker BuildKit named contexts:

```text
rust_executor_server/ -> Rust and C++ bridge source
/tmp/TRTLLM-v1.1.0    -> TensorRT-LLM headers/source matching runtime ABI
base image            -> TensorRT-LLM runtime libraries
```

## Run

Use the compose stack in `docker_b256`:

```bash
docker compose \
  -f "inference_experiment/trtllm_port/docker_b256/docker-compose.yml" \
  up -d
```

Override the executor image when testing a new tag:

```bash
INDICXLIT_RUST_EXECUTOR_IMAGE=indicxlit-trtllm:rust-executor-test \
docker compose -f "inference_experiment/trtllm_port/docker_b256/docker-compose.yml" up -d
```

## Validate

```bash
curl -f http://127.0.0.1:8000/v2/health/ready
curl -sS http://127.0.0.1:8000/metrics | grep indicxlit_http_requests_total
```

Smoke inference:

```bash
curl -sS -X POST http://127.0.0.1:8000/v2/models/indicxlit/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[{"name":"text_input","shape":[1,1],"datatype":"BYTES","data":["bharat"]},{"name":"target_lang","shape":[1,1],"datatype":"BYTES","data":["hi"]},{"name":"max_tokens","shape":[1,1],"datatype":"INT32","data":[32]},{"name":"beam_width","shape":[1,1],"datatype":"INT32","data":[5]},{"name":"topk","shape":[1,1],"datatype":"INT32","data":[5]}]}'
```

Expected top output is `भारत`.
