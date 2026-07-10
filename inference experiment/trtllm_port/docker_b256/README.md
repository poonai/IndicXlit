# IndicXlit TensorRT-LLM b256 Docker image

This packages the FP16 beam-5 max-batch-256 TensorRT-LLM Triton core model for
remote load testing.

It serves only the core `indicxlit_tensorrt_llm` model, not the Python
preprocess/postprocess ensemble. The included k6 script sends pre-tokenized
Triton HTTP inference requests with `beam_width=5`, `num_return_sequences=5`,
and `cross_attention_mask` populated.

## Build context

From the repo root:

```bash
bash "inference experiment/trtllm_port/docker_b256/prepare_context.sh"
```

This creates:

```text
inference experiment/trtllm_port/docker_b256/context/
```

## Build image

```bash
docker build \
  -t indicxlit-trtllm:b256-fp16 \
  "inference experiment/trtllm_port/docker_b256/context"
```

The Dockerfile defaults to:

```text
nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3
```

If your target environment uses a different TensorRT-LLM/Triton container, pass:

```bash
docker build \
  --build-arg BASE_IMAGE=<your-triton-trtllm-image> \
  -t indicxlit-trtllm:b256-fp16 \
  "inference experiment/trtllm_port/docker_b256/context"
```

## Run server

```bash
docker run --rm --gpus all \
  -p 8000:8000 \
  -p 8002:8002 \
  indicxlit-trtllm:b256-fp16
```

Health:

```bash
curl -f http://localhost:8000/v2/health/ready
```

Smoke:

```bash
python3 "inference experiment/trtllm_port/docker_b256/context/scripts/smoke_core.py" \
  --url http://localhost:8000
```

## k6, 256 VUs

```bash
k6 run \
  -e URL=http://localhost:8000 \
  -e VUS=256 \
  -e DURATION=60s \
  "inference experiment/trtllm_port/docker_b256/context/k6/core_b256.js"
```

The k6 script posts to:

```text
/v2/models/indicxlit_tensorrt_llm/infer
```

