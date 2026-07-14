# Rust Inference Engine Setup

This folder is the portable setup path for building the IndicXlit TensorRT-LLM
engine from a released converted checkpoint using the stock TensorRT-LLM Docker
image. The direct C++ runner currently expects continuous KV bindings for the
decoder, so the build uses paged KV for the encoder and continuous KV for the
decoder.

## Build Engine

From the repo root:

```bash
cd rust_inference
bash scripts/download_checkpoint_and_build_engine.sh
```

The script:

- downloads `indicxlit-trtllm-checkpoint-en-hi-fp16.tar.gz` from the GitHub release
- extracts it into `rust_inference/checkpoints/trtllm_checkpoint_en_hi_fp16/`
- runs `trtllm-build` inside `nvcr.io/nvidia/tritonserver:26.02-trtllm-python-py3`
- writes the engine to `rust_inference/engines/en_hi_beam5_fp16_b256_continuous_decoder_kv/`

Generated checkpoints, downloaded archives, and engines are intentionally not
committed.

## Build Runtime Image

After the engine exists, build the Rust/C++ executor image with the engine baked
in:

```bash
cd rust_inference
bash scripts/build_rust_executor_image.sh
```

Default image tag:

```text
indicxlit-rust-inference:continuous-decoder-kv
```

## Run Stack

After the image is built, bring up inference, Prometheus, and Grafana:

```bash
cd rust_inference
sudo docker compose up
```

Exposed services:

- inference API: `http://localhost:8000/v2/models/indicxlit/infer`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

Grafana credentials default to `admin` / `admin`. The dashboard is provisioned
under the `IndicXlit` folder.

## Dakshina Evaluation

With the inference stack running, evaluate Hindi Dakshina through the Rust HTTP
runtime:

```bash
cd rust_inference
python3 scripts/evaluate_dakshina.py --download-dakshina
```

Quick smoke run:

```bash
python3 scripts/evaluate_dakshina.py --download-dakshina --limit 512
```

The default output is:

```text
rust_inference/artifacts/dakshina_hi_rust_http_eval.json
```

Useful overrides:

```bash
python3 scripts/evaluate_dakshina.py \
  --url http://127.0.0.1:8000/v2/models/indicxlit/infer \
  --dakshina-tsv /path/to/hi.translit.sampled.test.tsv \
  --batch-size 256 \
  --beam-width 5 \
  --topk 5 \
  --save-rows
```

## Overrides

Useful environment overrides:

```bash
BUILD_IMAGE=nvcr.io/nvidia/tritonserver:26.02-trtllm-python-py3 \
ENGINE_DIR="$PWD/engines/custom_engine" \
bash scripts/download_checkpoint_and_build_engine.sh
```

For the runtime image:

```bash
IMAGE_TAG=indicxlit-rust-inference:test \
bash scripts/build_rust_executor_image.sh
```

For the compose stack:

```bash
INDICXLIT_IMAGE=indicxlit-rust-inference:test \
INDICXLIT_PORT=8000 \
GRAFANA_PORT=3000 \
PROMETHEUS_PORT=9090 \
sudo docker compose up
```

The default release asset URL is:

```text
https://github.com/poonai/IndicXlit/releases/download/trtllm-checkpoint-en-hi-fp16-v1/indicxlit-trtllm-checkpoint-en-hi-fp16.tar.gz
```
