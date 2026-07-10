# IndicXlit TensorRT-LLM / Triton Handoff

Last updated: 2026-07-10

This file is the handoff for continuing the IndicXlit TensorRT-LLM port in a
fresh instance. It records the current verified state, exact artifact paths,
install/build requirements, commands, results, and known limitations.

## Current State

The core model port is working.

- Fairseq Hindi Dakshina README-level evaluation was verified.
- The Fairseq checkpoint weights were converted directly to TensorRT-LLM
  encoder/decoder checkpoints.
- TensorRT-LLM beam-5 Dakshina parity is achieved after fixing encoder
  attention masks.
- A local Triton model repository exists with custom Python pre/postprocess and
  TensorRT-LLM backend serving.
- The `tensorrtllm` Triton backend was built locally and tested.

Do not treat the early parts of this project as speculative anymore. The
important remaining questions are serving shape, operational packaging, and
optimization.

## Key Paths

Workspace:

```text
/IndicXlit/inference experiment/trtllm_port/
```

Main status files:

```text
PHASE_STATUS.md
CHAT_HANDOFF.md
README.md
```

Core TensorRT-LLM scripts:

```text
convert_indicxlit_to_trtllm.py
run_trtllm_greedy.py
compare_trtllm_parity.py
evaluate_dakshina_trtllm.py
evaluate_dakshina_fairseq.py
verify_trtllm_weight_port.py
probe_trtllm_env.py
```

Triton prototype:

```text
triton_indicxlit/
triton_indicxlit/README.md
triton_indicxlit/scripts/run_triton.sh
triton_indicxlit/scripts/dry_run_pipeline.py
triton_indicxlit/model_repository/
```

TensorRT-LLM environment:

```text
/IndicXlit/.venv-trtllm
/IndicXlit/inference experiment/trtllm_port/env_trtllm.sh
```

Fairseq verification environment:

```text
/IndicXlit/.venv-fairseq
```

## Git State to Expect

The Triton work is currently uncommitted in this workspace.

Expected changed/untracked files include:

```text
.gitignore
inference experiment/trtllm_port/PHASE_STATUS.md
inference experiment/trtllm_port/CHAT_HANDOFF.md
inference experiment/trtllm_port/triton_indicxlit/**
```

Generated engine artifacts are intentionally ignored and should not be committed:

```text
inference experiment/trtllm_port/artifacts/trtllm_engines_en_hi_beam5_triton/
```

## Environment That Worked

Current cloud machine used for successful TRT/Triton work:

- GPU: NVIDIA GeForce RTX 4090, 24 GB VRAM
- Driver: 580.159.04
- CUDA runtime capability reported around CUDA 13
- RAM: about 503 GiB
- OS: Ubuntu 22.04 style package environment

Isolated TensorRT-LLM Python environment:

- Python: 3.10.12
- `tensorrt-llm==1.2.1`
- `tensorrt==10.14.1.48.post1`
- `torch==2.9.1+cu128`
- `nvidia-cublas==13.6.0.2`

Activate it with:

```bash
source "inference experiment/trtllm_port/env_trtllm.sh"
```

Do not install TensorRT-LLM into the production/app `.venv`.

## Reinstall Notes for a Fresh Instance

Prefer recreating the isolated `.venv-trtllm` rather than modifying the app
environment.

High-level requirements:

- Python 3.10 with venv/dev headers
- large disk, at least 150 GB practical
- `tensorrt-llm==1.2.1`
- matching TensorRT Python and C++ libraries
- local Triton Server
- TensorRT-LLM Triton backend built from source or provided by a matching
  NVIDIA container/image

The previous successful TensorRT C++ package versions were pinned to match the
Python TensorRT wheel:

```text
libnvinfer10=10.14.1.48-1+cuda13.0
libnvinfer-plugin10=10.14.1.48-1+cuda13.0
libnvinfer-dev=10.14.1.48-1+cuda13.0
libnvinfer-plugin-dev=10.14.1.48-1+cuda13.0
libnvinfer-headers-dev=10.14.1.48-1+cuda13.0
libnvinfer-headers-plugin-dev=10.14.1.48-1+cuda13.0
```

Other build/runtime packages installed during the working session:

```text
cmake
rapidjson-dev
nlohmann-json3-dev
```

The venv also needed CMake:

```bash
python -m pip install "cmake==3.31.10"
```

Triton standalone used in the successful run:

```text
/tmp/tritonserver-2.70.0/tritonserver
server_version: 2.70.0
```

Extra runtime compatibility libraries were placed under:

```text
/tmp/tritonserver-2.70.0/cuda13
/tmp/tritonserver-2.70.0/compat
/tmp/tritonserver-2.70.0/dcgm
/tmp/tritonserver-2.70.0/libarchive
```

The launch script already wires these into `LD_LIBRARY_PATH`.

## TensorRT-LLM Checkpoints and Engines

Converted TensorRT-LLM checkpoint:

```text
artifacts/trtllm_checkpoint_en_hi/encoder/config.json
artifacts/trtllm_checkpoint_en_hi/encoder/rank0.safetensors
artifacts/trtllm_checkpoint_en_hi/decoder/config.json
artifacts/trtllm_checkpoint_en_hi/decoder/rank0.safetensors
```

Correctness/parity engine:

```text
artifacts/trtllm_engines_en_hi_beam5/
```

This engine has `remove_input_padding=disable` and is the one used for the
attention-mask-fixed Dakshina parity result.

Triton serving engine:

```text
artifacts/trtllm_engines_en_hi_beam5_triton/
```

This engine is separate because Triton `inflight_fused_batching` requires:

- `remove_input_padding=enable`
- `kv_cache_type=paged`

Rebuild Triton-compatible engines:

```bash
source "inference experiment/trtllm_port/env_trtllm.sh"
OUT="inference experiment/trtllm_port/artifacts/trtllm_engines_en_hi_beam5_triton"

trtllm-build \
  --checkpoint_dir "inference experiment/trtllm_port/artifacts/trtllm_checkpoint_en_hi/encoder" \
  --output_dir "$OUT/encoder" \
  --max_batch_size 128 \
  --max_input_len 128 \
  --max_seq_len 128 \
  --max_beam_width 5 \
  --max_num_tokens 16384 \
  --kv_cache_type paged \
  --remove_input_padding enable \
  --bert_attention_plugin auto

trtllm-build \
  --checkpoint_dir "inference experiment/trtllm_port/artifacts/trtllm_checkpoint_en_hi/decoder" \
  --output_dir "$OUT/decoder" \
  --max_batch_size 128 \
  --max_input_len 1 \
  --max_seq_len 64 \
  --max_encoder_input_len 4096 \
  --max_beam_width 5 \
  --max_num_tokens 8192 \
  --kv_cache_type paged \
  --remove_input_padding enable \
  --gpt_attention_plugin auto
```

## Weight Port Verification

`verify_trtllm_weight_port.py` confirmed the converted tensors exactly match the
mapped Fairseq tensors:

- encoder max absolute diff: `0.0`
- decoder max absolute diff: `0.0`
- exact tensor port: `true`
- Fairseq embedding scale: `16.0`
- TensorRT-LLM encoder `has_embedding_scale`: `true`
- TensorRT-LLM encoder `has_position_embedding`: `true`

Important model details:

- Fairseq special ids: `<s>=0`, `<pad>=1`, `</s>=2`, `<unk>=3`
- Target language token is prepended to source input, e.g.
  `__hi__ b h a r a t`
- Decoder prompt is `</s>` and is stripped from final output.
- Fairseq uses sinusoidal positional embeddings, generated in the converter.
- Q/K/V projection weights are concatenated into TensorRT-LLM fused `qkv`
  tensors.

## Dakshina / README Evaluation

Dakshina dataset source:

```text
https://github.com/google-research-datasets/dakshina
```

Local extracted file:

```text
artifacts/dakshina_data/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv
```

Evaluation filtering:

- first two TSV columns
- ASCII-only roman side
- Devanagari-only native side
- lowercased roman side
- deduped pairs

Rows evaluated: `4502`

Fairseq beam-5 README-equivalent result:

- raw Top-1: `2725/4502 = 60.53%`
- raw Top-5: `3921/4502 = 87.09%`
- README Hindi target: `60.56%`
- gap: `-0.03` percentage points, effectively rounding/script noise

TensorRT-LLM beam-5 corrected result:

- raw Top-1: `2725/4502 = 60.53%`
- raw Top-5: `3920/4502 = 87.07%`
- throughput: about `799` items/s in the recorded run

Critical fix:

- TRT runner/evaluator must pass encoder attention masks.
- Without this, mixed-length batches let decoder cross-attention attend to
  padded encoder positions and the score dropped significantly.

The relevant code path creates:

```python
attention_mask = encoder_input_ids != PAD_ID
```

and passes it to `EncDecModelRunner.generate`.

## Benchmark State

`benchmark_batch_decode.py` supports:

```bash
--backend trtllm
```

Correctness-valid masked results on RTX 4090 fp32 correctness-first engine:

- batch 1: `81.10` items/s, `12.33` ms/item
- batch 32: `884.10` items/s, `1.13` ms/item
- batch 128: `2204.87` items/s, `0.45` ms/item

Command shape:

```bash
source "inference experiment/trtllm_port/env_trtllm.sh"
python "inference experiment/benchmark_batch_decode.py" \
  --backend trtllm \
  --direction en2indic \
  --lang hi \
  --beam-width 1 \
  --topk 1 \
  --model-instances 1 \
  --batch-sizes 1,32,128 \
  --target-items 512 \
  --min-repeats 4 \
  --warmup 3 \
  --words-file "inference experiment/trtllm_port/words_en_hi.txt" \
  --csv "inference experiment/trtllm_port/artifacts/trtllm_benchmark_batch_decode.csv"
```

Batch 1000 does not fit the current setup:

- stable engine rejects it because `max_batch_size=512`
- experimental `max_batch_size=1000` decoder build failed trying to allocate
  about `25.17` GB during TensorRT tactic selection on a 24 GB RTX 4090

## Triton Backend Build State

The old `triton-inference-server/tensorrtllm_backend` repo is documentation and
redirect-style context. The actual source used was TensorRT-LLM v1.2.1:

```text
/tmp/TensorRT-LLM-triton-v1.2.1/triton_backend/inflight_batcher_llm
```

The backend build output was:

```text
/tmp/TensorRT-LLM-triton-v1.2.1/triton_backend/inflight_batcher_llm/build/libtriton_tensorrtllm.so
/tmp/TensorRT-LLM-triton-v1.2.1/triton_backend/inflight_batcher_llm/build/trtllmExecutorWorker
```

Copied to Triton:

```text
/tmp/tritonserver-2.70.0/tritonserver/backends/tensorrtllm/libtriton_tensorrtllm.so
/tmp/tritonserver-2.70.0/tritonserver/backends/tensorrtllm/trtllmExecutorWorker
```

If rebuilding, ensure `ldd` on `libtriton_tensorrtllm.so` resolves libraries
from:

```text
.venv-trtllm/lib/python3.10/site-packages/tensorrt_llm/libs
.venv-trtllm/lib/python3.10/site-packages/torch/lib
.venv-trtllm/lib/python3.10/site-packages/nvidia/cuda_runtime/lib
.venv-trtllm/lib/python3.10/site-packages/nvidia/cu13/lib
.venv-trtllm/lib/python3.10/site-packages/nvidia/cudnn/lib
.venv-trtllm/lib/python3.10/site-packages/nvidia/cublas/lib
.venv-trtllm/lib/python3.10/site-packages/nvidia/nccl/lib
```

The working `run_triton.sh` sets these paths.

## Triton Serving

Launch:

```bash
"inference experiment/trtllm_port/triton_indicxlit/scripts/run_triton.sh"
```

Default ports:

- HTTP: `8010`
- metrics: `8012`
- gRPC disabled by default in this script

Expected READY models:

```text
indicxlit_preprocess
indicxlit_tensorrt_llm
indicxlit_postprocess
indicxlit_ensemble
```

Single smoke request:

```bash
curl -X POST localhost:8010/v2/models/indicxlit_ensemble/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [
      {"name": "text_input", "shape": [1, 1], "datatype": "BYTES", "data": ["bharat"]},
      {"name": "target_lang", "shape": [1, 1], "datatype": "BYTES", "data": ["hi"]},
      {"name": "max_tokens", "shape": [1, 1], "datatype": "INT32", "data": [32]},
      {"name": "beam_width", "shape": [1, 1], "datatype": "INT32", "data": [5]},
      {"name": "topk", "shape": [1, 1], "datatype": "INT32", "data": [5]},
      {"name": "rescore", "shape": [1, 1], "datatype": "BOOL", "data": [false]}
    ],
    "outputs": [
      {"name": "text_output"},
      {"name": "candidates_json"}
    ]
  }'
```

Verified output:

```json
{
  "text_output": "भारत",
  "candidates_json": ["भारत", "भरत", "अभारत", "बारत", "बहरत"]
}
```

Concurrent probe result:

- `16` single-item HTTP requests
- concurrency `8`
- all successful
- measured local HTTP+Python pre/postprocess rate: `38.07` items/s
- Triton metrics showed:
  - `17` successful `indicxlit_ensemble` requests
  - `17` successful `indicxlit_tensorrt_llm` requests
  - zero failures

Metrics command:

```bash
curl -sS localhost:8012/metrics | rg 'nv_trt_llm|indicxlit_ensemble|indicxlit_tensorrt_llm'
```

## Triton Serving Caveat

The current static ensemble is non-decoupled:

```text
model_transaction_policy {
  decoupled: false
}
```

This supports normal single-item requests through the ensemble. Send many
normal requests concurrently and let TensorRT-LLM's inflight batcher schedule
them internally.

A single client request carrying batch tensors, for example shape `[4, 1]`, was
rejected:

```text
Batch size > 1 requires the tensorrt_llm backend to be using decoupled transaction policy
```

Changing the core backend to `decoupled: true` allows that path in the
TensorRT-LLM backend, but Triton rejects the current static ensemble:

```text
step of model 'indicxlit_ensemble' receives inputs originated from different
decoupled models
```

Conclusion:

- Working shape now: non-decoupled static ensemble, many single-item HTTP
  requests.
- To support true client-side batched tensors in one request, build a custom
  BLS/Python wrapper or a different decoupled model layout.

## Known Warnings

During Triton inference the backend logs warnings like:

```text
CrossAttentionMask is not provided...
Default padding attention mask will be used...
```

Single-item serving output is correct for the smoke tests. For larger
correctness validation through Triton, verify whether the backend's default mask
is sufficient for padded/mixed-length traffic. The direct TensorRT-LLM parity
runner remains the correctness reference because it explicitly passes
`attention_mask`.

Triton backend unload can be slow or time out in some runs. Use fresh server
processes for config changes instead of relying on live reload.


## 2026-07-10 Current Optimization / Packaging State

The active path moved from the original FP32/b128 Triton experiment to FP16
engines and direct C++ executor benchmarking.

### Installer

A repo-local installer was added and committed in `9d16fc6`:

```text
inference experiment/trtllm_port/scripts/install_trtllm_triton.sh
inference experiment/trtllm_port/scripts/INSTALL_TRTLLM_TRITON.md
```

It installs the system packages, `.venv-trtllm`, TensorRT-LLM Python package,
standalone Triton, and the TensorRT-LLM Triton backend. It does not build model
engines.

### FP16 conversion and engines

`convert_indicxlit_to_trtllm.py` now supports:

```bash
--dtype float32|float16|bfloat16
```

The working FP16 checkpoint is:

```text
artifacts/trtllm_checkpoint_en_hi_fp16/
```

Working FP16 Triton/direct executor engines:

```text
artifacts/trtllm_engines_en_hi_beam5_triton_fp16_b256/
artifacts/trtllm_engines_en_hi_beam5_triton_fp16_b512/
```

These are generated artifacts and should not be committed.

### Direct C++ executor benchmark results

Direct C++ executor bindings were tested with:

```text
benchmark_cpp_executor_bindings.py
```

This bypasses Triton and uses the TensorRT-LLM C++ executor bindings directly.
Key result: throughput increases when the actual engine max batch is increased,
so Triton was not the only bottleneck.

Observed long-run FP16 results:

```text
b256 engine, window 256: ~3265 req/s
b256 engine, window 384: ~3344 req/s
b512 engine, window 256: ~3061 req/s
b512 engine, window 512: ~3600 req/s
b512 engine, window 768: ~3601 req/s
```

GPU memory stayed around 10 GiB for the b512 run, and sampled GPU utilization
peaked around 37-38%. This suggests the current path is not GPU-compute
saturated; host/executor scheduling overhead is still significant.

### Current Triton config direction

The active core backend config points to the FP16 b256 engine:

```text
triton_indicxlit/model_repository/indicxlit_tensorrt_llm/config.pbtxt
```

Important current settings:

```text
max_batch_size: 256
gpt_model_path: artifacts/trtllm_engines_en_hi_beam5_triton_fp16_b256/decoder
encoder_model_path: artifacts/trtllm_engines_en_hi_beam5_triton_fp16_b256/encoder
kv_cache_free_gpu_mem_fraction: 0.35
cross_kv_cache_fraction: 0.35
```

The core model accepts optional `cross_attention_mask`.

### Language behavior

The Triton Python preprocess/postprocess models now normalize `target_lang` and
fall back to `hi` when it is missing or empty. Preprocess validates the target
language against `lang_list.txt` and returns a Triton error for unsupported
languages.

### Docker/k6 packaging

A Docker packaging source tree exists at:

```text
inference experiment/trtllm_port/docker_b256/
```

Commit only the source files:

```text
Dockerfile
README.md
prepare_context.sh
scripts/run_server.sh
scripts/smoke_core.py
k6/core_b256.js
```

Do not commit `docker_b256/context/`; it is generated by `prepare_context.sh`
and contains copied model repository files, assets, and engine logs/artifacts.

The prepared tarball was generated for transfer/testing:

```text
artifacts/indicxlit_trtllm_b256_docker_context.tar.gz
```

It is generated and should not be committed.

### Commit hygiene

Do not commit local/generated payloads:

```text
.downloads/
.venv-triton-py312/
artifacts/trtllm_checkpoint_*/
artifacts/trtllm_engines_*/
artifacts/triton_benchmark_*/
artifacts/cpp_executor_*/
artifacts/nsight_*/
docker_b256/context/
```

## Immediate Next Steps

1. Commit the Triton model repository and documentation changes if they are
   acceptable.
2. In a fresh instance, reinstall/rebuild TensorRT-LLM, Triton Server, and the
   `tensorrtllm` backend as needed.
3. Rebuild or copy `artifacts/trtllm_engines_en_hi_beam5_triton/`.
4. Start Triton with `triton_indicxlit/scripts/run_triton.sh`.
5. Re-run the smoke request and concurrent probe.
6. Decide serving strategy:
   - keep static ensemble and concurrent single-item requests, or
   - implement BLS/Python wrapper for true batched request payloads.
7. Only after serving shape is stable, benchmark realistic HTTP traffic.

## Hard Rules

- Do not install TensorRT-LLM into the app `.venv`.
- Do not commit `.venv-trtllm`, `.venv-fairseq`, downloaded datasets, model
  assets, or TensorRT engine binaries.
- Do not modify production Flask/app serving code until the serving strategy is
  decided.
- Keep the Fairseq path as the default production behavior until final
  integration is proven.
