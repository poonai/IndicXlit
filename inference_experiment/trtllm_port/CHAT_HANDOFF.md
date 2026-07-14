# IndicXlit TensorRT-LLM / Triton Handoff

Last updated: 2026-07-13

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
/IndicXlit/inference_experiment/trtllm_port/
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
/IndicXlit/inference_experiment/trtllm_port/env_trtllm.sh
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
inference_experiment/trtllm_port/PHASE_STATUS.md
inference_experiment/trtllm_port/CHAT_HANDOFF.md
inference_experiment/trtllm_port/triton_indicxlit/**
```

Generated engine artifacts are intentionally ignored and should not be committed:

```text
inference_experiment/trtllm_port/artifacts/trtllm_engines_en_hi_beam5_triton/
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
source "inference_experiment/trtllm_port/env_trtllm.sh"
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
source "inference_experiment/trtllm_port/env_trtllm.sh"
OUT="inference_experiment/trtllm_port/artifacts/trtllm_engines_en_hi_beam5_triton"

trtllm-build \
  --checkpoint_dir "inference_experiment/trtllm_port/artifacts/trtllm_checkpoint_en_hi/encoder" \
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
  --checkpoint_dir "inference_experiment/trtllm_port/artifacts/trtllm_checkpoint_en_hi/decoder" \
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
source "inference_experiment/trtllm_port/env_trtllm.sh"
python "inference_experiment/benchmark_batch_decode.py" \
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
  --words-file "inference_experiment/trtllm_port/words_en_hi.txt" \
  --csv "inference_experiment/trtllm_port/artifacts/trtllm_benchmark_batch_decode.csv"
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
"inference_experiment/trtllm_port/triton_indicxlit/scripts/run_triton.sh"
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
inference_experiment/trtllm_port/scripts/install_trtllm_triton.sh
inference_experiment/trtllm_port/scripts/INSTALL_TRTLLM_TRITON.md
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
inference_experiment/trtllm_port/docker_b256/
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

## 2026-07-12 Cross-Attention + Contiguous KV Cache FMHA (decoder)

Status: working and verified. The decoder can now be built with
`--gpt_attention_plugin auto --remove_input_padding enable
--kv_cache_type continuous` and the cross-attention layers use fused FMHA
instead of falling back to unfused MHA.

### What changed

The warning that used to fire at build time:

```text
Fall back to unfused MHA because of cross attention + contiguous kv cache.
```

is gone. The build log now shows `context_fmha = True` for the decoder and no
`FMHA Kernel doesn't exist` message. The fp16 `Q_CONTIGUOUS_KV` sm86 kernel
for head_size=64 (the IndicXlit decoder head size) is selected for the
cross-attention layers.

The patch itself (already in the patched TensorRT-LLM source tree, see below):

- `cpp/tensorrt_llm/common/attentionOp.cpp`
  - removed the `isCrossAttention() && useKVCache() && !mPagedKVCache` fallback
    branch in `AttentionOp::initialize()`
  - `fmhaParams.kvPtr = params.cross_kv;` is now set whenever `isCrossAttention()`
    (previously only when `!useKVCache()`)
- `cpp/tensorrt_llm/kernels/fmhaDispatcher.cpp`
  - added a guard that TRTLLM-GEN does not support `Q_CONTIGUOUS_KV` context
    FMHA (only relevant on sm100+; the A10 uses `FusedMHARunnerV2`, which does
    support it)

### Patched TensorRT-LLM source tree

```text
/home/ubuntu/TensorRT-LLM   (commit b172f0be4 "Support cross-attention with contiguous KV cache")
```

This is the main branch (reports `1.3.0rc21`), not the 1.2.1 release. It is
built out of tree at `/home/ubuntu/TensorRT-LLM/cpp/build` (Ninja, sm86 only,
`CMAKE_CUDA_ARCHITECTURES=86-real`).

### Environment note (this instance differs from the 2026-07-10 state)

This cloud instance is NOT the RTX 4090 described above. Current box:

- GPU: NVIDIA A10, compute capability 8.6 (sm86), 24 GB
- nvcc: CUDA 12.8 (V12.8.93); driver supports CUDA 13 runtime
- The `.venv-trtllm` was upgraded past 1.2.1 compatibility:
  - `torch==2.11.0+cu130`
  - `transformers==5.5.4` (main-branch requirement)
  - `tensorrt==10.16.1.11`
  - effective `tensorrt_llm` is main-branch `1.3.0rc21` (built from source,
    replacing the 1.2.1 wheel)

Because torch is 2.11.0+cu130 (1.2.1 pins torch<=2.10), the v1.2.1 pip wheel
no longer loads (`libth_common.so` ABI break: `c10::impl::PyObjectSlot`).
The whole C++ stack + Python bindings were therefore rebuilt from the patched
main-branch source. Original wheels/libs are backed up in
`.venv-trtllm/.../tensorrt_llm/libs/*.bak_pre_xattn_patch`.

### Build fixes needed for the main-branch source on this box

The main-branch source assumes CUDA 13 + NCCL 2.29 + a newer toolchain than
this box has. These were required to complete the build (all on the patched
source tree or system, not the model code):

- `cuda::maximum<>` -> `cub::Max()` guards (CUDA 12.8 CCCL): added
  `trtllmGenKernels/blockScaleMoe/DevKernel.cu:912`; the other 4 files
  (decoderMaskedMultiheadAttention, sageAttention, eagleDecoding, debugUtils)
  were already patched in the working tree.
- `cudaMemcpyBatchAsync` (CUDA 13 only) guarded with
  `#if CUDA_VERSION >= 13000` in `thop/asyncUlyssesOp.cpp` (falls back to the
  per-copy `cudaMemcpyAsync` loop that already exists).
- System header/library alignment so the build links against the venv stack:
  - `/usr/include/nvtx3` -> venv `nvidia/cuda_nvrtc/include/nvtx3`
  - `/usr/include/nccl.h` -> venv `nvidia/nccl/include/nccl.h` (2.29, has
    `ncclWindow_t`)
  - `/usr/lib/x86_64-linux-gnu/libnccl.so.2.26.2` -> venv
    `nvidia/nccl/lib/libnccl.so.2` (2.29, has `ncclCommWindowRegister`).
    NOTE: `ldconfig` resets the `libnccl.so.2` symlink back to 2.26, so the
    `libnccl.so.2.26.2` file itself was replaced; do not run ldconfig.
    Originals backed up as `*.bak`.
  - `apt-get install libcufile-dev` (provides `cufile.h` for the runtime module)

### Python package changes for sm86

The main-branch `_torch` (PyTorch-native model) path imports Sm100 (Blackwell)
CUTLASS-DSL ops that are not generated for an sm86-only build. To let the
classic EncDec/trtllm-build path work, two eager imports in
`tensorrt_llm/__init__.py` are wrapped in try/except:
`tensorrt_llm._torch.models` and `tensorrt_llm.visual_gen`. These are only
needed for the new PyTorch-native model path, not for the encoder-decoder
runner.

### Rebuilt engine (the patched FMHA one)

```text
artifacts/trtllm_engines_en_hi_beam5_host_fp16_b256_continuous_kv_fmha/
  encoder/   (paged KV, bert plugin, rebuilt against TRT 10.16)
  decoder/   (continuous KV, gpt attention plugin auto, context_fmha=True)
```

Rebuild command (decoder only):

```bash
source "inference_experiment/trtllm_port/env_trtllm.sh"
CKPT="inference_experiment/trtllm_port/artifacts/trtllm_checkpoint_en_hi_fp16"
OUT="inference_experiment/trtllm_port/artifacts/trtllm_engines_en_hi_beam5_host_fp16_b256_continuous_kv_fmha"
trtllm-build \
  --checkpoint_dir "$CKPT/decoder" --output_dir "$OUT/decoder" \
  --max_batch_size 256 --max_input_len 1 --max_seq_len 64 \
  --max_encoder_input_len 4096 --max_beam_width 5 --max_num_tokens 8192 \
  --kv_cache_type continuous --remove_input_padding enable --gpt_attention_plugin auto
```

There is also `scripts/build_host_b256_engine.sh --kv-cache-type continuous`
which builds encoder (paged) + decoder (continuous) together.

### Verification (patched FMHA engine)

- Build log: no `Fall back to unfused MHA` and no
  `FMHA Kernel doesn't exist`; `context_fmha = True`.
- Smoke: `bharat -> भारत` (beam-5, candidates
  `भारत, भरत, ...`).
- Fairseq greedy parity (22-word list): **22/22 exact match,
  exact_match_rate = 1.000, passes_greedy_gate = true**.
  Report: `artifacts/trtllm_fmha_continuous_kv_parity_report.json`.
- Throughput on the A10: ~71 items/s greedy batch-22 (vs ~34 items/s beam-5
  batch-16 for the older pre-patch contiguous engine that used the unfused
  fallback). The contiguous-KV path is faster than paged for the direct
  fixed-batch runner here.

The full 4502-row Dakshina eval was NOT re-run on this instance because the
Dakshina dataset is not present
(`artifacts/dakshina_data/.../hi.translit.sampled.test.tsv` is missing). The
previously recorded TRT beam-5 result (`60.53%` Top-1) was with the
unfused-fallback path; the FMHA path is numerically equivalent (both correct),
but a Dakshina re-run is the recommended final confirmation once the dataset is
restored.

### Triton / Docker note

The continuous-KV + FMHA cross-attention engine is for the **direct host
runner** path. Triton `inflight_fused_batching` still requires
`remove_input_padding=enable` + `kv_cache_type=paged`, so the Triton/Docker
serving engines are paged. As of this update no Docker compose stack is
running (docker daemon active, zero containers).

### 2026-07-12 Patched v1.1.0 Triton Docker image (base-image version match)

Status: working. The patched cross-attention build is now inside a Triton
Docker image.

Approach (the key insight): the `tritonserver:26.02-trtllm-python-py3` base
image bundles trtllm **1.1.0** with the **`tensorrt_llm::`** namespace (no
`_v1`) and a prebuilt Triton backend plugin (`libtriton_tensorrtllm.so`).
Rather than rebuild the backend plugin (its C++ source was removed from
trtllm main and main-branch uses the `_v1` namespace), match the base image's
exact version: build the patched libs from the **v1.1.0** source and overlay
only the runtime libs. The base image's existing backend plugin then links
unchanged (verified: 0 unsatisfied `tensorrt_llm::` symbols).

What was built (inside a `tritonserver:26.02-trtllm` container, so TRT
10.13.3.9 + torch 2.9.0 + CUDA 13 nvcc all match the runtime):

- v1.1.0 source: `https://github.com/NVIDIA/TensorRT-LLM` tag `v1.1.0`
  (cloned to `/tmp/TRTLLM-v1.1.0`).
- Cross-attention patch applied to v1.1.0
  `cpp/tensorrt_llm/common/attentionOp.cpp` (identical to the main-branch
  patch: `kvPtr` set whenever `isCrossAttention()`, and the contiguous-KV
  fallback warning removed).
- fmha_v2 cubins generated with the **full** build_wheel.py env — the critical
  missing flag earlier was `GENERATE_CUBIN=1` (without it the generated
  `fmha_cubin.h` is malformed):
  ```
  TORCH_CUDA_ARCH_LIST=9.0 ENABLE_SM89_QMMA=1 ENABLE_HMMA_FP32=1 \
  GENERATE_CUBIN=1 SCHEDULING_MODE=1 ENABLE_SM100=1 ENABLE_SM120=1 \
  GENERATE_CU_TRTLLM=true python3 setup.py
  ```
  then move `generated/fmha_cubin.{h,cpp}` → `cubin/` and
  `generated/*sm*.cu` → `fmha_v2_cu/`.
- Submodules initialized: `3rdparty/{cutlass,json,NVTX,pybind11,nanobind,cppzmq,cxxopts,xgrammar}`
  (xgrammar's `.cc` are globbed into batch_manager; missing it causes link
  errors — re-run cmake after init so the GLOB refreshes).
- Build: `cmake ... -DBUILD_PYT=ON -DENABLE_UCX=OFF -DCMAKE_CUDA_ARCHITECTURES=86-real -GNinja`,
  then `ninja libtensorrt_llm.so libth_common.so libnvinfer_plugin_tensorrt_llm.so`.
- Built artifacts in the build container `/build/tensorrt_llm/`:
  `libtensorrt_llm.so` (~510 MB, no `_v1` symbols), `libth_common.so`,
  `libnvinfer_plugin_tensorrt_llm.so`, `libdecoder_attention_{0,1}.so`.
- Paged engine rebuilt in-container with the patched plugin
  (`context_fmha=True`, no cross-attention fallback).

Docker image:

```text
indicxlit-trtllm:b256-fp16-patched
```

Built with `docker_b256/context/Dockerfile.patched` (FROM
`tritonserver:26.02-trtllm-python-py3`, same as the original Dockerfile plus a
`COPY patched_libs_flat .../tensorrt_llm/libs/` overlay). The context was
prepared with `docker_b256/prepare_context.sh` then the engine replaced with
the freshly built paged engine and the patched libs added.

Verification (running container, Triton HTTP):

```text
curl bharat -> text_output: भारत
candidates_json: ["भारत", "भरत", "अभारत", "बारत", "बहरत"]
```

Matches the handoff's known-good Triton output. The base image's
`libtriton_tensorrtllm.so` and `tritonserver` are reused unchanged.

Build containers left in place for reference: `trtllm-v110-build` (the working
v1.1.0 build) and `trtllm-v121-build` (v1.2.1 attempt, abandoned). Sources at
`/tmp/TRTLLM-v1.1.0` and `/tmp/TRTLLM-v1.2.1`.

Caveat: Triton still serves with paged KV, so the cross-attention patch is a
no-op at *runtime* for this image (it only changes the contiguous-KV path).
The value of this image is that the whole TensorRT-LLM runtime is the patched
build, consistent and reproducible from the v1.1.0 source + the patch.

### 2026-07-12 Continuous-KV + fused-attention serving container

Status: working. A container that actually serves with `kv_cache_type=
continuous` (exercising the patched fused cross-attention path).

Why a separate image from the Triton one: the v1.1.0 Triton `tensorrtllm`
backend is paged-only — its `KvCacheConfig` has no cache-type field and the
executor's inflight scheduler requires paged KV. Continuous KV therefore cannot
go through Triton's inflight backend. Instead this image serves via the
**EncDecModelRunner** (the same runtime path as `run_trtllm_greedy.py`) behind
a small FastAPI HTTP server.

Image:

```text
indicxlit-trtllm:b256-fp16-continuous-kv
```

Built from `docker_b256/context_cont/Dockerfile` (FROM
`tritonserver:26.02-trtllm-python-py3`), overlaying the patched v1.1.0 libs and
bundling:

- paged encoder + **continuous-KV decoder** (`context_fmha=True`, no cross-attention
  fallback) at `/models/engines/b256_cont`
- vocab + lang list at `/models/assets/en2indic`
- `docker_b256/server_continuous_kv.py` → `/opt/indicxlit/server.py`
  (FastAPI; loads EncDecModelRunner once, exposes the Triton-ensemble-compatible
  endpoint `POST /v2/models/indicxlit/infer`, plus `/v2/health/ready`)
- `INDICXLIT_MODEL_ROOT`, `ENGINE_DIR`, `INDICXLIT_LANG_LIST`, `INDICXLIT_PORT`
  env vars

Run + smoke:

```bash
docker run -d --gpus all -p 8000:8000 indicxlit-trtllm:b256-fp16-continuous-kv
# then:
curl -X POST localhost:8000/v2/models/indicxlit/infer -H 'Content-Type: application/json' -d '{
  "inputs": [
    {"name":"text_input","shape":[1,1],"datatype":"BYTES","data":["bharat"]},
    {"name":"target_lang","shape":[1,1],"datatype":"BYTES","data":["hi"]},
    {"name":"max_tokens","shape":[1,1],"datatype":"INT32","data":[32]},
    {"name":"beam_width","shape":[1,1],"datatype":"INT32","data":[5]},
    {"name":"topk","shape":[1,1],"datatype":"INT32","data":[5]}
  ]
}'
# -> text_output: भारत ; candidates: ["भारत","भरत","भरात","भराट","भर्त"]
```

Trade-offs vs the Triton paged image:

- This exercises the patch at runtime (fused cross-attention on continuous KV).
- No Triton inflight batching / dynamic request scheduling — it is a
  fixed-batch per-request runner (the request is processed to completion before
  the next). For higher-throughput continuous-KV serving, wrap the runner in a
  batching queue or use the C++ executor with a continuous-KV config.
- Same Triton-style HTTP request shape, so existing single-item clients work
  unchanged.

### 2026-07-12 KV-cache type: build-time vs runtime, and the continuous+Triton mismatch test

These are the architectural findings that explain why there are two serving
images (paged/Triton vs continuous/runner) and why they cannot be merged.

**Where the KV-cache "type" lives.** `--kv_cache_type` at `trtllm-build` time
does not allocate any KV memory. It selects which variant of the GPT-attention
plugin is instantiated: in `cpp/tensorrt_llm/common/attentionOp.cpp`,
`AttentionOp` is templated on the KV buffer type — `KVBlockArray` for paged vs
`KVLinearBuffer` for continuous. That bakes three things into the `.engine`:

1. the attention kernel layout it dispatches (`Q_PAGED_KV` vs
   `Q_CONTIGUOUS_KV` in `fmhaRunner.cpp:165`);
2. the role/shape of the KV-cache input the engine declares;
3. how `enqueueContext`/`enqueueDecode` interpret the KV pointer passed in
   (`attentionOp.cpp:2033`, the `KVBlockArray` vs `KVLinearBuffer` branches).

The actual KV-cache bytes are **external** — owned and allocated by the
runtime (the executor / Triton backend / `EncDecModelRunner`), which must hand
the right buffer format to the engine every step. The runtime's
`KvCacheConfig` (`kv_cache_free_gpu_mem_fraction`, `cross_kv_cache_fraction`,
`max_tokens_in_paged_kv_cache`, ...) is purely pool sizing/management; it does
**not** override the engine's type. Rule: build picks the format; runtime must
speak the same format.

**Triton's backend is paged-only.** The `tritonserver:26.02-trtllm` backend's
executor manages a paged block pool — that block pool is what lets the inflight
scheduler add/remove requests of different lengths dynamically. Its
`KvCacheConfig` has no cache-type field. So continuous KV cannot go through
Triton's inflight backend.

**Empirical test — continuous engine served through Triton's paged/inflight
backend = hard load failure (not silent corruption).** Tested by mounting the
continuous-KV decoder into `indicxlit-trtllm:b256-fp16-patched` over the paged
decoder and starting tritonserver. The model goes `UNAVAILABLE` and the server
exits with:

```text
TrtGptModelInflightBatching requires GPT attention/Mamba Conv 1d plugin
with packed input and paged KV cache.
```

This is a deliberate guard in the backend (it checks the engine's plugin
variant at model-instance creation). So the wrong combo fails loudly before
serving any request — you never get silently-wrong results. Even without the
guard it would break: the inflight scheduler assigns/free blocks, but
continuous KV is a fixed per-slot linear buffer with nothing to dynamically
reassign, and the plugin would receive a block-array pointer where it expects
a linear buffer.

**Conclusion / serving matrix:**

| Image | KV type | Serving path | Patch exercised at runtime? |
|---|---|---|---|
| `indicxlit-trtllm:b256-fp16-native` | paged | Triton inflight | no (unpatched runtime) |
| `indicxlit-trtllm:b256-fp16-patched` | paged | Triton inflight | no (patched runtime, but Triton uses paged) |
| `indicxlit-trtllm:b256-fp16-continuous-kv` | **continuous** | FastAPI + `EncDecModelRunner` | **yes** |

**Current running state (2026-07-12):** docker daemon active, no serving
container running. Only the two build containers are up (`trtllm-v110-build`,
`trtllm-v121-build`). Port 8000 is free. To serve continuous KV, run the
FastAPI image (command in the section above).

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

## 2026-07-13 Rust Server / EncDecModelRunner Handoff

User direction changed after the direct C++ port investigation: **do not
continue replacing Python `EncDecModelRunner` tensor prep with a hand-written
C++ runner unless there is a strong reason.** The preferred next path is:

1. Keep the Rust batch server as the HTTP/scheduling layer.
2. Call the Python TensorRT-LLM `EncDecModelRunner` path through FFI /
   embedded Python / a narrow native bridge.
3. Let Python/TensorRT-LLM own the exact encoder/decoder tensor preparation,
   padding/removal, cross-attention masks, packed FMHA masks, continuous KV
   cache layout, and dynamic decode session behavior.
4. Benchmark throughput. If it is good enough, stop there. Correctness and
   preserving TensorRT-LLM semantics are more important than rewriting every
   helper in Rust/C++.

### Current Findings

- The Python direct runner that works for continuous KV is
  `tensorrt_llm.runtime.enc_dec_model_runner.EncDecModelRunner`.
- It uses C++/CUDA internally for the TensorRT engines and dynamic decoding,
  but Python builds and coordinates the tensor inputs.
- The important Python code paths are:
  - `/home/ubuntu/TensorRT-LLM/tensorrt_llm/runtime/enc_dec_model_runner.py`
    - `process_input`: remove-padding input prep.
    - `encoder_run`: flattened encoder inputs, position IDs, lengths.
    - `generate`: builds cross-attention mask and calls decoder session.
  - `/home/ubuntu/TensorRT-LLM/tensorrt_llm/runtime/generation.py`
    - `_prepare_cross_attention_mask`.
    - context phase calls
      `torch.ops.tensorrt_llm.pack_fmha_mask_by_input(...)`.
    - generation phase tiles the dense cross-attention mask by beam and passes
      an empty packed mask with shape
      `(batch_size, ceil(mask_cols / 32))`.
- The packed FMHA mask Python op is a C++/CUDA TensorRT-LLM helper:
  - `/home/ubuntu/TensorRT-LLM/cpp/tensorrt_llm/thop/fmhaPackMaskOp.cpp`
  - `/home/ubuntu/TensorRT-LLM/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmhaPackedMask.h`
  - `/home/ubuntu/TensorRT-LLM/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmhaPackedMask.cu`
- Python handles variable-length words in one batch by padding at the API
  boundary, then using remove-padding internally. For decoder context,
  `context_lengths` are length 1 for each request. Encoder lengths stay
  untiled in context phase and are beam-tiled for generation.

### Direct C++ Runner Experiment Status

The experimental direct C++ bridge lives under:

```text
inference_experiment/trtllm_port/docker_b256/rust_executor_server/
```

The image used for smoke tests was:

```text
indicxlit-trtllm:rust-executor
```

Known-good behavior before removing Rust fallbacks:

- Single input works.
- Beam search works for `namaste` with beam 5:
  top result `नमस्ते`.
- HTTP multi-line requests were previously correct only because Rust split
  them into single-word calls or same-length groups.

Direct mixed-length batch findings:

- Removed the Rust multi-word/same-length fallback so one request with
  `namaste`, `bharat`, `kiran` goes through one `engine.infer_batch`.
- First direct mixed batch failed with:

```text
cudaMemcpyAsync H2D: an illegal memory access was encountered
```

- A mismatch was found in `cross_attention_packed_mask` handling:
  context phase allocated `batch_size * 128` rows but advertised only
  `activeRows` rows to TensorRT. Python's context packed mask is
  `total_aligned_rows x aligned_cols/32`, where rows are rounded to 128.
- Generation phase should use Python's placeholder shape
  `(batch_size, ceil(mask_cols / 32))`, not `activeRows`.
- A second mismatch was found in length metadata:
  context phase should bind untiled `encoder_input_lengths`; generation phase
  should bind beam-tiled lengths.
- I started patching the direct C++ runner to call
  `tk::invokeBuildPackedMask<bool>` instead of hand-packing on CPU. That build
  succeeded, but the mixed batch still failed with the same CUDA illegal memory
  access.
- The next debugging step would have been dumping decoder engine I/O metadata
  and checking mask dtype/shape against the engine, but the user interrupted
  and redirected the plan.

### Important Caution

The direct C++ runner currently contains partial experimental changes in the
untracked `rust_executor_server` tree. Treat it as a scratch experiment, not a
stable serving path. If continuing with Python `EncDecModelRunner` via FFI, do
not rely on those partial direct-runner changes for correctness.

### Recommended Next Implementation

Build a Rust batch server that owns:

- HTTP/Triton-compatible request parsing.
- Request coalescing and response demultiplexing.
- Backpressure and batch-delay policy.
- Optional worker pool/process management.

But delegate inference to Python `EncDecModelRunner` through a narrow boundary:

- Rust sends a batch of words plus `target_lang`, `beam_width`, `topk`,
  `max_tokens`.
- Python returns one candidates JSON string per input word.
- Python keeps the model runner alive and reuses allocated GPU state as much as
  `EncDecModelRunner` already supports.

Possible bridge options:

1. Embedded Python with PyO3:
   - Rust process imports a small Python module.
   - The Python module owns global tokenizer/assets and a persistent
     `EncDecModelRunner`.
   - Rust calls a `infer_batch(words, lang, beam_width, topk, max_tokens)`
     Python function.
2. Long-lived Python worker process:
   - Rust communicates over Unix socket/stdin/gRPC.
   - Simpler isolation and easier debugging.
   - Slightly more IPC overhead, but probably acceptable compared to GPU
     decode time.
3. C ABI shim around Python:
   - Exposes `init` and `infer_batch` symbols.
   - Internally still uses Python and `EncDecModelRunner`.
   - More fragile than a worker process unless deployment packaging is locked.

Given the user goal, option 1 or 2 is preferable. Start with option 2 if speed
of implementation matters; switch to embedded Python only if IPC overhead is
measurable and significant.

### Suggested Validation

Use the Dakshina Hindi test TSV already downloaded at:

```text
inference_experiment/trtllm_port/artifacts/dakshina_data/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv
```

Previously measured quick 200-row Rust HTTP eval:

```text
rows: 200
top1: 94/200 = 0.47
top3: 121/200 = 0.605
throughput: ~31.25 items/s
```

After implementing Python `EncDecModelRunner` behind Rust batching, repeat:

1. Single word smoke: `namaste -> नमस्ते`.
2. Mixed one-request batch: `namaste`, `bharat`, `kiran`.
3. Concurrent requests with mixed word lengths.
4. 200-row Dakshina eval.
5. Nsight Systems profile only after correctness is stable.

Stop criterion: if throughput is close to the Python direct runner and accuracy
matches, keep the Python runner via FFI/worker and avoid a full C++ rewrite.

### 2026-07-13 IPC Worker Implementation Result

Implemented the recommended IPC path in:

```text
inference_experiment/trtllm_port/docker_b256/rust_executor_server/
```

Current serving shape:

- Rust still owns HTTP parsing, request batching, config compatibility checks,
  and response demuxing.
- `src/ffi.rs` no longer links the experimental C++ bridge. It launches one
  long-lived Python worker process and talks JSON-lines over stdin/stdout.
- `python/encdec_worker.py` owns `EncDecModelRunner.from_engine(...)`, keeps it
  loaded, and runs the known-good Python `generate(...)` path for batches.
- Docker image `indicxlit-trtllm:rust-executor` now builds Rust only and copies
  the Python worker into `/opt/indicxlit/python/encdec_worker.py`.

Smoke results:

```text
GET /v2/health/ready -> {"health":"ready"}
namaste -> नमस्ते
namaste / bharat / kiran in one request -> नमस्ते / भारत / किरण
```

Concurrent one-word HTTP probe returned sensible outputs:

```text
namaste नमस्ते
bharat  भारत
kiran   किरण
dilli   दिल्ली
shakti  शक्ति
vidya   विद्या
suraj   सुरज
pustak  पुस्तक
```

Quick HTTP Dakshina check through the full Rust -> Python-worker stack:

```text
rows: 200
top1: 122/200 = 0.61
top3: 162/200 = 0.81
throughput: ~48.9 items/s
```

The current detached smoke container is:

```text
indicxlit-rust-smoke
http://127.0.0.1:18000/v2/health/ready
http://127.0.0.1:18000/v2/models/indicxlit/infer
```

Stop it with:

```bash
sudo docker rm -f indicxlit-rust-smoke
```

### 2026-07-13 Direct C++ Beam Search Fix

The standalone direct C++ runner under:

```text
inference_experiment/trtllm_port/docker_b256/rust_executor_server/cpp/
```

now has a working batched beam-search smoke path for small batches.

Root cause found:

- The context step initialized `contextLengths` and `sequenceLength` for only
  `batchSize` rows.
- `DynamicDecodeLayer` beam search reads those buffers as `[batchSize *
  beamWidth]`.
- That left batch index > 0 beam state partly zero/uninitialized, producing
  corrupt outputs like the second `namaste` becoming `["न","ने","ना"]`.

Patch applied:

- In `direct_enc_dec_runner.cc`, context-phase decode buffers now initialize
  `batchSize * beamWidth` rows for `contextLengths` and `sequenceLength`.
- Also aligned the direct dynamic-decode bridge closer to TensorRT-LLM:
  `endIds` is `[batchSize]`, `sequenceLimitLength` is passed, and
  `beamSearchSteps` is set.
- A guarded diagnostic dump exists behind `INDICXLIT_DEBUG_RAW=1`.

Verified direct smoke:

```text
beam=5 namaste namaste
namaste ["नमस्ते","नामस्ते","नमसते"]
namaste ["नमस्ते","नामस्ते","नमसते"]

beam=5 namaste bharat kiran
namaste ["नमस्ते","नामस्ते","नमसते"]
bharat  ["भारत","भरत","भार्त"]
kiran   ["किरन","किरण","किरान"]

beam=1 namaste bharat kiran
namaste ["नमस्ते"]
bharat  ["भारत"]
kiran   ["किरन"]
```

Quick 200-row Dakshina correctness check via direct smoke in microbatches of 8:

```text
top1: 122/200 = 0.61
top3: 161/200 = 0.805
```

Important remaining issue:

- A single direct C++ call with batch size above 8 currently fails with
  `std::runtime_error("unknown error")` / exit 139, even though the engine config
  says `max_batch_size: 256`.
- Batch sizes 4 and 8 pass. Batch sizes 12, 16, 24, 32, 64, 96, 128, and 200
  failed in the smoke binary.
- This is separate from the beam corruption bug and must be fixed before wiring
  this direct C++ runner into the Rust server for real throughput testing.

### 2026-07-13 Resolution of the "batch > 8 crash"

This was **not a real crash**. It was two stacked bugs in `direct_smoke.cc`
itself; the direct C++ runner, encoder/decoder engines, and beam-search path
are all correct at large batches.

Root causes:

1. `direct_smoke.cc` hardcoded `config.max_batch_size = 8`. Any batch > 8 was
   rejected by the correct validation in `DirectEncDecRunner::infer()`:
   `throw std::runtime_error("invalid direct runner batch or beam size")`.
   The engine config (`max_batch_size: 256`) was never consulted because the
   smoke binary clamped the runtime cap below it.
2. `direct_smoke.cc` called `check(indicxlit_infer_batch(..., &error), error)`.
   C++ does not specify argument evaluation order; GCC read `error` (the right
   argument) while it was still `nullptr`, before `indicxlit_infer_batch`
   populated it. `check()` then fell through to its default message
   `"unknown error"`. Because `main()` has no try/catch, the throw escaped,
   `std::terminate` was called, and `abort()` produced the SIGABRT/exit-134.
   (The handoff's "exit 139" was a misread of the abort signal.)

Verification (after the fix, in the `tritonserver:26.02-trtllm-python-py3`
container with `b256_cont` engine + `en2indic` assets):

```text
INDICXLIT_MAX_BATCH_SIZE=256 ./indicxlit_direct_smoke <words...>
batch sizes 4, 8, 9, 12, 16, 24, 32, 64, 96, 128, 200, 256
all return code=0 with correct beam-5 candidates.
```

Fixes applied to `cpp/direct_smoke.cc`:

- `max_batch_size` is now configurable via `INDICXLIT_MAX_BATCH_SIZE` and
  defaults to 256 (matches the engine config).
- `check()` is now called as `int code = indicxlit_infer_batch(..., &error);
  check(code, error);` so the error pointer is read after the call returns.

Conclusion: the direct C++ runner is safe to wire into the Rust server for
throughput testing at any batch size up to the engine's `max_batch_size` (256).

### 2026-07-13 Rust Server Switched to Direct TensorRT C++ Runtime

The Rust executor server is now integrated with the fixed direct C++ TensorRT
runtime instead of the Python IPC worker.

Files changed:

```text
docker_b256/rust_executor_server/src/ffi.rs
docker_b256/rust_executor_server/build.rs
docker_b256/rust_executor_server/Dockerfile
```

Implementation shape:

- `src/ffi.rs` calls the C ABI in `libindicxlit_trtllm_bridge.so` directly:
  `indicxlit_create`, `indicxlit_infer_batch`, `indicxlit_destroy`.
- The Dockerfile copies the validated bridge from
  `cpp_build_direct/libindicxlit_trtllm_bridge.so` into `/opt/indicxlit/lib`.
- Runtime `LD_LIBRARY_PATH` includes `/opt/indicxlit/lib`.
- The image was rebuilt as:

```text
indicxlit-trtllm:rust-executor
```

Current running container:

```text
indicxlit-rust-smoke
host port 8000 -> container port 8000
```

Verified HTTP smokes:

```text
GET /v2/health/ready -> {"health":"ready"}
POST namaste -> नमस्ते
POST namaste / bharat / kiran -> नमस्ते / भारत / किरन
```

256-row HTTP batch probe through Rust -> direct C++:

```text
rows: 256
top1: 149/256 = 0.5820
top5: 216/256 = 0.8438
elapsed: ~0.093 s
throughput: ~2766 items/s
```

This throughput number is a single warm request on the already-loaded container,
not a full benchmark suite result. It is good enough to hand off to the external
benchmark suite.

### 2026-07-13 Docker Compose Switched from Triton to Rust/Direct-C++

The production-style compose stack now uses the Rust/direct-C++ image instead
of the Triton inference image.

Files updated:

```text
docker_b256/docker-compose.yml
docker_b256/nginx.conf
docker_b256/prometheus.yml
docker_b256/README.md
```

Key changes:

- `inference-1` and `inference-2` now use:
  `indicxlit-trtllm:rust-executor`
- NGINX still exposes host port `8000` and load-balances to both replicas on
  container port `8000`.
- Prometheus no longer scrapes stale Triton metrics on `8002`; it now keeps
  NGINX and DCGM.
- README now describes Rust/direct-C++ serving, not Triton ensemble serving.

Compose is currently running:

```text
indicxlit-b256-inference-1-1  healthy
indicxlit-b256-inference-2-1  healthy
indicxlit-b256-nginx-1        host 8000 -> container 8000
prometheus                    host 9090
grafana                       host 3000
```

Verified through the public NGINX endpoint:

```text
GET http://127.0.0.1:8000/v2/health/ready -> ready
POST namaste/bharat/kiran -> नमस्ते / भारत / किरन
256-row HTTP batch through NGINX:
  top1: 149/256
  top5: 216/256
  elapsed: ~0.114 s
  throughput: ~2248 items/s
```

### 2026-07-13 Rust Runtime Metrics Added with metrics-rs

The Rust/direct-C++ executor now exports Prometheus metrics using
`metrics` + `metrics-exporter-prometheus`.

Files updated:

```text
docker_b256/rust_executor_server/Cargo.toml
docker_b256/rust_executor_server/Dockerfile
docker_b256/rust_executor_server/src/main.rs
docker_b256/prometheus.yml
docker_b256/grafana/dashboards/indicxlit-b256.json
docker_b256/README.md
```

Runtime endpoint:

```text
GET /metrics
```

Prometheus scrape job:

```text
job_name: indicxlit-rust
targets:
  - inference-1:8000
  - inference-2:8000
```

Metrics currently emitted:

```text
indicxlit_http_requests_total{status}
indicxlit_http_request_words_total{status}
indicxlit_http_request_duration_seconds{status,quantile}
indicxlit_http_request_words{status,quantile}
indicxlit_queue_wait_duration_seconds{quantile}
indicxlit_batches_total{status,mode}
indicxlit_batch_words_total{status,mode}
indicxlit_batch_inference_duration_seconds{status,mode,quantile}
indicxlit_batch_requests{status,mode,quantile}
indicxlit_batch_words{status,mode,quantile}
indicxlit_config_max_batch_size
indicxlit_config_max_beam_width
indicxlit_config_max_num_tokens
indicxlit_config_batch_delay_microseconds
indicxlit_config_static_scheduler
```

The exporter renders summaries by default, so the Grafana dashboard uses the
emitted `quantile` series and `_sum/_count` averages instead of
`histogram_quantile(..._bucket...)`.

Build/deploy notes:

- The build stage now installs Rust via rustup because apt Cargo 1.75 cannot
  parse current Rust-2024 transitive crates from the metrics stack.
- Runtime image remains based on `indicxlit-trtllm:b256-fp16-continuous-kv` and
  does not include the Rust toolchain.
- Rebuilt image: `indicxlit-trtllm:rust-executor`.
- Recreated `inference-1`, `inference-2`, `prometheus`, and `grafana`.

Verification:

```text
POST http://127.0.0.1:8000/v2/models/indicxlit/infer
  namaste/bharat/kiran -> नमस्ते / भारत / किरन

Prometheus targets:
  indicxlit-rust inference-1:8000 up
  indicxlit-rust inference-2:8000 up
  nginx nginx-exporter:9113 up
  dcgm dcgm-exporter:9400 up

Observed series after probe:
  indicxlit_http_requests_total{instance="inference-1:8000",status="ok"} 1
  indicxlit_batch_inference_duration_seconds_count{instance="inference-1:8000",mode="merged",status="ok"} 1
  indicxlit_queue_wait_duration_seconds_count{instance="inference-1:8000"} 1
```

Follow-up routing fix:

- Prometheus was scraping inference-2 correctly and its config gauges were
  visible, but request/batch counters were absent because public traffic through
  NGINX was sticking to inference-1.
- Direct request to `http://inference-2:8000/v2/models/indicxlit/infer` worked
  and immediately produced inference-2 metrics.
- Changed `docker_b256/nginx.conf` upstream policy from `least_conn` to
  `random two least_conn`.
- Recreated nginx.
- Validation burst through `http://127.0.0.1:8000`:

```text
before:
  inference-1:8000 282320 requests
  inference-2:8000 1 request
after 40 public requests:
  inference-1:8000 282340 requests
  inference-2:8000 21 requests
```

### 2026-07-13 Single Container, Two Worker Runtime, No NGINX/MPS

The compose stack was changed again to remove the two-container MPS layout and
serve the Rust executor directly on host port `8000`.

Current serving shape:

```text
client -> host :8000 -> one Rust/direct-C++ container
                    -> one shared work queue
                    -> worker-0 / worker-1
                    -> one TensorRT-LLM engine per worker
```

Implementation details:

- Rust server uses `crossbeam-channel` with one `Sender<WorkItem>` and cloned
  receivers, so whichever worker is free receives the next request.
- `INDICXLIT_WORKERS` defaults to `2`.
- Each worker thread pins itself with `core_affinity`.
- Metrics include `worker` labels and CPU-core gauges:

```text
indicxlit_config_workers 2
indicxlit_worker_cpu_core{worker="worker-0"} 0
indicxlit_worker_cpu_core{worker="worker-1"} 1
```

Compose changes:

- `inference` publishes `8000:8000` directly.
- Removed `mps`, `inference-1`, `inference-2`, `nginx`, and
  `nginx-exporter` from the active compose stack.
- Prometheus now scrapes only:

```text
indicxlit-rust inference:8000
dcgm dcgm-exporter:9400
```

Validation:

```text
GET http://127.0.0.1:8000/v2/health/ready -> ready
POST namaste/bharat/kiran -> नमस्ते / भारत / किरन
Prometheus target indicxlit-rust inference:8000 -> up
```

### 2026-07-13 Direct C++ Runtime Optimizations After Nsight

Nsight showed the remaining host-side bottleneck was mostly bridge/runtime glue,
not Rust queueing:

- many small CUDA API calls, especially `cudaMemcpyAsync`
- repeated `DynamicDecodeAdapter` construction per batch
- an extra DynamicDecode CUDA stream plus stream synchronizations
- repeated host staging vector allocation/copy setup inside each inference

Implemented optimizations in the Rust executor C++ bridge:

- `direct_enc_dec_runner.cc`
  - create one `DynamicDecodeAdapter` in `DirectEncDecRunner` construction
  - pass the runner CUDA stream into DynamicDecode
  - cache decoder setup by `(batch_size, beam_width, cum_log_probs)`
  - reuse host staging vectors as runner members instead of allocating them per
    request/decode step
  - remove redundant stream synchronizations where same-stream ordering is
    sufficient
  - keep explicit syncs before final host-visible output copies
- `direct_dynamic_decode.{h,cc}`
  - add a constructor accepting an external `cudaStream_t`
  - wrap the runner stream without taking ownership when provided

Important build note:

- Building the bridge against `/home/ubuntu/TensorRT-LLM` produced an ABI
  mismatch with the current runtime image:

```text
undefined symbol: _ZN12tensorrt_llm3_v16common13TllmExceptionD1Ev
```

- Rebuilt successfully against `/tmp/TRTLLM-v1.1.0`, which matches the
  `indicxlit-trtllm:b256-fp16-continuous-kv` runtime ABI. That source tree is
  the patched TensorRT-LLM tree used for FMHA with continuous KV in this stack.

Rebuild/restart commands used:

```bash
sudo docker run --rm --gpus all --entrypoint /bin/bash \
  -v '/home/ubuntu/IndicXlit/inference_experiment/trtllm_port/docker_b256/rust_executor_server:/src' \
  -v '/tmp/TRTLLM-v1.1.0:/tmp/TRTLLM-v1.1.0:ro' \
  -w /src indicxlit-trtllm:b256-fp16-continuous-kv \
  -lc 'apt-get update >/dev/null && apt-get install -y --no-install-recommends cmake make g++ >/dev/null && cmake -S cpp -B cpp_build_direct -DTRTLLM_ROOT=/tmp/TRTLLM-v1.1.0 && cmake --build cpp_build_direct --target indicxlit_trtllm_bridge -j "$(nproc)"'

sudo docker build --no-cache -t indicxlit-trtllm:rust-executor \
  -f 'inference_experiment/trtllm_port/docker_b256/rust_executor_server/Dockerfile' \
  'inference_experiment/trtllm_port/docker_b256/rust_executor_server'

sudo docker compose -f 'inference_experiment/trtllm_port/docker_b256/docker-compose.yml' \
  up -d --force-recreate inference
```

Correctness validation after optimization:

```text
GET /v2/health/ready -> {"health":"ready"}

POST namaste/bharat/kiran -> नमस्ते / भारत / किरन

Dakshina Hindi lexicon, full sampled test TSV over Rust HTTP endpoint:
  rows: 4502
  batch size: 256
  beam/topk: 5/5
  top1: 0.5982
  top5: 0.8645
  elapsed: 1.464 s
  items/s: 3075.67
  final partial batch size: 150
```

Artifacts:

```text
inference_experiment/trtllm_port/artifacts/rust_executor_dakshina_hi_beam5_b256_eval_after_opt.json
inference_experiment/trtllm_port/artifacts/rust_executor_dakshina_hi_beam5_b256_full_eval_after_opt.json
```

Runtime status after validation:

```text
one inference container on host :8000
INDICXLIT_WORKERS=2
worker-0 pinned to CPU 0
worker-1 pinned to CPU 1
both workers received successful requests in /metrics
```

### 2026-07-13 Two Worker GPU Waiting Check

Question investigated: one worker appeared slower than the other, possibly due
to NVIDIA GPU-side waiting/serialization.

Current Prometheus counters after real traffic did not show persistent worker
imbalance:

```text
worker-0 successful batches: ~2348
worker-1 successful batches: ~2344
worker-0 cumulative batch inference: ~57.44 s
worker-1 cumulative batch inference: ~57.38 s
```

Controlled 4096 request / concurrency 64 / one-word request test with two
workers:

```text
client rps: 1613.8
worker-0: 2068 requests, 129 batches, batch_avg 15.12 ms
worker-1: 2028 requests, 126 batches, batch_avg 15.15 ms
```

Nsight report captured inside the container:

```text
inference_experiment/trtllm_port/artifacts/nsight/indicxlit_inside_two_worker_wait_20260713.nsys-rep
inference_experiment/trtllm_port/artifacts/nsight/indicxlit_inside_two_worker_wait_20260713.sqlite
```

Nsight findings:

```text
two main worker stream groups:
  group A streams 20/22 busy: 1794.65 ms
  group B streams 30/32 busy: 1810.31 ms
  overlap between groups: 486.92 ms
  overlap fraction: ~27%

main streams only:
  stream 22 busy: 1566.29 ms
  stream 32 busy: 1582.38 ms
  overlap: 373.15 ms
  overlap fraction: ~24%

cudaStreamSynchronize:
  total: ~88 ms over capture
  split evenly across the two batch threads

cudaMemcpyAsync:
  ~537k calls
  ~3.30 s CUDA API time
  split evenly across the two batch threads
```

Interpretation:

- The two Rust workers are balanced.
- The observed waiting is mostly GPU stream/kernel serialization, not one Rust
  worker waiting on the other in the queue.
- Two workers do not double throughput because the TensorRT/TensorRT-LLM kernels
  already occupy enough GPU resources that concurrent worker streams overlap
  only partially.

Quick one-worker comparison, same 4096 request / concurrency 64 / one-word
test:

```text
INDICXLIT_WORKERS=1 -> 1590.6 rps
INDICXLIT_WORKERS=2 -> 1613.8 rps
```

So on this synthetic workload, two workers are only slightly faster. The real
benchmark suite should compare `INDICXLIT_WORKERS=1` vs `2` with representative
batch/request shapes before assuming two workers are better.

### 2026-07-13 Engine Rebuild / CUDA Concurrency API Check

Question investigated: if rebuilding the engine is allowed, can we use a CUDA or
TensorRT API to get more "two things at a time" behavior from one GPU?

Relevant TensorRT APIs found in local TRT headers:

```text
IBuilderConfig::setMaxAuxStreams(nbStreams)
ICudaEngine::getNbAuxStreams()
IExecutionContext::setAuxStreams(cudaStream_t* auxStreams, int32_t nbStreams)
```

Current runtime engine properties queried from the compatible runtime image:

```text
encoder num_aux_streams = 0
decoder num_aux_streams = 1
```

So the decoder already has TensorRT internal auxiliary-stream execution. Nsight
also showed this: each Rust worker has a main stream plus an aux stream group.

Public `trtllm-build` v1.1.0 options do not expose `setMaxAuxStreams()`
directly. Tested the available rebuild knob most likely to affect variable
small batches:

```text
--multiple_profiles enable
```

Runtime-compatible rebuild command was run inside
`indicxlit-trtllm:b256-fp16-continuous-kv` so the engine plan matched TRT
10.13.3.9. Output:

```text
inference_experiment/trtllm_port/artifacts/trtllm_engines_en_hi_beam5_runtime_fp16_b256_continuous_kv_multi_profiles
```

Build log still reported:

```text
Total optimization profiles added: 1
```

Correctness/perf test by mounting that rebuilt engine into the Rust executor:

```text
namaste/bharat/kiran -> नमस्ते / भारत / किरन
4096 one-word requests, concurrency 64, workers=2:
  rebuilt --multiple_profiles engine: 1620.8 rps
  previous baseline: ~1613.8 rps
```

Conclusion:

- Rebuilding with public `trtllm-build` profile knobs does not materially change
  the two-worker overlap issue for this decoder.
- `CUDA_DEVICE_MAX_CONNECTIONS` was also tested at 8/16/32:

```text
8  -> 1405 rps
16 -> 1409 rps
32 -> 1530 rps
```

- The remaining promising paths are:
  1. restore two separate Rust server processes under NVIDIA MPS, matching the
     old two-inference-server topology that worked better
  2. patch TensorRT-LLM builder internals to force a different
     `setMaxAuxStreams()` policy, then rebuild and measure
  3. reduce per-step CUDA API/memcpy count in our direct runner, which Nsight
     still shows as a large overhead

### 2026-07-13 Restored NGINX + Two Rust Processes + MPS, 36-SM Engines

The compose stack was changed from one Rust process with two workers back to
two Rust inference processes behind NGINX. Each Rust process has exactly one
TensorRT worker:

```text
client -> host :8000 -> nginx
                    -> inference-1:8000, INDICXLIT_WORKERS=1
                    -> inference-2:8000, INDICXLIT_WORKERS=1
                    -> shared NVIDIA MPS daemon
```

Files updated:

```text
docker_b256/docker-compose.yml
docker_b256/nginx.conf
docker_b256/prometheus.yml
docker_b256/README.md
```

MPS details:

- `mps` service runs `nvidia-cuda-mps-control -d`
- `inference-1` and `inference-2` share:

```text
CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50
```

The first MPS attempt used the old 72-SM engine with
`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50`. TensorRT warned:

```text
The current device only supports 36 multiprocessors but the engine file requires 72.
Deadlocking is likely.
```

So both engines were rebuilt under MPS with the active-thread cap already set to
50, using the patched runtime image `indicxlit-trtllm:b256-fp16-continuous-kv`
so the TensorRT plan version and patched FMHA/continuous-KV libraries match the
serving image.

36-SM engine artifact:

```text
inference_experiment/trtllm_port/artifacts/trtllm_engines_en_hi_beam5_runtime_fp16_b256_continuous_kv_fmha_36sm
```

Build shape:

```text
encoder:
  max_batch_size=256
  max_input_len=128
  max_seq_len=128
  max_beam_width=5
  max_num_tokens=32768
  kv_cache_type=paged
  remove_input_padding=enable
  bert_attention_plugin=auto

decoder:
  max_batch_size=256
  max_input_len=1
  max_seq_len=64
  max_encoder_input_len=4096
  max_beam_width=5
  max_num_tokens=8192
  kv_cache_type=continuous
  remove_input_padding=enable
  gpt_attention_plugin=auto
```

The compose inference services now bind-mount that artifact to:

```text
/models/engines/b256_cont
```

Validation after rebuild/deploy:

```text
docker compose ps:
  mps up
  inference-1 healthy
  inference-2 healthy
  nginx host 8000 -> container 8000
  prometheus host 9090
  grafana host 3000

nvidia-smi:
  two indicxlit-rust-executor-server processes shown as M+C
  nvidia-cuda-mps-server running

logs:
  no TensorRT 36-vs-72 SM mismatch warning after deploying the 36-SM engines

GET http://127.0.0.1:8000/v2/health/ready -> ready
POST namaste/bharat/kiran -> नमस्ते / भारत / किरन
GET http://127.0.0.1:8000/nginx_status -> 43 requests after probe burst

metrics after 40 public requests:
  inference-1 indicxlit_config_workers 1, ok requests 19
  inference-2 indicxlit_config_workers 1, ok requests 22
```

## CUDA 595 Upgrade + MPS Static SM Partitioning

Goal: try NVIDIA MPS static SM partitioning instead of dynamic
`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`.

Host state after upgrade:

```text
NVIDIA-SMI 595.71.05
Driver Version: 595.71.05
CUDA Version: 13.2
GPU: NVIDIA A10, 72 SM
```

Packages installed:

```text
nvidia-driver-595-server
nvidia-compute-utils-595-server
nvidia-utils-595-server
```

The install built the 595 DKMS module for `6.8.0-1046-nvidia`. The old 580
kernel module remained loaded at first, so NVML reported a driver/library
mismatch. Because all GPU clients were stopped, the module was switched live
without reboot:

```bash
sudo systemctl stop nvidia-persistenced.service
sudo rmmod nvidia_uvm nvidia_peermem nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia
sudo modprobe nvidia_uvm
sudo modprobe nvidia_peermem
```

Docker initially failed to start GPU containers because `/run/cdi/nvidia.yaml`
still referenced 580 libraries such as `libcuda.so.580.105.08`. Regenerated the
CDI spec and restarted Docker:

```bash
sudo nvidia-ctk cdi generate --output=/run/cdi/nvidia.yaml
sudo systemctl restart docker
```

Manual static MPS test succeeded:

```text
nvidia-cuda-mps-control -d -S
sm_partition add GPU-f6b31fa7-854b-4642-d739-82eec1c300db 9
sm_partition add GPU-f6b31fa7-854b-4642-d739-82eec1c300db 9

lspart:
GPU-f6 free chunks 0, used chunks 18, free SM 0, used SM 72
partition 1: 9 chunks, 36 SM
partition 2: 9 chunks, 36 SM
```

Compose changes:

- `mps-entrypoint.sh` now starts `nvidia-cuda-mps-control -d -S`.
- It creates two static partitions, default `CUDA_MPS_STATIC_PARTITION_CHUNKS=9`.
- It writes generated partition IDs to `/tmp/nvidia-mps/partition-1` and
  `/tmp/nvidia-mps/partition-2`.
- New `static-mps-client-entrypoint.sh` waits for its partition file, exports
  `CUDA_MPS_SM_PARTITION`, unsets `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`, then
  starts `/opt/indicxlit/bin/indicxlit-rust-executor-server`.
- `inference-1` uses `CUDA_MPS_PARTITION_INDEX=1`.
- `inference-2` uses `CUDA_MPS_PARTITION_INDEX=2`.

Runtime validation:

```text
docker compose ps:
  mps up
  inference-1 healthy
  inference-2 healthy
  nginx host 8000 -> container 8000
  prometheus host 9090
  grafana host 3000

lspart from mps container:
  partition z/MMww...: 9 chunks, 36 SM, clients Yes
  partition MAzzPA8...: 9 chunks, 36 SM, clients Yes

nvidia-smi:
  two indicxlit-rust-executor-server processes shown as M+C
  nvidia-cuda-mps-server running

POST http://127.0.0.1:8000/v2/models/indicxlit/infer
  namaste/bharat/kiran -> नमस्ते / भारत / किरन

Direct backend checks:
  http://inference-1:8000 -> नमस्ते / भारत / किरन
  http://inference-2:8000 -> नमस्ते / भारत / किरन

Prometheus targets:
  dcgm-exporter:9400 up
  inference-1:8000 up
  inference-2:8000 up
```

### 2026-07-13 CPU Pinning, Batch-Cap Fix, and Non-MPS Nsight Profile

Implemented the first two optimization candidates from the previous section:

1. `docker_b256/docker-compose.yml` now pins the two static-MPS inference
   containers to separate host CPUs:
   - `inference-1`: `cpuset: "0"`
   - `inference-2`: `cpuset: "1"`
2. The Rust batcher now treats `INDICXLIT_MAX_BATCH_SIZE` as maximum total
   words/items per merged TensorRT call, not maximum HTTP requests. This fixes
   oversized merged batches when each HTTP request contains multiple words. The
   implementation keeps a single pending `WorkItem` when adding it would exceed
   the word cap.

The Rust image was rebuilt:

```bash
sudo docker build -t indicxlit-trtllm:rust-executor \
  'inference_experiment/trtllm_port/docker_b256/rust_executor_server'
```

Host `cargo` is not installed, so `cargo fmt` was not run on the host. The
Docker build compiled the patched Rust server successfully.

Static-MPS validation after the fix:

```text
30 s, 64 concurrent HTTP requests, 8 words/request, through NGINX :8000:
  ok requests: 16984
  errors: 0
  requests/s: 566.1
  words/s: 4529.1
  p50 latency: 114.2 ms
  p95 latency: 136.4 ms
  p99 latency: 151.2 ms

inference-1 metrics:
  ok requests: 8496
  words: 67968
  batches: 475
  avg words/batch: 143.1
  avg requests/batch: 17.9
  avg engine time/batch: 57.5 ms
  avg queue wait/request: 36.8 ms
  cpu core: 0

inference-2 metrics:
  ok requests: 8488
  words: 67904
  batches: 460
  avg words/batch: 147.6
  avg requests/batch: 18.5
  avg engine time/batch: 59.7 ms
  avg queue wait/request: 44.3 ms
  cpu core: 1
```

Non-MPS single-process profile was then run to get CUDA kernel data, because
Nsight Systems did not expose CUDA kernels cleanly under the static-MPS setup.
The non-MPS profile used the full-GPU runtime-compatible engine:

```text
inference_experiment/trtllm_port/artifacts/trtllm_engines_en_hi_beam5_runtime_fp16_b256_continuous_kv_multi_profiles
```

Clean profile artifact:

```text
inference_experiment/trtllm_port/artifacts/nsight/indicxlit_non_mps_single_clean_20260713.nsys-rep
inference_experiment/trtllm_port/artifacts/nsight/indicxlit_non_mps_single_clean_20260713.sqlite
```

Important profiling note: the standalone Rust server readiness endpoint is
`/v2/health/ready`; `/health` is not valid. `/metrics` also works as a
readiness probe. An earlier non-MPS run waited on `/health`, started load near
the end of the profile, and then produced client connection errors after `nsys`
closed the process. Ignore
`indicxlit_non_mps_single_fixed_batch_20260713.nsys-rep` for throughput
analysis.

Clean non-MPS load result:

```text
40 s, 64 concurrent HTTP requests, 8 words/request, direct to :18000:
  ok requests: 11563
  errors: 0
  requests/s: 289.1
  words/s: 2312.6
  p50 latency: 216.6 ms
  p95 latency: 231.1 ms
  p99 latency: 242.3 ms

server metrics:
  ok requests: 11563
  words: 92504
  batches: 362
  avg words/batch: 255.5
  avg requests/batch: 31.9
  avg engine time/batch: 110.8 ms
  avg queue wait/request: 99.7 ms
```

Nsight Systems summary for the clean non-MPS profile:

```text
CUDA API:
  cudaMemcpyAsync: 6,079,742 calls, 27.31 s API time
  cudaStreamSynchronize: 4,353 calls, 4.17 s API time
  CUDA kernel launches: ~334,488 calls across runtime/driver launch APIs

GPU activity:
  kernel records: 336,660, total kernel time 7.85 s
  memcpy records: 6,080,133, total memcpy time 8.35 s
  memset records: 9,766, total memset time 0.013 s
  kernel timeline span during load: 40.16 s

Memcpy:
  Device-to-device: 6,024,344 copies, 213.9 GB, 8.29 s
  D2D copy size modes:
    65,536 bytes: 2,775,120 copies
    11,264 bytes: 2,775,120 copies
    1,612 bytes: 462,520 copies
  Host-to-device: 51,414 copies, 204.8 MB, 0.052 s
  Device-to-host: 4,375 copies, 122.5 MB, 0.009 s

Top GPU kernels by total time:
  decoder masked_multihead_attention continuous-KV kernels:
    43,440 instances, 4.32 s total, ~99.5 us each
  TRT GEMM / fused Myelin kernels:
    next largest groups, individually 0.57 s, 0.50 s, 0.38 s, 0.27 s
  updateCacheIndirectionKernel:
    3,982 instances, 0.236 s total
  addBiasSoftMax:
    3,982 instances, 0.201 s total
  packFlashAttentionMask:
    362 instances, 0.135 s total

CUDA streams with kernel work:
  stream 22: 264,984 kernels, 7.28 s kernel time
  stream 20: 71,676 kernels, 0.57 s kernel time

OS runtime:
  futex time is high in aggregate, but this is mostly thread waits/idle waits
  across the profiled process and not the primary bottleneck for the hot path.
```

Interpretation:

- Our Rust queue/batcher is now doing what it should for the non-MPS run:
  batches are essentially full at 255.5/256 words.
- The main optimization opportunity visible in the non-MPS profile is not
  request batching; it is the TensorRT-LLM per-step execution shape: millions of
  small D2D copies and hundreds of thousands of kernel launches for 362 batches
  / 3,982 decode steps.
- The dominant GPU compute is decoder masked MHA. The patched continuous-KV
  fused cross-attention path is present too (`fmha_v2_flash_attention...` rows
  appear), but it is not the largest cost in this profile.
- The HTTP test client still uses fresh `urllib` connections; use persistent
  clients for final benchmark numbers. Server-side metrics are more reliable
  than OSRT totals from this client-heavy load.

Practical next optimization options:

1. Benchmark with a production client or persistent HTTP/gRPC client before
   tuning more; per-request connection churn is still noise.
2. For throughput, keep the two static-MPS Rust processes. In this test they
   reached ~4529 words/s combined vs ~2313 words/s for one full-GPU non-MPS
   process, even though the non-MPS process filled batches completely.
3. To improve the single-process path, investigate TensorRT-LLM internals that
   create the D2D copy storm:
   - TensorRT reformatting copy nodes for decoder inputs
   - KV/cache indirection and beam bookkeeping
   - input tensor packing/scatter between our bridge and the decoder session
4. Consider reducing beam width/top-k only if accuracy allows. Beam-related
   kernels (`updateCacheIndirectionKernel`, `beamStage*`, `addBiasSoftMax`) are
   visible every decode step.
5. If we want to change the engine/runtime, the most promising concrete target
   is reducing per-step shape churn/reformatting, not increasing batch size.

### 2026-07-13 Nsight Profile Attempt on Static MPS Runtime

Profile artifacts:

```text
inference_experiment/trtllm_port/artifacts/nsight/indicxlit_static_mps_inference1_20260713_static_mps_rebuilt.nsys-rep
inference_experiment/trtllm_port/artifacts/nsight/indicxlit_static_mps_inference1_20260713_static_mps_rebuilt.sqlite
inference_experiment/trtllm_port/artifacts/nsight/indicxlit_static_mps_inference2_20260713_static_mps_rebuilt.nsys-rep
inference_experiment/trtllm_port/artifacts/nsight/indicxlit_static_mps_inference2_20260713_static_mps_rebuilt.sqlite
inference_experiment/trtllm_port/artifacts/nsight/indicxlit_static_mps_server_20260713_static_mps_server.nsys-rep
inference_experiment/trtllm_port/artifacts/nsight/indicxlit_static_mps_server_20260713_static_mps_server.sqlite
inference_experiment/trtllm_port/artifacts/nsight/indicxlit_static_mps_systemwide_20260713.nsys-rep
inference_experiment/trtllm_port/artifacts/nsight/indicxlit_static_mps_systemwide_20260713.sqlite
```

Temporary profiling compose overrides added:

```text
docker_b256/docker-compose.nsys.yml
docker_b256/docker-compose.nsys-mps.yml
```

Important profiling limitation: with this MPS setup, Nsight CLI traces of the
Rust clients and MPS wrapper did **not** capture CUDA kernel rows. Client reports
show CUDA runtime init/deserialization and OSRT waits, but `cuda_gpu_kern_sum`
is empty. The MPS-wrapper profile also only captured shell/control-daemon OSRT
activity. A host system-wide capture produced a large report, but still no CUDA
kernel/activity tables, only OSRT/system events. For kernel-level analysis, use
a non-MPS single-process run, Nsight Compute on selected kernels, or another
MPS-aware CUPTI capture path.

Successful load run during host system-wide capture:

```text
30 s, 64 concurrent requests, 8 words/request, through NGINX:
  ok requests: 10185
  errors: 0
  requests/s: 339.5
  words/s: 2716.0
  p50 latency: 192.9 ms
  p95 latency: 222.9 ms
  p99 latency: 292.4 ms
```

Service metrics after the run:

```text
inference-1:
  ok requests: 5092
  words: 40736
  TensorRT batches: 283
  avg words/batch: 144.0
  avg requests/batch: 18.0
  avg engine time/batch: 103.7 ms
  avg queue wait/request: 76.5 ms

inference-2:
  ok requests: 5093
  words: 40744
  TensorRT batches: 289
  avg words/batch: 141.0
  avg requests/batch: 17.6
  avg engine time/batch: 101.4 ms
  avg queue wait/request: 68.1 ms
```

Current bottleneck interpretation:

- Both static MPS partitions are active and balanced.
- The engine is usually not seeing full 256-word batches; observed average is
  about 141-144 words per engine call.
- Queue wait is material, but it is buying batching. Increasing batching should
  improve throughput at latency cost.
- Both Rust containers report `indicxlit_worker_cpu_core{worker="worker-0"} 0`;
  if Docker cpusets are not isolating them, both workers may be pinned to host
  CPU 0 and should be separated.
- The HTTP load used new Python `urllib` connections, so the system-wide OSRT
  profile is dominated by `poll`, socket/connect/close, and condition waits.
  Use persistent clients for cleaner server-side conclusions.

Next optimization candidates:

1. Fix CPU affinity at container level: pin `inference-1` and `inference-2` to
   different host CPU cores with Docker `cpuset`, or pass a worker CPU offset
   into the Rust pinning logic.
2. Tune batching for throughput: current average batch is about 56% of the
   256-word engine capacity. Increase `INDICXLIT_BATCH_DELAY_US` for throughput
   runs, or implement a central front queue that fills and dispatches batches to
   whichever partition is free.
3. Use persistent HTTP clients or a binary/gRPC/internal protocol for benchmark
   runs; avoid per-request TCP connection churn.
4. For deeper GPU work, profile a non-MPS single process to inspect kernel mix,
   or use Nsight Compute/another MPS-compatible CUPTI route. Nsight Systems CLI
   did not expose CUDA kernels in the static-MPS runs.

Current note: startup still prints TensorRT's generic engine-plan warning:

```text
Using an engine plan file across different models of devices is not supported
and is likely to affect performance or even cause errors or deadlock.
```

The previous explicit 36-SM-vs-72-SM mismatch warning is not present; the
running engine is the 36-SM artifact.

### 2026-07-13 Static-MPS Runtime-Image Engine Rebuild

Goal: remove the remaining generic TensorRT warning:

```text
Using an engine plan file across different models of devices is not supported
and is likely to affect performance or even cause errors or deadlock.
```

First attempt rebuilt on the host venv under static MPS:

```text
artifacts/trtllm_engines_en_hi_beam5_runtime_fp16_b256_continuous_kv_fmha_static_mps_36sm
```

That artifact was **not usable** by the Rust serving image because the host venv
used TensorRT-LLM `1.3.0rc21`, while the serving image/runtime is the patched
TensorRT-LLM `1.1.0` stack with TensorRT `10.13.3.9`. Runtime failed with:

```text
IRuntime::deserializeCudaEngine: Error Code 6: API Usage Error
The engine plan file is not compatible with this version of TensorRT
```

Successful rebuild was done inside `indicxlit-trtllm:b256-fp16-continuous-kv`
with compose `mps` running in static mode. The builder container used the same
MPS Docker volumes and `CUDA_MPS_SM_PARTITION` from `/tmp/nvidia-mps/partition-1`.

Runtime-compatible artifact:

```text
inference_experiment/trtllm_port/artifacts/trtllm_engines_en_hi_beam5_runtime_fp16_b256_continuous_kv_fmha_static_mps_36sm_runtime
```

Build log:

```text
inference_experiment/trtllm_port/artifacts/build_static_mps_36sm_runtime_20260713_102722.log
```

Build shape stayed the same:

```text
encoder:
  max_batch_size=256
  max_input_len=128
  max_seq_len=128
  max_beam_width=5
  max_num_tokens=32768
  kv_cache_type=paged
  remove_input_padding=enable
  bert_attention_plugin=auto

decoder:
  max_batch_size=256
  max_input_len=1
  max_seq_len=64
  max_encoder_input_len=4096
  max_beam_width=5
  max_num_tokens=8192
  kv_cache_type=continuous
  remove_input_padding=enable
  gpt_attention_plugin=auto
```

Compose now mounts the runtime-compatible static-MPS artifact.

Validation:

```text
inference-1 healthy
inference-2 healthy
NGINX up on host :8000

lspart:
  partition z/MMww...: 9 chunks, 36 SM, clients Yes
  partition MAzzPA8...: 9 chunks, 36 SM, clients Yes

startup logs:
  no TensorRT "engine plan across different models" warning
  no 36-SM-vs-72-SM mismatch warning

POST http://127.0.0.1:8000/v2/models/indicxlit/infer
  namaste/bharat/kiran -> नमस्ते / भारत / किरन

Direct backend checks:
  inference-1 -> नमस्ते / भारत / किरन
  inference-2 -> नमस्ते / भारत / किरन

Prometheus targets:
  dcgm-exporter:9400 up
  inference-1:8000 up
  inference-2:8000 up
```

### 2026-07-13 Avoided KV Tiling `cudaMemcpyAsync` Storm

Nsight showed that the largest D2D copy storm was not TensorRT internal. It
matched our explicit `tileContextCaches()` implementation in
`docker_b256/rust_executor_server/cpp/direct_enc_dec_runner.cc`.

Root cause:

```text
for each request batch after decoder context phase:
  for each decoder layer (6):
    for each batch row (~256):
      for each beam (5):
        cudaMemcpyAsync self KV row
        cudaMemcpyAsync cross KV row
```

That creates about `batch * beam * layers * 2` D2D memcpy launches per engine
batch. In the previous clean non-MPS profile this appeared as:

```text
D2D copies: 6,024,344
D2D bytes: 213.9 GB
Dominant copy sizes:
  65,536 bytes: 2,775,120 copies  (self KV row)
  11,264 bytes: 2,775,120 copies  (cross KV row for ~44-token encoder max)
```

Patch:

- Added `direct_cache_tile.{h,cu}` with `launchTileBeamCopyHalf(...)`.
- Replaced the nested per-row/per-beam `cudaMemcpyAsync` loops with one CUDA
  kernel for self KV and one CUDA kernel for cross KV per decoder layer.
- Updated `cpp/CMakeLists.txt` to build CUDA sources.
- Rebuilt `cpp_build_direct/libindicxlit_trtllm_bridge.so` against the mounted
  v1.1.0 TensorRT-LLM source tree `/tmp/TRTLLM-v1.1.0` and rebuilt
  `indicxlit-trtllm:rust-executor`.

Correctness smoke after rebuild:

```text
POST /v2/models/indicxlit/infer
namaste/bharat/kiran -> नमस्ते / भारत / किरन
```

Clean non-MPS short profile after patch:

```text
artifact:
  artifacts/nsight/indicxlit_non_mps_tile_kv_20260713.nsys-rep
  artifacts/nsight/indicxlit_non_mps_tile_kv_20260713.sqlite

25 s, 64 concurrent HTTP requests, 8 words/request, direct to :18000:
  ok requests: 21162
  errors: 0
  requests/s: 846.5
  words/s: 6771.8
  p50 latency: 71.5 ms
  p95 latency: 78.8 ms
  p99 latency: 82.7 ms

server metrics:
  words: 169296
  batches: 662
  avg words/batch: 255.7
  avg engine time/batch: 37.7 ms
  avg queue wait/request: 26.7 ms
```

Nsight comparison vs previous clean non-MPS profile:

```text
Before:
  D2D copies: 6,024,344
  D2D bytes: 213.9 GB
  total memcpy records: 6,080,133
  engine time/batch: ~110.8 ms
  throughput: ~2312.6 words/s

After:
  D2D copies: 867,664
  D2D bytes: 1.42 GB
  total memcpy records: 969,657
  engine time/batch: ~37.7 ms
  throughput: ~6771.8 words/s
```

The previous 65,536-byte and 11,264-byte D2D copy modes disappeared. The new
explicit tiling kernel appears as:

```text
indicxlit::tileBeamCopyHalfKernel: 7,944 launches, 1.51 s total
```

Remaining D2D copy mode:

```text
1,612 bytes: 846,480 copies, 1.24 s
```

This is no longer KV tiling. It is likely per-step logits / beam-search decode
bookkeeping around padded vocab size 806 (`806 * sizeof(half) = 1612`). It is a
smaller target than the KV tiling bug and probably sits inside TensorRT-LLM
DynamicDecode / beam search rather than our obvious bridge loops.

Static-MPS production topology after patch:

```text
30 s, 64 concurrent HTTP requests, 8 words/request, through NGINX :8000:
  ok requests: 25420
  errors: 0
  requests/s: 847.3
  words/s: 6778.7
  p50 latency: 75.2 ms
  p95 latency: 92.0 ms
  p99 latency: 99.0 ms

inference-1:
  ok requests: 12700
  words: 101600
  batches: 686
  avg words/batch: 148.1
  avg engine time/batch: 38.2 ms
  avg queue wait/request: 21.2 ms
  cpu core: 0

inference-2:
  ok requests: 12720
  words: 101760
  batches: 683
  avg words/batch: 149.0
  avg engine time/batch: 38.5 ms
  avg queue wait/request: 20.9 ms
  cpu core: 1
```

Next copy-related target, if needed: inspect TensorRT-LLM DynamicDecode beam
search for the remaining 1,612-byte D2D copies. That size is one padded-vocab
logit row, and it fires many times per decode step.

### 2026-07-13 Fused Bridge Kernels After KV Copy Fix

Follow-up after the KV tiling fix: additional bridge-side kernel work was done
while intentionally ignoring TensorRT-LLM DynamicDecode internals.

Changes:

- `tileLogits()` now uses `launchTileBeamCopyHalf(...)` instead of per-beam
  `cudaMemcpyAsync`.
- Decoder generation `input_ids` now aliases `newTokens.ptr` directly after
  context step, removing the `newTokens -> inputIds` D2D copy.
- Added `launchPrepareDecodeInputs(...)`, which fuses these per-step bridge
  tasks into one kernel:
  - fill `position_ids`
  - fill `request_types`
  - fill `last_token_ids`
  - fill `context_lengths`
  - initialize `sequence_length` / `sequence_limit_length` during context step
  - set `cross_kv_cache_gen`
  - build the prefix-form `cross_attention_mask` directly on GPU
- Added `launchTileBeamCopyPairHalf(...)`, which fuses self-KV and cross-KV
  tiling into one kernel per decoder layer instead of two kernels per layer.

Correctness:

```text
POST /v2/models/indicxlit/infer
namaste/bharat/kiran -> नमस्ते / भारत / किरन
```

Static-MPS production topology after fusion:

```text
30 s, 64 concurrent HTTP requests, 8 words/request, through NGINX :8000:
  ok requests: 26268
  errors: 0
  requests/s: 875.6
  words/s: 7004.8
  p50 latency: 72.6 ms
  p95 latency: 89.0 ms
  p99 latency: 95.5 ms

inference-1:
  ok requests: 13139
  words: 105107
  batches: 718
  avg words/batch: 146.4
  avg engine time/batch: 36.5 ms
  avg queue wait/request: 19.6 ms
  cpu core: 0

inference-2:
  ok requests: 13130
  words: 105040
  batches: 708
  avg words/batch: 148.4
  avg engine time/batch: 36.9 ms
  avg queue wait/request: 20.5 ms
  cpu core: 1
```

Short non-MPS Nsight profile after fusion:

```text
artifact:
  artifacts/nsight/indicxlit_non_mps_fused_bridge_20260713.nsys-rep
  artifacts/nsight/indicxlit_non_mps_fused_bridge_20260713.sqlite

20 s, 64 concurrent HTTP requests, 8 words/request, direct to :18000:
  ok requests: 18880
  errors: 0
  requests/s: 944.0
  words/s: 7552.0
  p50 latency: 63.7 ms
  p95 latency: 69.9 ms
  p99 latency: 72.9 ms

server metrics:
  words: 151040
  batches: 591
  avg words/batch: 255.6
  avg engine time/batch: 33.8 ms
  avg queue wait/request: 23.0 ms
```

Nsight copy/launch comparison vs the post-KV-copy-fix profile:

```text
Post KV-copy fix, before fusion:
  total memcpy records: 969,657
  D2D copies: 867,664
  D2D bytes: 1.42 GB
  dominant remaining copy: 1,612 bytes x 846,480
  indicxlit tileBeamCopyHalfKernel: 7,944 launches

After fusion:
  total memcpy records: 63,873
  total memcpy bytes: 320.7 MB
  1,612-byte copy mode: gone
  indicxlit tileBeamCopyPairHalfKernel: 3,546 launches, 1.42 s total
  indicxlit prepareDecodeInputsKernel: 6,501 launches, 0.016 s total
  indicxlit tileBeamCopyHalfKernel: 591 launches, 0.008 s total (logit tiling)
```

Remaining bridge-side custom-kernel cost is mostly `tileBeamCopyPairHalfKernel`.
It copies self/cross KV once after context to expand batch rows across beams.
Further bridge-side work would need either a more optimized/vectorized copy
kernel or a runtime layout change that avoids materializing beam-tiled KV at all.

### 2026-07-13 Engine Batch Size Test: b256 vs b512 with Custom C++ Runner

Question: does increasing TensorRT engine `max_batch_size` improve throughput
for the custom Rust + C++ direct runner?

Built a full-GPU runtime-compatible b512 engine inside the same runtime image
used by the Rust executor:

```text
inference_experiment/trtllm_port/artifacts/trtllm_engines_en_hi_beam5_runtime_fp16_b512_continuous_kv_fused_bridge_test
```

Build log:

```text
inference_experiment/trtllm_port/artifacts/build_logs/build_b512_runtime_fused_bridge_test_20260713.log
```

Build shape:

```text
encoder:
  max_batch_size=512
  max_input_len=128
  max_seq_len=128
  max_beam_width=5
  max_num_tokens=65536
  kv_cache_type=paged
  remove_input_padding=enable
  bert_attention_plugin=auto

decoder:
  max_batch_size=512
  max_input_len=1
  max_seq_len=64
  max_encoder_input_len=4096
  max_beam_width=5
  max_num_tokens=8192
  kv_cache_type=continuous
  remove_input_padding=enable
  gpt_attention_plugin=auto
```

The b512 decoder built successfully. The encoder emitted the expected TensorRT
warning that `max_num_tokens=65536` is larger than recommended, but the engine
serialized and loaded.

Test method: single full-GPU non-MPS Rust executor, one worker, custom C++
runner, fused bridge kernels, 8 words/request, 20-second runs, direct to host
`:18000`.

Results:

```text
b256 engine, cap=256, concurrency=64:
  ok requests: 20053
  errors: 0
  requests/s: 1002.7
  words/s: 8021.2
  p50/p95/p99: 60.7 / 66.7 / 69.3 ms
  avg words/batch: 255.1
  avg engine time/batch: 32.1 ms
  avg queue wait/request: 21.1 ms

b256 engine, cap=256, concurrency=128:
  ok requests: 19919
  errors: 0
  requests/s: 996.0
  words/s: 7967.6
  p50/p95/p99: 125.5 / 131.9 / 134.5 ms
  avg words/batch: 255.4
  avg engine time/batch: 32.4 ms
  avg queue wait/request: 85.1 ms

b512 engine, cap=512, concurrency=64:
  ok requests: 16575
  errors: 0
  requests/s: 828.8
  words/s: 6630.0
  p50/p95/p99: 74.2 / 81.3 / 86.0 ms
  avg words/batch: 256.5
  avg engine time/batch: 34.0 ms
  avg queue wait/request: 30.8 ms

b512 engine, cap=512, concurrency=128:
  ok requests: 21328
  errors: 0
  requests/s: 1066.4
  words/s: 8531.2
  p50/p95/p99: 113.3 / 125.2 / 131.3 ms
  avg words/batch: 507.8
  avg engine time/batch: 60.1 ms
  avg queue wait/request: 39.4 ms
```

Interpretation:

- b512 only helps when the queue has enough work to fill it. At concurrency 64,
  average b512 batch stayed around 256 words, so b512 was worse than b256.
- When filled at concurrency 128, b512 improved single-process full-GPU
  throughput from about 8.0k words/s to about 8.5k words/s, roughly +7%.
- Per-batch engine time almost doubles from ~32 ms at 256 words to ~60 ms at
  508 words, but work per batch also doubles, so per-word engine time improves
  slightly.
- Latency tradeoff: b512 at concurrency 128 has lower latency than b256 at
  concurrency 128 because it drains the larger queue more efficiently, but it is
  still higher latency than b256 at concurrency 64.
- This was tested on full GPU non-MPS. The current production static-MPS stack
  uses 36-SM b256 engines. Do not mount this full-GPU b512 engine into the
  36-SM MPS partitions. If we want production MPS b512, rebuild b512 under the
  static MPS partition and test global concurrency high enough to fill each
  worker.

Decision: increasing engine batch size can help the custom C++ runner, but the
benefit is modest and workload-dependent. For low/moderate concurrency, b256 is
better. For high concurrency where a single worker can fill ~512 words per
batch, b512 gives a measurable throughput gain.

### 2026-07-13 Patched Runtime Build Repaired for Rust + Static MPS

Goal: make the engine build use the patched TensorRT-LLM runtime image and
produce a plan that the Rust executor can actually load under the 36-SM static
MPS partition.

What changed:

- `scripts/build_host_b256_engine.sh` now supports a Dockerized build path
  against `indicxlit-trtllm:b256-fp16-continuous-kv`.
- The build wrapper can join the live compose MPS volumes with
  `--volumes-from indicxlit-b256-inference-1-1` and source
  `docker_b256/static-mps-client-env.sh`, which makes `trtllm-build` see the
  36-SM static partition instead of the full 72-SM GPU.
- `docker_b256/docker-compose.yml` now mounts the rebuilt artifact:
  `artifacts/trtllm_engines_en_hi_beam5_runtime_fp16_b256_continuous_kv_fmha_static_mps_36sm_runtime_patchedbuild_36sm`

Successful verification on 2026-07-13:

- `inference-1` and `inference-2` came back healthy on `http://127.0.0.1:8000`
- no TensorRT 36-vs-72 SM mismatch warning in the startup logs
- sample request:

```bash
curl -X POST http://127.0.0.1:8000/v2/models/indicxlit/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [
      {"name":"text_input","shape":[1,1],"datatype":"BYTES","data":["bharat"]},
      {"name":"target_lang","shape":[1,1],"datatype":"BYTES","data":["hi"]},
      {"name":"max_tokens","shape":[1,1],"datatype":"INT32","data":[32]},
      {"name":"beam_width","shape":[1,1],"datatype":"INT32","data":[5]},
      {"name":"topk","shape":[1,1],"datatype":"INT32","data":[5]}
    ]
  }'
```

Response:

```json
{"model_name":"indicxlit","outputs":[{"data":["भारत"],"datatype":"BYTES","name":"text_output","shape":[1,1]},{"data":["[\"भारत\",\"भरत\",\"भार्त\",\"भराट\",\"भरात\"]"],"datatype":"BYTES","name":"candidates_json","shape":[1,1]}]}
```

Net result: patched build path is now compatible with the Rust executor and the
compose stack is using the corrected 36-SM engine artifact.

### 2026-07-14 Rust Executor Production Build Cleanup

Goal: make the Rust executor build process reproducible and remove the hidden
dependency on a prebuilt `cpp_build_direct/libindicxlit_trtllm_bridge.so`.

Changes made:

- `docker_b256/rust_executor_server/Dockerfile` now builds
  `libindicxlit_trtllm_bridge.so` from `cpp/` during the image build.
- `scripts/build_rust_executor_image.sh` wraps the BuildKit command and passes
  TensorRT-LLM source as a named Docker build context.
- `docker_b256/rust_executor_server/.dockerignore` excludes local generated
  build outputs.
- `scripts/prepare_rust_runtime_base_context.sh` prepares a generated
  `docker_b256/context_runtime` base-image context from explicit engine,
  asset, and patched-library inputs.
- `docker_b256/docker-compose.yml` can select the Rust executor image with
  `INDICXLIT_RUST_EXECUTOR_IMAGE`.

Important ABI finding:

- Building the Rust bridge against `/home/ubuntu/TensorRT-LLM`
  (`b172f0be4 Support cross-attention with contiguous KV cache`) succeeded at
  compile time but failed at process startup:

```text
undefined symbol: _ZN12tensorrt_llm3_v16common13TllmExceptionD1Ev
```

- That means `/home/ubuntu/TensorRT-LLM` is not ABI-compatible with the
  TensorRT-LLM runtime libraries currently inside
  `indicxlit-trtllm:b256-fp16-continuous-kv`.
- The running stack was restored by rebuilding `indicxlit-trtllm:rust-executor`
  against `/tmp/TRTLLM-v1.1.0`, which matches the current runtime image ABI.

Current verified state:

- `inference-1` and `inference-2` healthy under compose.
- startup logs show both workers load under static MPS partitions.
- sample request inside `inference-1`:
  `bharat -> भारत`.

Production implication:

To truly use `/home/ubuntu/TensorRT-LLM` as the single source of truth, rebuild
the base image/runtime libraries from that same checkout first, then rebuild:

1. `indicxlit-trtllm:b256-fp16-continuous-kv`
2. `indicxlit-trtllm:rust-executor`
3. the 36-SM engine artifact

Until that is done, the validated Rust executor bridge source root is
`/tmp/TRTLLM-v1.1.0`.
