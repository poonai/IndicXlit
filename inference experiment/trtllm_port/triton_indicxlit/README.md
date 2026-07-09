# IndicXlit Triton TensorRT-LLM Prototype

This folder contains a separate Triton model repository prototype for serving
the IndicXlit TensorRT-LLM encoder-decoder engine with custom Python
preprocessing and postprocessing.

## Layout

```text
model_repository/
  indicxlit_preprocess/      Python backend: roman word -> token ids
  indicxlit_tensorrt_llm/    TensorRT-LLM backend: encoder-decoder engine
  indicxlit_postprocess/     Python backend: token ids -> native text/rescore
  indicxlit_ensemble/        Triton ensemble wiring
```

The TensorRT-LLM backend points at:

```text
/IndicXlit/inference experiment/trtllm_port/artifacts/trtllm_engines_en_hi_beam5_triton/encoder
/IndicXlit/inference experiment/trtllm_port/artifacts/trtllm_engines_en_hi_beam5_triton/decoder
```

These are separate from the parity-check engines because Triton
`inflight_fused_batching` requires engines built with packed input and paged KV
cache.

## Run With Triton

This machine has a local standalone Triton Server install at:

```text
/tmp/tritonserver-2.70.0/tritonserver
```

The local install was verified with Triton `server_version 2.70.0`. The
standalone Triton release does not ship the TensorRT-LLM backend, so the backend
was built from the TensorRT-LLM v1.2.1 `triton_backend/inflight_batcher_llm`
source and installed here:

```text
/tmp/tritonserver-2.70.0/tritonserver/backends/tensorrtllm/libtriton_tensorrtllm.so
/tmp/tritonserver-2.70.0/tritonserver/backends/tensorrtllm/trtllmExecutorWorker
```

Launch the local server:

```bash
"inference experiment/trtllm_port/triton_indicxlit/scripts/run_triton.sh"
```

The public ensemble model is `indicxlit_ensemble`. The current HTTP port is
`8010` unless overridden with `HTTP_PORT`.

## Build Triton-Compatible Engines

The Triton backend uses:

- `gpt_model_type=inflight_fused_batching`
- `remove_input_padding=enable`
- `kv_cache_type=paged`
- beam width up to `5`

Rebuild the local engine artifacts with:

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

## Verified Requests

Python-stage checks passed:

```bash
curl -X POST localhost:8010/v2/models/indicxlit_preprocess/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[{"name":"TEXT","shape":[1,1],"datatype":"BYTES","data":["bharat"]},{"name":"TARGET_LANG","shape":[1,1],"datatype":"BYTES","data":["hi"]},{"name":"REQUEST_OUTPUT_LEN","shape":[1,1],"datatype":"INT32","data":[32]}]}'
```

The postprocess model was also loaded and decoded Hindi token IDs for
`भारत` correctly.

Full ensemble request:

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

Concurrent single-item HTTP requests also work through the ensemble and hit the
TensorRT-LLM backend successfully. A local probe with 16 requests at concurrency
8 completed with zero Triton failures and the TensorRT-LLM metrics reported 17
successful `indicxlit_tensorrt_llm` requests total after the smoke tests.

Client-side batch tensors, for example one HTTP request with shape `[4, 1]`, are
not supported by this ensemble shape in non-decoupled mode. The TensorRT-LLM
backend reported:

```text
Batch size > 1 requires the tensorrt_llm backend to be using decoupled transaction policy
```

Switching the backend to decoupled mode lets the core backend accept that path,
but Triton rejects the current static ensemble because postprocessing combines
outputs from a decoupled model with other tensors. For server-side batching,
send many normal single-item requests and let the TensorRT-LLM inflight batcher
schedule them internally.

## Local Dry Run

Without Triton, validate the Python preprocessing/postprocessing against the
actual TensorRT-LLM engine:

```bash
source "inference experiment/trtllm_port/env_trtllm.sh"
python "inference experiment/trtllm_port/triton_indicxlit/scripts/dry_run_pipeline.py"
```
