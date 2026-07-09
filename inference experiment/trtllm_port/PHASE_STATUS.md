# TensorRT-LLM Port Phase Status

## Phase 0: Baseline Lock

Status: complete in this workspace.

Artifacts:

- `artifacts/fairseq_baseline/manifest.json`
- `artifacts/fairseq_baseline/*_benchmark.csv`
- `artifacts/fairseq_baseline/*.json`

## Phase 1: TensorRT-LLM Environment Spike

Status: complete on the current cloud machine.

Current cloud environment:

- GPU: NVIDIA GeForce RTX 4090, 24 GB VRAM.
- Driver: 580.159.04, reported CUDA capability/runtime support: 13.0.
- Disk after install: about 60 GB free on the 80 GB filesystem.
- RAM: about 503 GiB total.

Installed isolated environment:

- `.venv-trtllm` uses Python 3.10.12.
- `tensorrt-llm==1.2.1`
- `tensorrt==10.14.1.48.post1`
- `torch==2.9.1+cu128`
- `nvidia-cublas==13.6.0.2`

Activation:

```bash
source "inference experiment/trtllm_port/env_trtllm.sh"
```

Validation:

- `tensorrt_llm` imports successfully after activation.
- `trtllm-build --help` runs successfully after activation.
- `torch.cuda.is_available()` is true inside `.venv-trtllm`.
- Latest probe artifact: `artifacts/trtllm_env_probe.json`.

Notes:

- Python 3.11 could not use the `tensorrt-llm==1.2.1` wheel because PyPI/NVIDIA
  exposed compatible x86_64 wheels for CPython 3.10 and 3.12, not 3.11.
- Python 3.10 needed `python3.10-venv`, `libpython3.10`, and `python3.10-dev`.
- `nvidia-cublas` was needed because TensorRT-LLM imports require
  `libcublasLt.so.13`.
- The activation helper sets `LD_LIBRARY_PATH` for CUDA 13 cuBLAS and TensorRT
  shared libraries from the venv.

Previous local-machine findings:

Findings:

- CUDA is available through PyTorch.
- `tensorrt_llm` is not installed.
- `trtllm-build` is not installed.
- A pip dry-run for `tensorrt-llm==1.2.1` attempted to resolve a large,
  incompatible runtime stack for this app venv, including a different Torch
  version and many CUDA/TensorRT packages. It was cancelled before installing
  anything.
- A separate `.venv-trtllm` was created to keep the Fairseq app venv untouched.
- `tensorrt~=10.14.1` installed successfully in `.venv-trtllm`.
- Installing `tensorrt-llm==1.2.1` in `.venv-trtllm` still failed with
  `OSError: [Errno 28] No space left on device` during package installation.
- The failing install built a `tensorrt_llm-1.2.1` wheel of about 2.5 GB and
  downloaded a large Torch/CUDA dependency stack, including Torch 2.9.1 and
  multiple CUDA 12 libraries. The local 32 GB filesystem is not enough.

Decision:

- Do not install TensorRT-LLM into the existing Fairseq app venv.
- Continue Phase 3+ on a cloud machine with at least 150 GB disk, preferably
  using NVIDIA's prebuilt TensorRT-LLM container.

## Phase 2: Checkpoint Introspection and Mapping Spec

Status: complete for dry-run mapping.

Artifacts:

- `artifacts/indicxlit_checkpoint_mapping.json`

Result:

- 267 tensors found in the Fairseq checkpoint.
- 261 required inference tensors mapped in the descriptive report.
- 0 missing required tensors.
- 6 metadata/sentinel tensors intentionally ignored.
- The downloaded en2indic checkpoint has `54` source embeddings and `806`
  target embeddings. The Fairseq vocabulary is `4` specials + dictionary file
  entries + `22` language tokens from `en2indic/lang_list.txt`.

## Phase 3: Greedy TensorRT-LLM Export

Status: complete for fp32 greedy smoke and baseline parity.

Model assets:

- Downloaded official en2indic v1.0 model release into the ignored model
  assets directory:
  `app/ai4bharat/transliteration/transformer/models/en2indic/v1.0`.
- Checkpoint: `transformer/indicxlit.pt`.
- Dictionaries: `corpus-bin/dict.*.txt`.

Converter:

- Script: `convert_indicxlit_to_trtllm.py`.
- Output checkpoint dir: `artifacts/trtllm_checkpoint_en_hi/`.
- Writes TensorRT-LLM `EncoderModel` and `DecoderModel` checkpoints:
  `config.json` + `rank0.safetensors` under `encoder/` and `decoder/`.
- Q/K/V Fairseq projection weights are concatenated into TensorRT-LLM fused
  `qkv` tensors.
- Precision: fp32.
- Includes encoder and decoder final layernorms from the Fairseq checkpoint.
- Includes Fairseq-compatible embedding scale `sqrt(256)=16.0`.
- Includes generated sinusoidal positional embedding tables for encoder and
  decoder. Fairseq uses sinusoidal/non-checkpoint positions because
  `encoder_learned_pos=false`, `decoder_learned_pos=false`, and
  `no_token_positional_embeddings=false`.

Engine build:

- Output engine dir: `artifacts/trtllm_engines_en_hi/`.
- Encoder build:
  - `max_batch_size=512`
  - `max_input_len=128`
  - `max_seq_len=128`
  - `max_beam_width=1`
  - `remove_input_padding=disable`
- Decoder build:
  - `max_batch_size=512`
  - `max_input_len=1`
  - `max_seq_len=64`
  - `max_encoder_input_len=4096`
  - `max_beam_width=1`
  - `remove_input_padding=disable`

Runner:

- Script: `run_trtllm_greedy.py`.
- Uses Fairseq-compatible ids: `<s>=0`, `<pad>=1`, `</s>=2`, `<unk>=3`;
  dictionary symbols follow, then `__lang__` tokens.
- Uses `</s>` as the one-token decoder prompt and strips that prompt from
  generated output before decoding.

Smoke results:

- Batch 1: passed, output shape `[1, 1, 33]`, first output `bharat -> भारत`,
  about `3.01` items/s.
- Batch 32: passed, output shape `[32, 1, 33]`, first output `bharat -> भारत`,
  about `91.47` items/s.
- Batch 128: passed, output shape `[128, 1, 33]`, first output `bharat -> भारत`,
  about `330.30` items/s.
- Baseline word list: passed, output shape `[22, 1, 33]`, exact greedy match
  against `artifacts/fairseq_baseline/greedy_postprocess.json`.

Weight-port verification:

- `verify_trtllm_weight_port.py` confirms the converted TensorRT-LLM checkpoint
  tensors exactly match the mapped Fairseq tensors:
  - encoder max absolute diff: `0.0`
  - decoder max absolute diff: `0.0`
  - exact tensor port: `true`
  - Fairseq embedding scale: `16.0`
  - TensorRT-LLM encoder `has_embedding_scale`: `true`
  - Fairseq uses token positions: `true`
  - TensorRT-LLM encoder `has_position_embedding`: `true`

Parity:

- `compare_trtllm_parity.py` wrote
  `artifacts/trtllm_parity_report.json`.
- Compared outputs: `22`.
- Missing outputs: `0`.
- Exact matches: `22`.
- Exact match rate: `1.0`.
- Greedy gate: passed.

Benchmark:

- `benchmark_batch_decode.py` now supports `--backend trtllm` for the ported
  greedy TensorRT-LLM engine.
- Latest single-instance benchmark artifact:
  `artifacts/trtllm_benchmark_batch_decode.csv`.
- Command shape:

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

- Correctness-valid masked results on the current RTX 4090 fp32
  correctness-first engine:
  - batch 1: `81.10` items/s, `12.33` ms/item
  - batch 32: `884.10` items/s, `1.13` ms/item
  - batch 128: `2204.87` items/s, `0.45` ms/item

Beam-4 Dakshina check:

- `run_trtllm_greedy.py` now supports beam decoding. Defaults are
  `--beam-width 4 --topk 4`; pass `--beam-width 1 --topk 1` for the earlier
  greedy path.
- Beam-4 engine dir:
  `artifacts/trtllm_engines_en_hi_beam4/`.
- Dakshina source:
  `https://github.com/google-research-datasets/dakshina`, release archive
  `dakshina_dataset_v1.0.tar`.
- Extracted Hindi lexicon test file:
  `artifacts/dakshina_data/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv`.
- Evaluation script:
  `evaluate_dakshina_trtllm.py`.
- Evaluation filtering mirrors the repo's old Dakshina script for Hindi:
  first two TSV columns, ASCII-only roman side, Devanagari-only native side,
  lowercased roman side, deduped pairs.
- Filtered/evaluated rows: `4502`.
- README Hindi Dakshina Top-1 target: `60.56`.
- Fairseq verification environment:
  - isolated venv: `.venv-fairseq`
  - `fairseq==0.12.2`
  - `torch==2.13.0+cu130`
  - GPU: NVIDIA GeForce RTX 4090
  - app dependency install and official `word_prob_dicts` rescoring archive
    are present locally, both ignored by git.
- Fairseq verification script:
  `evaluate_dakshina_fairseq.py`.
- Fairseq beam-5 README-equivalent full Hindi Dakshina result:
  - artifact: `artifacts/dakshina_hi_fairseq_beam5_eval.json`
  - rows: `4502`
  - raw Top-1: `2725/4502 = 60.53%`
  - raw Top-5: `3921/4502 = 87.09%`
  - throughput: `159.34` items/s
  - README target gap: `60.53%` vs `60.56%`, lower by `0.03` percentage
    points. This verifies the README Hindi Dakshina claim within rounding and
    evaluation-script noise.
- Fairseq package-rescored beam-5 result using app alpha `0.9`:
  - rescored Top-1: `3284/4502 = 72.95%`
  - rescored Top-5: `3921/4502 = 87.09%`
- Fairseq package-default beam-4 full Hindi Dakshina result:
  - artifact: `artifacts/dakshina_hi_fairseq_beam4_eval.json`
  - rows: `4502`
  - raw Top-1: `2716/4502 = 60.33%`
  - raw Top-4: `3812/4502 = 84.67%`
  - throughput: `171.81` items/s
- Fairseq package-rescored beam-4 result using app alpha `0.9`:
  - rescored Top-1: `3253/4502 = 72.26%`
  - rescored Top-4: `3812/4502 = 84.67%`
- The README Hindi Dakshina table matches raw Fairseq beam output, not the
  package's default rescored API output.
- TensorRT-LLM attention-mask fix:
  - Root cause of the earlier `57.89%` beam-4 result was missing encoder
    attention masks in the TRT runner/evaluator. In mixed-length batches,
    decoder cross-attention could attend to padded encoder positions.
  - `run_trtllm_greedy.py` now creates `attention_mask = encoder_input_ids !=
    PAD_ID` and passes it to `EncDecModelRunner.generate`.
  - `evaluate_dakshina_trtllm.py` uses the same mask for all TRT Dakshina
    evaluations.
- Corrected TRT beam-4 result with attention mask:
  - artifact: `artifacts/dakshina_hi_trtllm_beam4_eval_masked_b32.json`
  - batch size: `32`
  - Top-1: `2715/4502 = 60.31%`
  - Top-4: `3821/4502 = 84.87%`
  - throughput: `805.74` items/s
  - Fairseq beam-4 comparison: `2716/4502 = 60.33%` Top-1 and
    `3812/4502 = 84.67%` Top-4.
  - Top-1 gap vs Fairseq beam-4: `-0.02` percentage points, one row.
- Corrected TRT beam-5 README-equivalent result with attention mask:
  - beam-5 engine dir: `artifacts/trtllm_engines_en_hi_beam5/`
  - artifact: `artifacts/dakshina_hi_trtllm_beam5_eval_masked_b32.json`
  - batch size: `32`
  - Top-1: `2725/4502 = 60.53%`
  - Top-5: `3920/4502 = 87.07%`
  - throughput: `799.00` items/s
  - Fairseq beam-5 comparison: `2725/4502 = 60.53%` Top-1 and
    `3921/4502 = 87.09%` Top-5.
  - README Hindi Dakshina target: `60.56`; corrected TRT is within `0.03`
    percentage points.

Known caveats before optimization:

- Initial builds with padding removal enabled failed for larger batches because
  decoder cross-attention mask dimensions used packed encoder-token lengths
  beyond the profile. The current correctness-first engine disables padding
  removal and uses a larger decoder `max_encoder_input_len` profile.
- The current stable decoder profile is clean for the locked benchmark batch
  sizes `1,32,128`. A wider experimental profile for `256/512` and the 22-word
  list exceeded 24 GB VRAM at load time on the RTX 4090, so those rows need a
  separate optimization/profile pass before reporting.
- A true batch `1000` run does not fit the current setup:
  - The stable engine rejects it because `max_batch_size=512`.
  - A separate experimental `max_batch_size=1000` encoder build succeeded, but
    the matching decoder build with `max_encoder_input_len=32768` failed while
    trying to allocate about `25.17` GB during TensorRT tactic selection, above
    the RTX 4090's 24 GB VRAM.
- README-level Dakshina parity is now achieved for TensorRT-LLM after the
  attention-mask fix. The corrected beam-5 TRT result is `60.53%` vs README
  `60.56%` and Fairseq beam-5 `60.53%`.
- The current export is fp32 and correctness-first. INT8/FP8/FP16 optimization,
  padding-removal tuning, and larger-batch throughput work are still future
  phases.

## Triton TensorRT-LLM Serving Spike

Status: working for local non-decoupled ensemble serving.

Artifacts:

- Triton prototype repo:
  `triton_indicxlit/model_repository/`
- Launch helper:
  `triton_indicxlit/scripts/run_triton.sh`
- Triton-compatible engine dir:
  `artifacts/trtllm_engines_en_hi_beam5_triton/`

Environment:

- Local standalone Triton Server `2.70.0` installed at
  `/tmp/tritonserver-2.70.0/tritonserver`.
- Built TensorRT-LLM v1.2.1 `triton_backend/inflight_batcher_llm` locally and
  installed:
  - `/tmp/tritonserver-2.70.0/tritonserver/backends/tensorrtllm/libtriton_tensorrtllm.so`
  - `/tmp/tritonserver-2.70.0/tritonserver/backends/tensorrtllm/trtllmExecutorWorker`
- The launch helper wires the Triton process to `.venv-trtllm` libraries for
  TensorRT-LLM, TensorRT, Torch, CUDA, cuDNN, cuBLAS, and NCCL.

Engine build:

- A separate beam-5 engine was required for Triton inflight batching:
  - `remove_input_padding=enable`
  - `kv_cache_type=paged`
  - encoder `max_batch_size=128`, `max_input_len=128`,
    `max_num_tokens=16384`
  - decoder `max_batch_size=128`, `max_seq_len=64`,
    `max_encoder_input_len=4096`, `max_num_tokens=8192`
- The earlier correctness/parity engine remains separate because it uses
  `remove_input_padding=disable`.

Verified:

- Triton loaded all models as `READY`:
  - `indicxlit_preprocess`
  - `indicxlit_tensorrt_llm`
  - `indicxlit_postprocess`
  - `indicxlit_ensemble`
- Direct TensorRT-LLM backend request for `bharat` returned token IDs decoding
  to `भारत`.
- Full ensemble HTTP request returned:
  - `text_output`: `भारत`
  - candidates: `["भारत", "भरत", "अभारत", "बारत", "बहरत"]`
- Concurrent HTTP probe:
  - `16` single-item requests at concurrency `8`
  - all completed successfully
  - local measured rate: `38.07` items/s through HTTP + Python pre/postprocess
  - Triton metrics after smoke tests showed `17` successful
    `indicxlit_ensemble` requests and `17` successful
    `indicxlit_tensorrt_llm` requests with zero failures.

Serving caveat:

- The current static ensemble works in non-decoupled mode and should be driven
  with many normal single-item requests so TensorRT-LLM can schedule them
  internally.
- A single HTTP request carrying client-side batch tensors, such as shape
  `[4, 1]`, is rejected by the TensorRT-LLM backend unless decoupled mode is
  enabled.
- Enabling decoupled mode on the core backend made Triton reject this static
  ensemble because postprocessing combines tensors from a decoupled model with
  other ensemble values. Supporting client-side batched tensors would need a
  custom BLS/Python wrapper or a different decoupled model layout.
