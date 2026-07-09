# IndicXlit TensorRT-LLM Chat Handoff

This is a curated handoff from the Codex session. It is not a verbatim chat
transcript; it preserves the decisions, measurements, failed attempts, and next
steps needed to continue the work.

## Goal

Port the current IndicXlit Fairseq transformer transliteration inference path to
a TensorRT-LLM encoder-decoder experiment. Keep all work isolated in this
workspace until correctness and performance are proven. The production Flask/app
path should remain unchanged until the final integration phase.

Initial target:

- Direction: `en2indic`
- Language: Hindi (`hi`)
- Correctness target: greedy parity first, then beam search
- Precision target: fp32 first for parity, fp16 later only after correctness is
  understood

## Why TensorRT-LLM

The main bottleneck is not model size or GPU memory. The model is small:

- 6 encoder layers
- 6 decoder layers
- hidden dim 256
- FFN dim 1024
- 4 attention heads
- char-level target vocab around 806 symbols
- Fairseq checkpoint around 138 MB

The bottleneck is the autoregressive decode loop:

```text
for each output step:
    decoder forward
    log_softmax
    topk / beam select
    reorder incremental state
    finalize EOS candidates
```

Fairseq runs a lot of this loop through Python and generic beam-search
bookkeeping. GPU utilization stayed low because the GPU receives many small
bursts of work and waits between decode steps. TensorRT-LLM is relevant because
it can own the encoder-decoder generation loop, KV-cache management, and beam
search in optimized runtime code.

## Measurements and Experiments Already Done

### CPU/Fairseq Baseline

Before CUDA was enabled, direct batch decode on CPU plateaued roughly around
120-135 items/s depending on batch size. Flask/gevent queued concurrent requests
because the process was compute-bound and the app did not batch HTTP requests.

### CUDA Setup

The original `.venv` was CPU-only PyTorch. It was updated to:

- `torch 2.12.1+cu130`
- `torch.version.cuda = 13.0`
- `torch.cuda.is_available() = True`
- GPU observed: NVIDIA GeForce RTX 5060 Ti, about 8 GB VRAM

This is the existing Fairseq app venv. Do not install TensorRT-LLM into it.

### Fairseq GPU Throughput

Greedy-like decode:

- `beam=1 topk=1`
- large batches plateaued around 400-440 items/s in the benchmark script

Default-ish beam decode:

- `beam=8 topk=5`
- large batches plateaued around 270 items/s in earlier tests
- artifacts in `artifacts/fairseq_baseline/` contain the current locked
  baseline rows

Important caveat: one-shot `nvidia-smi` row-boundary samples underreport
utilization. Trust throughput more than the sampled utilization number.

### Multiple Model Instances

The benchmark was modified with `--model-instances N` to load independent model
replicas and run one Python thread per instance.

Observed behavior:

- 2 replicas improved aggregate throughput by roughly 25-30%.
- 4 replicas regressed.
- Per-instance throughput dropped as instances increased.

Interpretation: multiple replicas add some concurrency, but contention appears
quickly. This does not solve the decode-loop problem.

### Custom Greedy Decoder Experiment

`../greedy_decode_experiment.py` implements a specialized Fairseq greedy decoder
outside app/site-packages. It bypasses generic `SequenceGenerator` beam logic for
`beam=1`.

Observed earlier:

- Fairseq batch 512: around 420 items/s
- custom greedy batch 512: around 490-507 items/s

This supported the theory that Fairseq generation overhead is meaningful.

## TensorRT-LLM Workspace Artifacts

Current workspace:

```text
inference experiment/trtllm_port/
```

Important files:

- `README.md`: workspace overview
- `PHASE_STATUS.md`: current phase status and blocker
- `capture_fairseq_baseline.py`: captures Fairseq output/benchmark artifacts
- `inspect_indicxlit_checkpoint.py`: inspects checkpoint and emits mapping
  report
- `probe_trtllm_env.py`: checks TensorRT-LLM environment availability
- `words_en_hi.txt`: fixed baseline word list

Important artifacts:

- `artifacts/fairseq_baseline/manifest.json`
- `artifacts/fairseq_baseline/*.json`
- `artifacts/fairseq_baseline/*_benchmark.csv`
- `artifacts/indicxlit_checkpoint_mapping.json`
- `artifacts/trtllm_env_probe.json`

## Phase Status

### Phase 0: Baseline Lock

Complete.

The baseline capture tool created four cases:

- `greedy_no_postprocess`
- `beam8_top5_no_postprocess`
- `greedy_postprocess`
- `beam8_top5_postprocess`

Each case has JSON output artifacts and benchmark CSV rows.

### Phase 1: TensorRT-LLM Environment

Blocked locally by disk capacity.

Findings:

- Current app `.venv` should not receive TensorRT-LLM.
- A separate `.venv-trtllm` was created.
- `tensorrt~=10.14.1` installed successfully there.
- `tensorrt-llm==1.2.1` failed due `OSError: [Errno 28] No space left on
  device`.
- The failing install built a `tensorrt_llm-1.2.1` wheel around 2.5 GB and
  pulled a large Torch/CUDA dependency stack.
- The local 32 GB filesystem is not enough.

Cloud recommendation:

- GPU: L4/A10/A10G/A100/RTX 4090 class
- VRAM: 16 GB minimum, 24 GB comfortable
- CPU: 8 vCPU minimum
- RAM: 32 GB minimum, 64 GB comfortable
- Disk: 150 GB minimum, 250 GB comfortable
- Prefer NVIDIA's prebuilt TensorRT-LLM container over pip in the app venv.

### Phase 2: Checkpoint Mapping

Complete for dry-run mapping.

Result from `inspect_indicxlit_checkpoint.py`:

- 267 tensors found
- 261 required inference tensors mapped
- 0 missing required tensors
- 6 metadata/sentinel tensors intentionally ignored

The mapping report uses descriptive target names. Final converter names must be
matched against the TensorRT-LLM version installed on the cloud machine.

## Full Phase Plan

### Phase 3: Greedy TensorRT-LLM Export

Implement only after TensorRT-LLM is available in a suitable environment.

Tasks:

- Create an IndicXlit-specific converter or adapt TensorRT-LLM's encoder-decoder
  NMT converter.
- Convert only `en2indic` first.
- Build encoder and decoder engines with:
  - `tp_size=1`
  - `pp_size=1`
  - `max_beam_width=1`
  - `max_batch_size=512`
  - `max_input_len=128`
  - `max_seq_len=64`
  - fp32 precision first
- Write a runner accepting already-preprocessed char-token strings.

Success gate:

- Converter completes with no missing weights.
- Engine build succeeds.
- Runner returns non-empty output for one Hindi transliteration input.
- Batch sizes 1, 32, and 128 run without crashing.

### Phase 4: Greedy Parity

Compare against Phase 0 fixed baseline.

Compare:

- preprocessed source string
- source token ids
- generated token ids
- raw decoded char string
- final postprocessed output

Success gate:

- At least 95% exact final output match against Fairseq greedy.
- Any mismatch has a token-id diff artifact.
- Batch output order matches input order.

### Phase 5: Beam Search

Only after greedy parity is stable.

Tasks:

- Rebuild decoder engine with `max_beam_width=8`.
- Run with `num_beams=8`.
- Match Fairseq best output first.
- Treat full `topk=5` candidate parity as optional v2 if TRT-LLM runner does not
  expose the same candidate shape naturally.

Success gate:

- Beam run completes for batch sizes 1, 32, and 128.
- Best output matches Fairseq beam best output for at least 95% of fixed words.
- Throughput is recorded against Fairseq `beam=8 topk=5`.

### Phase 6: Performance Benchmark

Benchmark TensorRT-LLM against the locked Fairseq baseline.

Use:

- batch sizes 1, 32, 128, 512
- greedy and beam
- postprocess off and on
- fixed target item count

Record:

- items/s
- wall ms/item
- CPU cores
- RSS
- GPU memory
- continuous GPU utilization if practical

Success gate:

- Greedy TRT-LLM beats Fairseq greedy by at least 25%, or explain why not.
- Beam TRT-LLM beats Fairseq beam by at least 50%, or explain why not.
- CPU usage drops for beam decode.
- No correctness regression versus parity gates.

### Phase 7: App Integration Decision

Only integrate if correctness and performance justify it.

Tasks:

- Add runtime-selectable backend:
  - `fairseq` default
  - `trtllm` opt-in
- Share existing preprocessing/postprocessing.
- Validate engine files at startup.
- Fail fast if selected backend assets are missing.

Success gate:

- Existing Fairseq app behavior unchanged by default.
- TRT-LLM backend serves the same API shape.
- HTTP smoke test passes.
- End-to-end API benchmark shows a real gain.
- Rollback is one config change.

## Important Warnings for the Next Agent

- Do not edit `.venv` or install TensorRT-LLM into it.
- Do not modify production app serving code until Phase 7.
- Do not commit `.venv-trtllm`; it is ignored.
- The Fairseq checkpoint stores `args=None`; config lives under `ckpt["cfg"]`.
- IndicXlit preprocessing prefixes inputs with the target language token, e.g.
  `__hi__ b h a r a t`.
- Fairseq special ids are expected to be `<s>=0`, `<pad>=1`, `</s>=2`,
  `<unk>=3`; verify against dictionaries/runtime before final converter work.

