# TensorRT-LLM Port Phase Status

## Phase 0: Baseline Lock

Status: complete in this workspace.

Artifacts:

- `artifacts/fairseq_baseline/manifest.json`
- `artifacts/fairseq_baseline/*_benchmark.csv`
- `artifacts/fairseq_baseline/*.json`

## Phase 1: TensorRT-LLM Environment Spike

Status: blocked by local disk capacity.

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
- 261 required inference tensors mapped.
- 0 missing required tensors.
- 6 metadata/sentinel tensors intentionally ignored.
