# TensorRT-LLM IndicXlit Port Workspace

This directory contains experiment-only tooling for evaluating a TensorRT-LLM
port of the IndicXlit Fairseq transformer. Nothing here is wired into the app.

## Phase gates

1. `capture_fairseq_baseline.py` locks Fairseq outputs and benchmark metrics.
2. `probe_trtllm_env.py` checks whether TensorRT-LLM tooling is available.
3. `inspect_indicxlit_checkpoint.py` validates the checkpoint/config/dictionary
   mapping needed before writing a TensorRT-LLM converter.

Artifacts are written under `inference experiment/trtllm_port/artifacts/` by
default.

