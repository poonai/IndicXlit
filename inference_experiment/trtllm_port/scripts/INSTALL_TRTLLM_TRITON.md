# TensorRT-LLM / Triton installer

Script:

```bash
inference_experiment/trtllm_port/scripts/install_trtllm_triton.sh
```

It installs the runtime/build stack used by the TensorRT-LLM port:

- Ubuntu build/runtime packages
- optional pinned TensorRT Debian packages
- `.venv-trtllm`
- `tensorrt-llm==1.2.1`
- standalone Triton under `/tmp/tritonserver-2.70.0`
- TensorRT-LLM Triton backend built from NVIDIA/TensorRT-LLM

## Recommended fresh-machine flow

1. Install base packages and Python environment:

```bash
sudo bash "inference_experiment/trtllm_port/scripts/install_trtllm_triton.sh" \
  --system-deps \
  --venv \
  --verify
```

2. If the NVIDIA TensorRT apt repository is configured, install pinned C++ libs:

```bash
sudo bash "inference_experiment/trtllm_port/scripts/install_trtllm_triton.sh" \
  --tensorrt-apt
```

3. Install standalone Triton from an archive:

```bash
sudo bash "inference_experiment/trtllm_port/scripts/install_trtllm_triton.sh" \
  --triton \
  --triton-archive /path/to/tritonserver-2.70.0-linux-amd64.tar.gz
```

or from a URL:

```bash
sudo bash "inference_experiment/trtllm_port/scripts/install_trtllm_triton.sh" \
  --triton \
  --triton-url "https://..."
```

4. Build/install the TensorRT-LLM Triton backend:

```bash
sudo bash "inference_experiment/trtllm_port/scripts/install_trtllm_triton.sh" \
  --backend \
  --verify
```

## One-shot

If apt repositories and a Triton archive are ready:

```bash
sudo bash "inference_experiment/trtllm_port/scripts/install_trtllm_triton.sh" \
  --all \
  --triton-archive /path/to/tritonserver-2.70.0-linux-amd64.tar.gz
```

## Activate env

```bash
source "inference_experiment/trtllm_port/env_trtllm.sh"
```

## Start Triton

```bash
LOAD_MODEL=indicxlit_tensorrt_llm \
"inference_experiment/trtllm_port/triton_indicxlit/scripts/run_triton.sh"
```

## Notes

- The script does not build model engines.
- Backend build can take time and needs network access to clone/fetch dependencies.
- The pinned TensorRT Debian packages require NVIDIA apt repositories to already
  be configured on the host.
- If you use the Docker image route, this installer is not required inside the
  container.

