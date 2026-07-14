# Source this file before running TensorRT-LLM Phase 3+ commands.
#
# Usage:
#   source "inference_experiment/trtllm_port/env_trtllm.sh"
#   python "inference_experiment/trtllm_port/probe_trtllm_env.py"

TRTLLM_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRTLLM_VENV="$TRTLLM_REPO_ROOT/.venv-trtllm"
TRTLLM_PYTHON="${TRTLLM_VENV}/bin/python"
if [[ "${INDICXLIT_BUILD_IN_CONTAINER:-0}" == "1" || ! -x "${TRTLLM_PYTHON}" ]]; then
  TRTLLM_PYTHON="$(command -v python3)"
fi
TRTLLM_SITE="$("${TRTLLM_PYTHON}" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

if [[ "${INDICXLIT_BUILD_IN_CONTAINER:-0}" != "1" && -d "${TRTLLM_VENV}/bin" ]]; then
  export PATH="$TRTLLM_VENV/bin:$PATH"
fi
export LD_LIBRARY_PATH="$TRTLLM_SITE/tensorrt_llm/libs:$TRTLLM_SITE/torch/lib:$TRTLLM_SITE/tensorrt_libs:$TRTLLM_SITE/nvidia/cuda_runtime/lib:$TRTLLM_SITE/nvidia/cu13/lib:$TRTLLM_SITE/nvidia/cudnn/lib:$TRTLLM_SITE/nvidia/cublas/lib:$TRTLLM_SITE/nvidia/nccl/lib:$TRTLLM_SITE/nvidia/cuda_nvrtc/lib:$TRTLLM_SITE/nvidia/cusolver/lib:$TRTLLM_SITE/nvidia/cusparse/lib:$TRTLLM_SITE/nvidia/curand/lib:$TRTLLM_SITE/nvidia/cufft/lib:$TRTLLM_SITE/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
