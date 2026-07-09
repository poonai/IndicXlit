# Source this file before running TensorRT-LLM Phase 3+ commands.
#
# Usage:
#   source "inference experiment/trtllm_port/env_trtllm.sh"
#   python "inference experiment/trtllm_port/probe_trtllm_env.py"

TRTLLM_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRTLLM_VENV="$TRTLLM_REPO_ROOT/.venv-trtllm"
TRTLLM_SITE="$TRTLLM_VENV/lib/python3.10/site-packages"

export PATH="$TRTLLM_VENV/bin:$PATH"
export LD_LIBRARY_PATH="$TRTLLM_SITE/nvidia/cu13/lib:$TRTLLM_SITE/tensorrt_libs:${LD_LIBRARY_PATH:-}"
