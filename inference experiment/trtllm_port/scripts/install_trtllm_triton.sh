#!/usr/bin/env bash
set -euo pipefail

# Bootstrap the TensorRT-LLM + standalone Triton setup used by this port.
#
# This script is intentionally explicit and idempotent-ish. It does not build
# model engines; it installs the runtime/build environment needed to convert,
# build, run direct TensorRT-LLM, and run Triton TensorRT-LLM serving.
#
# Typical fresh-machine usage:
#   bash "inference experiment/trtllm_port/scripts/install_trtllm_triton.sh" --all \
#     --triton-archive /path/to/tritonserver-2.70.0-linux-amd64.tar.gz
#
# If you already have Triton extracted at /tmp/tritonserver-2.70.0/tritonserver:
#   bash ".../install_trtllm_triton.sh" --system-deps --venv --backend --verify

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PORT_DIR="${REPO_ROOT}/inference experiment/trtllm_port"

TRTLLM_VERSION="1.2.1"
TENSORRT_DEB_VERSION="10.14.1.48-1+cuda13.0"
PYTHON_BIN="python3.10"
TRTLLM_VENV="${REPO_ROOT}/.venv-trtllm"
TRITON_PREFIX="/tmp/tritonserver-2.70.0"
TRITON_HOME="${TRITON_PREFIX}/tritonserver"
TRITON_ARCHIVE=""
TRITON_URL=""
TRITON_SHORT_TAG="r25.06"
BACKEND_REPO="/tmp/TensorRT-LLM-triton-v${TRTLLM_VERSION}"
BACKEND_GIT_URL="https://github.com/NVIDIA/TensorRT-LLM.git"
BACKEND_GIT_REF="v${TRTLLM_VERSION}"
TRT_ROOT="/usr/local/tensorrt"

DO_SYSTEM_DEPS=0
DO_TENSORRT_APT=0
DO_VENV=0
DO_TRITON=0
DO_BACKEND=0
DO_VERIFY=0

usage() {
  cat <<USAGE
Usage:
  $0 [options]

Main options:
  --all                 Run system deps, TensorRT apt packages, venv, Triton, backend, verify.
  --system-deps         Install base apt build/runtime dependencies.
  --tensorrt-apt        Install pinned TensorRT Debian packages if NVIDIA apt repo is configured.
  --venv                Create/update ${TRTLLM_VENV} and install TensorRT-LLM Python packages.
  --triton              Install standalone Triton from --triton-archive or --triton-url.
  --backend             Clone/build TensorRT-LLM Triton backend and copy it into Triton.
  --verify              Print versions and check expected binaries/libraries.

Config:
  --triton-archive PATH Use a local Triton server .tar.gz/.tgz archive.
  --triton-url URL      Download Triton archive from URL.
  --triton-prefix PATH  Extraction prefix. Default: ${TRITON_PREFIX}
  --trtllm-version VER  Default: ${TRTLLM_VERSION}
  --backend-repo PATH   Default: ${BACKEND_REPO}
  --backend-ref REF     Default: ${BACKEND_GIT_REF}
  --triton-tag TAG      Triton dependency tag for backend build. Default: ${TRITON_SHORT_TAG}
  --python-bin PATH     Python for venv. Default: ${PYTHON_BIN}
  --trt-root PATH       TensorRT root for backend build. Default: ${TRT_ROOT}

Notes:
  - Run as root or with passwordless sudo for apt/backend install.
  - This script expects Ubuntu 22.04-style packages.
  - Standalone Triton archive is not hardcoded because release URLs differ by distribution.
USAGE
}

log() {
  printf '[install_trtllm_triton] %s\n' "$*" >&2
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    log "Missing required command: $1"
    exit 1
  }
}

apt_install() {
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
}

extract_triton_archive() {
  local archive="$1"
  local tmp_extract
  tmp_extract="$(mktemp -d /tmp/tritonserver-extract.XXXXXX)"

  tar -xzf "${archive}" -C "${tmp_extract}"

  if [[ -x "${tmp_extract}/tritonserver/bin/tritonserver" ]]; then
    rsync -a "${tmp_extract}/" "${TRITON_PREFIX}/"
  elif [[ -x "${tmp_extract}/bin/tritonserver" ]]; then
    mkdir -p "${TRITON_HOME}"
    rsync -a "${tmp_extract}/" "${TRITON_HOME}/"
  else
    log "Archive did not contain bin/tritonserver in an expected layout."
    log "First few archive entries:"
    tar -tzf "${archive}" | sed -n '1,40p' >&2 || true
    rm -rf "${tmp_extract}"
    exit 1
  fi

  rm -rf "${tmp_extract}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      DO_SYSTEM_DEPS=1
      DO_TENSORRT_APT=1
      DO_VENV=1
      DO_TRITON=1
      DO_BACKEND=1
      DO_VERIFY=1
      shift
      ;;
    --system-deps) DO_SYSTEM_DEPS=1; shift ;;
    --tensorrt-apt) DO_TENSORRT_APT=1; shift ;;
    --venv) DO_VENV=1; shift ;;
    --triton) DO_TRITON=1; shift ;;
    --backend) DO_BACKEND=1; shift ;;
    --verify) DO_VERIFY=1; shift ;;
    --triton-archive) TRITON_ARCHIVE="$2"; shift 2 ;;
    --triton-url) TRITON_URL="$2"; shift 2 ;;
    --triton-prefix)
      TRITON_PREFIX="$2"
      TRITON_HOME="${TRITON_PREFIX}/tritonserver"
      shift 2
      ;;
    --trtllm-version)
      TRTLLM_VERSION="$2"
      BACKEND_GIT_REF="v${TRTLLM_VERSION}"
      BACKEND_REPO="/tmp/TensorRT-LLM-triton-v${TRTLLM_VERSION}"
      shift 2
      ;;
    --backend-repo) BACKEND_REPO="$2"; shift 2 ;;
    --backend-ref) BACKEND_GIT_REF="$2"; shift 2 ;;
    --triton-tag) TRITON_SHORT_TAG="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --trt-root) TRT_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      log "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ ${DO_SYSTEM_DEPS} -eq 0 && ${DO_TENSORRT_APT} -eq 0 && ${DO_VENV} -eq 0 && ${DO_TRITON} -eq 0 && ${DO_BACKEND} -eq 0 && ${DO_VERIFY} -eq 0 ]]; then
  usage
  exit 0
fi

if [[ ${DO_SYSTEM_DEPS} -eq 1 ]]; then
  log "Installing base system dependencies"
  apt_install \
    ca-certificates \
    curl \
    git \
    git-lfs \
    build-essential \
    pkg-config \
    "${PYTHON_BIN}" \
    "${PYTHON_BIN}-venv" \
    "${PYTHON_BIN}-dev" \
    cmake \
    ninja-build \
    rapidjson-dev \
    nlohmann-json3-dev \
    libopenmpi-dev \
    openmpi-bin \
    libnuma-dev \
    libb64-dev \
    zlib1g-dev \
    patchelf \
    unzip \
    rsync
  git lfs install || true
fi

if [[ ${DO_TENSORRT_APT} -eq 1 ]]; then
  log "Installing pinned TensorRT Debian packages (${TENSORRT_DEB_VERSION})"
  log "This requires NVIDIA CUDA/TensorRT apt repositories to already be configured."
  apt_install \
    "libnvinfer10=${TENSORRT_DEB_VERSION}" \
    "libnvinfer-plugin10=${TENSORRT_DEB_VERSION}" \
    "libnvinfer-dev=${TENSORRT_DEB_VERSION}" \
    "libnvinfer-plugin-dev=${TENSORRT_DEB_VERSION}" \
    "libnvinfer-headers-dev=${TENSORRT_DEB_VERSION}" \
    "libnvinfer-headers-plugin-dev=${TENSORRT_DEB_VERSION}" \
    "libnvonnxparsers10=${TENSORRT_DEB_VERSION}" \
    "libnvonnxparsers-dev=${TENSORRT_DEB_VERSION}"
fi

if [[ ${DO_VENV} -eq 1 ]]; then
  log "Creating/updating TensorRT-LLM venv at ${TRTLLM_VENV}"
  need_cmd "${PYTHON_BIN}"
  "${PYTHON_BIN}" -m venv "${TRTLLM_VENV}"
  # shellcheck disable=SC1091
  source "${TRTLLM_VENV}/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install \
    "cmake==3.31.10" \
    "tensorrt-llm==${TRTLLM_VERSION}" \
    "tritonclient[http]"

  log "Writing env_trtllm.sh"
  cat > "${PORT_DIR}/env_trtllm.sh" <<'ENVEOF'
# Source this file before running TensorRT-LLM commands.
TRTLLM_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRTLLM_VENV="$TRTLLM_REPO_ROOT/.venv-trtllm"
TRTLLM_SITE="$("$TRTLLM_VENV/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

export PATH="$TRTLLM_VENV/bin:$PATH"
export LD_LIBRARY_PATH="$TRTLLM_SITE/tensorrt_llm/libs:$TRTLLM_SITE/torch/lib:$TRTLLM_SITE/tensorrt_libs:$TRTLLM_SITE/nvidia/cuda_runtime/lib:$TRTLLM_SITE/nvidia/cu13/lib:$TRTLLM_SITE/nvidia/cudnn/lib:$TRTLLM_SITE/nvidia/cublas/lib:$TRTLLM_SITE/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
ENVEOF
fi

if [[ ${DO_TRITON} -eq 1 ]]; then
  log "Installing standalone Triton into ${TRITON_PREFIX}"
  mkdir -p "${TRITON_PREFIX}"
  if [[ -n "${TRITON_ARCHIVE}" ]]; then
    [[ -f "${TRITON_ARCHIVE}" ]] || {
      log "Triton archive not found: ${TRITON_ARCHIVE}"
      exit 1
    }
    extract_triton_archive "${TRITON_ARCHIVE}"
  elif [[ -n "${TRITON_URL}" ]]; then
    need_cmd curl
    TMP_ARCHIVE="/tmp/tritonserver-install.tar.gz"
    curl -L "${TRITON_URL}" -o "${TMP_ARCHIVE}"
    extract_triton_archive "${TMP_ARCHIVE}"
  elif [[ -x "${TRITON_HOME}/bin/tritonserver" ]]; then
    log "Triton already present: ${TRITON_HOME}/bin/tritonserver"
  else
    log "No Triton archive/url supplied and ${TRITON_HOME}/bin/tritonserver is missing."
    log "Pass --triton-archive PATH or --triton-url URL."
    exit 1
  fi
fi

if [[ ${DO_BACKEND} -eq 1 ]]; then
  log "Building TensorRT-LLM Triton backend"
  [[ -x "${TRITON_HOME}/bin/tritonserver" ]] || {
    log "Triton not found at ${TRITON_HOME}/bin/tritonserver. Run --triton first or set --triton-prefix."
    exit 1
  }
  [[ -d "${TRTLLM_VENV}" ]] || {
    log "TensorRT-LLM venv missing at ${TRTLLM_VENV}. Run --venv first."
    exit 1
  }

  if [[ ! -d "${BACKEND_REPO}/.git" ]]; then
    log "Cloning TensorRT-LLM ${BACKEND_GIT_REF} to ${BACKEND_REPO}"
    git clone --branch "${BACKEND_GIT_REF}" --depth 1 "${BACKEND_GIT_URL}" "${BACKEND_REPO}"
  else
    log "Using existing backend repo: ${BACKEND_REPO}"
  fi

  # Some checkouts need LFS payloads for C++ builds. Ignore failure if LFS is not
  # required by the selected ref/sparse state.
  git -C "${BACKEND_REPO}" lfs pull || true

  # shellcheck disable=SC1091
  source "${PORT_DIR}/env_trtllm.sh"
  export LD_LIBRARY_PATH="${TRITON_HOME}/lib64:${LD_LIBRARY_PATH:-}"

  BACKEND_DIR="${BACKEND_REPO}/triton_backend/inflight_batcher_llm"
  [[ -x "${BACKEND_DIR}/scripts/build.sh" ]] || chmod +x "${BACKEND_DIR}/scripts/build.sh"
  (
    cd "${BACKEND_DIR}"
    bash scripts/build.sh -t "${TRT_ROOT}" -s "${TRITON_SHORT_TAG}"
  )

  mkdir -p "${TRITON_HOME}/backends/tensorrtllm"
  cp "${BACKEND_DIR}/build/libtriton_tensorrtllm.so" "${TRITON_HOME}/backends/tensorrtllm/"
  cp "${BACKEND_DIR}/build/trtllmExecutorWorker" "${TRITON_HOME}/backends/tensorrtllm/"

  log "Installed TensorRT-LLM backend into ${TRITON_HOME}/backends/tensorrtllm"
fi

if [[ ${DO_VERIFY} -eq 1 ]]; then
  log "Verification"
  if [[ -x "${TRTLLM_VENV}/bin/python" ]]; then
    "${TRTLLM_VENV}/bin/python" - <<'PY'
import importlib.metadata as md
for pkg in ["tensorrt_llm", "tensorrt", "torch", "tritonclient"]:
    try:
        print(f"{pkg}={md.version(pkg)}")
    except Exception as exc:
        print(f"{pkg}=MISSING ({exc})")
PY
  else
    log "Missing venv python: ${TRTLLM_VENV}/bin/python"
  fi

  if [[ -x "${TRITON_HOME}/bin/tritonserver" ]]; then
    "${TRITON_HOME}/bin/tritonserver" --version || true
  else
    log "Missing Triton binary: ${TRITON_HOME}/bin/tritonserver"
  fi

  if [[ -f "${TRITON_HOME}/backends/tensorrtllm/libtriton_tensorrtllm.so" ]]; then
    log "TensorRT-LLM backend present."
    ldd "${TRITON_HOME}/backends/tensorrtllm/libtriton_tensorrtllm.so" | sed -n '1,120p' || true
  else
    log "TensorRT-LLM backend missing from ${TRITON_HOME}/backends/tensorrtllm"
  fi
fi

log "Done."

