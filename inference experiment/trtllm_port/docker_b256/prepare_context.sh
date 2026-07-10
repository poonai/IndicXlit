#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PORT_DIR}/../.." && pwd)"
CTX="${SCRIPT_DIR}/context"

ENGINE_SRC="${PORT_DIR}/artifacts/trtllm_engines_en_hi_beam5_triton_fp16_b256"
MODEL_REPO_SRC="${PORT_DIR}/triton_indicxlit/model_repository"
MODEL_ASSETS_SRC="${REPO_ROOT}/app/ai4bharat/transliteration/transformer/models/en2indic"

if [[ ! -d "${ENGINE_SRC}/encoder" || ! -d "${ENGINE_SRC}/decoder" ]]; then
  echo "Missing b256 FP16 engine at ${ENGINE_SRC}" >&2
  exit 1
fi

rm -rf "${CTX}"
mkdir -p "${CTX}/scripts" "${CTX}/k6" "${CTX}/engines" "${CTX}/model_assets"

cp "${SCRIPT_DIR}/Dockerfile" "${CTX}/Dockerfile"
cp "${SCRIPT_DIR}/scripts/run_server.sh" "${CTX}/scripts/run_server.sh"
cp "${SCRIPT_DIR}/scripts/smoke_core.py" "${CTX}/scripts/smoke_core.py"
cp "${SCRIPT_DIR}/k6/core_b256.js" "${CTX}/k6/core_b256.js"

cp -a "${MODEL_REPO_SRC}" "${CTX}/model_repository"
cp -a "${ENGINE_SRC}" "${CTX}/engines/trtllm_engines_en_hi_beam5_triton_fp16_b256"
cp -a "${MODEL_ASSETS_SRC}" "${CTX}/model_assets/en2indic"

chmod +x "${CTX}/scripts/run_server.sh" "${CTX}/scripts/smoke_core.py"

printf 'Prepared Docker context:\n  %s\n\nBuild:\n  docker build -t indicxlit-trtllm:b256-fp16 "%s"\n\nRun:\n  docker run --rm --gpus all -p 8000:8000 -p 8002:8002 indicxlit-trtllm:b256-fp16\n' "${CTX}" "${CTX}"
