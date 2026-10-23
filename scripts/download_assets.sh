#!/bin/bash
# Download Qwen3-4B, DAPO-Math-17K, and AIME 2024, 2025, and 2026.
#
# Usage:
#   bash /root/LiSAO/scripts/download_assets.sh
#
# Env: HF_ENDPOINT (default https://hf-mirror.com), DATA_ROOT, SKIP_EVAL=1
set -exuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LISAO_ROOT=${LISAO_ROOT:-"$(cd "${SCRIPT_DIR}/.." && pwd)"}
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/mirror_env.sh"

DATA_ROOT=${DATA_ROOT:-/root/data}
MODEL_DIR=${MODEL_DIR:-${DATA_ROOT}/models/Qwen3-4B}
TRAIN_DATA_DIR=${TRAIN_DATA_DIR:-${DATA_ROOT}/datasets/dapo-math-17k}
AIME2024_DIR=${AIME2024_DIR:-${DATA_ROOT}/datasets/aime-2024}
AIME2025_DIR=${AIME2025_DIR:-${DATA_ROOT}/datasets/aime-2025}
AIME2026_DIR=${AIME2026_DIR:-${DATA_ROOT}/datasets/aime-2026}
SKIP_EVAL=${SKIP_EVAL:-0}

mkdir -p "${DATA_ROOT}/models" "${DATA_ROOT}/datasets"

if ! command -v hf >/dev/null 2>&1; then
  pip_mirror install -U "huggingface_hub[cli]"
fi

echo "HF_ENDPOINT=${HF_ENDPOINT} HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET} PIP_INDEX_URL=${PIP_INDEX_URL}"

find "${MODEL_DIR}" "${TRAIN_DATA_DIR}" "${AIME2024_DIR}" "${AIME2025_DIR}" "${AIME2026_DIR}" \
  -name '*.lock' -delete 2>/dev/null || true

hf download Qwen/Qwen3-4B --local-dir "${MODEL_DIR}"
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir "${TRAIN_DATA_DIR}"

if [[ "${SKIP_EVAL}" != "1" ]]; then
  hf download --repo-type dataset zhuzilin/aime-2024 --local-dir "${AIME2024_DIR}"
  hf download --repo-type dataset zhuzilin/aime-2025 --local-dir "${AIME2025_DIR}"
  hf download --repo-type dataset math-ai/aime26 --local-dir "${AIME2026_DIR}/raw"
  SRC_2026=""
  for cand in \
    "${AIME2026_DIR}/raw/aime2026.jsonl" \
    "${AIME2026_DIR}/raw/aime-2026.jsonl" \
    "${AIME2026_DIR}/raw/aime26.jsonl"; do
    if [[ -f "${cand}" ]]; then
      SRC_2026=${cand}
      break
    fi
  done
  if [[ -z "${SRC_2026}" ]]; then
    echo "ERROR: math-ai/aime26 downloaded but no jsonl under ${AIME2026_DIR}/raw" >&2
    ls -la "${AIME2026_DIR}/raw" >&2 || true
    exit 1
  fi
  python3 "${LISAO_ROOT}/examples/retool/convert_aime_jsonl.py" \
    "${SRC_2026}" \
    "${AIME2026_DIR}/aime-2026.jsonl"
fi

echo "Done. MODEL_DIR=${MODEL_DIR}"
ls -lh "${TRAIN_DATA_DIR}" || true
if [[ "${SKIP_EVAL}" != "1" ]]; then
  ls -lh "${AIME2024_DIR}" "${AIME2025_DIR}" "${AIME2026_DIR}" || true
fi
