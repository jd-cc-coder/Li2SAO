#!/bin/bash
# Download ReTool RL assets via HF mirror. Skips SFT data/base model; pulls the
# already-SFT'd Qwen3-4B checkpoint plus DAPO math train and AIME 2024/2025/2026.
#
# Usage:
#   bash /root/LiSAO/examples/retool/download_assets.sh
#
# Env: HF_ENDPOINT (default https://hf-mirror.com), DATA_ROOT, SKIP_EVAL=1
set -exuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LISAO_ROOT=${LISAO_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
# shellcheck disable=SC1091
source "${LISAO_ROOT}/scripts/mirror_env.sh"

DATA_ROOT=${DATA_ROOT:-/root/data}
MODEL_DIR=${MODEL_DIR:-${DATA_ROOT}/models/qwen3-4b-sft}
TRAIN_DATA_DIR=${TRAIN_DATA_DIR:-${DATA_ROOT}/datasets/dapo-math-17k}
AIME2024_DIR=${AIME2024_DIR:-${DATA_ROOT}/datasets/aime-2024}
AIME2025_DIR=${AIME2025_DIR:-${DATA_ROOT}/datasets/aime-2025}
AIME2026_DIR=${AIME2026_DIR:-${DATA_ROOT}/datasets/aime-2026}
SKIP_EVAL=${SKIP_EVAL:-0}
SFT_REPO=${SFT_REPO:-font-info/qwen3-4b-sft-SGLang-RL}

mkdir -p "${DATA_ROOT}/models" "${DATA_ROOT}/datasets"

if ! command -v hf >/dev/null 2>&1; then
  pip_mirror install -U "huggingface_hub[cli]"
fi

# generate_with_retool.py needs jinja2; tool_sandbox.py needs psutil.
pip_mirror install "jinja2>=3.0.0" "psutil>=5.8.0"

echo "HF_ENDPOINT=${HF_ENDPOINT} HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET} PIP_INDEX_URL=${PIP_INDEX_URL}"
echo "SFT_REPO=${SFT_REPO} MODEL_DIR=${MODEL_DIR}"

find "${MODEL_DIR}" "${TRAIN_DATA_DIR}" "${AIME2024_DIR}" "${AIME2025_DIR}" "${AIME2026_DIR}" \
  -name '*.lock' -delete 2>/dev/null || true

hf download "${SFT_REPO}" --local-dir "${MODEL_DIR}"
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir "${TRAIN_DATA_DIR}"

if [[ "${SKIP_EVAL}" != "1" ]]; then
  hf download --repo-type dataset zhuzilin/aime-2024 --local-dir "${AIME2024_DIR}"
  hf download --repo-type dataset zhuzilin/aime-2025 --local-dir "${AIME2025_DIR}"
  # math-ai/aime26 is problem/answer jsonl; convert to slime prompt/label.
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
  python3 "${SCRIPT_DIR}/convert_aime_jsonl.py" \
    "${SRC_2026}" \
    "${AIME2026_DIR}/aime-2026.jsonl"
fi

echo "Done. MODEL_DIR=${MODEL_DIR}"
ls -lh "${MODEL_DIR}" | head || true
ls -lh "${TRAIN_DATA_DIR}" || true
if [[ "${SKIP_EVAL}" != "1" ]]; then
  ls -lh "${AIME2024_DIR}" "${AIME2025_DIR}" "${AIME2026_DIR}" || true
  wc -l "${AIME2024_DIR}/aime-2024.jsonl" "${AIME2025_DIR}/aime-2025.jsonl" \
    "${AIME2026_DIR}/aime-2026.jsonl"
fi
