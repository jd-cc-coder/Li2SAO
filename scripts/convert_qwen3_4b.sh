#!/bin/bash
# Convert Qwen3-4B HF checkpoint to Megatron torch_dist format.
#
# Usage:
#   bash /root/LiSAO/scripts/convert_qwen3_4b.sh
set -exuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/mirror_env.sh"

LISAO_ROOT=${LISAO_ROOT:-/root/LiSAO}
LISAO_SLIME_ROOT=${LISAO_SLIME_ROOT:-${LISAO_ROOT}/lisao_slime}
DATA_ROOT=${DATA_ROOT:-/root/data}
MODEL_DIR=${MODEL_DIR:-${DATA_ROOT}/models/Qwen3-4B}
SAVE_DIR=${SAVE_DIR:-${MODEL_DIR}_torch_dist}
MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM}

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "ERROR: HF checkpoint not found at ${MODEL_DIR}. Run download_assets.sh first." >&2
  exit 1
fi

cd "${LISAO_SLIME_ROOT}"
# shellcheck disable=SC1091
source scripts/models/qwen3-4B.sh

export PYTHONPATH="${MEGATRON_PATH}:${LISAO_SLIME_ROOT}:${PYTHONPATH:-}"

python tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint "${MODEL_DIR}" \
  --save "${SAVE_DIR}"

echo "Converted torch_dist saved to ${SAVE_DIR}"
