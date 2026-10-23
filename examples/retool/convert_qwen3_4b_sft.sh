#!/bin/bash
# Convert the ReTool SFT HF checkpoint (Qwen3-4B-Instruct-2507 + rotary 5M)
# to Megatron torch_dist. Skip SFT training; run after download_assets.sh.
#
# Usage:
#   bash /root/LiSAO/examples/retool/convert_qwen3_4b_sft.sh
set -exuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LISAO_ROOT=${LISAO_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
DATA_ROOT=${DATA_ROOT:-/root/data}

export MODEL_DIR=${MODEL_DIR:-${DATA_ROOT}/models/qwen3-4b-sft}
export SAVE_DIR=${SAVE_DIR:-${MODEL_DIR}_torch_dist}
# Instruct-2507 / SFT ckpt uses rotary base 5e6 (not the base Qwen3-4B 1e6).
export MODEL_ARGS_ROTARY_BASE=${MODEL_ARGS_ROTARY_BASE:-5000000}

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "ERROR: HF checkpoint not found at ${MODEL_DIR}. Run download_assets.sh first." >&2
  exit 1
fi

bash "${LISAO_ROOT}/scripts/convert_qwen3_4b.sh"
echo "Converted torch_dist saved to ${SAVE_DIR} (rotary-base=${MODEL_ARGS_ROTARY_BASE})"
