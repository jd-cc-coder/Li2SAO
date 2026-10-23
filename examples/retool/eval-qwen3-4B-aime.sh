#!/bin/bash
# ReTool eval-only: AIME 2024 + 2025 + 2026 with code_interpreter sandbox.
#
# Sandboxed ReTool evaluation on AIME 2024, 2025, and 2026.
# This is separate from the single-turn evaluator, which does not call the interpreter.
# Scoring uses the same binary math grader as training.
#
# Usage:
#   # SFT baseline (after convert_qwen3_4b_sft.sh)
#   LOAD=/root/data/models/qwen3-4b-sft_torch_dist \
#     bash /root/LiSAO/examples/retool/eval-qwen3-4B-aime.sh
#   # Post-train GRPO latest (all three AIME years)
#   LOAD=/root/data/ckpts/qwen3-4B-retool-grpo \
#     bash /root/LiSAO/examples/retool/eval-qwen3-4B-aime.sh
#   # One saved iteration; subset years
#   LOAD=/root/data/ckpts/qwen3-4B-retool-lisao CKPT_STEP=100 \
#   EVAL_AIME_YEARS=2025,2026 \
#     bash /root/LiSAO/examples/retool/eval-qwen3-4B-aime.sh
#   # SFT base + every iter_* (pass@k table at the end)
#   EVAL_ALL=1 LOAD=/root/data/ckpts/qwen3-4B-retool-grpo-inflight-2t6p \
#     bash /root/LiSAO/examples/retool/eval-qwen3-4B-aime.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
if [[ "${EVAL_ALL:-0}" == "1" ]]; then
  exec bash "${SCRIPT_DIR}/eval-qwen3-4B-aime-all.sh"
fi
set -x

SKIP_CLEANUP=${SKIP_CLEANUP:-0}
SKIP_RAY_START=${SKIP_RAY_START:-0}
if [[ "${SKIP_CLEANUP}" != "1" ]]; then
  pkill -9 sglang 2>/dev/null || true
  sleep 3
  ray stop --force 2>/dev/null || true
  pkill -9 ray 2>/dev/null || true
  pkill -9 python 2>/dev/null || true
  sleep 3
fi

export PYTHONUNBUFFERED=1

LISAO_ROOT=${LISAO_ROOT:-/root/LiSAO}
LISAO_SLIME_ROOT=${LISAO_SLIME_ROOT:-${LISAO_ROOT}/lisao_slime}
RETOOL_SRC=${RETOOL_SRC:-${LISAO_SLIME_ROOT}/examples/retool}
EXAMPLE_RETOOL=${EXAMPLE_RETOOL:-${LISAO_ROOT}/examples/retool}
DATA_ROOT=${DATA_ROOT:-/root/data}
MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM}

# shellcheck disable=SC1091
source "${EXAMPLE_RETOOL}/_eval_config.sh"
parse_eval_aime_years
require_retool_aime_jsonl
set_retool_eval_tag

MODEL_DIR=${MODEL_DIR:-${DATA_ROOT}/models/qwen3-4b-sft}
REF_LOAD=${REF_LOAD:-${MODEL_DIR}_torch_dist}
TRAINED_CKPT=${DATA_ROOT}/ckpts/qwen3-4B-retool-grpo
if [[ -n "${LOAD:-}" ]]; then
  :
elif [[ -f "${TRAINED_CKPT}/latest_checkpointed_iteration.txt" ]]; then
  LOAD=${TRAINED_CKPT}
else
  LOAD=${REF_LOAD}
fi
PROMPT_DATA=${PROMPT_DATA:-$(retool_aime_jsonl "${EVAL_AIME_YEAR_LIST[0]}")}

NUM_GPUS=${NUM_GPUS:-8}
N_SAMPLES=${N_SAMPLES:-16}
# Keep modest: each sample can issue up to 16 sandbox Python calls.
SGLANG_SERVER_CONCURRENCY=${SGLANG_SERVER_CONCURRENCY:-16}
MODEL_MAX_CONTEXT_LEN=${MODEL_MAX_CONTEXT_LEN:-32768}
EVAL_MAX_CONTEXT_LEN=${EVAL_MAX_CONTEXT_LEN:-${MODEL_MAX_CONTEXT_LEN}}
EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN:-${EVAL_MAX_CONTEXT_LEN}}
ENABLE_THINKING=${ENABLE_THINKING:-0}
EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-1.0}
APPLY_CHAT_TEMPLATE_KWARGS=${APPLY_CHAT_TEMPLATE_KWARGS:-}
if [[ -z "${APPLY_CHAT_TEMPLATE_KWARGS}" ]]; then
  if [[ "${ENABLE_THINKING}" == "1" ]]; then
    APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking": true}'
  else
    APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
  fi
fi

TP_SIZE=${TP_SIZE:-2}
DP_SIZE=$((NUM_GPUS / TP_SIZE))
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.7}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-${DP_SIZE}}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-8192}

EVAL_OUT_DIR=${EVAL_OUT_DIR:-${DATA_ROOT}/eval/retool}
EVAL_LOG=${EVAL_LOG:-${EVAL_OUT_DIR}/${EVAL_TAG}_$(date -u +%Y%m%dT%H%M%SZ).log}
EVAL_CONFIG=${EVAL_CONFIG:-${EVAL_OUT_DIR}/${EVAL_TAG}_eval.yaml}
mkdir -p "${EVAL_OUT_DIR}"
export EVAL_OUT_DIR EVAL_TAG DATA_ROOT

for p in "${MODEL_DIR}" "${REF_LOAD}" "${RETOOL_SRC}" "${EXAMPLE_RETOOL}"; do
  [[ -d "${p}" ]] || { echo "ERROR: missing ${p}" >&2; exit 1; }
done
[[ -f "${PROMPT_DATA}" ]] || { echo "ERROR: missing placeholder prompt-data ${PROMPT_DATA}" >&2; exit 1; }
[[ -f "${LISAO_SLIME_ROOT}/train.py" ]] || {
  echo "ERROR: missing ${LISAO_SLIME_ROOT}/train.py (is LiSAO mounted?)" >&2
  exit 1
}
[[ -f "${RETOOL_SRC}/generate_with_retool.py" && -f "${RETOOL_SRC}/tool_sandbox.py" ]] || {
  echo "ERROR: retool generate/sandbox missing under ${RETOOL_SRC}" >&2
  exit 1
}
[[ -f "${EXAMPLE_RETOOL}/eval_rm.py" ]] || {
  echo "ERROR: missing ${EXAMPLE_RETOOL}/eval_rm.py" >&2
  exit 1
}
[[ -f "${LOAD}/latest_checkpointed_iteration.txt" ]] || {
  echo "ERROR: missing megatron ckpt under ${LOAD}" >&2
  echo "  For SFT baseline: LOAD=${REF_LOAD} (after convert_qwen3_4b_sft.sh)" >&2
  echo "  For post-train: LOAD=/root/data/ckpts/qwen3-4B-retool-grpo (or -lisao)" >&2
  exit 1
}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l | tr -d ' ' || true)
NVLINK_COUNT=${NVLINK_COUNT:-0}
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
if (( NUM_GPUS % TP_SIZE != 0 )); then
  echo "ERROR: NUM_GPUS=${NUM_GPUS} not divisible by TP_SIZE=${TP_SIZE}" >&2
  exit 1
fi
if (( NUM_GPUS < 2 )); then
  echo "ERROR: NUM_GPUS=${NUM_GPUS} need >=2 (rollout-num-gpus-per-engine=2)" >&2
  exit 1
fi
if (( GLOBAL_BATCH_SIZE % DP_SIZE != 0 )); then
  echo "ERROR: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} not divisible by DP_SIZE=${DP_SIZE}" >&2
  exit 1
fi

CKPT_STEP=${CKPT_STEP:-}
if [[ -n "${CKPT_STEP}" ]]; then
  if ! [[ "${CKPT_STEP}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: CKPT_STEP must be an integer, got ${CKPT_STEP}" >&2
    exit 1
  fi
  ITER_DIR=$(printf "%s/iter_%07d" "${LOAD}" "${CKPT_STEP}")
  [[ -d "${ITER_DIR}" ]] || {
    echo "ERROR: missing ${ITER_DIR} (CKPT_STEP=${CKPT_STEP})" >&2
    exit 1
  }
fi

write_retool_eval_config "${EVAL_CONFIG}"

echo "HAS_NVLINK=${HAS_NVLINK} LOAD=${LOAD} CKPT_STEP=${CKPT_STEP:-<latest>} AIME_YEARS=${EVAL_AIME_YEAR_LIST[*]}"
echo "NUM_GPUS=${NUM_GPUS} TP_SIZE=${TP_SIZE} DP_SIZE=${DP_SIZE} GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "APPLY_CHAT_TEMPLATE_KWARGS=${APPLY_CHAT_TEMPLATE_KWARGS} EVAL_TEMPERATURE=${EVAL_TEMPERATURE}"
echo "EVAL_MAX_CONTEXT_LEN=${EVAL_MAX_CONTEXT_LEN} EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN}"
echo "N_SAMPLES=${N_SAMPLES} SGLANG_SERVER_CONCURRENCY=${SGLANG_SERVER_CONCURRENCY}"
echo "eval generate=generate_with_retool.generate (sandbox)  eval RM=eval_rm → lisao.eval.math_rm"

# Instruct-2507 / SFT: rotary-base 5e6
# shellcheck disable=SC1091
source "${LISAO_SLIME_ROOT}/scripts/models/qwen3-4B-Instruct-2507.sh"

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_DIR}"
  --ref-load "${REF_LOAD}"
  --load "${LOAD}"
  --save /tmp/retool_aime_eval_dummy
  --save-interval 999999
)
if [[ -n "${CKPT_STEP}" ]]; then
  CKPT_ARGS+=(--ckpt-step "${CKPT_STEP}")
fi

ROLLOUT_ARGS=(
  --prompt-data "${PROMPT_DATA}"
  --input-key prompt
  --label-key label
  --apply-chat-template
  --reward-key score
  --num-rollout 0
  --rollout-batch-size 1
  --n-samples-per-prompt 1
  --rollout-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
  --rollout-max-context-len "${EVAL_MAX_CONTEXT_LEN}"
  --rollout-temperature "${EVAL_TEMPERATURE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
)
if [[ -n "${APPLY_CHAT_TEMPLATE_KWARGS}" ]]; then
  ROLLOUT_ARGS+=(--apply-chat-template-kwargs "${APPLY_CHAT_TEMPLATE_KWARGS}")
fi

EVAL_ARGS=(
  --eval-interval 1
  --eval-config "${EVAL_CONFIG}"
  --n-samples-per-eval-prompt "${N_SAMPLES}"
  --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
  --eval-max-context-len "${EVAL_MAX_CONTEXT_LEN}"
  --eval-temperature "${EVAL_TEMPERATURE}"
  --eval-top-p 1
  --log-passrate
  --custom-eval-rollout-log-function-path eval_log.log_eval_rollout_data
)

PERF_ARGS=(
  --tensor-model-parallel-size "${TP_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}"
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --eps-clip 0.2
  --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --lr-decay-iters 1
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --no-load-optim
  --no-load-rng
  --finetune
  --override-opt-param-scheduler
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 2
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
  --sglang-cuda-graph-backend-prefill disabled
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

CUSTOM_ARGS=(
  --custom-generate-function-path generate_with_retool.generate
  --custom-rm-path eval_rm.reward_func
)

_cvd=${CUDA_VISIBLE_DEVICES-}
if [[ -z "${_cvd}" || "${_cvd}" == "NoDevFiles" || "${_cvd}" == "-1" || "${_cvd}" == "void" ]]; then
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%,}
fi
unset _cvd
export CUDA_VISIBLE_DEVICES NUM_GPUS
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

if [[ "${SKIP_RAY_START}" == "1" ]]; then
  export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
  echo "SKIP_RAY_START=1: reuse Ray cluster MASTER_ADDR=${MASTER_ADDR}"
  ray status
else
  unset RAY_ADDRESS RAY_GCS_ADDRESS
  export MASTER_ADDR=127.0.0.1
  ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

RUNTIME_ENV_VARS=(
  "\"PYTHONPATH\": \"${MEGATRON_PATH}:${LISAO_SLIME_ROOT}:${RETOOL_SRC}:${EXAMPLE_RETOOL}\""
  "\"CUDA_VISIBLE_DEVICES\": \"${CUDA_VISIBLE_DEVICES}\""
  "\"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES\": \"1\""
  "\"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\""
  "\"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\""
  "\"DATA_ROOT\": \"${DATA_ROOT}\""
  "\"EVAL_OUT_DIR\": \"${EVAL_OUT_DIR}\""
  "\"EVAL_TAG\": \"${EVAL_TAG}\""
  "\"MASTER_ADDR\": \"${MASTER_ADDR}\""
)
if [[ -n "${NCCL_IB_DISABLE:-}" ]]; then
  RUNTIME_ENV_VARS+=("\"NCCL_IB_DISABLE\": \"${NCCL_IB_DISABLE}\"")
fi
if [[ -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
  RUNTIME_ENV_VARS+=("\"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME}\"")
fi
if [[ -n "${GLOO_SOCKET_IFNAME:-}" ]]; then
  RUNTIME_ENV_VARS+=("\"GLOO_SOCKET_IFNAME\": \"${GLOO_SOCKET_IFNAME}\"")
fi
if [[ -n "${no_proxy:-}" ]]; then
  RUNTIME_ENV_VARS+=("\"no_proxy\": \"${no_proxy}\"")
  RUNTIME_ENV_VARS+=("\"NO_PROXY\": \"${no_proxy}\"")
fi
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    $(IFS=,; echo "${RUNTIME_ENV_VARS[*]}")
  }
}"

cd "${LISAO_SLIME_ROOT}"

echo "Eval log: ${EVAL_LOG}"
echo "Eval JSON dir: ${EVAL_OUT_DIR}"

set +e
ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 "${LISAO_SLIME_ROOT}/train.py" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  --num-gpus-per-node "${NUM_GPUS}" \
  --colocate \
  ${MODEL_ARGS[@]} \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${CUSTOM_ARGS[@]}" \
  2>&1 | tee "${EVAL_LOG}"
status=${PIPESTATUS[0]}
set -e

echo "AIME ${EVAL_AIME_YEAR_LIST[*]} ReTool eval exit=${status}"
echo "  full log: ${EVAL_LOG}"
echo "  metrics:  ${EVAL_OUT_DIR}/${EVAL_TAG}_latest.json"
echo "  per set:  metrics['eval/aime2024'] / eval/aime2025 / eval/aime2026  (boxed accuracy)"
echo "  pass@k:   metrics['eval/aimeYYYY-pass@1/2/4/...']  (N_SAMPLES>1 + --log-passrate)"
echo "  sandbox:  metrics['eval/aimeYYYY-sandbox_hit_ratio']  (share of samples with <interpreter>)"
echo "  sweep:    EVAL_ALL=1 to eval SFT base + every iter_* then write \${EVAL_OUT_DIR}/\${EVAL_TAG}_all_summary.md"
exit "${status}"
