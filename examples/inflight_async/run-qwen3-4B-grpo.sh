#!/bin/bash
# In-flight GRPO on Qwen3-4B for the single-turn math experiment.
#
# Defaults: 8 samples per prompt, 16 prompts, global batch 128,
# asymmetric clip 0.2 and 0.28, max policy lag 16, 500 steps, thinking off.
# Checkpoints are saved every 100 steps. Evaluation is a separate offline job.
#
# Prerequisites:
#   bash /root/LiSAO/scripts/setup_env.sh
#   bash /root/LiSAO/scripts/download_assets.sh
#   bash /root/LiSAO/scripts/convert_qwen3_4b.sh
#
# Default layout is 4 training GPUs and 4 rollout GPUs on one 8 GPU machine.
set -exuo pipefail

# Multi-node: SKIP_CLEANUP=1 SKIP_RAY_START=1 so we do not kill the Ray cluster.
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
DATA_ROOT=${DATA_ROOT:-/root/data}
MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM}

MODEL_DIR=${MODEL_DIR:-${DATA_ROOT}/models/Qwen3-4B}
REF_LOAD=${REF_LOAD:-${MODEL_DIR}_torch_dist}
DATA_PATH=${DATA_PATH:-${DATA_ROOT}/datasets/dapo-math-17k/dapo-math-17k.jsonl}
SAVE_DIR=${SAVE_DIR:-${DATA_ROOT}/ckpts/qwen3-4B-inflight-async}

NUM_ROLLOUT=${NUM_ROLLOUT:-500}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
# Must be rollout_batch_size * n_samples_per_prompt for GRPO.
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
SAVE_INTERVAL=${SAVE_INTERVAL:-100}
NUM_GPUS=${NUM_GPUS:-8}
ACTOR_GPUS=${ACTOR_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-$((NUM_GPUS - ACTOR_GPUS))}
# Per-engine client concurrency. Worker in-flight groups ≈ this * num_engines
# (num_engines = ROLLOUT_GPUS / gpus_per_engine). Keep low for long thinking KV.
SGLANG_SERVER_CONCURRENCY=${SGLANG_SERVER_CONCURRENCY:-16}
RUN_EVAL=${RUN_EVAL:-0}
# Default thinking off (shorter responses, less truncation). Set ENABLE_THINKING=1 to enable.
ENABLE_THINKING=${ENABLE_THINKING:-0}
APPLY_CHAT_TEMPLATE_KWARGS=${APPLY_CHAT_TEMPLATE_KWARGS:-}
if [[ -z "${APPLY_CHAT_TEMPLATE_KWARGS}" ]]; then
  if [[ "${ENABLE_THINKING}" == "1" ]]; then
    APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking": true}'
  else
    APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
  fi
fi

# Qwen3-4B native context is 32768 (~32K). Response budget may equal the
# window; slime clamps max_new_tokens to context - prompt (SGLang requires
# prompt + completion <= context).
MODEL_MAX_CONTEXT_LEN=${MODEL_MAX_CONTEXT_LEN:-32768}
ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-${MODEL_MAX_CONTEXT_LEN}}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-${ROLLOUT_MAX_CONTEXT_LEN}}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-$((ROLLOUT_MAX_RESPONSE_LEN / 4))}
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}
# Actor parallel: world = TP * PP * CP * DP == ACTOR_GPUS.
# Long sequences → large [T, V] logits (V≈151936 fp32). Default CP=1 on 4 actor
# GPUs (with TP=2); raise CP_SIZE if actor logits OOM.
TP_SIZE=${TP_SIZE:-2}
CP_SIZE=${CP_SIZE:-1}
# Packing budget per GPU (not the generate cap). A sample longer than this still
# forms its own microbatch (no truncate); then CP + chunked logprobs matter.
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}
# Chunk softmax over T to avoid a second full [T, V/TP] alloc (common OOM).
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-8192}
# Free GPU memory for long logits (Adam states → CPU). Set 0 to disable.
OPTIMIZER_CPU_OFFLOAD=${OPTIMIZER_CPU_OFFLOAD:-1}
# Drop a finished group when its oldest sampled policy is more than this many
# versions behind the trainer. The default limit is 16.
MAX_POLICY_LAG=${MAX_POLICY_LAG:-16}

# WandB default: online. Opt out with WANDB_MODE=offline or USE_WANDB=0.
# Multi-node Ray workers do not inherit ~/.netrc, so copy the login key into
# WANDB_KEY / runtime env when the head node already ran `wandb login`.
USE_WANDB=${USE_WANDB:-1}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_PROJECT=${WANDB_PROJECT:-lisao-inflight-async}
WANDB_GROUP=${WANDB_GROUP:-qwen3-4B}
WANDB_DIR=${WANDB_DIR:-${DATA_ROOT}/wandb}
WANDB_KEY=${WANDB_KEY:-${WANDB_API_KEY:-}}
if [[ -z "${WANDB_KEY}" && -f "${HOME}/.netrc" ]]; then
  WANDB_KEY=$(python3 -c 'import netrc; a=netrc.netrc().authenticators("api.wandb.ai"); print(a[2] if a and a[2] else "")' 2>/dev/null || true)
fi
export WANDB_DIR WANDB_MODE
if [[ -n "${WANDB_KEY}" ]]; then
  export WANDB_API_KEY="${WANDB_KEY}"
elif [[ "${USE_WANDB}" == "1" && "${WANDB_MODE}" == "online" ]]; then
  echo "WARNING: WANDB_MODE=online but no WANDB_API_KEY / ~/.netrc. Run \`wandb login\` in this container or export WANDB_API_KEY." >&2
fi
mkdir -p "${WANDB_DIR}"

for p in "${MODEL_DIR}" "${REF_LOAD}"; do
  [[ -d "${p}" ]] || { echo "ERROR: missing ${p}" >&2; exit 1; }
done
[[ -f "${DATA_PATH}" ]] || { echo "ERROR: missing ${DATA_PATH}" >&2; exit 1; }
[[ -f "${LISAO_SLIME_ROOT}/train_async.py" ]] || {
  echo "ERROR: missing ${LISAO_SLIME_ROOT}/train_async.py (is LiSAO mounted?)" >&2
  exit 1
}
[[ -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]] || {
  echo "ERROR: ${REF_LOAD} is not a megatron ckpt (run convert_qwen3_4b.sh)" >&2
  exit 1
}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-${DATA_ROOT}/datasets/aime-2025/aime-2025.jsonl}
if [[ "${RUN_EVAL}" == "1" && ! -f "${EVAL_DATA_PATH}" ]]; then
  echo "ERROR: RUN_EVAL=1 but missing ${EVAL_DATA_PATH}" >&2
  exit 1
fi
mkdir -p "${SAVE_DIR}"

ACTOR_PARALLEL=$((TP_SIZE * CP_SIZE))
if (( ACTOR_GPUS % ACTOR_PARALLEL != 0 )); then
  echo "ERROR: ACTOR_GPUS=${ACTOR_GPUS} not divisible by TP_SIZE*CP_SIZE=${ACTOR_PARALLEL}" >&2
  exit 1
fi
if (( ROLLOUT_GPUS < 2 )); then
  echo "ERROR: ROLLOUT_GPUS=${ROLLOUT_GPUS} need >=2 (rollout-num-gpus-per-engine=2)" >&2
  exit 1
fi

# grep exits 1 when there is no NVLink; pipefail + set -e would abort the script.
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l | tr -d ' ' || true)
NVLINK_COUNT=${NVLINK_COUNT:-0}
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK=${HAS_NVLINK} ACTOR_GPUS=${ACTOR_GPUS} ROLLOUT_GPUS=${ROLLOUT_GPUS} NUM_ROLLOUT=${NUM_ROLLOUT}"
echo "ROLLOUT_BATCH=${ROLLOUT_BATCH_SIZE} N_SAMPLES=${N_SAMPLES_PER_PROMPT} GLOBAL_BATCH=${GLOBAL_BATCH_SIZE} SAVE_INTERVAL=${SAVE_INTERVAL}"
echo "SGLANG_SERVER_CONCURRENCY=${SGLANG_SERVER_CONCURRENCY} (in-flight groups ≈ ${SGLANG_SERVER_CONCURRENCY} * num_engines)"
echo "CONTEXT=${ROLLOUT_MAX_CONTEXT_LEN} RESPONSE=${ROLLOUT_MAX_RESPONSE_LEN} TP=${TP_SIZE} CP=${CP_SIZE} MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU}"
echo "LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE} OPTIMIZER_CPU_OFFLOAD=${OPTIMIZER_CPU_OFFLOAD}"
echo "USE_WANDB=${USE_WANDB} WANDB_MODE=${WANDB_MODE} WANDB_DIR=${WANDB_DIR} WANDB_PROJECT=${WANDB_PROJECT}"
echo "ENABLE_THINKING=${ENABLE_THINKING} APPLY_CHAT_TEMPLATE_KWARGS=${APPLY_CHAT_TEMPLATE_KWARGS}"
echo "MAX_POLICY_LAG=${MAX_POLICY_LAG} OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN} OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR}"

# shellcheck disable=SC1091
source "${LISAO_SLIME_ROOT}/scripts/models/qwen3-4B.sh"

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_DIR}"
  --ref-load "${REF_LOAD}"
  --save "${SAVE_DIR}"
  # Last rollout always saves when save_interval is set.
  --save-interval "${SAVE_INTERVAL}"
)

ROLLOUT_ARGS=(
  --rollout-function-path lisao.inflight_async.rollout.generate_rollout_inflight_async
  --prompt-data "${DATA_PATH}"
  --input-key prompt
  --label-key label
  --apply-chat-template
  --rollout-shuffle
  # deepscaler needs </think>+\\boxed{}; flexible grades boxed or final integer.
  --rm-type math
  --custom-rm-path lisao.eval.math_rm.async_rm_math_flexible
  --custom-reward-post-process-path lisao.sao.overlong.post_process_rewards
  --rollout-top-p 1
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  # Full native window; effective new tokens = min(response, context - prompt).
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
  --rollout-temperature 1
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --balance-data
  --update-weights-interval 1
)
if [[ -n "${MAX_POLICY_LAG}" ]]; then
  ROLLOUT_ARGS+=(--max-policy-lag "${MAX_POLICY_LAG}")
fi
if [[ -n "${APPLY_CHAT_TEMPLATE_KWARGS}" ]]; then
  ROLLOUT_ARGS+=(--apply-chat-template-kwargs "${APPLY_CHAT_TEMPLATE_KWARGS}")
fi

PERF_ARGS=(
  --tensor-model-parallel-size "${TP_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size "${CP_SIZE}"
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  # Already on: activation recompute (saves act mem; does NOT shrink final logits).
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}"
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-rollout-logprobs
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)
if [[ "${OPTIMIZER_CPU_OFFLOAD}" == "1" ]]; then
  OPTIMIZER_ARGS+=(
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
  )
fi

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 2
  --sglang-mem-fraction-static 0.7
  # slime default 512 * 2 engines floods KV on long thinking.
  --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
  # Safe with memory saver (colocate/offload); breakable prefill CUDA graph crashes.
  --sglang-cuda-graph-backend-prefill disabled
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

WANDB_ARGS=()
if [[ "${USE_WANDB}" == "1" ]]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --wandb-mode "${WANDB_MODE}"
    --wandb-dir "${WANDB_DIR}"
    --disable-wandb-random-suffix
  )
  # Optional: pass key explicitly. Otherwise rely on `wandb login` (~/.netrc).
  if [[ -n "${WANDB_KEY}" ]]; then
    WANDB_ARGS+=(--wandb-key "${WANDB_KEY}")
  fi
fi

# Ray job entrypoint is not assigned GPUs; Megatron validate_args now probes cuda:0.
# Keep devices visible on the driver without stealing a GPU from placement groups.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
fi
export CUDA_VISIBLE_DEVICES
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
if [[ "${SKIP_RAY_START}" == "1" ]]; then
  echo "SKIP_RAY_START=1: reuse Ray cluster MASTER_ADDR=${MASTER_ADDR}"
  ray status
else
  ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

# Build runtime env JSON (optional WANDB_API_KEY for online sync).
RUNTIME_ENV_VARS=(
  "\"PYTHONPATH\": \"${MEGATRON_PATH}:${LISAO_SLIME_ROOT}\""
  "\"CUDA_VISIBLE_DEVICES\": \"${CUDA_VISIBLE_DEVICES}\""
  "\"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES\": \"1\""
  "\"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\""
  "\"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\""
  "\"WANDB_DIR\": \"${WANDB_DIR}\""
  "\"WANDB_MODE\": \"${WANDB_MODE}\""
  "\"MASTER_ADDR\": \"${MASTER_ADDR}\""
  "\"OVERLONG_BUFFER_LEN\": \"${OVERLONG_BUFFER_LEN}\""
  "\"OVERLONG_PENALTY_FACTOR\": \"${OVERLONG_PENALTY_FACTOR}\""
)
if [[ -n "${WANDB_KEY}" ]]; then
  RUNTIME_ENV_VARS+=("\"WANDB_API_KEY\": \"${WANDB_KEY}\"")
fi
if [[ -n "${NCCL_IB_DISABLE:-}" ]]; then
  RUNTIME_ENV_VARS+=("\"NCCL_IB_DISABLE\": \"${NCCL_IB_DISABLE}\"")
fi
if [[ -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
  RUNTIME_ENV_VARS+=("\"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME}\"")
fi
if [[ -n "${NCCL_IB_HCA:-}" ]]; then
  RUNTIME_ENV_VARS+=("\"NCCL_IB_HCA\": \"${NCCL_IB_HCA}\"")
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

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 "${LISAO_SLIME_ROOT}/train_async.py" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${ACTOR_GPUS}" \
  --rollout-num-gpus "${ROLLOUT_GPUS}" \
  ${MODEL_ARGS[@]} \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${WANDB_ARGS[@]}"

echo "Training finished. ckpt -> ${SAVE_DIR}"
if [[ "${USE_WANDB}" == "1" ]]; then
  echo "WandB mode=${WANDB_MODE} dir=${WANDB_DIR} project=${WANDB_PROJECT}"
  if [[ "${WANDB_MODE}" == "offline" ]]; then
    echo "  Offline run; later sync: WANDB_API_KEY=... wandb sync ${WANDB_DIR}/wandb/offline-run-*"
  else
    echo "  Online: https://wandb.ai (project ${WANDB_PROJECT})"
  fi
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  echo "Starting offline AIME 2024, 2025, and 2026 evaluation."
  LOAD="${SAVE_DIR}" \
  MODEL_DIR="${MODEL_DIR}" \
  REF_LOAD="${REF_LOAD}" \
  DATA_ROOT="${DATA_ROOT}" \
  NUM_GPUS="${NUM_GPUS}" \
  EVAL_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN}" \
  EVAL_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN}" \
  ENABLE_THINKING="${ENABLE_THINKING}" \
  APPLY_CHAT_TEMPLATE_KWARGS="${APPLY_CHAT_TEMPLATE_KWARGS}" \
  EVAL_AIME_YEARS="${EVAL_AIME_YEARS:-2024,2025,2026}" \
    bash "${LISAO_ROOT}/examples/inflight_async/eval-qwen3-4B-aime.sh"
fi
