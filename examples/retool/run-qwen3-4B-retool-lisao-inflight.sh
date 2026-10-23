#!/bin/bash
# In-flight LiSAO for the ReTool math experiment.
#
# Same LiSAO settings as single-turn reasoning. GAE skips interpreter
# observation tokens. The critic runs inside the actor process.
#
# Prerequisites:
#   bash /root/LiSAO/scripts/setup_env.sh
#   bash /root/LiSAO/examples/retool/download_assets.sh
#   bash /root/LiSAO/examples/retool/convert_qwen3_4b_sft.sh
set -exuo pipefail

SKIP_CLEANUP=${SKIP_CLEANUP:-0}
SKIP_RAY_START=${SKIP_RAY_START:-0}
FAST_MODE=${FAST_MODE:-1}
for arg in "$@"; do
  case "${arg}" in
    --fast_mode|--fast-mode) FAST_MODE=1 ;;
  esac
done

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

MODEL_DIR=${MODEL_DIR:-${DATA_ROOT}/models/qwen3-4b-sft}
REF_LOAD=${REF_LOAD:-${MODEL_DIR}_torch_dist}
DATA_PATH=${DATA_PATH:-${DATA_ROOT}/datasets/dapo-math-17k/dapo-math-17k.jsonl}
SAVE_DIR=${SAVE_DIR:-${DATA_ROOT}/ckpts/qwen3-4B-retool-lisao-inflight}

NUM_ROLLOUT=${NUM_ROLLOUT:-500}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-128}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
SAVE_INTERVAL=${SAVE_INTERVAL:-100}
NUM_GPUS=${NUM_GPUS:-8}
ACTOR_GPUS=${ACTOR_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-$((NUM_GPUS - ACTOR_GPUS))}
SGLANG_SERVER_CONCURRENCY=${SGLANG_SERVER_CONCURRENCY:-16}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.7}
RUN_EVAL=${RUN_EVAL:-0}
ENABLE_THINKING=${ENABLE_THINKING:-0}
APPLY_CHAT_TEMPLATE_KWARGS=${APPLY_CHAT_TEMPLATE_KWARGS:-}
if [[ -z "${APPLY_CHAT_TEMPLATE_KWARGS}" ]]; then
  if [[ "${ENABLE_THINKING}" == "1" ]]; then
    APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking": true}'
  else
    APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
  fi
fi

MODEL_MAX_CONTEXT_LEN=${MODEL_MAX_CONTEXT_LEN:-32768}
ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-${MODEL_MAX_CONTEXT_LEN}}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-${ROLLOUT_MAX_CONTEXT_LEN}}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-$((ROLLOUT_MAX_RESPONSE_LEN / 4))}
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}
TP_SIZE=${TP_SIZE:-2}
CP_SIZE=${CP_SIZE:-1}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-8192}
OPTIMIZER_CPU_OFFLOAD=${OPTIMIZER_CPU_OFFLOAD:-0}
MAX_POLICY_LAG=${MAX_POLICY_LAG:-16}

NUM_CRITIC_ONLY_STEPS=${NUM_CRITIC_ONLY_STEPS:-10}
CRITIC_TRAIN_EPOCH=${CRITIC_TRAIN_EPOCH:-2}
GAE_ALPHA=${GAE_ALPHA:-1.5}
USE_SKIP_OBS_GAE=${USE_SKIP_OBS_GAE:-1}
TIS_CLIP_LOW=${TIS_CLIP_LOW:-0.7}
TIS_CLIP=${TIS_CLIP:-6.0}
EPS_CLIP=${EPS_CLIP:-0.3}
EPS_CLIP_HIGH=${EPS_CLIP_HIGH:-5.0}
ACTOR_LR=${ACTOR_LR:-1e-6}
CRITIC_LR=${CRITIC_LR:-5e-6}

USE_WANDB=${USE_WANDB:-1}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_PROJECT=${WANDB_PROJECT:-lisao-retool}
WANDB_GROUP=${WANDB_GROUP:-qwen3-4B-lisao-inflight}
WANDB_DIR=${WANDB_DIR:-${DATA_ROOT}/wandb}
WANDB_KEY=${WANDB_KEY:-${WANDB_API_KEY:-}}
if [[ -z "${WANDB_KEY}" && -f "${HOME}/.netrc" ]]; then
  WANDB_KEY=$(python3 -c 'import netrc; a=netrc.netrc().authenticators("api.wandb.ai"); print(a[2] if a and a[2] else "")' 2>/dev/null || true)
fi
export WANDB_DIR WANDB_MODE
if [[ -n "${WANDB_KEY}" ]]; then
  export WANDB_API_KEY="${WANDB_KEY}"
elif [[ "${USE_WANDB}" == "1" && "${WANDB_MODE}" == "online" ]]; then
  echo "WARNING: WANDB_MODE=online but no WANDB_API_KEY / ~/.netrc. Run \`wandb login\` or export WANDB_API_KEY." >&2
fi
mkdir -p "${WANDB_DIR}"

for p in "${MODEL_DIR}" "${REF_LOAD}" "${RETOOL_SRC}"; do
  [[ -d "${p}" ]] || { echo "ERROR: missing ${p}" >&2; exit 1; }
done
[[ -f "${DATA_PATH}" ]] || { echo "ERROR: missing ${DATA_PATH}" >&2; exit 1; }
[[ -f "${LISAO_SLIME_ROOT}/train_async.py" ]] || {
  echo "ERROR: missing ${LISAO_SLIME_ROOT}/train_async.py (is LiSAO mounted?)" >&2
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
[[ -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]] || {
  echo "ERROR: ${REF_LOAD} is not a megatron ckpt (run convert_qwen3_4b_sft.sh)" >&2
  exit 1
}
mkdir -p "${SAVE_DIR}" "${SAVE_DIR}/critic"

if (( N_SAMPLES_PER_PROMPT != 1 )); then
  echo "WARNING: LiSAO is single-rollout; N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT} (expected 1)" >&2
fi

ACTOR_PARALLEL=$((TP_SIZE * CP_SIZE))
if (( ACTOR_GPUS % ACTOR_PARALLEL != 0 )); then
  echo "ERROR: ACTOR_GPUS=${ACTOR_GPUS} not divisible by TP_SIZE*CP_SIZE=${ACTOR_PARALLEL}" >&2
  exit 1
fi
if (( ROLLOUT_GPUS < 2 )); then
  echo "ERROR: ROLLOUT_GPUS=${ROLLOUT_GPUS} need >=2 (rollout-num-gpus-per-engine=2)" >&2
  exit 1
fi

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l | tr -d ' ' || true)
NVLINK_COUNT=${NVLINK_COUNT:-0}
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK=${HAS_NVLINK} ACTOR_GPUS=${ACTOR_GPUS} ROLLOUT_GPUS=${ROLLOUT_GPUS} NUM_ROLLOUT=${NUM_ROLLOUT}"
echo "ROLLOUT_BATCH=${ROLLOUT_BATCH_SIZE} N_SAMPLES=${N_SAMPLES_PER_PROMPT} GLOBAL_BATCH=${GLOBAL_BATCH_SIZE}"
echo "CONTEXT=${ROLLOUT_MAX_CONTEXT_LEN} RESPONSE=${ROLLOUT_MAX_RESPONSE_LEN} TP=${TP_SIZE} CP=${CP_SIZE}"
echo "OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN} USE_SKIP_OBS_GAE=${USE_SKIP_OBS_GAE}"
echo "CRITIC_TRAIN_EPOCH=${CRITIC_TRAIN_EPOCH} NUM_CRITIC_ONLY_STEPS=${NUM_CRITIC_ONLY_STEPS}"
echo "FAST_MODE=${FAST_MODE} DIS clip=[${TIS_CLIP_LOW}, ${TIS_CLIP}]"
echo "MAX_POLICY_LAG=${MAX_POLICY_LAG} SGLANG_SERVER_CONCURRENCY=${SGLANG_SERVER_CONCURRENCY}"
echo "MODEL_DIR=${MODEL_DIR} RETOOL_SRC=${RETOOL_SRC}"

# Instruct-2507 / SFT: rotary-base 5e6
# shellcheck disable=SC1091
source "${LISAO_SLIME_ROOT}/scripts/models/qwen3-4B-Instruct-2507.sh"

MEGATRON_CONFIG=${MEGATRON_CONFIG:-${SAVE_DIR}/megatron_critic.yaml}
cat > "${MEGATRON_CONFIG}" <<EOF
megatron:
  - name: default
    role: actor
    overrides:
      lr: ${ACTOR_LR}
      save: ${SAVE_DIR}
  - name: default
    role: critic
    overrides:
      lr: ${CRITIC_LR}
      save: ${SAVE_DIR}/critic
      freeze_params_name_list:
        - self_attention
      lambd: 1.0
      gae_alpha: null
EOF
cp -f "${EXAMPLE_RETOOL}/megatron_critic.yaml" "${SAVE_DIR}/megatron_critic.template.yaml"

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_DIR}"
  --ref-load "${REF_LOAD}"
  --save "${SAVE_DIR}"
  --save-interval "${SAVE_INTERVAL}"
  --megatron-config-path "${MEGATRON_CONFIG}"
)

ROLLOUT_ARGS=(
  --rollout-function-path lisao.inflight_async.rollout.generate_rollout_inflight_async
  --prompt-data "${DATA_PATH}"
  --input-key prompt
  --label-key label
  --apply-chat-template
  --rollout-shuffle
  --reward-key score
  --custom-reward-post-process-path lisao.sao.overlong.post_process_rewards
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
  --rollout-temperature 1
  --rollout-top-p 1
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
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}"
)

SAO_ARGS=(
  --advantage-estimator ppo
  --normalize-advantages
  --use-dis
  --tis-clip-low "${TIS_CLIP_LOW}"
  --tis-clip "${TIS_CLIP}"
  --custom-tis-function-path lisao.sao.dis.dis_tis_function
  --eps-clip "${EPS_CLIP}"
  --eps-clip-high "${EPS_CLIP_HIGH}"
  --gae-alpha "${GAE_ALPHA}"
  --num-critic-only-steps "${NUM_CRITIC_ONLY_STEPS}"
  --critic-train-epoch "${CRITIC_TRAIN_EPOCH}"
  --entropy-coef 0.00
)
if [[ "${USE_SKIP_OBS_GAE}" == "1" ]]; then
  SAO_ARGS+=(--use-skip-obs-gae)
fi
if [[ "${FAST_MODE}" == "1" ]]; then
  SAO_ARGS+=(--fast_mode)
fi

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${ACTOR_LR}"
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
  --custom-rm-path generate_with_retool.reward_func
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
  if [[ -n "${WANDB_KEY}" ]]; then
    WANDB_ARGS+=(--wandb-key "${WANDB_KEY}")
  fi
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
fi
export CUDA_VISIBLE_DEVICES
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export OVERLONG_BUFFER_LEN OVERLONG_PENALTY_FACTOR

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
if [[ "${SKIP_RAY_START}" == "1" ]]; then
  echo "SKIP_RAY_START=1: reuse Ray cluster MASTER_ADDR=${MASTER_ADDR}"
  ray status
else
  ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

RUNTIME_ENV_VARS=(
  "\"PYTHONPATH\": \"${MEGATRON_PATH}:${LISAO_SLIME_ROOT}:${RETOOL_SRC}:${EXAMPLE_RETOOL}\""
  "\"CUDA_VISIBLE_DEVICES\": \"${CUDA_VISIBLE_DEVICES}\""
  "\"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES\": \"1\""
  "\"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\""
  "\"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\""
  "\"WANDB_DIR\": \"${WANDB_DIR}\""
  "\"WANDB_MODE\": \"${WANDB_MODE}\""
  "\"MASTER_ADDR\": \"${MASTER_ADDR}\""
  "\"OVERLONG_BUFFER_LEN\": \"${OVERLONG_BUFFER_LEN}\""
  "\"OVERLONG_PENALTY_FACTOR\": \"${OVERLONG_PENALTY_FACTOR}\""
  "\"DATA_ROOT\": \"${DATA_ROOT}\""
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
  "${SAO_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${CUSTOM_ARGS[@]}" \
  "${WANDB_ARGS[@]}"

echo "Training finished. actor ckpt -> ${SAVE_DIR}"
echo "  critic ckpt -> ${SAVE_DIR}/critic"
if [[ "${USE_WANDB}" == "1" ]]; then
  echo "WandB mode=${WANDB_MODE} dir=${WANDB_DIR} project=${WANDB_PROJECT}"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  echo "Starting ReTool AIME eval (sandbox)..."
  LOAD="${SAVE_DIR}" \
  MODEL_DIR="${MODEL_DIR}" \
  REF_LOAD="${REF_LOAD}" \
  DATA_ROOT="${DATA_ROOT}" \
  NUM_GPUS="${NUM_GPUS}" \
  ENABLE_THINKING="${ENABLE_THINKING}" \
  APPLY_CHAT_TEMPLATE_KWARGS="${APPLY_CHAT_TEMPLATE_KWARGS}" \
    bash "${EXAMPLE_RETOOL}/eval-qwen3-4B-aime.sh"
fi
