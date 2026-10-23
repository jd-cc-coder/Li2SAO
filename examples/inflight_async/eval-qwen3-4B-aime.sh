#!/bin/bash
# Offline single-turn evaluation on AIME 2024, 2025, and 2026.
#
# In-flight training does not run evaluation. This starts a separate
# train.py job with num-rollout 0. Generation is text only.
#
# Usage:
#   LOAD=/root/data/models/Qwen3-4B_torch_dist \
#     bash /root/LiSAO/examples/inflight_async/eval-qwen3-4B-aime.sh
#   LOAD=/root/data/ckpts/qwen3-4B-li2sao CKPT_STEP=500 \
#     bash /root/LiSAO/examples/inflight_async/eval-qwen3-4B-aime.sh
#   EVAL_ALL=1 LOAD=/root/data/ckpts/qwen3-4B-li2sao \
#     bash /root/LiSAO/examples/inflight_async/eval-qwen3-4B-aime.sh
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
DATA_ROOT=${DATA_ROOT:-/root/data}
MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM}

MODEL_DIR=${MODEL_DIR:-${DATA_ROOT}/models/Qwen3-4B}
REF_LOAD=${REF_LOAD:-${MODEL_DIR}_torch_dist}
# Prefer trained ckpt; fall back to converted torch_dist for pre-train baseline.
TRAINED_CKPT=${DATA_ROOT}/ckpts/qwen3-4B-inflight-async
if [[ -n "${LOAD:-}" ]]; then
  :
elif [[ -f "${TRAINED_CKPT}/latest_checkpointed_iteration.txt" ]]; then
  LOAD=${TRAINED_CKPT}
else
  LOAD=${REF_LOAD}
fi
# Evaluate AIME 2024, 2025, and 2026 in one job.
EVAL_AIME_YEARS=${EVAL_AIME_YEARS:-2024,2025,2026}
EVAL_PROMPT_DATA_ARGS=()
if [[ -n "${EVAL_AIME_YEARS}" ]]; then
  IFS=',' read -ra _ys <<< "${EVAL_AIME_YEARS}"
  for y in "${_ys[@]}"; do
    y="${y// /}"
    [[ -n "${y}" ]] || continue
    if [[ ! "${y}" =~ ^202[456]$ ]]; then
      echo "ERROR: unknown AIME year ${y} (want 2024, 2025, and/or 2026)" >&2
      exit 1
    fi
    _path="${DATA_ROOT}/datasets/aime-${y}/aime-${y}.jsonl"
    EVAL_PROMPT_DATA_ARGS+=("aime${y}" "${_path}")
  done
  unset _ys y _path
  if [[ ${#EVAL_PROMPT_DATA_ARGS[@]} -eq 0 ]]; then
    echo "ERROR: EVAL_AIME_YEARS is empty" >&2
    exit 1
  fi
  EVAL_DATA_PATH=${EVAL_PROMPT_DATA_ARGS[1]}
else
  EVAL_DATA_PATH=${EVAL_DATA_PATH:-${DATA_ROOT}/datasets/aime-2025/aime-2025.jsonl}
  EVAL_PROMPT_DATA_ARGS=(aime2025 "${EVAL_DATA_PATH}")
fi
# Placeholder prompt-data (unused when num_rollout=0, but path must exist).
PROMPT_DATA=${PROMPT_DATA:-${EVAL_DATA_PATH}}

NUM_GPUS=${NUM_GPUS:-8}
# COLOCATE=1 (default): actor and SGLang share NUM_GPUS. COLOCATE=0: disagg.
COLOCATE=${COLOCATE:-1}
ACTOR_GPUS=${ACTOR_GPUS:-${NUM_GPUS}}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-${ACTOR_GPUS}}
N_SAMPLES=${N_SAMPLES:-16}
# Per-engine client concurrency (slime default 512). In-flight requests ≈ this * num_engines.
# Colocate 8 GPU / TP=2 → 4 engines; AIME 30×16=480 so default 512 will fire all at once.
# Long 32K eval can fill KV — drop to 16/32 if engines OOM or stall.
SGLANG_SERVER_CONCURRENCY=${SGLANG_SERVER_CONCURRENCY:-512}
# Align with training: thinking off by default, temp=1, full Qwen3-4B native 32768 (~32K).
# Response may equal context; generate clamps to context - prompt for SGLang.
MODEL_MAX_CONTEXT_LEN=${MODEL_MAX_CONTEXT_LEN:-32768}
EVAL_MAX_CONTEXT_LEN=${EVAL_MAX_CONTEXT_LEN:-${MODEL_MAX_CONTEXT_LEN}}
EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN:-${EVAL_MAX_CONTEXT_LEN}}
ENABLE_THINKING=${ENABLE_THINKING:-0}
EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-1.0}
# Default thinking off. Set ENABLE_THINKING=1 to re-enable.
APPLY_CHAT_TEMPLATE_KWARGS=${APPLY_CHAT_TEMPLATE_KWARGS:-}
if [[ -z "${APPLY_CHAT_TEMPLATE_KWARGS}" ]]; then
  if [[ "${ENABLE_THINKING}" == "1" ]]; then
    APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking": true}'
  else
    APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
  fi
fi
# Megatron requires global_batch_size % (micro_batch_size * dp_size) == 0.
# With TP=2, PP=1: dp_size = ACTOR_GPUS / TP (eval-only placeholder).
TP_SIZE=${TP_SIZE:-2}
DP_SIZE=$((ACTOR_GPUS / TP_SIZE))
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-${DP_SIZE}}
EVAL_OUT_DIR=${EVAL_OUT_DIR:-${DATA_ROOT}/eval}
if [[ "${EVAL_AIME_YEARS}" == *","* ]]; then
  EVAL_TAG=${EVAL_TAG:-aime}
else
  EVAL_TAG=${EVAL_TAG:-aime${EVAL_AIME_YEARS// /}}
fi
EVAL_LOG=${EVAL_LOG:-${EVAL_OUT_DIR}/${EVAL_TAG}_$(date -u +%Y%m%dT%H%M%SZ).log}
mkdir -p "${EVAL_OUT_DIR}"
export EVAL_OUT_DIR EVAL_TAG DATA_ROOT

for p in "${MODEL_DIR}" "${REF_LOAD}"; do
  [[ -d "${p}" ]] || { echo "ERROR: missing ${p}" >&2; exit 1; }
done
_i=1
while [[ "${_i}" -lt "${#EVAL_PROMPT_DATA_ARGS[@]}" ]]; do
  [[ -f "${EVAL_PROMPT_DATA_ARGS[$_i]}" ]] || {
    echo "ERROR: missing ${EVAL_PROMPT_DATA_ARGS[$_i]}" >&2
    exit 1
  }
  _i=$((_i + 2))
done
unset _i
[[ -f "${LISAO_SLIME_ROOT}/train.py" ]] || {
  echo "ERROR: missing ${LISAO_SLIME_ROOT}/train.py (is LiSAO mounted?)" >&2
  exit 1
}
[[ -f "${LISAO_SLIME_ROOT}/lisao/eval/eval_log.py" ]] || {
  echo "ERROR: missing lisao/eval/eval_log.py; sync latest LiSAO code into the mount" >&2
  exit 1
}
[[ -f "${LISAO_SLIME_ROOT}/lisao/eval/math_rm.py" ]] || {
  echo "ERROR: missing lisao/eval/math_rm.py; sync latest LiSAO code into the mount" >&2
  exit 1
}
[[ -f "${LOAD}/latest_checkpointed_iteration.txt" ]] || {
  echo "ERROR: missing megatron ckpt under ${LOAD}" >&2
  echo "  For pre-train baseline: LOAD=${REF_LOAD} (after convert_qwen3_4b.sh)" >&2
  echo "  For post-train: LOAD=${TRAINED_CKPT}" >&2
  exit 1
}

# grep exits 1 when there is no NVLink; pipefail + set -e would abort the script.
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l | tr -d ' ' || true)
NVLINK_COUNT=${NVLINK_COUNT:-0}
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
if (( ACTOR_GPUS % TP_SIZE != 0 )); then
  echo "ERROR: ACTOR_GPUS=${ACTOR_GPUS} not divisible by TP_SIZE=${TP_SIZE}" >&2
  exit 1
fi
if (( ROLLOUT_GPUS < 2 || ROLLOUT_GPUS % 2 != 0 )); then
  echo "ERROR: ROLLOUT_GPUS=${ROLLOUT_GPUS} must be >=2 and even (rollout-num-gpus-per-engine=2)" >&2
  exit 1
fi
if (( GLOBAL_BATCH_SIZE % DP_SIZE != 0 )); then
  echo "ERROR: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} not divisible by DP_SIZE=${DP_SIZE}" >&2
  exit 1
fi
if [[ -z "${SGLANG_MEM_FRACTION_STATIC}" ]]; then
  if [[ "${COLOCATE}" == "1" ]]; then
    SGLANG_MEM_FRACTION_STATIC=0.7
  else
    SGLANG_MEM_FRACTION_STATIC=0.85
  fi
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

echo "HAS_NVLINK=${HAS_NVLINK} LOAD=${LOAD} CKPT_STEP=${CKPT_STEP:-<latest>} EVAL_AIME_YEARS=${EVAL_AIME_YEARS:-2025} EVAL_PROMPT_DATA=${EVAL_PROMPT_DATA_ARGS[*]}"
echo "COLOCATE=${COLOCATE} ACTOR_GPUS=${ACTOR_GPUS} ROLLOUT_GPUS=${ROLLOUT_GPUS} NUM_GPUS=${NUM_GPUS}"
echo "TP_SIZE=${TP_SIZE} DP_SIZE=${DP_SIZE} GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "APPLY_CHAT_TEMPLATE_KWARGS=${APPLY_CHAT_TEMPLATE_KWARGS} EVAL_TEMPERATURE=${EVAL_TEMPERATURE}"
echo "EVAL_MAX_CONTEXT_LEN=${EVAL_MAX_CONTEXT_LEN} EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN}"
echo "N_SAMPLES=${N_SAMPLES} SGLANG_SERVER_CONCURRENCY=${SGLANG_SERVER_CONCURRENCY} (in-flight ≈ ${SGLANG_SERVER_CONCURRENCY} * num_engines)"

# shellcheck disable=SC1091
source "${LISAO_SLIME_ROOT}/scripts/models/qwen3-4B.sh"

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_DIR}"
  --ref-load "${REF_LOAD}"
  --load "${LOAD}"
  --save /tmp/lisao_aime2025_eval_dummy
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
  # Same RM as training: boxed preferred, else final integer (AIME-friendly).
  --rm-type math
  --custom-rm-path lisao.eval.math_rm.async_rm_math_flexible
  --num-rollout 0
  --rollout-batch-size 1
  --n-samples-per-prompt 1
  --rollout-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
  --rollout-max-context-len "${EVAL_MAX_CONTEXT_LEN}"
  --rollout-temperature "${EVAL_TEMPERATURE}"
  # Placeholder for Megatron init only (num_rollout=0); must be divisible by DP.
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
)
if [[ -n "${APPLY_CHAT_TEMPLATE_KWARGS}" ]]; then
  ROLLOUT_ARGS+=(--apply-chat-template-kwargs "${APPLY_CHAT_TEMPLATE_KWARGS}")
fi

EVAL_ARGS=(
  --eval-interval 1
  --eval-prompt-data "${EVAL_PROMPT_DATA_ARGS[@]}"
  --eval-input-key prompt
  --eval-label-key label
  --n-samples-per-eval-prompt "${N_SAMPLES}"
  --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
  --eval-max-context-len "${EVAL_MAX_CONTEXT_LEN}"
  --eval-temperature "${EVAL_TEMPERATURE}"
  --eval-top-p 1
  --log-passrate
  --custom-eval-rollout-log-function-path lisao.eval.eval_log.log_eval_rollout_data
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
  --max-tokens-per-gpu 4096
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
  # Placeholder: num_rollout=0 would otherwise make lr_decay_steps=0 and crash.
  --lr-decay-iters 1
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  # Eval does not train. A training ckpt's scheduler has lr_decay_steps from
  # the train job (e.g. 64000); eval's placeholder is 1 * GBS (e.g. 4).
  # Megatron asserts they match unless we skip optim/scheduler restore.
  --no-load-optim
  --no-load-rng
  --finetune
  --override-opt-param-scheduler
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 2
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
)
if [[ "${COLOCATE}" == "1" ]]; then
  # Colocate eval enables memory saver; breakable prefill CUDA graph is incompatible.
  SGLANG_ARGS+=(--sglang-cuda-graph-backend-prefill disabled)
fi

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

# Per-node GPU list (8), never cluster-wide 16.
# ray job submit driver has 0 GPUs. Empty CUDA_VISIBLE_DEVICES plus
# RAY_EXPERIMENTAL_NOSET leaves MegatronTrainRayActor with no CUDA.
_cvd=${CUDA_VISIBLE_DEVICES-}
if [[ -z "${_cvd}" || "${_cvd}" == "NoDevFiles" || "${_cvd}" == "-1" || "${_cvd}" == "void" ]]; then
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%,}
fi
unset _cvd
export CUDA_VISIBLE_DEVICES NUM_GPUS
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
if [[ "${COLOCATE}" == "1" ]]; then
  EVAL_NEED_GPUS=${EVAL_NEED_GPUS:-${ACTOR_GPUS}}
  if (( ROLLOUT_GPUS > EVAL_NEED_GPUS )); then
    EVAL_NEED_GPUS=${ROLLOUT_GPUS}
  fi
else
  EVAL_NEED_GPUS=${EVAL_NEED_GPUS:-$((ACTOR_GPUS + ROLLOUT_GPUS))}
fi
export EVAL_NEED_GPUS

# Submit shell may not see GPUs (2-node job driver uses Ray workers).
if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "WARN: nvidia-smi not usable in this process. Continuing; Ray workers must have GPUs." >&2
fi

if [[ "${SKIP_RAY_START}" == "1" ]]; then
  export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
  echo "SKIP_RAY_START=1: reuse Ray cluster MASTER_ADDR=${MASTER_ADDR}"
  ray status
  python3 - <<'PY'
import os
import ray

ray.init(address="auto", ignore_reinit_error=True)
alive = [n for n in ray.nodes() if n.get("Alive")]
gpus = float(ray.cluster_resources().get("GPU", 0) or 0)
ips = sorted({n["NodeManagerAddress"] for n in alive})
need = int(os.environ.get("EVAL_NEED_GPUS", os.environ["NUM_GPUS"]))
print(f"eval ray: nodes={len(alive)} gpus={gpus:g} ips={ips} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
if gpus + 1e-6 < need:
    raise SystemExit(f"ERROR: eval needs {need} GPUs, Ray reports {gpus}")
PY
else
  unset RAY_ADDRESS RAY_GCS_ADDRESS
  export MASTER_ADDR=127.0.0.1
  ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
  python3 - <<'PY'
import os
import ray

ray.init(address="auto", ignore_reinit_error=True)
alive = [n for n in ray.nodes() if n.get("Alive")]
gpus = float(ray.cluster_resources().get("GPU", 0) or 0)
ips = sorted({n["NodeManagerAddress"] for n in alive})
need = int(os.environ["NUM_GPUS"])
print(f"eval ray: nodes={len(alive)} gpus={gpus:g} ips={ips} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
if len(alive) != 1:
    raise SystemExit(
        f"ERROR: single-node eval expects 1 Ray node, got {len(alive)} {ips}. "
        "Stop any existing Ray cluster and rerun this script."
    )
if gpus + 1e-6 < need:
    raise SystemExit(f"ERROR: eval needs {need} GPUs, Ray reports {gpus}")
PY
fi

RUNTIME_ENV_VARS=(
  "\"PYTHONPATH\": \"${MEGATRON_PATH}:${LISAO_SLIME_ROOT}\""
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

CLUSTER_ARGS=(
  --actor-num-nodes 1
  --actor-num-gpus-per-node "${ACTOR_GPUS}"
  --num-gpus-per-node "${NUM_GPUS}"
)
if [[ "${COLOCATE}" == "1" ]]; then
  CLUSTER_ARGS+=(--colocate)
else
  CLUSTER_ARGS+=(--rollout-num-gpus "${ROLLOUT_GPUS}")
fi

set +e
ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 "${LISAO_SLIME_ROOT}/train.py" \
  "${CLUSTER_ARGS[@]}" \
  ${MODEL_ARGS[@]} \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  2>&1 | tee "${EVAL_LOG}"
status=${PIPESTATUS[0]}
set -e

echo "AIME ${EVAL_AIME_YEARS:-2025} eval exit=${status}"
echo "  full log: ${EVAL_LOG}"
echo "  metrics:  ${EVAL_OUT_DIR}/${EVAL_TAG}_latest.json  (and timestamped *.json)"
echo "  pass@k:   metrics['eval/aimeYYYY-pass@1/2/4/...']  (N_SAMPLES>1 + --log-passrate)"
echo "            or datasets.aimeYYYY.pass_rate in the JSON"
echo "  mean rw:  metrics['eval/aimeYYYY']  (avg of all sample rewards; = Pass@1 if N_SAMPLES=1)"
exit "${status}"
