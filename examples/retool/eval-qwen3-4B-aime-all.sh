#!/bin/bash
# Sweep ReTool AIME eval (sandbox + pass@k) over SFT base and every Megatron
# ``iter_*`` under the training save dir, then print a comparison table.
#
# Usage:
#   LOAD=/root/data/ckpts/qwen3-4B-retool-grpo-inflight-2t6p \
#     bash /root/LiSAO/examples/retool/eval-qwen3-4B-aime-all.sh
#   EVAL_ALL=1 LOAD=... bash /root/LiSAO/examples/retool/eval-qwen3-4B-aime.sh
#
# Env knobs (in addition to those of eval-qwen3-4B-aime.sh):
#   CKPT_DIR / LOAD     training save dir
#   REF_LOAD            SFT megatron ckpt (default: qwen3-4b-sft_torch_dist)
#   EVAL_BASE=1         also eval the SFT baseline
#   CKPT_STEPS=99,199   only these iterations; empty = every iter_*
#   SKIP_EXISTING=0     reuse ${EVAL_TAG}_latest.json when present
#   CONTINUE_ON_ERROR=1 keep going after a failed ckpt
#   SUMMARIZE_ONLY=0    rebuild the table from existing JSONs, skip eval
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
EVAL_SCRIPT="${SCRIPT_DIR}/eval-qwen3-4B-aime.sh"
SUMMARIZE_SCRIPT="${SCRIPT_DIR}/summarize_aime_eval.py"

LISAO_ROOT=${LISAO_ROOT:-/root/LiSAO}
DATA_ROOT=${DATA_ROOT:-/root/data}
MODEL_DIR=${MODEL_DIR:-${DATA_ROOT}/models/qwen3-4b-sft}
REF_LOAD=${REF_LOAD:-${MODEL_DIR}_torch_dist}
CKPT_DIR=${CKPT_DIR:-${LOAD:-${DATA_ROOT}/ckpts/qwen3-4B-retool-grpo}}
EVAL_OUT_DIR=${EVAL_OUT_DIR:-${DATA_ROOT}/eval/retool}
EVAL_TAG_PREFIX=${EVAL_TAG_PREFIX:-aime}
EVAL_BASE=${EVAL_BASE:-1}
SKIP_EXISTING=${SKIP_EXISTING:-0}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
SUMMARIZE_ONLY=${SUMMARIZE_ONLY:-0}

mkdir -p "${EVAL_OUT_DIR}"
JOBS_JSONL="${EVAL_OUT_DIR}/${EVAL_TAG_PREFIX}_all_jobs.jsonl"

list_ckpt_steps() {
  local dir="$1"
  local d name
  shopt -s nullglob
  for d in "${dir}"/iter_*; do
    [[ -d "${d}" ]] || continue
    name=$(basename "${d}")
    [[ "${name}" =~ ^iter_([0-9]+)$ ]] || continue
    printf '%d\n' "$((10#${BASH_REMATCH[1]}))"
  done | sort -n
}

append_job() {
  python3 -c '
import json, sys
name, step, tag, load, ckpt_step, status, exit_code = sys.argv[1:8]
job = {
    "name": name,
    "step": None if step == "" else int(step),
    "tag": tag,
    "load": load or None,
    "ckpt_step": None if ckpt_step == "" else int(ckpt_step),
    "status": status,
    "exit_code": None if exit_code == "" else int(exit_code),
}
print(json.dumps(job, ensure_ascii=False))
' "$@" >> "${JOBS_JSONL}"
}

summarize() {
  python3 "${SUMMARIZE_SCRIPT}" \
    --jobs "${JOBS_JSONL}" \
    --out-dir "${EVAL_OUT_DIR}" \
    --prefix "${EVAL_TAG_PREFIX}"
}

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "ERROR: missing ${EVAL_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${SUMMARIZE_SCRIPT}" ]]; then
  echo "ERROR: missing ${SUMMARIZE_SCRIPT}" >&2
  exit 1
fi
if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "ERROR: missing CKPT_DIR=${CKPT_DIR}" >&2
  exit 1
fi

STEPS=()
if [[ -n "${CKPT_STEPS:-}" ]]; then
  IFS=',' read -ra _raw_steps <<< "${CKPT_STEPS}"
  for s in "${_raw_steps[@]}"; do
    s="${s//[[:space:]]/}"
    [[ -n "${s}" ]] || continue
    if ! [[ "${s}" =~ ^[0-9]+$ ]]; then
      echo "ERROR: CKPT_STEPS entries must be integers, got ${s}" >&2
      exit 1
    fi
    STEPS+=("${s}")
  done
else
  mapfile -t STEPS < <(list_ckpt_steps "${CKPT_DIR}")
fi

if [[ ${#STEPS[@]} -eq 0 && "${EVAL_BASE}" != "1" ]]; then
  echo "ERROR: no iter_* under ${CKPT_DIR} and EVAL_BASE=0" >&2
  exit 1
fi

if [[ ${#STEPS[@]} -gt 0 ]]; then
  echo "Sweep ReTool AIME: base=${EVAL_BASE} ckpts=${STEPS[*]} dir=${CKPT_DIR}"
else
  echo "Sweep ReTool AIME: base=${EVAL_BASE} ckpts=<none> dir=${CKPT_DIR}"
fi
echo "Results -> ${EVAL_OUT_DIR}  prefix=${EVAL_TAG_PREFIX}"
echo "N_SAMPLES=${N_SAMPLES:-16} EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-1.0} YEARS=${EVAL_AIME_YEARS:-2024,2025,2026}"

if [[ "${SUMMARIZE_ONLY}" == "1" ]]; then
  if [[ ! -s "${JOBS_JSONL}" ]]; then
    : > "${JOBS_JSONL}"
    if [[ "${EVAL_BASE}" == "1" ]]; then
      append_job "base" "0" "${EVAL_TAG_PREFIX}_base" "${REF_LOAD}" "" "ok" "0"
    fi
    if [[ ${#STEPS[@]} -gt 0 ]]; then
      for step in "${STEPS[@]}"; do
        append_job "step${step}" "${step}" "${EVAL_TAG_PREFIX}_step${step}" "${CKPT_DIR}" "${step}" "ok" "0"
      done
    fi
  fi
  summarize
  exit $?
fi

: > "${JOBS_JSONL}"
fail_count=0

run_one() {
  local name="$1"
  local step="$2"
  local tag="$3"
  local load="$4"
  local ckpt_step="$5"
  local latest="${EVAL_OUT_DIR}/${tag}_latest.json"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${latest}" ]]; then
    echo "SKIP existing ${latest}"
    append_job "${name}" "${step}" "${tag}" "${load}" "${ckpt_step}" "ok" "0"
    return 0
  fi

  echo "======== eval ${name}  LOAD=${load}  CKPT_STEP=${ckpt_step:-<none>}  TAG=${tag} ========"
  local status=0
  EVAL_ALL=0 \
  LOAD="${load}" \
  CKPT_STEP="${ckpt_step}" \
  EVAL_TAG="${tag}" \
  EVAL_OUT_DIR="${EVAL_OUT_DIR}" \
  REF_LOAD="${REF_LOAD}" \
  MODEL_DIR="${MODEL_DIR}" \
  DATA_ROOT="${DATA_ROOT}" \
  LISAO_ROOT="${LISAO_ROOT}" \
    bash "${EVAL_SCRIPT}" || status=$?

  if [[ "${status}" -eq 0 ]]; then
    append_job "${name}" "${step}" "${tag}" "${load}" "${ckpt_step}" "ok" "0"
    summarize || true
    return 0
  fi

  echo "ERROR: eval ${name} failed with exit=${status}" >&2
  append_job "${name}" "${step}" "${tag}" "${load}" "${ckpt_step}" "fail" "${status}"
  fail_count=$((fail_count + 1))
  summarize || true
  if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
    exit "${status}"
  fi
}

if [[ "${EVAL_BASE}" == "1" ]]; then
  run_one "base" "0" "${EVAL_TAG_PREFIX}_base" "${REF_LOAD}" ""
fi

if [[ ${#STEPS[@]} -gt 0 ]]; then
  for step in "${STEPS[@]}"; do
    iter_dir=$(printf "%s/iter_%07d" "${CKPT_DIR}" "${step}")
    if [[ ! -d "${iter_dir}" ]]; then
      echo "ERROR: missing ${iter_dir}" >&2
      append_job "step${step}" "${step}" "${EVAL_TAG_PREFIX}_step${step}" "${CKPT_DIR}" "${step}" "missing" ""
      fail_count=$((fail_count + 1))
      if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
        summarize || true
        exit 1
      fi
      continue
    fi
    run_one "step${step}" "${step}" "${EVAL_TAG_PREFIX}_step${step}" "${CKPT_DIR}" "${step}"
  done
fi

echo "======== ReTool AIME sweep table ========"
summarize
summary_status=$?

if [[ "${fail_count}" -gt 0 ]]; then
  echo "Sweep finished with ${fail_count} failed/missing checkpoint(s)." >&2
  exit 1
fi
exit "${summary_status}"
