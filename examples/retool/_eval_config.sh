# Shared eval YAML writer. Source from train / eval scripts.
# Pins generate → sandbox and RM → lisao.eval.math_rm (same 0/1 flexible
# grader as examples/lisao; eval_rm is a thin wrapper).
# Default years: AIME 2024 + 2025 + 2026.

retool_aime_jsonl() {
  local year=$1
  echo "${DATA_ROOT}/datasets/aime-${year}/aime-${year}.jsonl"
}

parse_eval_aime_years() {
  local raw=${EVAL_AIME_YEARS:-2024,2025,2026}
  EVAL_AIME_YEAR_LIST=()
  local y
  local _ys
  IFS=',' read -ra _ys <<< "${raw}"
  for y in "${_ys[@]}"; do
    y="${y// /}"
    [[ -n "${y}" ]] || continue
    if [[ ! "${y}" =~ ^202[456]$ ]]; then
      echo "ERROR: unknown AIME year ${y} (want 2024, 2025, and/or 2026)" >&2
      return 1
    fi
    EVAL_AIME_YEAR_LIST+=("${y}")
  done
  if [[ ${#EVAL_AIME_YEAR_LIST[@]} -eq 0 ]]; then
    echo "ERROR: EVAL_AIME_YEARS is empty" >&2
    return 1
  fi
}

require_retool_aime_jsonl() {
  local y f
  for y in "${EVAL_AIME_YEAR_LIST[@]}"; do
    f=$(retool_aime_jsonl "${y}")
    if [[ ! -f "${f}" ]]; then
      echo "ERROR: missing ${f}" >&2
      echo "  Run: bash ${EXAMPLE_RETOOL:-/root/LiSAO/examples/retool}/download_assets.sh" >&2
      return 1
    fi
  done
}

set_retool_eval_tag() {
  if [[ ${#EVAL_AIME_YEAR_LIST[@]} -eq 1 ]]; then
    EVAL_TAG=${EVAL_TAG:-aime${EVAL_AIME_YEAR_LIST[0]}}
  else
    EVAL_TAG=${EVAL_TAG:-aime}
  fi
}

write_retool_eval_config() {
  local dest=$1
  local y path
  mkdir -p "$(dirname "${dest}")"
  {
    echo "eval:"
    echo "  datasets:"
    for y in "${EVAL_AIME_YEAR_LIST[@]}"; do
      path=$(retool_aime_jsonl "${y}")
      cat <<EOF
    - name: aime${y}
      path: "${path}"
      input_key: prompt
      label_key: label
      custom_generate_function_path: generate_with_retool.generate
      custom_rm_path: eval_rm.reward_func
EOF
    done
  } > "${dest}"
}
