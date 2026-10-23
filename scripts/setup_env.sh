#!/bin/bash
# Install lisao_slime (editable) so both ``slime`` and ``lisao`` are importable.
# All network downloads go through mirrors (pip / HF / GitHub proxy).
#
# Usage (LiSAO repo mounted at /root/LiSAO):
#   bash /root/LiSAO/scripts/setup_env.sh
set -exuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/mirror_env.sh"

LISAO_ROOT=${LISAO_ROOT:-/root/LiSAO}
LISAO_SLIME_ROOT=${LISAO_SLIME_ROOT:-${LISAO_ROOT}/lisao_slime}
MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM}

if [[ ! -d "${LISAO_SLIME_ROOT}/slime" ]]; then
  echo "ERROR: ${LISAO_SLIME_ROOT}/slime not found." >&2
  exit 1
fi
if [[ ! -d "${LISAO_SLIME_ROOT}/lisao/inflight_async" ]]; then
  echo "ERROR: ${LISAO_SLIME_ROOT}/lisao/inflight_async not found." >&2
  exit 1
fi

echo "mirrors: PIP_INDEX_URL=${PIP_INDEX_URL} HF_ENDPOINT=${HF_ENDPOINT} GITHUB_PROXY=${GITHUB_PROXY}"

cd "${LISAO_SLIME_ROOT}"
pip_mirror install -e . --no-deps
# CLI only; do not install hf_transfer (it bypasses HF_ENDPOINT).
pip_mirror install -U "huggingface_hub[cli]"

# Weight convert / Megatron bridge (pinned same as slime Dockerfile).
# Install via ghfast.top GitHub proxy (override with MBRIDGE_PIP_URL).
MBRIDGE_COMMIT="${MBRIDGE_COMMIT:-89eb10887887bc74853f89a4de258c0702932a1c}"
MBRIDGE_PIP_URL="${MBRIDGE_PIP_URL:-git+https://ghfast.top/https://github.com/ISEEKYAN/mbridge.git@${MBRIDGE_COMMIT}}"
if ! python -c "import mbridge" >/dev/null 2>&1; then
  echo "mbridge missing; installing ${MBRIDGE_PIP_URL}"
  pip install "${MBRIDGE_PIP_URL}" --no-deps
else
  echo "mbridge already importable; skip install"
fi

# lisao_slime root exposes packages: slime/ and lisao/
# ${PYTHONPATH:-}: set -u fails if PYTHONPATH is unset.
export PYTHONPATH="${MEGATRON_PATH}:${LISAO_SLIME_ROOT}:${PYTHONPATH:-}"
EXPORT_LINE="export PYTHONPATH=\"${MEGATRON_PATH}:${LISAO_SLIME_ROOT}:\${PYTHONPATH:-}\""

if ! grep -qF "${LISAO_SLIME_ROOT}" /root/.bashrc 2>/dev/null; then
  {
    echo ""
    echo "# lisao_slime"
    echo "${EXPORT_LINE}"
  } >> /root/.bashrc
fi

python - <<'PY'
import lisao
import lisao.inflight_async
import mbridge
import slime
print("setup_env ok:", "lisao", lisao.__file__, "slime", slime.__file__, "mbridge", mbridge.__file__)
PY

echo "PYTHONPATH=${PYTHONPATH}"
echo "PIP_INDEX_URL=${PIP_INDEX_URL} HF_ENDPOINT=${HF_ENDPOINT} GITHUB_PROXY=${GITHUB_PROXY}"
echo "setup_env.sh done."
