# China mirrors for pip / Hugging Face / GitHub. Source from other scripts:
#   # shellcheck disable=SC1091
#   source "$(dirname "${BASH_SOURCE[0]}")/mirror_env.sh"
#
# Override:
#   PIP_INDEX_URL / PIP_TRUSTED_HOST / HF_ENDPOINT / GITHUB_PROXY
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# GitHub prefix proxy (used by setup_env mbridge install).
export GITHUB_PROXY="${GITHUB_PROXY:-https://ghfast.top}"

# Newer huggingface_hub defaults to Xet (cas-server.xethub.hf.co), which bypasses
# HF_ENDPOINT and often 401s behind mirrors. Force classic HTTP via the mirror.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
# Keep off so downloads stay on HF_ENDPOINT (do not use hf_transfer CDN).
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

# pip via Tsinghua index (all positional/option args forwarded).
pip_mirror() {
  pip "$@" -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}"
}
