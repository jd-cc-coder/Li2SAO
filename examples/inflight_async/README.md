# Single-turn GRPO

In-flight GRPO for single-turn mathematical reasoning. The policy is Qwen3-4B, the training set is DAPO-Math-17K, and evaluation is AIME 2024, 2025, and 2026 without tools.

| Script | Role |
| --- | --- |
| `run-qwen3-4B-grpo.sh` | 8 samples per prompt, clip 0.2 and 0.28, 500 steps |
| `eval-qwen3-4B-aime.sh` | Offline evaluation of one checkpoint |
| `eval-qwen3-4B-aime-all.sh` | Base model plus every saved iteration |

```bash
bash /root/LiSAO/scripts/setup_env.sh
bash /root/LiSAO/scripts/download_assets.sh
bash /root/LiSAO/scripts/convert_qwen3_4b.sh
bash /root/LiSAO/examples/inflight_async/run-qwen3-4B-grpo.sh

LOAD=/root/data/ckpts/qwen3-4B-inflight-async \
  bash /root/LiSAO/examples/inflight_async/eval-qwen3-4B-aime-all.sh
```

The default layout is 4 training GPUs and 4 rollout GPUs. Thinking is off, the context window is 32768, and the maximum policy lag is 16. Responses in the last quarter of the window receive the DAPO overlong penalty.

LiSAO and Li2SAO for this task are in [`examples/lisao`](../lisao/README.md).
