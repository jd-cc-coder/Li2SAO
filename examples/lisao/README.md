# Single-turn LiSAO and Li2SAO

These scripts train Qwen3-4B on DAPO-Math-17K with the same in-flight pipeline as GRPO. Evaluation is offline and text only, on AIME 2024, 2025, and 2026.

| Script | Role |
| --- | --- |
| `run-qwen3-4B-lisao.sh` | 1 sample per prompt, GAE, DIS, 10 critic warmup steps |
| `run-qwen3-4B-li2sao.sh` | 8 samples during bootstrap, then 1 sample and GAE |
| `megatron_critic.yaml` | Frozen attention critic template |

Shared critic settings: learning rate 5e-6, 2 updates per batch, attention frozen, MLP blocks and the value head trained inside the actor process. LiSAO waits 10 steps before the first actor update. Li2SAO updates both networks from the first step.

Li2SAO bootstrap uses group-relative actor advantages and a group-centered critic target with ranking coefficient 0.1 and margin 0.1. Value clip is 0.2 during the bootstrap and inactive after the switch. The switch opens after step 60 when smoothed residual explained variance exceeds 0.2 and ranking accuracy exceeds 0.65 for 10 steps, and it is forced by step 150. Leftover bootstrap groups are dropped. Clip and DIS stay at the LiSAO values 0.3 and 5.0, and the interval from 0.7 to 6.0.

```bash
bash /root/LiSAO/scripts/setup_env.sh
bash /root/LiSAO/scripts/download_assets.sh
bash /root/LiSAO/scripts/convert_qwen3_4b.sh
bash /root/LiSAO/examples/lisao/run-qwen3-4B-lisao.sh
bash /root/LiSAO/examples/lisao/run-qwen3-4B-li2sao.sh

LOAD=/root/data/ckpts/qwen3-4B-li2sao \
  bash /root/LiSAO/examples/inflight_async/eval-qwen3-4B-aime-all.sh
```

GRPO for this task is in [`examples/inflight_async`](../inflight_async/README.md).
