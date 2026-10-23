# ReTool

ReTool trains the same three algorithms on DAPO-Math-17K, starting from the tool-use supervised checkpoint `font-info/qwen3-4b-sft-SGLang-RL`. The model interleaves reasoning with a Python interpreter. A trajectory has at most 16 turns, and each interpreter call times out after 120 seconds.

Evaluation uses that same sandbox on AIME 2024, 2025, and 2026. The single-turn scripts do not call the interpreter, so they are not a valid score for these checkpoints.

| Script | Role |
| --- | --- |
| `download_assets.sh` | SFT checkpoint, DAPO-Math-17K, and AIME 2024, 2025, and 2026 |
| `convert_qwen3_4b_sft.sh` | Hugging Face weights to Megatron `torch_dist` |
| `run-qwen3-4B-retool-grpo-inflight.sh` | GRPO |
| `run-qwen3-4B-retool-lisao-inflight.sh` | LiSAO, with skip-observation GAE |
| `run-qwen3-4B-retool-li2sao-inflight.sh` | Li2SAO, with skip-observation GAE |
| `eval-qwen3-4B-aime.sh` | One checkpoint, all three years |
| `eval-qwen3-4B-aime-all.sh` | SFT baseline plus every saved iteration |

Algorithm settings match the single-turn scripts. The reward is the same binary math grader. Responses in the last quarter of the 32768 token window receive the DAPO overlong penalty.

```bash
bash /root/LiSAO/scripts/setup_env.sh
bash /root/LiSAO/examples/retool/download_assets.sh
bash /root/LiSAO/examples/retool/convert_qwen3_4b_sft.sh
bash /root/LiSAO/examples/retool/run-qwen3-4B-retool-grpo-inflight.sh
bash /root/LiSAO/examples/retool/run-qwen3-4B-retool-lisao-inflight.sh
bash /root/LiSAO/examples/retool/run-qwen3-4B-retool-li2sao-inflight.sh

LOAD=/root/data/ckpts/qwen3-4B-retool-li2sao-inflight \
  bash /root/LiSAO/examples/retool/eval-qwen3-4B-aime-all.sh
```

The default layout is 4 training GPUs and 4 rollout GPUs. Training does not evaluate.
