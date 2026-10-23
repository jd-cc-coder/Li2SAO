# LiSAO and Li2SAO

LiSAO is single-rollout asynchronous optimization without a pretrained value model. The critic is a second copy of the policy, with a scalar value head. Attention parameters whose names contain `self_attention` stay frozen. Training updates the MLP blocks and the value head. The actor is held fixed for 10 rollout steps, then each batch applies one policy update and two critic updates. The critic learning rate is 5e-6.

Advantages use length-adaptive GAE with alpha 1.5, so lambda is `1 - 1 / (1.5 L)` and is clamped to the unit interval. The discount is 1. The critic target uses lambda 1. DIS drops tokens whose importance ratio falls outside the interval from 0.7 to 6.0. The policy clip is 0.3 and 5.0. The entropy coefficient is 0.

On ReTool, GAE is computed only on model tokens. Pass `--use-skip-obs-gae` for that task. Single-turn responses have no observation tokens, so the same estimator is ordinary GAE.

Li2SAO starts the actor and critic together. While the group size is 8, actor advantages are the group-relative outcome rewards, and the critic fits group-centered returns plus a ranking loss with coefficient 0.1 and margin 0.1. Value clip is 0.2 during this bootstrap and inactive after the switch. The readiness gate uses residual explained variance above 0.2 and ranking accuracy above 0.65, smoothed with beta 0.9, for 10 consecutive steps after step 60. Step 150 forces the switch. The in-flight switch finishes the prefetched job, drops leftover bootstrap groups, and then samples one response per prompt. After that, the actor uses GAE and the critic fits absolute returns. Clip and DIS stay at the LiSAO values for the whole run.

The example scripts place the critic in the actor process with `--fast_mode`.

Launch scripts are in [`examples/lisao`](../../../examples/lisao/README.md) and [`examples/retool`](../../../examples/retool/README.md).
