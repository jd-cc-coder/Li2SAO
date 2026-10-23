# In-flight asynchronous rollout

When a weight update interrupts decoding, the generated prefix is kept and the request continues under the new policy for the remaining length. One trajectory can contain tokens from more than one policy version. The token ratio uses the log probability stored for the version that sampled the token.

Finished groups whose oldest policy is more than `--max-policy-lag` versions behind the trainer are dropped. The example scripts use 16.

If the user already set a custom generate function, such as ReTool, that function is wrapped rather than replaced. The wrapper records each segment and resumes an unfinished tool turn.

Li2SAO changes the group size only after the prefetched bootstrap job has been consumed. Leftover bootstrap groups are discarded, so the next batch contains only singleton responses.

Training scripts are in [`examples/inflight_async`](../../../examples/inflight_async/README.md), [`examples/lisao`](../../../examples/lisao/README.md), and [`examples/retool`](../../../examples/retool/README.md).
