import logging

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args, uses_inprocess_critic
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.utils.misc import should_run_periodic_action

logger = logging.getLogger(__name__)


def _li2sao_controller(args):
    if not getattr(args, "use_li2sao", False):
        return None
    from lisao.sao.li2sao import Li2SAOController

    return Li2SAOController.from_args(args)


def _commit_li2sao_sampling(args, rollout_manager, pending, actor_model=None):
    if pending is None:
        return
    from lisao.inflight_async.switch import resolve_inflight_switch
    from lisao.sao.li2sao import apply_sampling_to_args, sync_li2sao_loss_fields

    n_samples, rollout_batch_size, reason = pending
    mode = resolve_inflight_switch(args)
    logger.info(
        "Li2SAO apply sampling n=%s batch=%s reason=%s inflight_switch=%s",
        n_samples,
        rollout_batch_size,
        reason,
        mode,
    )
    ray.get(
        rollout_manager.set_li2sao_sampling.remote(
            n_samples,
            rollout_batch_size,
            drain_inflight=(mode == "drain"),
            inflight_switch=mode,
        )
    )
    apply_sampling_to_args(args, n_samples, rollout_batch_size)
    sync_li2sao_loss_fields(args, actor_model)


def train(args):
    configure_logger()
    release_train = args.release_train

    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout and not release_train:
        ray.get(rollout_manager.onload_weights.remote())

    # Always push actor weights to rollout once weights are loaded.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train(actor_trains_this_step):
        # Each model auto-offloads after train() when offload_train is set,
        # so we only need clear_memory for the non-offload case.
        if not args.offload_train:
            if not args.use_critic or actor_trains_this_step or uses_inprocess_critic(args):
                actor_model.clear_memory()
            else:
                critic_model.clear_memory()

    # train loop.
    controller = _li2sao_controller(args)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if release_train:
            actor_model.create()

        actor_trains = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if uses_inprocess_critic(args):
            train_results = ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
        elif args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains:
                train_results = ray.get(
                    actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs)
                )
            else:
                train_results = ray.get(value_refs)
        else:
            train_results = ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
        if controller is not None:
            from lisao.sao.li2sao import collect_li2sao_metrics, log_li2sao_controller

            pending = controller.observe(collect_li2sao_metrics(train_results))
            log_li2sao_controller(args, rollout_id, controller, pending[2] if pending else None)
            _commit_li2sao_sampling(args, rollout_manager, pending, actor_model)

        if release_train or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            force_sync = release_train or rollout_id == args.num_rollout - 1
            if actor_trains:
                actor_model.save_model(rollout_id, force_sync=force_sync)
            if args.use_critic and not uses_inprocess_critic(args):
                critic_model.save_model(rollout_id, force_sync=force_sync)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        offload_train(actor_trains)
        if args.offload_rollout and not release_train:
            ray.get(rollout_manager.onload_weights.remote())
        # Critic-only warmup does not change the policy; skip the SGLang sync.
        if actor_trains or release_train:
            actor_model.update_weights()

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
