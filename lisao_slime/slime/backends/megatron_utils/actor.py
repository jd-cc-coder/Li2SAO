import logging
import os
from argparse import Namespace
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from pathlib import Path

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from torch_memory_saver import torch_memory_saver
from transformers import AutoConfig, AutoTokenizer

from slime.ray.train_actor import TrainRayActor
from slime.utils import train_dump_utils
from slime.utils.data import process_rollout_data
from slime.utils.distributed_utils import get_gloo_group
from slime.utils.logging_utils import init_tracking
from slime.utils.arguments import uses_inprocess_critic
from slime.utils.memory_utils import clear_memory, print_memory
from slime.utils.misc import Box
from slime.utils.reloadable_process_group import (
    destroy_process_groups,
    monkey_patch_torch_dist,
    register_default_process_group,
    reload_process_groups,
)
from slime.utils.routing_replay import RoutingReplay
from slime.utils.timer import Timer, inverse_timer, timer, with_defer
from slime.utils.types import RolloutBatch

from ...utils.profile_utils import TrainProfiler
from ...utils.tensor_backper import TensorBackuper
from .checkpoint import load_checkpoint
from .cp_utils import prepare_routed_experts_for_routing_replay, slice_log_prob_with_cp
from .data import DataIterator, get_data_iterator, log_perf_data, log_rollout_data
from .hf_checkpoint_saver import save_hf_model_to_path
from .initialize import init, is_megatron_main_rank
from .loss import compute_advantages_and_returns, get_log_probs_and_entropy, get_values
from .model import forward_only, initialize_model_and_optimizer, save, train
from .update_weight.common import named_params_and_buffers
from .update_weight.update_weight_from_disk import UpdateWeightFromDisk
from .update_weight.update_weight_from_distributed import UpdateWeightFromDistributed
from .update_weight.update_weight_from_tensor import UpdateWeightFromTensor

logging.getLogger("megatron").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@contextmanager
def _use_megatron_args(args: Namespace):
    from megatron.training.global_vars import get_args as megatron_get_args
    from megatron.training.global_vars import set_args as megatron_set_args

    previous = megatron_get_args()
    megatron_set_args(args)
    try:
        yield
    finally:
        megatron_set_args(previous)


class MegatronTrainRayActor(TrainRayActor):
    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
    ) -> int | None:
        if args.debug_rollout_only:
            self.args = args
            return 0

        monkey_patch_torch_dist()
        super().init(args, role, with_ref, with_opd_teacher)
        # Destroying and recreating WORLD invalidates raw dist.group.WORLD references cached by external code.
        # Set SLIME_DESTROY_WORLD_PROCESS_GROUP=0 when such references may outlive a train sleep/wake cycle.
        if os.getenv("SLIME_DESTROY_WORLD_PROCESS_GROUP", "1").lower() not in {"0", "false", "no"}:
            register_default_process_group(timeout=timedelta(minutes=args.distributed_timeout_minutes))
        else:
            logger.info("Default WORLD process-group destruction is disabled")

        init(args)

        if is_megatron_main_rank():
            init_tracking(args, primary=False, role=role)

        self.prof = TrainProfiler(args)

        # read config and tokenizer serialized to prevent concurrent writing bug.
        for i in range(args.num_gpus_per_node):
            if i == dist.get_rank() % args.num_gpus_per_node:
                self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

        dist.barrier(group=get_gloo_group())

        if args.offload_train:
            if (x := args.train_memory_margin_bytes) > 0:
                logger.info(f"Set torch_memory_saver.memory_margin_bytes to {x}")
                torch_memory_saver.memory_margin_bytes = x

        self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id = initialize_model_and_optimizer(
            args, role
        )
        self.actor_model = self.model
        self.actor_optimizer = self.optimizer
        self.actor_opt_param_scheduler = self.opt_param_scheduler
        self.critic_model = None
        self.critic_optimizer = None
        self.critic_opt_param_scheduler = None
        self.critic_args = None
        if uses_inprocess_critic(args) and role == "actor":
            self._init_inprocess_critic(args)
            self._bind_train_modules("actor")

        vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
        if vpp_size > 1:
            from megatron.core.utils import get_model_config

            microbatch_group_size_per_vp_stage = get_model_config(self.model[0]).microbatch_group_size_per_vp_stage
        else:
            microbatch_group_size_per_vp_stage = 1
        self.train_parallel_config = {
            "dp_size": mpu.get_data_parallel_world_size(with_context_parallel=False),
            "cp_size": mpu.get_context_parallel_world_size(),
            "vpp_size": vpp_size,
            "microbatch_group_size_per_vp_stage": microbatch_group_size_per_vp_stage,
        }

        start_rollout_id = loaded_rollout_id + 1

        if role == "critic":
            if self.args.offload_train:
                self.sleep()
            return start_rollout_id

        self.weights_backuper = TensorBackuper.create(
            source_getter=lambda: self._actor_sync_named_params(),
            single_tag=None,
        )
        self._active_model_tag: str | None = "actor"
        self.weights_backuper.backup("actor")

        if with_ref:
            self.load_other_checkpoint("ref", args.ref_load)

        # Load teacher model for Megatron-based on-policy distillation
        if with_opd_teacher:
            self.load_other_checkpoint("teacher", args.opd_teacher_load)

        if self.args.keep_old_actor:
            # Load old_actor checkpoint
            self.load_other_checkpoint("old_actor", args.load)
            # Create rollout_actor as a copy of current actor
            if args.update_weights_interval == 1:
                self.weights_backuper.backup("rollout_actor")

        if self.args.vocab_size is None:
            # Prefer HF config vocab_size (which may include model-native padding)
            # over tokenizer vocab_size (which may be smaller, e.g. GPT-OSS).
            hf_vocab = getattr(self.hf_config, "vocab_size", None)
            self.args.vocab_size = hf_vocab if hf_vocab is not None else self.tokenizer.vocab_size

        update_weight_mode = self.args.update_weight_mode
        update_weight_transport = self.args.update_weight_transport

        if update_weight_mode == "delta":
            # Delta sync is disk-transport only: each engine's /pull_weights applies the published
            # deltas into a host-local checkpoint on every host it spans, and the engines reload
            # via vanilla update_weights_from_disk.
            assert not self.args.colocate, "--update-weight-mode=delta is not supported with --colocate"
            assert (
                update_weight_transport == "disk"
            ), "--update-weight-mode=delta requires --update-weight-transport=disk"
            from .update_weight.update_weight_from_disk_delta import UpdateWeightFromDiskDelta

            update_weight_cls = UpdateWeightFromDiskDelta
        elif update_weight_transport == "disk":
            update_weight_cls = UpdateWeightFromDisk
        elif self.args.colocate:
            update_weight_cls = UpdateWeightFromTensor
        else:
            assert update_weight_mode == "full"
            assert (
                update_weight_transport == "nccl"
            ), f"unsupported weight sync mode/transport: {update_weight_mode!r}/{update_weight_transport!r}"
            update_weight_cls = UpdateWeightFromDistributed
        self.weight_updater = update_weight_cls(
            self.args,
            self.model,
            weights_getter=lambda: self.weights_backuper.get("actor"),
            model_name=type(self.hf_config).__name__.lower() if self.args.model_name is None else self.args.model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
        )
        self.weight_updater.weight_version = getattr(self.args, "update_weight_start_version", 0)

        # empty cache after initialization
        clear_memory()

        if self.args.offload_train:
            # recover to actor in the end.
            self._switch_model("actor")
            self.sleep()

        self.rollout_engines = None

        self.rollout_data_postprocess = None
        if self.args.rollout_data_postprocess_path is not None:
            from slime.utils.misc import load_function

            self.rollout_data_postprocess = load_function(self.args.rollout_data_postprocess_path)

        self.prof.on_init_end()

        return start_rollout_id

    def _init_inprocess_critic(self, args: Namespace) -> None:
        import copy

        from slime.utils.arguments import parse_megatron_role_args

        if args.megatron_config_path is not None:
            critic_args = parse_megatron_role_args(args, args.megatron_config_path, role="critic")
        else:
            critic_args = copy.deepcopy(args)
            critic_args.untie_embeddings_and_output_weights = True
        self.critic_args = critic_args
        with _use_megatron_args(critic_args):
            (
                self.critic_model,
                self.critic_optimizer,
                self.critic_opt_param_scheduler,
                _,
            ) = initialize_model_and_optimizer(critic_args, "critic")
        if getattr(args, "lora_light_mode", False):
            logger.info("lora_light_mode: in-process critic replica with MLP LoRA (no offload)")
        else:
            logger.info("fast_mode: in-process critic replica, freeze-attention MLP, no offload")

    def _bind_train_modules(self, role: str) -> None:
        if role == "critic":
            assert self.critic_model is not None
            self.model = self.critic_model
            self.optimizer = self.critic_optimizer
            self.opt_param_scheduler = self.critic_opt_param_scheduler
            return
        self.model = self.actor_model
        self.optimizer = self.actor_optimizer
        self.opt_param_scheduler = self.actor_opt_param_scheduler

    @timer
    def sleep(self) -> None:
        assert self.args.offload_train

        clear_memory(clear_host_memory=True)
        print_memory("before offload model")
        if (
            self.role == "actor"
            and self.args.use_critic
            and not self.args.colocate
            and hasattr(self.weight_updater, "disconnect_rollout_engines")
        ):
            self.weight_updater.disconnect_rollout_engines()
        destroy_process_groups()

        torch_memory_saver.pause()

        print_memory("after offload model")

    @timer
    def wake_up(self) -> None:
        assert self.args.offload_train
        print_memory("before wake_up model")

        torch_memory_saver.resume()

        clear_memory()
        reload_process_groups()
        if self.role == "actor":
            self._switch_model("actor")
        print_memory("after wake_up model")

    def _get_rollout_data(self, rollout_data_ref: Box) -> RolloutBatch:
        # Fetch data through ray on CPU, not sure if this will be performance bottleneck.
        # Both first pp stage and the last pp stage will receive the data.
        rollout_data = process_rollout_data(
            self.args,
            rollout_data_ref,
            mpu.get_data_parallel_rank(with_context_parallel=False),
            mpu.get_data_parallel_world_size(with_context_parallel=False),
        )
        # TODO: this is ugly, move to somewhere else?
        # move tokens to GPU in advance
        device = torch.cuda.current_device()
        rollout_data["tokens"] = [
            t.to(device=device, dtype=torch.long, non_blocking=True) for t in rollout_data["tokens"]
        ]
        rollout_data["loss_masks"] = [
            t.to(device=device, dtype=torch.int, non_blocking=True) for t in rollout_data["loss_masks"]
        ]
        if "rollout_mask_sums" in rollout_data:
            # Promote precomputed per-rollout mask totals to GPU tensors here
            # (matching loss_masks) so the loss reducer can just divide.
            rollout_data["rollout_mask_sums"] = rollout_data["rollout_mask_sums"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
        if "multimodal_train_inputs" in rollout_data:
            # Move multimodal training tensors to GPU in advance
            rollout_data["multimodal_train_inputs"] = [
                (
                    {
                        key: value.to(device=device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                        for key, value in mm_dict.items()
                    }
                    if mm_dict is not None
                    else None
                )
                for mm_dict in rollout_data["multimodal_train_inputs"]
            ]

        for key in ["rollout_log_probs", "teacher_log_probs"]:
            if key not in rollout_data:
                continue
            rollout_data[key] = [
                slice_log_prob_with_cp(log_prob, total_length, response_length).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                for log_prob, total_length, response_length in zip(
                    rollout_data[key],
                    rollout_data["total_lengths"],
                    rollout_data["response_lengths"],
                    strict=False,
                )
            ]
        return rollout_data

    def _switch_model(self, target_tag: str) -> None:
        if target_tag not in self.weights_backuper.backup_tags:
            raise ValueError(f"Cannot switch to unknown model tag: {target_tag}")
        self.weights_backuper.restore(target_tag)
        self._active_model_tag = target_tag

    def fill_routing_replay(self, data_iterator, num_microbatches, rollout_data):
        if "rollout_routed_experts" not in rollout_data:
            raise ValueError(
                "rollout_routed_experts is required in rollout_data when use_rollout_routing_replay is set."
            )

        from megatron.core.transformer.transformer_block import get_num_layers_to_build
        from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

        from slime.utils.routing_replay import RoutingReplay

        for iterator in data_iterator:
            iterator.reset()

        for _ in range(sum(num_microbatches)):
            batch = data_iterator[0].get_next(["rollout_routed_experts", "tokens"])
            rollout_routed_experts = prepare_routed_experts_for_routing_replay(
                batch["rollout_routed_experts"],
                batch["tokens"],
                num_experts=self.args.num_experts,
                data_pad_size_multiplier=self.args.data_pad_size_multiplier,
                sequence_parallel=self.args.sequence_parallel,
                allgather_cp=self.args.allgather_cp,
            )

            routing_replay_offset = 0
            for vp_stage, model in enumerate(self.model):
                config = model.module.config
                num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
                offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
                for layer_id in range(offset, offset + num_layers_to_build):
                    # skip dense layer
                    if isinstance(config.moe_layer_freq, int):
                        if layer_id % config.moe_layer_freq != 0:
                            continue
                    elif isinstance(config.moe_layer_freq, list):
                        assert len(config.moe_layer_freq) == config.num_layers
                        if config.moe_layer_freq[layer_id] == 0:
                            continue
                    layer_routed_experts = rollout_routed_experts[:, layer_id]
                    RoutingReplay.all_routing_replays[routing_replay_offset].record(layer_routed_experts)
                    routing_replay_offset += 1
            assert routing_replay_offset == len(RoutingReplay.all_routing_replays)

        del rollout_data["rollout_routed_experts"]

        for iterator in data_iterator:
            iterator.reset()

    def compute_log_prob(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:

        with timer(f"{store_prefix}log_probs"):
            return forward_only(
                get_log_probs_and_entropy,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
                use_rollout_top_p_replay=True,
            )

    def _actor_sync_named_params(self):
        return named_params_and_buffers(
            self.args,
            self.model,
            convert_to_global_name=self.args.megatron_to_hf_mode == "raw",
        )

    def train(self, rollout_id: int, rollout_data_ref: Box, external_data=None):
        if self.args.debug_rollout_only:
            return None

        if self.args.offload_train:
            self.wake_up()

        with timer("data_preprocess"):
            rollout_data = self._get_rollout_data(rollout_data_ref)

        if self.role == "actor":
            self._log_inflight_train_metrics(rollout_id, rollout_data)

        if self.role == "critic":
            result = self.train_critic(rollout_id, rollout_data)
        elif uses_inprocess_critic(self.args):
            result = self.train_inprocess(rollout_id, rollout_data)
        else:
            self.train_actor(rollout_id, rollout_data, external_data=external_data)
            result = None

        if self.args.offload_train:
            del rollout_data
            self.sleep()

        li2sao_metrics = getattr(self.args, "_lisao_li2sao_metrics", None)
        if li2sao_metrics is not None:
            if result is None:
                result = {"li2sao": li2sao_metrics}
            elif isinstance(result, dict):
                result = dict(result)
                result["li2sao"] = li2sao_metrics
        return result

    def _log_inflight_train_metrics(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        """Log policy lag vs the actor/policy ``weight_version`` (not critic)."""
        if self.role != "actor":
            return
        if not rollout_data.get("inflight_weight_versions"):
            return
        try:
            from lisao.inflight_async.metrics import log_inflight_train_metrics
        except ImportError:
            return
        current = getattr(self.weight_updater, "weight_version", None)
        log_inflight_train_metrics(self.args, rollout_id, rollout_data, current)

    def train_critic(self, rollout_id: int, rollout_data: RolloutBatch):
        """Train critic and return CPU values (used as old-values for the next actor train)."""
        from lisao.sao.li2sao import apply_li2sao_loss_mode_from_batch

        apply_li2sao_loss_mode_from_batch(self.args, rollout_data)
        data_iterator = get_data_iterator(rollout_data)
        num_microbatches = rollout_data["num_microbatches"]
        global_batch_sizes = rollout_data["global_batch_sizes"]

        # Compute current critic values once (old_values for value clip and for actor GAE).
        rollout_data.update(forward_only(get_values, self.args, self.model, data_iterator, num_microbatches))

        compute_advantages_and_returns(self.args, rollout_data)
        try:
            from lisao.sao.metrics import log_critic_explained_variance

            log_critic_explained_variance(self.args, rollout_id, rollout_data)
        except ImportError:
            pass
        try:
            from lisao.sao.li2sao import log_li2sao_train_metrics

            log_li2sao_train_metrics(self.args, rollout_id)
        except ImportError:
            pass

        self.args.loss_type = "value_loss"
        critic_train_epoch = max(1, int(getattr(self.args, "critic_train_epoch", 1)))
        for _ in range(critic_train_epoch):
            train(
                rollout_id,
                self.model,
                self.optimizer,
                self.opt_param_scheduler,
                data_iterator,
                num_microbatches,
                global_batch_sizes,
            )

        if mpu.is_pipeline_last_stage() and "values" in rollout_data:
            from slime.backends.megatron_utils.data import tensors_to_cpu

            return {"values": tensors_to_cpu(rollout_data["values"])}
        return {}

    def train_inprocess(self, rollout_id: int, rollout_data: RolloutBatch):
        """Train the in-process critic replica, then the actor (no offload).

        Used by ``--fast_mode`` and ``--lora_light_mode``. Critic-only warmup
        (``num_critic_only_steps``) skips the policy update and the SGLang
        sync. ``critic_train_epoch`` is K value-loss passes.
        """
        from lisao.sao.lora import is_critic_only_warmup

        prev_args = self.args
        prev_loss_type = self.args.loss_type
        critic_only = is_critic_only_warmup(self.args, rollout_id)
        mode = "lora_light_mode" if getattr(self.args, "lora_light_mode", False) else "fast_mode"
        try:
            self._bind_train_modules("critic")
            self.args = self.critic_args
            with _use_megatron_args(self.critic_args):
                result = self.train_critic(rollout_id, rollout_data)
            if critic_only:
                logger.info(
                    "%s critic-only warmup rollout %s (num_critic_only_steps=%s): skip actor",
                    mode,
                    rollout_id,
                    getattr(prev_args, "num_critic_only_steps", 0),
                )
                return result

            self._bind_train_modules("actor")
            self.args = prev_args
            self.args.loss_type = prev_loss_type or "policy_loss"
            self.train_actor(rollout_id, rollout_data, external_data=result)
            return None
        finally:
            self.args = prev_args
            self.args.loss_type = prev_loss_type
            self._bind_train_modules("actor")

    def train_actor(self, rollout_id: int, rollout_data: RolloutBatch, external_data=None) -> None:
        try:
            from lisao.sao.li2sao import apply_li2sao_loss_mode_from_batch

            apply_li2sao_loss_mode_from_batch(self.args, rollout_data)
        except ImportError:
            pass
        # Create data iterator for log_probs and train.
        data_iterator = get_data_iterator(rollout_data)
        num_microbatches = rollout_data["num_microbatches"]
        global_batch_sizes = rollout_data["global_batch_sizes"]

        if self.args.use_rollout_routing_replay:
            self.fill_routing_replay(data_iterator, num_microbatches, rollout_data)

        with inverse_timer("train_wait"), timer("train"):
            if self.args.compute_advantages_and_returns:
                if "ref" in self.weights_backuper.backup_tags:
                    if self.args.use_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                    self._switch_model("ref")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="ref_",
                        )
                    )

                # Forward teacher model to get teacher_log_probs for Megatron-based OPD
                if "teacher" in self.weights_backuper.backup_tags:
                    if self.args.use_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                    self._switch_model("teacher")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="teacher_",
                        )
                    )

                self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
                can_reuse_log_probs_in_loss = (
                    len(num_microbatches) == 1
                    and self.args.loss_type == "policy_loss"
                    and self.args.kl_coef == 0
                    and not self.args.use_rollout_logprobs
                    and not self.args.get_mismatch_metrics
                    and not self.args.use_critic
                    and not self.args.keep_old_actor
                    and not self.args.use_opd
                    and (not self.args.use_routing_replay or self.args.use_rollout_routing_replay)
                    and self.args.advantage_estimator != "gspo"
                )
                if (
                    not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics
                ) and not can_reuse_log_probs_in_loss:
                    if self.args.use_routing_replay:
                        if self.args.use_rollout_routing_replay:
                            os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"
                        else:
                            os.environ["ROUTING_REPLAY_STAGE"] = "record"
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="",
                        )
                    )
                    if self.args.use_rollout_routing_replay:
                        RoutingReplay.clear_all_forward()

                if self.args.use_critic:
                    if external_data is not None and mpu.is_pipeline_last_stage():
                        values = external_data.get("values")
                        if values is not None:
                            from slime.backends.megatron_utils.data import tensors_to_gpu

                            rollout_data["values"] = tensors_to_gpu(values)
                if self._active_model_tag != "actor":
                    self._switch_model("actor")

                # Calculate adv and returns. Need to performed before training (instead of on the fly),
                # because we may need normalize the whole rollout.
                compute_advantages_and_returns(self.args, rollout_data)
                try:
                    from lisao.sao.li2sao import log_li2sao_train_metrics

                    log_li2sao_train_metrics(self.args, rollout_id)
                except ImportError:
                    pass

            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(self.args, rollout_id, rollout_data)

            log_rollout_data(
                rollout_id,
                self.args,
                rollout_data,
            )

            # Train
            if self.args.use_routing_replay:
                os.environ["ROUTING_REPLAY_STAGE"] = "replay_backward"
            with timer("actor_train"):
                train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    data_iterator,
                    num_microbatches,
                    global_batch_sizes,
                )

            self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        if self.args.use_routing_replay:
            RoutingReplay.clear_all()

        # update the cpu actor weight to the latest model
        self.weights_backuper.backup("actor")

        # Update ref model if needed
        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and "ref" in self.weights_backuper.backup_tags
        ):
            with timer("ref_model_update"):
                if is_megatron_main_rank():
                    logger.info(f"Updating ref model at rollout_id {rollout_id}")
                self.weights_backuper.backup("ref")

        log_perf_data(rollout_id, self.args, extra_metrics=self.weight_updater.pop_metrics())

    @timer
    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only:
            return

        if uses_inprocess_critic(self.args) and getattr(self, "actor_model", None) is not None:
            self._bind_train_modules("actor")

        # torch dist may trigger nccl communication during saving.
        if self.args.offload_train:
            self.wake_up()

        if self.args.async_save:
            from megatron.training.async_utils import maybe_finalize_async_save

            maybe_finalize_async_save(blocking=True)

        save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)

        if uses_inprocess_critic(self.args) and self.critic_model is not None:
            with _use_megatron_args(self.critic_args):
                save(
                    rollout_id,
                    self.critic_model,
                    self.critic_optimizer,
                    self.critic_opt_param_scheduler,
                )

        if force_sync and self.args.async_save:
            maybe_finalize_async_save(blocking=True)

        if self.args.save_hf is not None and self.role == "actor":
            save_hf_model_to_path(self.args, Path(self.args.save_hf.format(rollout_id=rollout_id)), self.model)

        if self.args.offload_train:
            self.sleep()

    @timer
    def update_weights(self) -> None:
        if self.role != "actor":
            logger.info("skip update_weights for role=%s (only actor/policy is synced to rollout)", self.role)
            return
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.args.use_fault_tolerance:
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.recover_updatable_engines.remote())
            dist.barrier(group=get_gloo_group())

        (
            rollout_engines,
            rollout_engine_lock,
            num_new_engines,
            engine_gpu_counts,
            engine_gpu_offsets,
            engine_parallel_configs,
        ) = ray.get(self.rollout_manager.get_updatable_engines_and_lock.remote())

        reconnect_rollout_engines = self.args.offload_train and self.args.use_critic and not self.args.colocate

        if not rollout_engines and not reconnect_rollout_engines:
            if dist.get_rank() == 0:
                logger.info("No updatable SGLang engines are running; skip weight update.")
            return

        if reconnect_rollout_engines:
            self.wake_up()
        elif self.args.offload_train:
            reload_process_groups()

        if num_new_engines > 0 or reconnect_rollout_engines:
            self.weight_updater.connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
                engine_parallel_configs=engine_parallel_configs,
            )
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_updatable_num_new_engines.remote())

        with torch_memory_saver.disable() if self.args.offload_train else nullcontext():
            print_memory("before update_weights")
            # Tell In-Flight Async clients (RolloutManager process) to wait
            # before resubmitting aborted generations, matching AReaL PauseState.
            if dist.get_rank() == 0:
                try:
                    ray.get(self.rollout_manager.set_inflight_weight_update_paused.remote(True))
                except Exception as e:  # noqa: BLE001
                    logger.debug("set_inflight_weight_update_paused(True) skipped: %s", e)
            try:
                self.weight_updater.update_weights()
            finally:
                if dist.get_rank() == 0:
                    try:
                        ray.get(self.rollout_manager.set_inflight_weight_update_paused.remote(False))
                    except Exception as e:  # noqa: BLE001
                        logger.debug("set_inflight_weight_update_paused(False) skipped: %s", e)
            print_memory("after update_weights")

            if getattr(self.args, "keep_old_actor", False):
                if self.args.update_weights_interval == 1:
                    logger.info("updating model queue: rollout_actor -> old_actor, actor -> rollout_actor")
                    # Queue-style update: rollout_actor params -> old_actor, actor params -> rollout_actor
                    # First copy rollout_actor to old_actor
                    self.weights_backuper.copy(src_tag="rollout_actor", dst_tag="old_actor")
                    # Then copy current actor to rollout_actor
                    self.weights_backuper.backup("rollout_actor")
                else:
                    self.weights_backuper.backup("old_actor")

        if reconnect_rollout_engines:
            self.sleep()
        elif self.args.offload_train:
            destroy_process_groups()

    def load_other_checkpoint(self, model_tag: str, path: str) -> None:
        old_args = self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune
        self.args.load = path
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True

        old_ckpt_step = None
        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.ref_ckpt_step
        elif model_tag == "teacher" and self.args.opd_teacher_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.opd_teacher_ckpt_step

        _, _ = load_checkpoint(
            self.model,
            None,
            None,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
        self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune = old_args

        if old_ckpt_step is not None:
            self.args.ckpt_step = old_ckpt_step

        self.weights_backuper.backup(model_tag)
        self._active_model_tag = model_tag
