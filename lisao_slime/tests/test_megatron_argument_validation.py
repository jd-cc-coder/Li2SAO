import importlib.util
import sys
import types
from pathlib import Path

import pytest

NUM_GPUS = 0


def load_arguments_module(monkeypatch):
    megatron_mod = types.ModuleType("megatron")
    training_mod = types.ModuleType("megatron.training")
    arguments_mod = types.ModuleType("megatron.training.arguments")
    tokenizer_pkg_mod = types.ModuleType("megatron.training.tokenizer")
    tokenizer_mod = types.ModuleType("megatron.training.tokenizer.tokenizer")
    transformers_mod = types.ModuleType("transformers")

    arguments_mod.parse_args = lambda *args, **kwargs: None
    arguments_mod.validate_args = lambda args: args
    tokenizer_mod._vocab_size_with_padding = lambda vocab_size, _args: vocab_size
    transformers_mod.AutoConfig = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: None)

    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.training", training_mod)
    monkeypatch.setitem(sys.modules, "megatron.training.arguments", arguments_mod)
    monkeypatch.setitem(sys.modules, "megatron.training.tokenizer", tokenizer_pkg_mod)
    monkeypatch.setitem(sys.modules, "megatron.training.tokenizer.tokenizer", tokenizer_mod)
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    module_path = Path(__file__).resolve().parents[1] / "slime" / "backends" / "megatron_utils" / "arguments.py"
    module_name = "test_megatron_argument_validation_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_slime_arguments_module(monkeypatch):
    router_pkg_mod = types.ModuleType("sglang_router")
    router_launch_mod = types.ModuleType("sglang_router.launch_router")
    sglang_arguments_mod = types.ModuleType("slime.backends.sglang_utils.arguments")
    sglang_external_mod = types.ModuleType("slime.backends.sglang_utils.external")
    logging_utils_mod = types.ModuleType("slime.utils.logging_utils")

    router_launch_mod.RouterArgs = object
    sglang_arguments_mod.sglang_parse_args = lambda *args, **kwargs: None
    sglang_arguments_mod.validate_args = lambda args: args
    sglang_external_mod.apply_external_engine_info_to_args = lambda *args, **kwargs: None
    logging_utils_mod.configure_logger = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "sglang_router", router_pkg_mod)
    monkeypatch.setitem(sys.modules, "sglang_router.launch_router", router_launch_mod)
    monkeypatch.setitem(sys.modules, "slime.backends.sglang_utils.arguments", sglang_arguments_mod)
    monkeypatch.setitem(sys.modules, "slime.backends.sglang_utils.external", sglang_external_mod)
    monkeypatch.setitem(sys.modules, "slime.utils.logging_utils", logging_utils_mod)

    module_path = Path(__file__).resolve().parents[1] / "slime" / "utils" / "arguments.py"
    module_name = "test_slime_argument_validation_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_qwen3_6_args(**overrides):
    values = dict(
        hidden_size=2048,
        num_attention_heads=16,
        num_layers=40,
        ffn_hidden_size=512,
        moe_ffn_hidden_size=512,
        moe_shared_expert_intermediate_size=512,
        moe_layer_freq=[1] * 40,
        untie_embeddings_and_output_weights=True,
        norm_epsilon=1e-6,
        layernorm_epsilon=1e-6,
        rotary_base=10000000,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def make_qwen3_6_hf_config():
    text_config = types.SimpleNamespace(
        hidden_size=2048,
        num_attention_heads=16,
        num_hidden_layers=40,
        intermediate_size=5632,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts=256,
        tie_word_embeddings=False,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_theta": 10000000},
    )
    return types.SimpleNamespace(text_config=text_config)


def make_allgather_cp_args(**overrides):
    values = dict(
        allgather_cp=True,
        context_parallel_size=2,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


@pytest.mark.unit
def test_hf_validate_all_moe_skips_dense_intermediate_size(monkeypatch):
    module = load_arguments_module(monkeypatch)

    module._hf_validate_args(make_qwen3_6_args(), make_qwen3_6_hf_config())


@pytest.mark.unit
def test_hf_validate_checks_moe_intermediate_size(monkeypatch):
    module = load_arguments_module(monkeypatch)

    with pytest.raises(AssertionError, match="moe_intermediate_size"):
        module._hf_validate_args(make_qwen3_6_args(moe_ffn_hidden_size=256), make_qwen3_6_hf_config())


@pytest.mark.unit
def test_hf_validate_checks_dense_intermediate_size_when_moe_has_dense_layers(monkeypatch):
    module = load_arguments_module(monkeypatch)

    args = make_qwen3_6_args(moe_layer_freq=[0] + [1] * 39)

    with pytest.raises(AssertionError, match="intermediate_size"):
        module._hf_validate_args(args, make_qwen3_6_hf_config())


@pytest.mark.unit
def test_allgather_cp_rejects_non_dsa_cp_models(monkeypatch):
    module = load_arguments_module(monkeypatch)
    args = make_allgather_cp_args()
    hf_config = types.SimpleNamespace(architectures=["Qwen3ForCausalLM"], model_type="qwen3")

    with pytest.raises(ValueError, match="only supported for DSA attention models"):
        module._validate_allgather_cp_supported(args, hf_config)


@pytest.mark.unit
@pytest.mark.parametrize(
    "hf_config",
    [
        types.SimpleNamespace(architectures=["DeepseekV32ForCausalLM"], model_type="deepseek_v3"),
        types.SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"], model_type="glm"),
    ],
)
def test_allgather_cp_allows_dsa_architectures(monkeypatch, hf_config):
    module = load_arguments_module(monkeypatch)

    module._validate_allgather_cp_supported(make_allgather_cp_args(), hf_config)


@pytest.mark.unit
def test_allgather_cp_ignores_cp_size_one(monkeypatch):
    module = load_arguments_module(monkeypatch)
    args = make_allgather_cp_args(context_parallel_size=1)

    module._validate_allgather_cp_supported(args)


@pytest.mark.unit
def test_update_weight_disk_dir_required_for_disk_transport(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(update_weight_transport="disk", update_weight_disk_dir=None)

    with pytest.raises(ValueError, match="update-weight-disk-dir"):
        module.slime_validate_args(args)


def make_slime_validate_args(**overrides):
    values = dict(
        eval_config=None,
        eval_prompt_data=None,
        kl_coef=0,
        use_kl_loss=False,
        ref_load=None,
        use_opd=False,
        opd_type=None,
        opd_teacher_load=None,
        megatron_to_hf_mode="raw",
        load=None,
        hf_checkpoint="/tmp/hf",
        ref_ckpt_step=None,
        ckpt_step=None,
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
        override_opt_param_scheduler=False,
        start_rollout_id=None,
        eval_interval=None,
        save_interval=None,
        save=None,
        kl_loss_coef=0,
        advantage_estimator="grpo",
        normalize_advantages=False,
        use_rollout_logprobs=False,
        use_tis=False,
        get_mismatch_metrics=False,
        custom_tis_function_path=None,
        use_dynamic_batch_size=False,
        max_tokens_per_gpu=None,
        log_probs_max_tokens_per_gpu=None,
        balance_by_flops=False,
        balance_data=False,
        eps_clip_high=None,
        eps_clip=0.2,
        eval_reward_key=None,
        reward_key="reward",
        dump_details=None,
        save_debug_rollout_data=None,
        save_debug_train_data=None,
        load_debug_rollout_data=None,
        rollout_external_engine_addrs=None,
        debug_train_only=False,
        actor_num_gpus_per_node=8,
        actor_num_nodes=1,
        offload=False,
        offload_train=None,
        offload_rollout=None,
        debug_rollout_only=False,
        colocate=False,
        rollout_num_gpus=8,
        train_memory_margin_bytes=0,
        eval_function_path=None,
        rollout_function_path="custom.rollout",
        num_steps_per_rollout=None,
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        global_batch_size=None,
        grpo_std_normalization=True,
        over_sampling_batch_size=None,
        num_epoch=None,
        num_rollout=1,
        rollout_global_dataset=False,
        enable_mtp_training=False,
        mtp_num_layers=None,
        use_rollout_routing_replay=False,
        use_routing_replay=False,
        custom_config_path=None,
        eval_max_context_len=None,
        rollout_max_context_len=None,
        rollout_max_prompt_len=None,
        train_backend="megatron",
        release_train=False,
        keep_old_actor=False,
        only_train_params_name_list=None,
        freeze_params_name_list=None,
        update_weight_transport="nccl",
        update_weight_disk_dir=None,
        update_weight_local_checkpoint_dir=None,
        update_weight_mode="full",
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


@pytest.mark.unit
def test_slime_validate_args_preserves_zero_rollout_gpus_under_colocate(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(colocate=True, rollout_num_gpus=0)

    module.slime_validate_args(args)

    assert args.rollout_num_gpus == 0
    assert args.offload_train is True
    assert args.offload_rollout is True


@pytest.mark.unit
def test_slime_validate_args_preserves_larger_rollout_gpus_under_colocate(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        colocate=True,
        actor_num_gpus_per_node=8,
        actor_num_nodes=1,
        rollout_num_gpus=12,
    )

    module.slime_validate_args(args)

    assert args.rollout_num_gpus == 12
    assert args.offload_train is True
    assert args.offload_rollout is True


@pytest.mark.unit
def test_slime_validate_args_preserves_zero_rollout_gpus_without_colocate(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(colocate=False, rollout_num_gpus=0)

    module.slime_validate_args(args)

    assert args.rollout_num_gpus == 0
    assert args.actor_num_gpus_per_node == 8
    assert args.actor_num_nodes == 1
    assert args.offload_train is False
    assert args.offload_rollout is False


def _write_megatron_ckpt(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "latest_checkpointed_iteration.txt").write_text("499")
    return ckpt


@pytest.mark.unit
def test_eval_only_skips_optimizer_restore_for_training_ckpt(monkeypatch, tmp_path):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        num_rollout=0,
        load=str(_write_megatron_ckpt(tmp_path)),
        megatron_to_hf_mode="raw",
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
        override_opt_param_scheduler=False,
    )

    module.slime_validate_args(args)

    assert args.no_load_optim is True
    assert args.no_load_rng is True
    assert args.finetune is True
    assert args.override_opt_param_scheduler is True


@pytest.mark.unit
def test_train_resume_keeps_optimizer_restore_for_training_ckpt(monkeypatch, tmp_path):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        num_rollout=500,
        load=str(_write_megatron_ckpt(tmp_path)),
        megatron_to_hf_mode="raw",
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
        override_opt_param_scheduler=False,
    )

    module.slime_validate_args(args)

    assert args.no_load_optim is False
    assert args.no_load_rng is False
    assert args.finetune is False
    assert args.override_opt_param_scheduler is False


@pytest.mark.unit
def test_update_weight_delta_requires_disk_transport(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        update_weight_mode="delta",
        update_weight_transport="nccl",
        update_weight_local_checkpoint_dir="/local/ckpt",
    )

    with pytest.raises(ValueError, match="requires --update-weight-transport=disk"):
        module.slime_validate_args(args)


@pytest.mark.unit
def test_update_weight_delta_rejects_colocate(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        update_weight_mode="delta",
        update_weight_transport="disk",
        update_weight_disk_dir="/shared/delta",
        update_weight_local_checkpoint_dir="/local/ckpt",
        colocate=True,
    )

    with pytest.raises(ValueError, match="not supported with --colocate"):
        module.slime_validate_args(args)


@pytest.mark.unit
def test_update_weight_delta_requires_local_checkpoint_dir(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        update_weight_mode="delta",
        update_weight_transport="disk",
        update_weight_disk_dir="/shared/delta",
        update_weight_local_checkpoint_dir=None,
    )

    with pytest.raises(ValueError, match="requires --update-weight-local-checkpoint-dir"):
        module.slime_validate_args(args)


@pytest.mark.unit
def test_use_dis_enables_tis_and_rollout_logprobs(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(use_dis=True, use_tis=False, use_rollout_logprobs=False)

    module.slime_validate_args(args)

    assert args.use_tis is True
    assert args.use_rollout_logprobs is True
    assert args.custom_tis_function_path == "lisao.sao.dis.dis_tis_function"


@pytest.mark.unit
def test_use_tis_and_rollout_logprobs_remain_mutex_without_dis(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(use_tis=True, use_rollout_logprobs=True)

    with pytest.raises(AssertionError, match="use_rollout_logprobs and use_tis"):
        module.slime_validate_args(args)


@pytest.mark.unit
def test_critic_train_epoch_must_be_positive(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(critic_train_epoch=0)

    with pytest.raises(ValueError, match="critic-train-epoch"):
        module.slime_validate_args(args)


@pytest.mark.unit
def test_lora_light_mode_requires_ppo(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(lora_light_mode=True, advantage_estimator="grpo")

    with pytest.raises(ValueError, match="lora_light_mode"):
        module.slime_validate_args(args)


@pytest.mark.unit
def test_fast_mode_requires_ppo(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(fast_mode=True, advantage_estimator="grpo")

    with pytest.raises(ValueError, match="fast_mode"):
        module.slime_validate_args(args)


@pytest.mark.unit
def test_lora_light_mode_skips_offload_and_keeps_critic_schedule(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        lora_light_mode=True,
        advantage_estimator="ppo",
        critic_train_epoch=2,
        num_critic_only_steps=10,
        offload_train=None,
    )

    module.slime_validate_args(args)

    assert args.use_critic is True
    assert args.offload_train is False
    assert args.critic_train_epoch == 2
    assert args.num_critic_only_steps == 10


@pytest.mark.unit
def test_fast_mode_skips_offload_and_keeps_critic_schedule(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        fast_mode=True,
        advantage_estimator="ppo",
        critic_train_epoch=2,
        num_critic_only_steps=10,
        offload_train=None,
    )

    module.slime_validate_args(args)

    assert args.use_critic is True
    assert args.offload_train is False
    assert args.critic_train_epoch == 2
    assert args.num_critic_only_steps == 10
    assert getattr(args, "lora_light_mode", False) is False


@pytest.mark.unit
def test_ppo_without_lora_light_mode_still_offloads_train(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(advantage_estimator="ppo", offload_train=None)

    module.slime_validate_args(args)

    assert args.use_critic is True
    assert args.offload_train is True


@pytest.mark.unit
def test_li2sao_validate_forces_boost_sampling_and_no_critic_warmup(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        use_li2sao=True,
        advantage_estimator="ppo",
        fast_mode=True,
        n_samples_per_prompt=1,
        rollout_batch_size=128,
        global_batch_size=128,
        num_critic_only_steps=10,
        li2sao_n_boost=8,
        li2sao_n_stable=1,
        li2sao_traj_per_step=128,
        li2sao_n_boost_min_steps=60,
        li2sao_n_boost_max_steps=150,
        offload_train=None,
    )

    module.slime_validate_args(args)

    assert args.n_samples_per_prompt == 8
    assert args.rollout_batch_size == 16
    assert args.num_critic_only_steps == 0
    assert args.global_batch_size == 128
    assert args.over_sampling_batch_size == 16


@pytest.mark.unit
def test_li2sao_requires_ppo(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        use_li2sao=True,
        advantage_estimator="grpo",
        li2sao_n_boost=8,
        li2sao_n_stable=1,
        li2sao_traj_per_step=128,
        li2sao_n_boost_min_steps=60,
        li2sao_n_boost_max_steps=150,
    )

    with pytest.raises(ValueError, match="li2sao"):
        module.slime_validate_args(args)


@pytest.mark.unit
def test_uses_inprocess_critic_helper(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    assert module.uses_inprocess_critic(types.SimpleNamespace(fast_mode=True, lora_light_mode=False))
    assert module.uses_inprocess_critic(types.SimpleNamespace(fast_mode=False, lora_light_mode=True))
    assert not module.uses_inprocess_critic(types.SimpleNamespace(fast_mode=False, lora_light_mode=False))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
