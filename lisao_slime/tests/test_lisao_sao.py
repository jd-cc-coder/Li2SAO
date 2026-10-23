"""CPU tests for LiSAO DIS, skip-obs GAE, length-adaptive λ, and overlong shaping."""

from types import SimpleNamespace

import pytest
import torch

from lisao.sao.dis import dis_tis_function
from lisao.sao.gae import compute_per_sample_lambd, skip_obs_gae
from lisao.sao.metrics import explained_variance
from lisao.sao.overlong import overlong_penalty, post_process_rewards
from slime.utils.ppo_utils import vanilla_gae

NUM_GPUS = 0


class _Args:
    def __init__(self, tis_clip_low=0.7, tis_clip=6.0):
        self.tis_clip_low = tis_clip_low
        self.tis_clip = tis_clip


def test_dis_masks_tokens_outside_trust_region():
    args = _Args()
    train_lp = [torch.tensor([-0.5, -1.0, -1.0, -1.0, -0.5, -0.5])]
    rollout_lp = [torch.tensor([-0.5, -2.5, -0.31, 0.00, -0.5, -0.5])]
    ratio = torch.exp(train_lp[0] - rollout_lp[0])
    pg_loss = -ratio
    out_pg, masks, metrics = dis_tis_function(
        args,
        pg_loss=pg_loss.clone(),
        train_log_probs=train_lp,
        rollout_log_probs=rollout_lp,
        loss_masks=[torch.ones(6)],
    )
    assert masks[0].shape == (6,)
    assert out_pg[2].item() == 0.0 and out_pg[3].item() == 0.0
    assert out_pg[0].item() != 0.0 and out_pg[1].item() != 0.0
    expected_mask = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    torch.testing.assert_close(metrics["tis_clipfrac"], 1.0 - expected_mask)
    torch.testing.assert_close(out_pg, pg_loss * expected_mask)


def test_dis_does_not_multiply_ratio_again():
    args = _Args(tis_clip_low=0.5, tis_clip=2.0)
    train_lp = [torch.zeros(2)]
    rollout_lp = [torch.zeros(2)]
    pg_loss = torch.tensor([-1.5, -2.0])
    out_pg, _, _ = dis_tis_function(
        args,
        pg_loss=pg_loss.clone(),
        train_log_probs=train_lp,
        rollout_log_probs=rollout_lp,
        loss_masks=[torch.ones(2)],
    )
    torch.testing.assert_close(out_pg, pg_loss)


def test_per_sample_lambd_formula():
    lengths = [10, 100, 1000]
    lambds = compute_per_sample_lambd(lengths, alpha=1.5)
    expected = torch.tensor([1 - 1 / (1.5 * L) for L in lengths])
    torch.testing.assert_close(lambds, expected, rtol=1e-6, atol=1e-6)


def test_vanilla_gae_accepts_per_row_lambda():
    rewards = torch.zeros(2, 4)
    rewards[:, -1] = 1.0
    values = torch.linspace(0.1, 0.4, 4).unsqueeze(0).repeat(2, 1)
    lambd = torch.tensor([0.5, 0.99])
    adv, _ = vanilla_gae(rewards, values, gamma=1.0, lambd=lambd)
    assert not torch.allclose(adv[0], adv[1])
    adv_lo, _ = vanilla_gae(rewards[:1], values[:1], gamma=1.0, lambd=0.5)
    torch.testing.assert_close(adv[0], adv_lo[0])


def test_skip_obs_gae_zeros_observation_tokens():
    loss_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]])
    values = torch.tensor([[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]])
    rewards = torch.zeros(1, 8)
    rewards[0, 7] = 1.0
    adv_skip, _ = skip_obs_gae(rewards, values, loss_mask, gamma=1.0, lambd=0.9)
    obs = (loss_mask[0] == 0).nonzero(as_tuple=True)[0]
    assert torch.all(adv_skip[0, obs] == 0)
    adv_van, _ = vanilla_gae(rewards, values, gamma=1.0, lambd=0.9)
    action = (loss_mask[0] == 1).nonzero(as_tuple=True)[0]
    assert not torch.allclose(adv_skip[0, action], adv_van[0, action])


def test_skip_obs_gae_matches_vanilla_when_all_action():
    rewards = torch.zeros(2, 5)
    rewards[:, -1] = 1.0
    values = torch.randn(2, 5)
    masks = torch.ones(2, 5)
    adv_skip, ret_skip = skip_obs_gae(rewards, values, masks, gamma=0.99, lambd=0.95)
    adv_van, ret_van = vanilla_gae(rewards, values, gamma=0.99, lambd=0.95)
    torch.testing.assert_close(adv_skip, adv_van, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ret_skip, ret_van, rtol=1e-5, atol=1e-5)


def test_chunked_gae_falls_back_for_per_row_lambda():
    from slime.utils.ppo_utils import chunked_gae

    rewards = torch.zeros(2, 8)
    rewards[:, -1] = 1.0
    values = torch.linspace(0.1, 0.8, 8).unsqueeze(0).repeat(2, 1)
    lambd = torch.tensor([0.4, 0.95])
    adv_c, _ = chunked_gae(rewards, values, gamma=1.0, lambd=lambd)
    adv_v, _ = vanilla_gae(rewards, values, gamma=1.0, lambd=lambd)
    torch.testing.assert_close(adv_c, adv_v, rtol=1e-5, atol=1e-5)


def test_explained_variance_perfect_fit_is_one():
    returns = torch.tensor([1.0, 2.0, 3.0, 4.0])
    ev = explained_variance(returns, returns)
    assert ev.item() == pytest.approx(1.0)


def test_explained_variance_constant_returns_is_zero():
    returns = torch.ones(4)
    values = torch.tensor([0.0, 1.0, 2.0, 3.0])
    ev = explained_variance(returns, values)
    assert ev.item() == pytest.approx(0.0)


def test_masked_pairs_and_half_split():
    from lisao.sao.metrics import masked_return_value_pairs, split_pairs_in_half

    returns = torch.tensor([0.0, 1.0, 2.0, 3.0])
    values = torch.tensor([0.0, 1.0, 2.0, 3.0])
    mask = torch.tensor([1.0, 0.0, 1.0, 1.0])
    rr, vv = masked_return_value_pairs(returns, values, mask)
    torch.testing.assert_close(rr, torch.tensor([0.0, 2.0, 3.0]))
    (pre_r, _), (suf_r, _) = split_pairs_in_half(rr, vv)
    torch.testing.assert_close(pre_r, torch.tensor([0.0]))
    torch.testing.assert_close(suf_r, torch.tensor([2.0, 3.0]))


def test_overlong_penalty_zero_inside_buffer():
    assert overlong_penalty(1000, max_response_len=16384, buffer_len=4096, penalty_factor=1.0) == 0.0


def test_overlong_penalty_linear_in_buffer():
    # halfway through the last 4096 tokens of a 16384 cap → -0.5
    penalty = overlong_penalty(16384 - 2048, max_response_len=16384, buffer_len=4096, penalty_factor=1.0)
    assert penalty == pytest.approx(-0.5)


def test_overlong_penalty_full_at_cap_and_truncated():
    at_cap = overlong_penalty(16384, max_response_len=16384, buffer_len=4096, penalty_factor=1.0)
    assert at_cap == pytest.approx(-1.0)
    truncated_early = overlong_penalty(
        8000, max_response_len=16384, buffer_len=4096, penalty_factor=1.0, truncated=True
    )
    assert truncated_early == pytest.approx(-1.0)


def test_overlong_post_process_shapes_reward(monkeypatch):
    monkeypatch.setenv("OVERLONG_BUFFER_LEN", "4096")
    monkeypatch.setenv("OVERLONG_PENALTY_FACTOR", "1.0")
    args = SimpleNamespace(rollout_max_response_len=16384, reward_key=None)

    class _Sample:
        def __init__(self, reward, response_length, status="completed"):
            self.reward = reward
            self.response_length = response_length
            self.status = status

        def get_reward_value(self, _args):
            return self.reward

    samples = [
        _Sample(1.0, 1000),
        _Sample(1.0, 16384 - 2048),
        _Sample(1.0, 8000, status="truncated"),
    ]
    raw, shaped = post_process_rewards(args, samples)
    assert raw == [1.0, 1.0, 1.0]
    assert shaped[0] == pytest.approx(1.0)
    assert shaped[1] == pytest.approx(0.5)
    assert shaped[2] == pytest.approx(0.0)


class _Block(torch.nn.Module):
    def __init__(self, hidden=4):
        super().__init__()
        self.linear_qkv = torch.nn.Linear(hidden, hidden, bias=False)
        self.linear_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.linear_fc1 = torch.nn.Linear(hidden, hidden * 2, bias=False)
        self.linear_fc2 = torch.nn.Linear(hidden * 2, hidden, bias=False)


class _ToyGPT(torch.nn.Module):
    def __init__(self, n_layers=2, hidden=4):
        super().__init__()
        self.decoder = torch.nn.Module()
        self.decoder.layers = torch.nn.ModuleList([_Block(hidden) for _ in range(n_layers)])
        self.output_layer = torch.nn.Linear(hidden, 8, bias=False)


def test_lora_default_targets_are_mlp_only():
    from lisao.sao.lora import inject_lora, parse_lora_target_modules

    assert parse_lora_target_modules(None) == ("fc1", "fc2")
    model = _ToyGPT()
    injected = inject_lora(model, lora_rank=2, lora_alpha=4.0, num_layers=2)
    assert injected
    assert all("linear_fc1" in name or "linear_fc2" in name for name in injected)
    assert not any("linear_qkv" in name or "linear_proj" in name for name in injected)


def test_lora_disabled_matches_base_forward_and_enable_adds_delta():
    from lisao.sao.lora import disable_lora, enable_lora, inject_lora, is_lora_enabled

    torch.manual_seed(0)
    model = _ToyGPT()
    x = torch.randn(2, 3, 4)
    base = model.decoder.layers[0].linear_fc1(x).detach()
    injected = inject_lora(model, lora_rank=2, lora_alpha=4.0, num_layers=2)
    assert injected
    assert any("linear_fc1" in name for name in injected)
    with torch.no_grad():
        model.decoder.layers[0].linear_fc1.lora_B.normal_()
    disable_lora()
    assert is_lora_enabled() is False
    torch.testing.assert_close(model.decoder.layers[0].linear_fc1(x), base)
    enable_lora()
    delta = model.decoder.layers[0].linear_fc1(x)
    assert not torch.allclose(delta, base)
    disable_lora()


def test_lora_layers_restricts_to_last_n():
    from lisao.sao.lora import inject_lora

    model = _ToyGPT(n_layers=4)
    injected = inject_lora(model, lora_rank=2, lora_alpha=4.0, lora_layers=1, num_layers=4)
    assert injected
    assert all(".layers.3." in name for name in injected)


def test_lora_and_critic_param_name_filters():
    from lisao.sao.lora import (
        filter_policy_sync_params,
        freeze_non_lora_critic_params,
        inject_lora,
        is_critic_trainable_param,
        is_lora_or_critic_param,
        is_policy_sync_param,
    )

    crash_name = "module.module.decoder.layers.0.mlp.linear_fc2.lora_A"
    assert is_lora_or_critic_param("decoder.layers.0.linear_fc1.lora_A")
    assert is_lora_or_critic_param(crash_name)
    assert is_lora_or_critic_param("decoder.layers.0.linear_fc2.lora_B")
    assert is_critic_trainable_param("output_layer.weight")
    assert is_policy_sync_param("decoder.layers.0.linear_qkv.weight")
    assert is_policy_sync_param("output_layer.weight")
    assert not is_policy_sync_param(crash_name)
    named = [
        ("decoder.layers.0.linear_fc1.weight", torch.zeros(1)),
        ("decoder.layers.0.linear_fc1.lora_A", torch.zeros(1)),
        ("output_layer.weight", torch.zeros(1)),
    ]
    synced = [n for n, _ in filter_policy_sync_params(named)]
    assert synced == ["decoder.layers.0.linear_fc1.weight", "output_layer.weight"]

    model = _ToyGPT()
    inject_lora(model, lora_rank=2, lora_alpha=4.0, num_layers=2)
    freeze_non_lora_critic_params(model)
    for name, param in model.named_parameters():
        if "lora_" in name or "output_layer" in name:
            assert param.requires_grad, name
        else:
            assert not param.requires_grad, name


def _install_fake_megatron_tp(monkeypatch):
    """Stub Megatron TP collectives: gather doubles seq dim 0, reduce-scatter halves it."""
    import sys
    from types import ModuleType

    def gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=True, group=None):
        del tensor_parallel_output_grad, group
        return torch.cat([x, x], dim=0)

    def reduce_scatter_to_sequence_parallel_region(x, group=None):
        del group
        return x.chunk(2, dim=0)[0].contiguous()

    def reduce_from_tensor_model_parallel_region(x, group=None):
        del group
        return x

    tp = ModuleType("megatron.core.tensor_parallel")
    tp.gather_from_sequence_parallel_region = gather_from_sequence_parallel_region
    tp.reduce_scatter_to_sequence_parallel_region = reduce_scatter_to_sequence_parallel_region
    tp.reduce_from_tensor_model_parallel_region = reduce_from_tensor_model_parallel_region
    core = ModuleType("megatron.core")
    core.tensor_parallel = tp
    megatron = ModuleType("megatron")
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.tensor_parallel", tp)


class _ColumnParallelLike(torch.nn.Module):
    """Mimic ColumnParallelLinear+SP: GEMM runs on the gathered sequence."""

    def __init__(self, hidden=8):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden, hidden))
        self.sequence_parallel = True
        self.parallel_mode = "column"

    def forward(self, x):
        gathered = torch.cat([x, x], dim=0)
        return (torch.nn.functional.linear(gathered, self.weight), None)


class _RowParallelLike(torch.nn.Module):
    """Mimic RowParallelLinear+SP: output is reduce-scattered along sequence."""

    def __init__(self, hidden=8):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden, hidden))
        self.sequence_parallel = True
        self.input_is_parallel = True
        self.parallel_mode = "row"

    def forward(self, x):
        full = torch.nn.functional.linear(x, self.weight)
        return (full.chunk(2, dim=0)[0].contiguous(), None)


def test_lora_column_parallel_sequence_parallel_matches_gathered_seq(monkeypatch):
    from lisao.sao.lora import disable_lora, enable_lora, inject_lora

    _install_fake_megatron_tp(monkeypatch)
    torch.manual_seed(0)
    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.decoder.layers[0].linear_qkv = _ColumnParallelLike()
    x = torch.randn(4, 8)
    base, _ = model.decoder.layers[0].linear_qkv(x)
    assert base.shape[0] == 8
    inject_lora(model, lora_rank=2, lora_alpha=4.0, lora_target_modules="qkv", num_layers=1)
    with torch.no_grad():
        model.decoder.layers[0].linear_qkv.lora_B.normal_()
    disable_lora()
    out_off, _ = model.decoder.layers[0].linear_qkv(x)
    torch.testing.assert_close(out_off, base)
    enable_lora()
    out_on, _ = model.decoder.layers[0].linear_qkv(x)
    assert out_on.shape == base.shape
    assert not torch.allclose(out_on, base)
    disable_lora()


def test_lora_row_parallel_sequence_parallel_matches_scattered_seq(monkeypatch):
    from lisao.sao.lora import disable_lora, enable_lora, inject_lora

    _install_fake_megatron_tp(monkeypatch)
    torch.manual_seed(0)
    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.decoder.layers[0].linear_proj = _RowParallelLike()
    x = torch.randn(8, 8)
    base, _ = model.decoder.layers[0].linear_proj(x)
    assert base.shape[0] == 4
    inject_lora(model, lora_rank=2, lora_alpha=4.0, lora_target_modules="proj", num_layers=1)
    assert model.decoder.layers[0].linear_proj.lora_B.sequence_parallel is True
    with torch.no_grad():
        model.decoder.layers[0].linear_proj.lora_B.normal_()
    disable_lora()
    out_off, _ = model.decoder.layers[0].linear_proj(x)
    torch.testing.assert_close(out_off, base)
    enable_lora()
    out_on, _ = model.decoder.layers[0].linear_proj(x)
    assert out_on.shape == base.shape
    assert not torch.allclose(out_on, base)
    disable_lora()


def test_lora_te_row_parallel_detected_via_parallel_mode(monkeypatch):
    from lisao.sao.lora import _is_row_parallel, disable_lora, enable_lora, inject_lora

    _install_fake_megatron_tp(monkeypatch)

    class _TERowLike(_RowParallelLike):
        def __init__(self, hidden=8):
            super().__init__(hidden)
            del self.input_is_parallel

    module = _TERowLike()
    assert _is_row_parallel(module)
    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.decoder.layers[0].linear_fc2 = module
    x = torch.randn(8, 8)
    base, _ = module(x)
    inject_lora(model, lora_rank=2, lora_alpha=4.0, lora_target_modules="fc2", num_layers=1)
    with torch.no_grad():
        model.decoder.layers[0].linear_fc2.lora_B.normal_()
    enable_lora()
    out, _ = model.decoder.layers[0].linear_fc2(x)
    assert out.shape == base.shape
    disable_lora()


def test_maybe_attach_lora_critic_only_on_critic_role():
    from argparse import Namespace

    from lisao.sao.lora import maybe_attach_lora_critic

    actor = _ToyGPT()
    maybe_attach_lora_critic(
        actor,
        Namespace(lora_light_mode=True, lora_rank=2, lora_alpha=4.0, lora_target_modules=None, lora_layers=-1),
        role="actor",
        post_process=True,
    )
    assert not any(hasattr(m, "lora_A") for m in actor.modules())

    critic = _ToyGPT()
    maybe_attach_lora_critic(
        critic,
        Namespace(
            lora_light_mode=True, lora_rank=2, lora_alpha=4.0, lora_target_modules=None, lora_layers=-1, num_layers=2
        ),
        role="critic",
        post_process=True,
    )
    assert any(hasattr(m, "lora_A") for m in critic.modules())
    assert critic.decoder.layers[0].linear_fc1.lora_A is not None
    assert not hasattr(critic.decoder.layers[0].linear_qkv, "lora_A")
    assert not critic.decoder.layers[0].linear_fc1.weight.requires_grad
    assert critic.decoder.layers[0].linear_fc1.lora_A.requires_grad
    assert critic.output_layer.weight.requires_grad


def test_lora_critic_only_warmup_gate():
    from lisao.sao.lora import is_critic_only_warmup

    args = SimpleNamespace(num_critic_only_steps=10)
    assert is_critic_only_warmup(args, 0) is True
    assert is_critic_only_warmup(args, 9) is True
    assert is_critic_only_warmup(args, 10) is False
    assert is_critic_only_warmup(SimpleNamespace(num_critic_only_steps=0), 0) is False
    assert is_critic_only_warmup(None, 0) is False


def _grpo_pair_adv(correct: bool) -> float:
    """GRPO A for a 0/1 pair (mean 0.5), matching slime unbiased std + 1e-6."""
    from lisao.sao.li2sao import GRPO_STD_EPS

    std = float(torch.tensor([1.0, 0.0]).std(unbiased=True).item())
    centered = 0.5 if correct else -0.5
    return centered / (std + GRPO_STD_EPS)


def test_li2sao_group_relative_replaces_gae_and_zeros_identical():
    from lisao.sao.li2sao import apply_group_relative_actor_advantages

    advantages = [torch.ones(3) * 2.0, torch.ones(3) * 2.0, torch.ones(3), torch.ones(3) * 0.5]
    n_grpo, mixed, mean_abs = apply_group_relative_actor_advantages(
        advantages,
        group_ids=[0, 0, 1, 2],
        raw_rewards=[1.0, 0.0, 1.0, 1.0],
    )
    pos, neg = _grpo_pair_adv(True), _grpo_pair_adv(False)
    assert n_grpo == 4
    assert mixed == pytest.approx(1.0)
    assert mean_abs == pytest.approx((abs(pos) + abs(neg)) / 4)
    torch.testing.assert_close(advantages[0], torch.ones(3) * pos)
    torch.testing.assert_close(advantages[1], torch.ones(3) * neg)
    torch.testing.assert_close(advantages[2], torch.zeros(3))
    torch.testing.assert_close(advantages[3], torch.zeros(3))


def test_li2sao_group_relative_uses_partition_and_zeros_obs():
    from lisao.sao.li2sao import apply_group_relative_actor_advantages

    advantages = [torch.ones(4) * 9.0]
    n_grpo, _, _ = apply_group_relative_actor_advantages(
        advantages,
        group_ids=[7, 7],
        raw_rewards=[1.0, 0.0],
        partition=[1],
        loss_masks=[torch.tensor([1.0, 1.0, 0.0, 0.0])],
    )
    assert n_grpo == 1
    neg = _grpo_pair_adv(False)
    torch.testing.assert_close(advantages[0], torch.tensor([neg, neg, 0.0, 0.0]))


def test_li2sao_last_action_value_skips_obs():
    from lisao.sao.li2sao import last_action_value

    values = torch.tensor([0.1, 0.2, 0.9, 0.3])
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    assert last_action_value(values, mask).item() == pytest.approx(0.2)
    assert last_action_value(values, None).item() == pytest.approx(0.3)


def test_li2sao_sequence_ev_perfect_and_constant():
    from lisao.sao.li2sao import sequence_explained_variance

    r = torch.tensor([0.0, 1.0, 0.0, 1.0])
    ev = sequence_explained_variance(r, r)
    assert ev is not None and ev.item() == pytest.approx(1.0)
    assert sequence_explained_variance(torch.ones(4), torch.zeros(4)) is None


def test_li2sao_controller_one_way_ev_and_timeout():
    from lisao.sao.li2sao import Li2SAOController

    ctl = Li2SAOController(
        ev_ema_beta=0.0,
        ev_threshold=0.2,
        ev_consecutive=2,
        boost_min_steps=3,
        boost_max_steps=6,
    )
    assert ctl.observe({"ev_seq": 0.9, "group_mixed_frac": 0.8}) is None
    assert ctl.observe({"ev_seq": 0.9, "group_mixed_frac": 0.8}) is None
    assert ctl.observe({"ev_seq": 0.9, "group_mixed_frac": 0.8}) is None
    switched = ctl.observe({"ev_seq": 0.9, "group_mixed_frac": 0.8})
    assert switched == (1, 128, "ev_ready")
    assert ctl.switched is True
    assert ctl.observe({"ev_seq": 0.0, "group_mixed_frac": 0.1}) is None

    timeout = Li2SAOController(boost_min_steps=2, boost_max_steps=2, ev_consecutive=99)
    assert timeout.observe({"ev_seq": 0.0, "group_mixed_frac": 1.0}) is None
    assert timeout.observe({"ev_seq": 0.0, "group_mixed_frac": 1.0}) == (1, 128, "timeout")


def test_li2sao_controller_ignores_low_mixed_frac():
    from lisao.sao.li2sao import Li2SAOController

    ctl = Li2SAOController(
        ev_ema_beta=0.0,
        ev_threshold=0.9,
        ev_consecutive=2,
        boost_min_steps=2,
        boost_max_steps=5,
    )
    for _ in range(4):
        assert ctl.observe({"ev_seq": 0.0, "group_mixed_frac": 0.0}) is None
    assert ctl.observe({"ev_seq": 0.0, "group_mixed_frac": 0.0}) == (1, 128, "timeout")


def test_li2sao_controller_seq_ignores_token_ev():
    from lisao.sao.li2sao import Li2SAOController

    ctl = Li2SAOController(
        ev_ema_beta=0.0,
        ev_threshold=0.2,
        ev_consecutive=2,
        boost_min_steps=3,
        boost_max_steps=4,
        switch_criterion="seq",
    )
    ready_token = {
        "ev_seq": 0.0,
        "ev_token": 0.9,
        "ev_prefix": 0.9,
        "ev_suffix": 0.9,
        "ev_residual": 0.9,
        "rank_acc": 1.0,
    }
    assert ctl.observe(ready_token) is None
    assert ctl.observe(ready_token) is None
    assert ctl.observe(ready_token) is None
    assert ctl.consecutive_ready == 0
    assert ctl.observe({"ev_seq": 0.0}) == (1, 128, "timeout")


def test_li2sao_controller_token_requires_all_halves():
    from lisao.sao.li2sao import Li2SAOController

    ctl = Li2SAOController(
        ev_ema_beta=0.0,
        ev_threshold=0.2,
        ev_consecutive=2,
        boost_min_steps=3,
        boost_max_steps=8,
        switch_criterion="token",
    )
    high_seq = {
        "ev_seq": 0.9,
        "ev_token": 0.9,
        "ev_prefix": 0.1,
        "ev_suffix": 0.9,
        "ev_residual": 0.9,
        "rank_acc": 1.0,
    }
    assert ctl.observe(high_seq) is None
    assert ctl.observe(high_seq) is None
    assert ctl.observe(high_seq) is None
    assert ctl.consecutive_ready == 0
    ready = {"ev_seq": 0.0, "ev_token": 0.9, "ev_prefix": 0.5, "ev_suffix": 0.6}
    assert ctl.observe(ready) is None
    switched = ctl.observe(ready)
    assert switched == (1, 128, "ev_ready")
    assert ctl.ev_ema == pytest.approx(0.9)
    assert ctl.ev_prefix_ema == pytest.approx(0.5)
    assert ctl.ev_suffix_ema == pytest.approx(0.6)


def test_li2sao_controller_residual_uses_ev_residual_not_seq():
    from lisao.sao.li2sao import Li2SAOController

    ctl = Li2SAOController(
        ev_ema_beta=0.0,
        ev_threshold=0.2,
        ev_consecutive=2,
        boost_min_steps=3,
        boost_max_steps=5,
        switch_criterion="residual",
    )
    high_seq = {"ev_seq": 0.9, "ev_residual": 0.0, "rank_acc": 1.0}
    assert ctl.observe(high_seq) is None
    assert ctl.observe(high_seq) is None
    assert ctl.observe(high_seq) is None
    assert ctl.consecutive_ready == 0
    ready = {"ev_seq": 0.0, "ev_residual": 0.8, "rank_acc": 0.1}
    assert ctl.observe(ready) is None
    switched = ctl.observe(ready)
    assert switched == (1, 128, "ev_ready")
    assert ctl.ev_ema == pytest.approx(0.8)


def test_li2sao_controller_residual_optional_rank_gate():
    from lisao.sao.li2sao import Li2SAOController

    ctl = Li2SAOController(
        ev_ema_beta=0.0,
        ev_threshold=0.2,
        ev_consecutive=2,
        boost_min_steps=3,
        boost_max_steps=8,
        switch_criterion="residual",
        rank_threshold=0.65,
    )
    high_ev_low_rank = {"ev_seq": 0.9, "ev_residual": 0.9, "rank_acc": 0.4}
    assert ctl.observe(high_ev_low_rank) is None
    assert ctl.observe(high_ev_low_rank) is None
    assert ctl.observe(high_ev_low_rank) is None
    assert ctl.consecutive_ready == 0
    ready = {"ev_residual": 0.9, "rank_acc": 0.8}
    assert ctl.observe(ready) is None
    switched = ctl.observe(ready)
    assert switched == (1, 128, "ev_ready")
    assert ctl.rank_ema == pytest.approx(0.8)


def test_mixed_group_switch_metrics_perfect_residual_and_rank():
    from lisao.sao.li2sao import mixed_group_switch_metrics

    last_values = [torch.tensor(0.5), torch.tensor(-0.5), torch.tensor(0.0), torch.tensor(0.0)]
    ev, rank, n_pairs = mixed_group_switch_metrics(
        last_values,
        group_ids=[0, 0, 1, 1],
        raw_rewards=[1.0, 0.0, 1.0, 1.0],
        partition=None,
    )
    assert ev == pytest.approx(1.0)
    assert rank == pytest.approx(1.0)
    assert n_pairs == 1


def test_mixed_group_switch_metrics_skips_collapsed_and_wrong_rank():
    from lisao.sao.li2sao import mixed_group_switch_metrics

    last_values = [torch.tensor(-0.4), torch.tensor(0.4)]
    ev, rank, n_pairs = mixed_group_switch_metrics(
        last_values,
        group_ids=[3, 3],
        raw_rewards=[1.0, 0.0],
        partition=None,
    )
    assert ev is not None
    assert ev < 0.0
    assert rank == pytest.approx(0.0)
    assert n_pairs == 1


def test_mixed_group_switch_metrics_drops_collapsed_groups():
    from lisao.sao.li2sao import mixed_group_switch_metrics

    ev, rank, n_pairs = mixed_group_switch_metrics(
        [torch.tensor(1.0), torch.tensor(1.0)],
        group_ids=[0, 0],
        raw_rewards=[1.0, 1.0],
        partition=None,
    )
    assert ev is None
    assert rank is None
    assert n_pairs == 0


def test_li2sao_controller_from_args_residual_rank():
    from lisao.sao.li2sao import Li2SAOController

    ctl = Li2SAOController.from_args(
        SimpleNamespace(
            li2sao_switch_criterion="residual",
            li2sao_switch_rank_threshold=0.7,
            li2sao_ev_threshold=0.2,
        )
    )
    assert ctl.switch_criterion == "residual"
    assert ctl.rank_threshold == pytest.approx(0.7)
    assert ctl.ev_threshold == pytest.approx(0.2)


def test_li2sao_apply_replaces_actor_adv_keeps_gae_returns():
    from lisao.sao.li2sao import apply_li2sao

    advantages = [torch.ones(3), torch.ones(3) * 2.0]
    returns = [torch.ones(3) * 4.0, torch.ones(3) * 5.0]
    values = [torch.zeros(3), torch.zeros(3)]
    args = SimpleNamespace(use_li2sao=True)
    rollout = {
        "advantages": advantages,
        "returns": returns,
        "values": values,
        "raw_reward": [1.0, 0.0],
        "group_indices": [3, 3],
        "loss_masks": [torch.ones(3), torch.ones(3)],
        "response_lengths": [3, 3],
        "total_lengths": [3, 3],
    }
    metrics = apply_li2sao(args, rollout)
    assert metrics is not None
    assert metrics["n_grpo"] == 2
    assert metrics["actor_mode"] == "grpo"
    pos, neg = _grpo_pair_adv(True), _grpo_pair_adv(False)
    torch.testing.assert_close(rollout["advantages"][0], torch.ones(3) * pos)
    torch.testing.assert_close(rollout["advantages"][1], torch.ones(3) * neg)
    torch.testing.assert_close(rollout["returns"][0], torch.ones(3) * 4.0)
    torch.testing.assert_close(rollout["returns"][1], torch.ones(3) * 5.0)
    assert metrics["ev_token"] is not None
    assert metrics["ev_prefix"] is not None
    assert metrics["ev_suffix"] is not None
    assert metrics["ev_residual"] == pytest.approx(0.0)
    assert metrics["rank_acc"] == pytest.approx(0.0)
    assert metrics["n_rank_pairs"] == 1


def test_li2sao_apply_replaces_when_dp_sharded_with_partition():
    from lisao.sao.li2sao import apply_li2sao

    advantages = [torch.ones(2) * 9.0, torch.ones(2) * 9.0]
    returns = [torch.ones(2), torch.ones(2) * 2.0]
    args = SimpleNamespace(use_li2sao=True)
    rollout = {
        "advantages": advantages,
        "returns": returns,
        "values": [torch.zeros(2), torch.zeros(2)],
        "raw_reward": [1.0, 0.0, 1.0, 0.0],
        "group_indices": [0, 0, 1, 1],
        "partition": [1, 2],
        "loss_masks": [torch.ones(2), torch.ones(2)],
        "response_lengths": [2, 2],
        "total_lengths": [2, 2],
    }
    metrics = apply_li2sao(args, rollout)
    assert metrics is not None
    assert metrics["actor_mode"] == "grpo"
    assert metrics["n_grpo"] == 2
    torch.testing.assert_close(rollout["advantages"][0], torch.ones(2) * _grpo_pair_adv(False))
    torch.testing.assert_close(rollout["advantages"][1], torch.ones(2) * _grpo_pair_adv(True))
    torch.testing.assert_close(rollout["returns"][0], torch.ones(2))
    torch.testing.assert_close(rollout["returns"][1], torch.ones(2) * 2.0)


def test_li2sao_apply_skips_replace_without_partition_when_dp_sharded():
    from lisao.sao.li2sao import apply_li2sao

    advantages = [torch.ones(2) * 9.0]
    args = SimpleNamespace(use_li2sao=True)
    rollout = {
        "advantages": advantages,
        "returns": [torch.ones(2)],
        "values": [torch.zeros(2)],
        "raw_reward": [1.0, 0.0, 1.0, 0.0],
        "group_indices": [0, 0, 1, 1],
        "loss_masks": [torch.ones(2)],
        "response_lengths": [2],
        "total_lengths": [2],
    }
    metrics = apply_li2sao(args, rollout)
    assert metrics is not None
    assert metrics["actor_mode"] == "gae"
    assert metrics["n_grpo"] == 0
    assert metrics["n_batch"] == 2
    torch.testing.assert_close(rollout["advantages"][0], torch.ones(2) * 9.0)


def test_process_rollout_data_keeps_partition(monkeypatch):
    from slime.utils.data import process_rollout_data

    class _Ref:
        def __init__(self, inner):
            self.inner = inner

    payload = {
        "partition": [1, 3],
        "total_lengths": [10, 20, 30, 40],
        "group_indices": [0, 0, 1, 1],
        "raw_reward": [1.0, 0.0, 1.0, 0.0],
    }
    monkeypatch.setattr("slime.utils.data.ray.get", lambda value: value)
    out = process_rollout_data(SimpleNamespace(), [_Ref(payload)], dp_rank=0, dp_size=1)
    assert out["partition"] == [1, 3]
    assert out["total_lengths"] == [20, 40]
    assert out["group_indices"] == [0, 0, 1, 1]
    assert out["raw_reward"] == [1.0, 0.0, 1.0, 0.0]


def test_li2sao_apply_reports_perfect_token_and_half_ev():
    from lisao.sao.li2sao import apply_li2sao

    returns = [torch.tensor([0.0, 1.0, 0.0, 1.0]), torch.tensor([1.0, 0.0, 1.0, 0.0])]
    args = SimpleNamespace(use_li2sao=True)
    rollout = {
        "advantages": [torch.ones(4), torch.ones(4)],
        "returns": returns,
        "values": [t.clone() for t in returns],
        "raw_reward": [1.0, 0.0],
        "group_indices": [1, 2],
        "loss_masks": [torch.ones(4), torch.ones(4)],
        "response_lengths": [4, 4],
        "total_lengths": [4, 4],
    }
    metrics = apply_li2sao(args, rollout)
    assert metrics is not None
    assert metrics["ev_token"] == pytest.approx(1.0)
    assert metrics["ev_prefix"] == pytest.approx(1.0)
    assert metrics["ev_suffix"] == pytest.approx(1.0)


def test_li2sao_apply_keeps_gae_when_n_is_one():
    from lisao.sao.li2sao import apply_li2sao

    advantages = [torch.ones(3) * 2.0]
    returns = [torch.ones(3) * 4.0]
    args = SimpleNamespace(use_li2sao=True)
    rollout = {
        "advantages": advantages,
        "returns": returns,
        "values": [torch.zeros(3)],
        "raw_reward": [1.0],
        "group_indices": [9],
        "loss_masks": [torch.ones(3)],
        "response_lengths": [3],
        "total_lengths": [3],
    }
    metrics = apply_li2sao(args, rollout)
    assert metrics is not None
    assert metrics["actor_mode"] == "gae"
    assert metrics["n_grpo"] == 0
    torch.testing.assert_close(rollout["advantages"][0], torch.ones(3) * 2.0)
    torch.testing.assert_close(rollout["returns"][0], torch.ones(3) * 4.0)


def test_li2sao_apply_keeps_gae_after_flatten_unique_group_ids():
    from lisao.sao.li2sao import apply_li2sao

    advantages = [torch.ones(2) * 3.0 for _ in range(8)]
    returns = [torch.ones(2) * 7.0 for _ in range(8)]
    args = SimpleNamespace(use_li2sao=True)
    rollout = {
        "advantages": advantages,
        "returns": returns,
        "values": [torch.zeros(2) for _ in range(8)],
        "raw_reward": [1.0, 0.0] * 4,
        "group_indices": list(range(50, 58)),
        "loss_masks": [torch.ones(2) for _ in range(8)],
        "response_lengths": [2] * 8,
        "total_lengths": [2] * 8,
    }
    metrics = apply_li2sao(args, rollout)
    assert metrics is not None
    assert metrics["actor_mode"] == "gae"
    assert metrics["n_batch"] == 1
    torch.testing.assert_close(rollout["advantages"][0], torch.ones(2) * 3.0)
    torch.testing.assert_close(rollout["returns"][0], torch.ones(2) * 7.0)


def test_gae_mix_alpha_clips_and_handles_missing():
    from lisao.sao.li2sao import gae_mix_alpha

    assert gae_mix_alpha(None, 0.2) == 0.0
    assert gae_mix_alpha(-1.0, 0.2) == 0.0
    assert gae_mix_alpha(0.0, 0.2) == 0.0
    assert gae_mix_alpha(0.1, 0.2) == pytest.approx(0.5)
    assert gae_mix_alpha(0.4, 0.2) == pytest.approx(1.0)
    assert gae_mix_alpha(0.1, 0.0) == 0.0


def test_mix_gae_grpo_advantage_zeros_obs_tokens():
    from lisao.sao.li2sao import mix_gae_grpo_advantage

    gae = torch.tensor([9.0, 9.0, 9.0])
    grpo = torch.tensor([1.0, 1.0, 1.0])
    mask = torch.tensor([0.0, 1.0, 1.0])
    mixed = mix_gae_grpo_advantage(gae, grpo, 0.5, mask)
    torch.testing.assert_close(mixed, torch.tensor([0.0, 5.0, 5.0]))


def test_li2sao_gae_mix_zero_ev_keeps_pure_grpo():
    from lisao.sao.li2sao import apply_li2sao

    gae = [torch.ones(3) * 9.0, torch.ones(3) * -9.0]
    args = SimpleNamespace(use_li2sao=True, li2sao_gae_mix=True, li2sao_ev_threshold=0.2)
    rollout = {
        "advantages": [a.clone() for a in gae],
        "returns": [torch.ones(3) * 4.0, torch.ones(3) * 5.0],
        "values": [torch.zeros(3), torch.zeros(3)],
        "raw_reward": [1.0, 0.0],
        "group_indices": [3, 3],
        "loss_masks": [torch.ones(3), torch.ones(3)],
        "response_lengths": [3, 3],
        "total_lengths": [3, 3],
    }
    metrics = apply_li2sao(args, rollout)
    assert metrics is not None
    assert metrics["actor_mode"] == "grpo"
    assert metrics["gae_mix_alpha"] == pytest.approx(0.0)
    pos, neg = _grpo_pair_adv(True), _grpo_pair_adv(False)
    torch.testing.assert_close(rollout["advantages"][0], torch.ones(3) * pos)
    torch.testing.assert_close(rollout["advantages"][1], torch.ones(3) * neg)


def test_li2sao_gae_mix_perfect_seq_ev_uses_gae():
    from lisao.sao.li2sao import apply_li2sao

    gae = [torch.ones(3) * 10.0, torch.ones(3) * -4.0]
    args = SimpleNamespace(use_li2sao=True, li2sao_gae_mix=True, li2sao_ev_threshold=0.2)
    rollout = {
        "advantages": [a.clone() for a in gae],
        "returns": [torch.ones(3), torch.zeros(3)],
        "values": [torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.0, 0.0, 0.0])],
        "raw_reward": [1.0, 0.0],
        "group_indices": [3, 3],
        "loss_masks": [torch.ones(3), torch.ones(3)],
        "response_lengths": [3, 3],
        "total_lengths": [3, 3],
    }
    metrics = apply_li2sao(args, rollout)
    assert metrics is not None
    assert metrics["ev_seq"] == pytest.approx(1.0)
    assert metrics["gae_mix_alpha"] == pytest.approx(1.0)
    assert metrics["actor_mode"] == "mix"
    torch.testing.assert_close(rollout["advantages"][0], gae[0])
    torch.testing.assert_close(rollout["advantages"][1], gae[1])


def test_li2sao_gae_mix_half_when_threshold_is_twice_ev():
    from lisao.sao.li2sao import apply_li2sao

    gae = [torch.ones(3) * 10.0, torch.ones(3) * -4.0]
    args = SimpleNamespace(use_li2sao=True, li2sao_gae_mix=True, li2sao_ev_threshold=2.0)
    rollout = {
        "advantages": [a.clone() for a in gae],
        "returns": [torch.ones(3), torch.zeros(3)],
        "values": [torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.0, 0.0, 0.0])],
        "raw_reward": [1.0, 0.0],
        "group_indices": [3, 3],
        "loss_masks": [torch.ones(3), torch.ones(3)],
        "response_lengths": [3, 3],
        "total_lengths": [3, 3],
    }
    metrics = apply_li2sao(args, rollout)
    assert metrics is not None
    assert metrics["ev_seq"] == pytest.approx(1.0)
    assert metrics["gae_mix_alpha"] == pytest.approx(0.5)
    pos, neg = _grpo_pair_adv(True), _grpo_pair_adv(False)
    torch.testing.assert_close(rollout["advantages"][0], torch.ones(3) * (0.5 * 10.0 + 0.5 * pos))
    torch.testing.assert_close(rollout["advantages"][1], torch.ones(3) * (0.5 * -4.0 + 0.5 * neg))


def test_li2sao_gae_mix_noop_when_n_is_one():
    from lisao.sao.li2sao import apply_li2sao

    args = SimpleNamespace(use_li2sao=True, li2sao_gae_mix=True, li2sao_ev_threshold=0.2)
    rollout = {
        "advantages": [torch.ones(3) * 2.0],
        "returns": [torch.ones(3) * 4.0],
        "values": [torch.zeros(3)],
        "raw_reward": [1.0],
        "group_indices": [9],
        "loss_masks": [torch.ones(3)],
        "response_lengths": [3],
        "total_lengths": [3],
    }
    metrics = apply_li2sao(args, rollout)
    assert metrics is not None
    assert metrics["actor_mode"] == "gae"
    assert metrics["gae_mix_alpha"] == pytest.approx(0.0)
    torch.testing.assert_close(rollout["advantages"][0], torch.ones(3) * 2.0)


def test_li2sao_gae_mix_residual_rank_stays_grpo_when_ev_res_is_zero():
    from lisao.sao.li2sao import apply_li2sao

    gae = [torch.ones(3) * 9.0, torch.ones(3) * -9.0]
    args = SimpleNamespace(
        use_li2sao=True,
        li2sao_gae_mix=True,
        li2sao_ev_threshold=0.2,
        li2sao_switch_criterion="residual",
        li2sao_critic_target="residual_rank",
        li2sao_switch_rank_threshold=0.65,
    )
    rollout = {
        "advantages": [a.clone() for a in gae],
        "returns": [torch.ones(3) * 4.0, torch.ones(3) * 5.0],
        "values": [torch.zeros(3), torch.zeros(3)],
        "raw_reward": [1.0, 0.0],
        "group_indices": [3, 3],
        "loss_masks": [torch.ones(3), torch.ones(3)],
        "response_lengths": [3, 3],
        "total_lengths": [3, 3],
    }
    metrics = apply_li2sao(args, rollout)
    assert metrics is not None
    assert metrics["ev_residual"] == pytest.approx(0.0)
    assert metrics["gae_mix_alpha"] == pytest.approx(0.0)
    assert metrics["actor_mode"] == "grpo"
    pos, neg = _grpo_pair_adv(True), _grpo_pair_adv(False)
    torch.testing.assert_close(rollout["advantages"][0], torch.ones(3) * pos)
    torch.testing.assert_close(rollout["advantages"][1], torch.ones(3) * neg)


def test_li2sao_gae_mix_residual_perfect_uses_gae():
    from lisao.sao.li2sao import apply_li2sao

    gae = [torch.ones(3) * 8.0, torch.ones(3) * 1.0]
    args = SimpleNamespace(
        use_li2sao=True,
        li2sao_gae_mix=True,
        li2sao_ev_threshold=0.2,
        li2sao_switch_criterion="residual",
        li2sao_critic_target="residual_rank",
    )
    rollout = {
        "advantages": [a.clone() for a in gae],
        "returns": [torch.ones(3), torch.zeros(3)],
        "values": [torch.tensor([0.0, 0.0, 0.5]), torch.tensor([0.0, 0.0, -0.5])],
        "raw_reward": [1.0, 0.0],
        "group_indices": [3, 3],
        "loss_masks": [torch.ones(3), torch.ones(3)],
        "response_lengths": [3, 3],
        "total_lengths": [3, 3],
    }
    metrics = apply_li2sao(args, rollout)
    assert metrics is not None
    assert metrics["ev_residual"] == pytest.approx(1.0)
    assert metrics["gae_mix_alpha"] == pytest.approx(1.0)
    assert metrics["actor_mode"] == "mix"
    torch.testing.assert_close(rollout["advantages"][0], gae[0])
    torch.testing.assert_close(rollout["advantages"][1], gae[1])


def test_li2sao_configure_args_forces_boost_sampling():
    from lisao.sao.li2sao import configure_li2sao_args

    args = SimpleNamespace(
        advantage_estimator="ppo",
        li2sao_n_boost=8,
        li2sao_n_stable=1,
        li2sao_traj_per_step=128,
        li2sao_n_boost_min_steps=60,
        li2sao_n_boost_max_steps=150,
        num_critic_only_steps=10,
        n_samples_per_prompt=1,
        rollout_batch_size=128,
        global_batch_size=128,
        over_sampling_batch_size=None,
        grpo_std_normalization=True,
    )
    configure_li2sao_args(args)
    assert args.n_samples_per_prompt == 8
    assert args.rollout_batch_size == 16
    assert args.num_critic_only_steps == 0
    assert args.over_sampling_batch_size == 16
    assert args.grpo_std_normalization is False

    args.li2sao_inflight_switch = "flatten"
    configure_li2sao_args(args)
    args.li2sao_inflight_switch = "nope"
    with pytest.raises(ValueError, match="inflight-switch"):
        configure_li2sao_args(args)


def test_li2sao_boost_grpo_loss_off_until_n1_then_restores_lisao():
    from lisao.sao.li2sao import (
        LOSS_MODE_BOOST_GRPO,
        LOSS_MODE_STABLE_LISAO,
        apply_li2sao_loss_mode,
        apply_li2sao_loss_mode_from_batch,
        apply_sampling_to_args,
        configure_li2sao_args,
    )

    args = SimpleNamespace(
        advantage_estimator="ppo",
        li2sao_n_boost=8,
        li2sao_n_stable=1,
        li2sao_traj_per_step=128,
        li2sao_n_boost_min_steps=60,
        li2sao_n_boost_max_steps=150,
        li2sao_boost_grpo_loss=True,
        li2sao_boost_eps_clip=0.2,
        li2sao_boost_eps_clip_high=0.28,
        num_critic_only_steps=0,
        n_samples_per_prompt=1,
        rollout_batch_size=128,
        global_batch_size=128,
        over_sampling_batch_size=None,
        grpo_std_normalization=True,
        eps_clip=0.3,
        eps_clip_high=5.0,
        use_tis=True,
        use_dis=True,
        use_rollout_logprobs=True,
    )
    configure_li2sao_args(args)
    assert args.use_tis is False
    assert args.use_dis is True
    assert args.use_rollout_logprobs is True
    assert args.eps_clip == 0.2
    assert args.eps_clip_high == 0.28
    assert args.li2sao_loss_mode == LOSS_MODE_BOOST_GRPO
    assert args.li2sao_stable_eps_clip == 0.3
    assert args.li2sao_stable_eps_clip_high == 5.0
    assert args.li2sao_stable_use_tis is True

    configure_li2sao_args(args)
    assert args.li2sao_stable_eps_clip == 0.3
    assert args.eps_clip == 0.2

    apply_sampling_to_args(args, 1, 128)
    assert args.n_samples_per_prompt == 1
    assert args.use_tis is True
    assert args.eps_clip == 0.3
    assert args.eps_clip_high == 5.0
    assert args.li2sao_loss_mode == LOSS_MODE_STABLE_LISAO

    leftover = {"group_indices": [7] * 8}
    apply_li2sao_loss_mode_from_batch(args, leftover)
    assert args.use_tis is False
    assert args.eps_clip == 0.2
    assert args.eps_clip_high == 0.28
    assert args.li2sao_loss_mode == LOSS_MODE_BOOST_GRPO

    apply_li2sao_loss_mode(args, 1)
    assert args.use_tis is True
    assert args.eps_clip == 0.3
    assert args.eps_clip_high == 5.0
    assert args.use_dis is True


def test_li2sao_configure_token_criterion_raises_defaults_unless_overridden():
    from lisao.sao.li2sao import (
        TOKEN_DEFAULT_BOOST_MAX_STEPS,
        TOKEN_DEFAULT_BOOST_MIN_STEPS,
        TOKEN_DEFAULT_EV_THRESHOLD,
        configure_li2sao_args,
    )

    args = SimpleNamespace(
        advantage_estimator="ppo",
        li2sao_n_boost=8,
        li2sao_n_stable=1,
        li2sao_traj_per_step=128,
        li2sao_n_boost_min_steps=60,
        li2sao_n_boost_max_steps=150,
        li2sao_ev_threshold=0.2,
        li2sao_switch_criterion="token",
        num_critic_only_steps=10,
        n_samples_per_prompt=1,
        rollout_batch_size=128,
        global_batch_size=128,
        over_sampling_batch_size=None,
        grpo_std_normalization=True,
    )
    configure_li2sao_args(args)
    assert args.li2sao_switch_criterion == "token"
    assert args.li2sao_ev_threshold == TOKEN_DEFAULT_EV_THRESHOLD
    assert args.li2sao_n_boost_min_steps == TOKEN_DEFAULT_BOOST_MIN_STEPS
    assert args.li2sao_n_boost_max_steps == TOKEN_DEFAULT_BOOST_MAX_STEPS

    args.li2sao_ev_threshold = 0.35
    args.li2sao_n_boost_min_steps = 90
    args.li2sao_n_boost_max_steps = 200
    configure_li2sao_args(args)
    assert args.li2sao_ev_threshold == 0.35
    assert args.li2sao_n_boost_min_steps == 90
    assert args.li2sao_n_boost_max_steps == 200

    args.li2sao_switch_criterion = "nope"
    with pytest.raises(ValueError, match="switch-criterion"):
        configure_li2sao_args(args)

    args.li2sao_switch_criterion = "seq"
    args.li2sao_critic_target = "nope"
    with pytest.raises(ValueError, match="critic-target"):
        configure_li2sao_args(args)

    args.li2sao_critic_target = "absolute"
    args.li2sao_switch_criterion = "residual"
    args.li2sao_ev_threshold = 0.2
    args.li2sao_n_boost_min_steps = 60
    args.li2sao_n_boost_max_steps = 150
    configure_li2sao_args(args)
    assert args.li2sao_switch_criterion == "residual"
    assert args.li2sao_ev_threshold == 0.2
    assert args.li2sao_n_boost_min_steps == 60
    assert args.li2sao_n_boost_max_steps == 150


def _critic_rollout(*, fork=True, extra_collapsed=False):
    """Two-sample mixed group (pos/neg) plus optional collapsed group."""
    prompt = 7
    if fork:
        resp_pos = [1, 1, 9, 9]
        resp_neg = [1, 1, 3, 3]
    else:
        resp_pos = [1, 2, 3, 4]
        resp_neg = [1, 2, 3, 4]
    tokens = [
        torch.tensor([prompt, *resp_pos], dtype=torch.long),
        torch.tensor([prompt, *resp_neg], dtype=torch.long),
    ]
    returns = [torch.ones(4) * 1.0, torch.ones(4) * 0.0]
    advantages = [torch.ones(4) * 9.0, torch.ones(4) * -9.0]
    raw = [1.0, 0.0]
    gids = [3, 3]
    loss_masks = [torch.ones(4), torch.ones(4)]
    if extra_collapsed:
        tokens.append(torch.tensor([prompt, 5, 5, 5, 5], dtype=torch.long))
        returns.append(torch.ones(4) * 1.0)
        advantages.append(torch.ones(4) * 4.0)
        raw.append(1.0)
        gids.append(8)
        loss_masks.append(torch.ones(4))
        tokens.append(torch.tensor([prompt, 6, 6, 6, 6], dtype=torch.long))
        returns.append(torch.ones(4) * 1.0)
        advantages.append(torch.ones(4) * 4.0)
        raw.append(1.0)
        gids.append(8)
        loss_masks.append(torch.ones(4))
    n_local = len(returns)
    return {
        "advantages": advantages,
        "returns": returns,
        "values": [torch.zeros(4) for _ in range(n_local)],
        "raw_reward": raw,
        "group_indices": gids,
        "loss_masks": loss_masks,
        "response_lengths": [4] * n_local,
        "total_lengths": [5] * n_local,
        "tokens": tokens,
    }


def test_critic_process_absolute_is_noop():
    from lisao.sao.critic_process import apply_critic_process_targets

    rollout = _critic_rollout()
    adv0 = [a.clone() for a in rollout["advantages"]]
    ret0 = [r.clone() for r in rollout["returns"]]
    metrics = apply_critic_process_targets(SimpleNamespace(li2sao_critic_target="absolute"), rollout)
    assert metrics is None
    assert "critic_keep" not in rollout
    for a, b in zip(adv0, rollout["advantages"], strict=True):
        torch.testing.assert_close(a, b)
    for a, b in zip(ret0, rollout["returns"], strict=True):
        torch.testing.assert_close(a, b)


def test_critic_process_n1_keeps_absolute_gae():
    from lisao.sao.critic_process import apply_critic_process_targets

    rollout = {
        "advantages": [torch.ones(3) * 2.0],
        "returns": [torch.ones(3) * 4.0],
        "raw_reward": [1.0],
        "group_indices": [9],
        "loss_masks": [torch.ones(3)],
        "response_lengths": [3],
        "total_lengths": [3],
        "tokens": [torch.arange(3)],
    }
    metrics = apply_critic_process_targets(SimpleNamespace(li2sao_critic_target="residual"), rollout)
    assert metrics is not None
    assert metrics["noop"] == 1
    assert "critic_keep" not in rollout
    torch.testing.assert_close(rollout["returns"][0], torch.ones(3) * 4.0)
    torch.testing.assert_close(rollout["advantages"][0], torch.ones(3) * 2.0)


def test_critic_process_mixed_residual_subtracts_raw_mean():
    from lisao.sao.critic_process import apply_critic_process_targets

    rollout = _critic_rollout()
    adv0 = [a.clone() for a in rollout["advantages"]]
    metrics = apply_critic_process_targets(SimpleNamespace(li2sao_critic_target="residual"), rollout)
    assert metrics is not None
    assert metrics["n_mixed_kept"] == 2
    assert metrics["n_collapsed_drop"] == 0
    torch.testing.assert_close(rollout["returns"][0], torch.ones(4) * 0.5)
    torch.testing.assert_close(rollout["returns"][1], torch.ones(4) * -0.5)
    for a, b in zip(adv0, rollout["advantages"], strict=True):
        torch.testing.assert_close(a, b)


def test_critic_process_drops_collapsed_group():
    from lisao.sao.critic_process import apply_critic_process_targets

    rollout = _critic_rollout(extra_collapsed=True)
    collapsed_ret = rollout["returns"][2].clone()
    metrics = apply_critic_process_targets(SimpleNamespace(li2sao_critic_target="residual"), rollout)
    assert metrics["n_mixed_kept"] == 2
    assert metrics["n_collapsed_drop"] == 2
    torch.testing.assert_close(rollout["critic_keep"][2], torch.zeros(4))
    torch.testing.assert_close(rollout["critic_keep"][3], torch.zeros(4))
    torch.testing.assert_close(rollout["returns"][2], collapsed_ret)


def test_critic_process_zeros_shared_prefix():
    from lisao.sao.critic_process import apply_critic_process_targets

    rollout = _critic_rollout(fork=True)
    apply_critic_process_targets(SimpleNamespace(li2sao_critic_target="residual"), rollout)
    expected = torch.tensor([0.0, 0.0, 1.0, 1.0])
    torch.testing.assert_close(rollout["critic_keep"][0], expected)
    torch.testing.assert_close(rollout["critic_keep"][1], expected)


def test_critic_process_last_token_without_local_opposite():
    from lisao.sao.critic_process import apply_critic_process_targets

    rollout = {
        "advantages": [torch.ones(4) * 3.0],
        "returns": [torch.ones(4) * 1.0],
        "raw_reward": [1.0, 1.0, 0.0, 0.0],
        "group_indices": [7, 7, 7, 7],
        "partition": [0],
        "loss_masks": [torch.ones(4)],
        "response_lengths": [4],
        "total_lengths": [5],
        "tokens": [torch.tensor([7, 1, 1, 1, 1])],
    }
    metrics = apply_critic_process_targets(SimpleNamespace(li2sao_critic_target="residual"), rollout)
    assert metrics["n_mixed_kept"] == 1
    assert metrics["n_last_token_only"] == 1
    torch.testing.assert_close(rollout["returns"][0], torch.ones(4) * 0.5)
    torch.testing.assert_close(rollout["critic_keep"][0], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(rollout["advantages"][0], torch.ones(4) * 3.0)


def test_critic_process_keeps_overlong_shape():
    from lisao.sao.critic_process import apply_critic_process_targets

    rollout = _critic_rollout()
    rollout["returns"] = [torch.tensor([1.0, 0.4]), torch.tensor([0.0, -0.3])]
    rollout["advantages"] = [torch.ones(2), torch.ones(2)]
    rollout["loss_masks"] = [torch.ones(2), torch.ones(2)]
    rollout["response_lengths"] = [2, 2]
    rollout["total_lengths"] = [3, 3]
    rollout["tokens"] = [
        torch.tensor([7, 1, 9]),
        torch.tensor([7, 1, 3]),
    ]
    apply_critic_process_targets(SimpleNamespace(li2sao_critic_target="residual"), rollout)
    torch.testing.assert_close(rollout["returns"][0], torch.tensor([0.5, -0.1]))
    torch.testing.assert_close(rollout["returns"][1], torch.tensor([-0.5, -0.8]))


def test_critic_rank_hinge_zero_when_ordered():
    from lisao.sao.critic_process import mixed_group_last_action_rank_loss

    v_pos = torch.tensor([0.0, 0.5], requires_grad=True)
    v_neg = torch.tensor([0.0, 0.0], requires_grad=True)
    loss, n = mixed_group_last_action_rank_loss(
        [v_pos, v_neg],
        [1, 1],
        [1.0, 0.0],
        [torch.ones(2), torch.ones(2)],
        margin=0.1,
    )
    assert n == 1
    assert loss.detach().item() == pytest.approx(0.0)


def test_critic_rank_hinge_violation():
    from lisao.sao.critic_process import mixed_group_last_action_rank_loss

    v_pos = torch.tensor([0.0, 0.0], requires_grad=True)
    v_neg = torch.tensor([0.0, 0.4], requires_grad=True)
    loss, n = mixed_group_last_action_rank_loss(
        [v_pos, v_neg],
        [1, 1],
        [1.0, 0.0],
        [torch.ones(2), torch.ones(2)],
        margin=0.1,
    )
    assert n == 1
    assert loss.detach().item() == pytest.approx(0.5)
    loss.backward()
    assert v_pos.grad is not None and v_neg.grad is not None


def test_critic_process_after_li2sao_does_not_touch_actor_a():
    from lisao.sao.critic_process import apply_critic_process_targets
    from lisao.sao.li2sao import apply_li2sao

    rollout = _critic_rollout()
    args = SimpleNamespace(use_li2sao=True, li2sao_critic_target="residual")
    apply_li2sao(args, rollout)
    adv = [a.clone() for a in rollout["advantages"]]
    apply_critic_process_targets(args, rollout)
    for a, b in zip(adv, rollout["advantages"], strict=True):
        torch.testing.assert_close(a, b)
    torch.testing.assert_close(rollout["returns"][0], torch.ones(4) * 0.5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
