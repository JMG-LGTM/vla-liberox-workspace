"""Exact CPU-only held-out losses, unequal batches and read-only evaluation."""
from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from post_training.iql.critic import PixelIQL
from post_training.diagnostics import evaluate_critic
from test_training import configuration, runner_dataset


class ActionQ(nn.Module):
    def __init__(self, multiplier):
        super().__init__()
        self.multiplier = nn.Parameter(torch.tensor(float(multiplier)))

    def forward(self, pixels, proprio, actions, action_mask):
        return actions[:, 0, 0] * self.multiplier


class ConstantQ(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)), requires_grad=False)

    def forward(self, pixels, proprio, actions, action_mask):
        return self.value.expand(len(proprio))


class StateV(nn.Module):
    def __init__(self):
        super().__init__()
        self.multiplier = nn.Parameter(torch.ones(()))

    def forward(self, pixels, proprio):
        return proprio[:, 0] * self.multiplier


@pytest.fixture
def source():
    model = PixelIQL(horizon=2, discount=.5, expectile=.8)
    model.q1, model.q2 = ActionQ(1), ActionQ(2)
    model.target_q1, model.target_q2 = ConstantQ(2), ConstantQ(4)
    model.value = StateV()
    actions = torch.zeros(3, 2, 7)
    actions[:, 0, 0] = torch.tensor([1., 2., 3.])
    proprio, next_proprio = torch.zeros(3, 8), torch.zeros(3, 8)
    proprio[:, 0], next_proprio[:, 0] = torch.tensor([1., 3., 5.]), torch.tensor([4., 6., 8.])
    batch = {"pixels": torch.zeros(3, 6, 4, 4, dtype=torch.uint8),
        "next_pixels": torch.ones(3, 6, 4, 4, dtype=torch.uint8),
        "proprio": proprio, "next_proprio": next_proprio, "actions": actions,
        "action_mask": torch.ones(3, 2, dtype=torch.bool),
        "reward": torch.tensor([-1., 0., -1.]), "bootstrap_mask": torch.tensor([1., 0., 1.]),
        "remaining_blocks": torch.tensor([1, 0, -1]),
        "success_terminal": torch.tensor([False, True, False]),
        "run_id": ["success_a", "success_b", "truncated_failure"]}
    return model, batch


def subset(batch, start, stop):
    return {key: value[start:stop] for key, value in batch.items()}


def test_exact_training_targets_and_sample_weighted_losses_across_unequal_batches(source):
    model, batch = source
    scores = model.scores(batch)
    full = evaluate_critic(model, [batch])
    split = evaluate_critic(model, [subset(batch, 0, 2), subset(batch, 2, 3)])
    # Current next V=[4,6,8], rewards=[-1,0,-1], gamma=.5, masks=[1,0,1]
    # give targets=[1,0,3]. Online Q1=[1,2,3], Q2=[2,4,6].
    # V uses target minQ=2, not online minQ=[1,2,3]: expectile losses=.8,.2,1.8.
    expected = {"count": 3, "q1_mse": 4 / 3, "q2_mse": 26 / 3,
        "q_loss": 5., "v_expectile_loss": 2.8 / 3,
        "q_abs_disagreement": 2., "target_mean": 4 / 3}
    for key, value in expected.items():
        assert full["losses"][key] == pytest.approx(value)
        assert split["losses"][key] == pytest.approx(value)
    assert full["q_distribution"]["mean"] == pytest.approx(float(scores["q"].mean()))
    assert full["value_distribution"]["mean"] == pytest.approx(3.)
    assert full["spearman_v_remaining_blocks"] == pytest.approx(-1.)
    assert full["losses"]["aggregation"] == "mean_per_transition_without_augmentation"
    assert full["losses"]["q_target_definition"].endswith("online_V(next_observation)")
    assert "target_Q1" in full["losses"]["v_target_definition"]
    json.dumps(full, allow_nan=False)


def test_empty_evaluation_reports_missing_losses_not_zero(source):
    model, _ = source
    result = evaluate_critic(model, [])
    assert result["count"] == result["losses"]["count"] == 0
    for key in ("q1_mse", "q2_mse", "q_loss", "v_expectile_loss", "q_abs_disagreement", "target_mean"):
        assert result["losses"][key] is None
    json.dumps(result, allow_nan=False)


def test_eval_uses_raw_inputs_no_grad_no_gpu_and_no_optimizer_or_target_updates(source, monkeypatch):
    model, batch = source
    model.train(); model.q2.eval()
    modes = {name: item.training for name, item in model.named_modules()}
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    before = deepcopy(model.state_dict())
    gradients = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    optimizer_before = deepcopy(optimizer.state_dict())
    forward = model.forward
    def observed(inputs, **kwargs):
        assert not torch.is_grad_enabled() and not any(m.training for m in model.modules())
        assert kwargs["stage"] == "both"
        assert inputs["pixels"].data_ptr() == batch["pixels"].data_ptr()
        assert inputs["next_pixels"].data_ptr() == batch["next_pixels"].data_ptr()
        return forward(inputs, **kwargs)
    monkeypatch.setattr(model, "forward", observed)
    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluation must not augment/update/communicate/use CUDA")
    monkeypatch.setattr(model, "update_targets", forbidden)
    monkeypatch.setattr(optimizer, "step", forbidden)
    monkeypatch.setattr(torch.cuda, "get_rng_state", forbidden)
    monkeypatch.setattr(torch.cuda, "set_rng_state", forbidden)
    monkeypatch.setattr(torch.distributed, "all_reduce", forbidden)
    monkeypatch.setattr("post_training.iql.critic.augment_batch", forbidden)
    evaluate_critic(model, [batch])
    assert all(torch.equal(before[k], value) for k, value in model.state_dict().items())
    assert modes == {name: item.training for name, item in model.named_modules()}
    assert optimizer_before == optimizer.state_dict()
    for old, parameter in zip(gradients, model.parameters()):
        assert (parameter.grad is None) if old is None else torch.equal(old, parameter.grad)


@pytest.mark.parametrize("fail", [False, True])
def test_global_rng_and_mixed_modes_restore_even_on_failure(source, monkeypatch, fail):
    model, batch = source
    model.train(); model.target_q1.eval()
    modes = {name: item.training for name, item in model.named_modules()}
    score = model.scores
    def consumes_rng(inputs, **kwargs):
        random.random(); np.random.random(); torch.rand(1)
        if fail:
            raise RuntimeError("synthetic failure after RNG consumption")
        return score(inputs, **kwargs)
    monkeypatch.setattr(model, "scores", consumes_rng)
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    if fail:
        with pytest.raises(RuntimeError, match="synthetic failure"):
            evaluate_critic(model, [batch])
    else:
        evaluate_critic(model, [batch])
    assert random.getstate() == python_rng
    restored_numpy = np.random.get_state()
    assert restored_numpy[0] == numpy_rng[0] and np.array_equal(restored_numpy[1], numpy_rng[1])
    assert restored_numpy[2:] == numpy_rng[2:]
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert modes == {name: item.training for name, item in model.named_modules()}


def test_dataloader_generator_is_restored(source):
    model, batch = source
    rows = [{key: value[i] for key, value in batch.items()} for i in range(3)]
    generator = torch.Generator().manual_seed(34)
    loader = DataLoader(rows, batch_size=2, shuffle=False, generator=generator)
    before = generator.get_state().clone()
    result = evaluate_critic(model, loader)
    assert result["losses"]["count"] == 3
    assert result["losses"]["q_loss"] == pytest.approx(5.)
    assert torch.equal(before, generator.get_state())


@pytest.mark.parametrize("mutation", ["q1_shape", "value_shape", "target_shape", "nonfinite_q", "nonfinite_v", "nonfinite_target"])
def test_wrong_loss_vectors_and_nonfinite_statistics_fail_closed(source, monkeypatch, mutation):
    model, batch = source
    forward = model.forward
    def broken(inputs, **kwargs):
        result = forward(inputs, **kwargs)
        if mutation == "q1_shape": result["q1"] = result["q1"].unsqueeze(-1)
        elif mutation == "value_shape": result["v_loss"] = result["v_loss"][:-1]
        elif mutation == "target_shape": result["target"] = result["target"].unsqueeze(-1)
        elif mutation == "nonfinite_q": result["q1"][0] = float("nan")
        elif mutation == "nonfinite_v": result["v_loss"][0] = float("inf")
        else: result["target"][0] = float("nan")
        return result
    monkeypatch.setattr(model, "forward", broken)
    model.train(); modes = {name: item.training for name, item in model.named_modules()}
    with pytest.raises((ValueError, FloatingPointError)):
        evaluate_critic(model, [batch])
    assert modes == {name: item.training for name, item in model.named_modules()}


def test_missing_transition_fields_use_existing_strict_training_contract(source):
    model, batch = source
    batch.pop("bootstrap_mask")
    with pytest.raises(KeyError, match="bootstrap_mask"):
        evaluate_critic(model, [batch])


def test_critic_only_runner_records_nested_losses_without_actor_training(runner_dataset, tmp_path):
    from post_training import training
    config = configuration(tmp_path, "critic", updates=2)
    summary = training.train(config, runner_dataset)
    assert summary["policy_optimizer_updates"] == 0
    assert summary["critic_optimizer_updates"] == 2
    records = [json.loads(line) for line in (Path(config["output_dir"]) / "metrics.jsonl").read_text().splitlines()]
    assert len(records) == 2
    for record in records:
        validation = record["validation"]
        losses = validation["losses"]
        assert validation["held_out"] is True and validation["split"] == "val"
        assert losses["count"] == validation["count"] > 0
        assert losses["q_loss"] == pytest.approx(.5 * (losses["q1_mse"] + losses["q2_mse"]))
        assert losses["v_expectile_loss"] >= 0 and losses["q_abs_disagreement"] >= 0
        json.dumps(record, allow_nan=False)
