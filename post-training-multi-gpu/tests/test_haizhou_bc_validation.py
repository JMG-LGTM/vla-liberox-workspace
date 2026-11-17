"""CPU-only checks for exact root thresholds and non-invasive held-out BC loss."""
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from post_training import training
from post_training.config import DEFAULTS, validate_config
from post_training.data import grouped_split, inspect_dataset
from post_training.policy import masked_flow_losses
from test_official_bc import mixed
from test_policy_export import base
from test_training import configuration, install_fake_policy, load_snapshot, runner_dataset


def test_haizhou_threshold_matches_independent_oracle_and_groups_branches():
    episodes = [dict(root_run_id=f"root_{i}") for i in range(5)]
    episodes += [dict(root_run_id="root_3"), dict(root_run_id="root_3")]
    result = grouped_split(episodes, .2, 7, "haizhou_hash_threshold")
    expected = {}
    for episode in episodes:
        root = episode["root_run_id"]
        fraction = int.from_bytes(hashlib.sha256(f"7:{root}".encode()).digest()[:8], "big") / float(2**64)
        expected[root] = "val" if fraction < .2 else "train"
    assert result == expected
    assert len(result) == 5
    assert result != grouped_split(episodes, .2, 7)
    assert grouped_split(list(reversed(episodes)), .2, 7, "haizhou_hash_threshold") == result


def test_haizhou_all_validation_fallback_preserves_first_selected_root():
    candidates = [f"candidate_{i}" for i in range(100)]
    roots = [root for root in candidates if int.from_bytes(hashlib.sha256(f"7:{root}".encode()).digest()[:8], "big")
             / float(2**64) < .2][:2]
    episodes = [dict(root_run_id=root) for root in reversed(roots)]
    result = grouped_split(episodes, .2, 7, "haizhou_hash_threshold")
    assert result[episodes[0]["root_run_id"]] == "train"
    assert result[episodes[1]["root_run_id"]] == "val"
    assert set(grouped_split(episodes, 0, 7, "haizhou_hash_threshold").values()) == {"train"}


class ValidationActor(torch.nn.Module):
    def __init__(self, *, invalid=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))
        self.frozen = torch.nn.Dropout(.5).eval()
        self.invalid = invalid
        self.seen_grad = []
        self.seen_training = []

    def forward(self, batch):
        self.seen_grad.append(torch.is_grad_enabled())
        self.seen_training.append(self.training)
        random.random(); np.random.random(); torch.rand(3)
        errors = batch["errors"] * self.weight
        if self.invalid:
            errors = errors * float("inf")
        return masked_flow_losses(errors, batch["action_mask"])


def validation_loader():
    # Three samples with losses 1,3,5; final batch has one sample. Masked padding
    # deliberately has huge values so including official-style tail padding fails.
    samples = []
    for value in (1., 3., 5.):
        errors = torch.full((8, 7), 10000.)
        errors[0] = value
        samples.append(dict(errors=errors, action_mask=torch.tensor([True] + [False] * 7)))
    return DataLoader(samples, batch_size=2, generator=torch.Generator().manual_seed(81))


def test_actor_validation_weighting_tail_masks_no_updates_rng_and_modes():
    actor = ValidationActor()
    actor.weight.grad = torch.tensor(9.)
    torch.manual_seed(99); random.seed(99); np.random.seed(99)
    before = training.rng_state(torch.device("cpu"))
    modes = [module.training for module in actor.modules()]
    loader = validation_loader()
    loader_rng = loader.generator.get_state().clone()
    train_generator = torch.Generator().manual_seed(17)
    train_rng = train_generator.get_state().clone()
    result = training.evaluate_actor(actor, loader, torch.device("cpu"), seed=17007)
    assert result["samples"] == 3 and result["val_actor_loss"] == 3.
    assert actor.weight.item() == 1. and actor.weight.grad.item() == 9.
    assert actor.seen_grad == [False, False] and actor.seen_training == [False, False]
    assert [module.training for module in actor.modules()] == modes
    assert torch.equal(loader.generator.get_state(), loader_rng)
    assert torch.equal(train_generator.get_state(), train_rng)
    actual = (random.random(), np.random.random(), torch.rand(3))
    training.restore_rng(before, torch.device("cpu"))
    expected = (random.random(), np.random.random(), torch.rand(3))
    assert actual[:2] == expected[:2] and torch.equal(actual[2], expected[2])


def test_actor_validation_fixed_stream_is_repeatable_after_training_rng_changes():
    class RandomActor(ValidationActor):
        def forward(self, batch):
            return super().forward(batch) + torch.randn(len(batch["action_mask"])) + random.random() + np.random.random()
    actor, loader = RandomActor(), validation_loader()
    first = training.evaluate_actor(actor, loader, torch.device("cpu"), seed=17007)
    torch.rand(50); random.random(); np.random.random()
    second = training.evaluate_actor(actor, loader, torch.device("cpu"), seed=17007)
    assert first == second


def test_actor_validation_restores_rng_and_modes_after_failure():
    actor = ValidationActor(invalid=True)
    before = training.rng_state(torch.device("cpu"))
    modes = [module.training for module in actor.modules()]
    with pytest.raises(FloatingPointError, match="Nonfinite held-out"):
        training.evaluate_actor(actor, validation_loader(), torch.device("cpu"), seed=17007)
    assert torch.equal(torch.get_rng_state(), before["torch"])
    assert random.getstate() == before["python"]
    assert [module.training for module in actor.modules()] == modes


def test_bc_validation_actual_runner_preserves_update_math_and_resumes_logs(
        runner_dataset, tmp_path, monkeypatch, base):
    root = install_fake_policy(monkeypatch, base)
    config = configuration(tmp_path, "bc", updates=4)
    config["model"]["checkpoint"] = str(root)
    config["training"].update(validation_interval=2, validation_batch_size=3)
    config["output_dir"] = str(tmp_path / "without_validation")
    baseline = training.train(config, runner_dataset)
    baseline_state = load_snapshot(baseline["checkpoint"])
    config["training"]["bc_validation"] = True
    config["output_dir"] = str(tmp_path / "with_validation")
    enabled = training.train(config, runner_dataset)
    enabled_state = load_snapshot(enabled["checkpoint"])
    assert enabled["last_bc_validation_update"] == 4
    for key, value in baseline_state["actor"].items():
        assert torch.equal(value, enabled_state["actor"][key])
    log = Path(config["output_dir"]) / "validation_metrics.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [row["optimizer_update"] for row in rows] == [2, 4]
    assert all(row["samples"] == 4 and row["held_out"] for row in rows)
    # Replaying an older full checkpoint must retire later validation records.
    earlier = Path(config["output_dir"]) / "checkpoints/update_00000002.pt"
    resumed = training.train(config, runner_dataset, resume=earlier)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [row["optimizer_update"] for row in rows] == [2, 4]
    resumed_state = load_snapshot(resumed["checkpoint"])
    for key, value in enabled_state["actor"].items():
        assert torch.equal(value, resumed_state["actor"][key])


def test_official_demonstration_can_be_held_out_with_real_tail_masks(mixed, tmp_path, monkeypatch, base):
    config = configuration(tmp_path, "bc", updates=1)
    # Choose a fixed seed with an official demo held out and a nonempty train.
    config["data"].update(split_strategy="haizhou_hash_threshold")
    for seed in range(100):
        config["data"]["seed"] = seed
        info = inspect_dataset(mixed[-1], config)
        official_val = [row for row in info["rows"] if row["source_type"] == "official_bc" and row["split"] == "val"]
        if official_val and info["train_count"]:
            break
    else:
        pytest.fail("No official held-out test fixture found")
    root = install_fake_policy(monkeypatch, base)
    config["model"]["checkpoint"] = str(root)
    config["training"].update(bc_validation=True, validation_batch_size=3)
    result = training.train(config, mixed[-1])
    row = json.loads((Path(config["output_dir"]) / "validation_metrics.jsonl").read_text())
    assert result["status"] == "COMPLETED" and row["samples"] == info["val_count"]
    assert np.isfinite(row["val_actor_loss"])


def test_validation_config_rejects_missing_heldout_and_invalid_strategy():
    config = copy.deepcopy(DEFAULTS)
    config["training"]["bc_validation"] = True
    config["data"]["validation_fraction"] = 0
    with pytest.raises(ValueError, match="held-out"):
        validate_config(config)
    config = copy.deepcopy(DEFAULTS)
    config["data"]["split_strategy"] = "unreviewed"
    with pytest.raises(ValueError, match="split_strategy"):
        validate_config(config)


_DDP_WORKER = r'''
import json, os, sys
from pathlib import Path
import torch
from post_training import policy, training
from post_training.config import load_config
from post_training.pi05_assets import checkpoint_identity, load_normalization, platform_model_settings
from post_training.policy import Pi05Policy
from test_training import StochasticFakeModel
from full_memory_fakes import TinyVisionFakeModel
torch.set_num_threads(1)
config = load_config(sys.argv[1])
identity = checkpoint_identity(config['model']['checkpoint'])
def fake_loader(model_config, device, *, verified_identity):
    assert identity == verified_identity
    model = TinyVisionFakeModel() if model_config['vision_gradient_checkpointing'] else StochasticFakeModel()
    return Pi05Policy(model, lambda value: {'actions': value['actions']},
        identity, platform_model_settings(model_config, identity), load_normalization(model_config['checkpoint']),
        observation_factory=lambda value: value,
        vision_gradient_checkpointing=model_config['vision_gradient_checkpointing']).to(device)
policy.load_policy = fake_loader
calls = []
gradient_views = []
original_finite_step = training._finite_step
def capture_finite_step(optimizer, parameters, clip, device):
    parameters = list(parameters)
    if config['distributed']['gradient_as_bucket_view']:
        gradient_views.append(all(parameter.grad is None or parameter.grad._is_view() for parameter in parameters))
        assert all(opt.optim.defaults['foreach'] is False for opt in optimizer.optimizers)
    return original_finite_step(optimizer, parameters, clip, device)
training._finite_step = capture_finite_step
original = training.evaluate_actor
def capture(*args, **kwargs):
    assert int(os.environ['RANK']) == 0
    result = original(*args, **kwargs)
    calls.append(result)
    return result
training.evaluate_actor = capture
summary = training.train(config, sys.argv[2], resume=sys.argv[3] if len(sys.argv)>3 else None)
Path(config['output_dir'], 'rank_' + os.environ['RANK'] + '_bc_validation.json').write_text(
    json.dumps(dict(summary=summary, validation_calls=calls, gradient_view_checks=gradient_views,
        memory_options=dict(bucket_view=config['distributed']['gradient_as_bucket_view'],
            foreach=config['training']['adamw_foreach'], vision_gc=config['model']['vision_gradient_checkpointing']))))
'''


def test_two_process_bc_heldout_zero_sharded_resume(mixed, tmp_path, base):
    config = configuration(tmp_path, "bc", updates=2)
    config["model"]["checkpoint"] = str(base[0])
    config["data"].update(split_strategy="haizhou_hash_threshold")
    for seed in range(100):
        config["data"]["seed"] = seed
        info = inspect_dataset(mixed[-1], config)
        if info["val_count"] and info["train_count"]:
            break
    config["training"].update(global_batch_size=4, per_device_batch_size=2, gradient_accumulation_steps=1,
        zero_optimizer=True, checkpoint_format="sharded", bc_validation=True,
        validation_batch_size=3, validation_interval=1, sampling="replacement")
    config["distributed"]["backend"] = "gloo"
    path = tmp_path / "bc_validation.yaml"
    path.write_text(yaml.safe_dump(config))
    worker = tmp_path / "bc_validation_worker.py"
    worker.write_text(_DDP_WORKER)
    project = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
        "PYTHONPATH": os.pathsep.join([str(project / "src"), str(project / "tests")])}
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               "--nproc_per_node=2", str(worker), str(path), str(mixed[-1])]
    directory = Path(config["output_dir"])
    def run(resume=None):
        result = subprocess.run(command + ([str(resume)] if resume else []), env=environment,
                                capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        ranks = [json.loads((directory / f"rank_{rank}_bc_validation.json").read_text()) for rank in range(2)]
        assert len(ranks[0]["validation_calls"]) == (1 if resume else 2)
        assert ranks[1]["validation_calls"] == []
        assert ranks[0]["summary"]["last_bc_validation_loss"] == ranks[1]["summary"]["last_bc_validation_loss"]
        assert all(row["samples"] == info["val_count"] for row in ranks[0]["validation_calls"])
        rows = [json.loads(line) for line in (directory / "validation_metrics.jsonl").read_text().splitlines()]
        assert [row["optimizer_update"] for row in rows] == [1, 2]
    run()
    final = directory / "checkpoints/update_00000002.pt"
    before = load_snapshot(final)
    assert before["contract"]["gradient_accumulation_steps"] == 1
    run(directory / "checkpoints/update_00000001.pt")
    after = load_snapshot(final)
    for key, value in before["actor"].items():
        assert torch.equal(value, after["actor"][key])
