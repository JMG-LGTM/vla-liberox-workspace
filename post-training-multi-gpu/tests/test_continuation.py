"""Frozen-source gates and actual two-process full-state continuation fixtures."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from post_training import continuation as c
from post_training.config import actor_learning_rate, batch_plan, load_config, validate_config
from post_training.io import sha256_file, stable_hash

FIXTURES = Path(__file__).parent / "fixtures/continuation"


def settings(root):
    return {"mode": c.MODE, "source_checkpoint": str(root / "resume_input/checkpoints" / f"update_{c.SOURCE_UPDATE:08d}.pt"),
        "source_model_sha256": c.SOURCE_SHA, "source_manifest_sha256": c.MANIFEST_SHA,
        "source_contract_sha256": c.CONTRACT_SHA, "source_config_sha256": c.CONFIG_SHA,
        "source_commit": c.SOURCE_CODE, "source_update": c.SOURCE_UPDATE,
        "target_updates": c.TARGET_UPDATE, "learning_rate_policy": c.POLICY}


def source_config():
    return load_config(FIXTURES / "source_config.yaml")


def source_contract():
    return json.loads((FIXTURES / "source_contract.json").read_text())


def extended_config(root):
    config = source_config()
    config["output_dir"] = str(root / "runs/production")
    config["training"]["optimizer_updates"] = c.TARGET_UPDATE
    config["training"]["continuation"] = settings(root)
    return config


def extended_contract(config, old=None):
    result = copy.deepcopy(source_contract() if old is None else old)
    result.update(batch_plan(config, 2))
    result["config_sha256"] = stable_hash(c.comparable_config(config))
    result["continuation"] = copy.deepcopy(config["training"]["continuation"])
    return result


def test_inactive_feature_keeps_original_config_and_contract_hashes():
    config = source_config()
    assert config["training"]["continuation"] is None
    assert stable_hash(c.comparable_config(config)) == c.CONFIG_SHA
    assert stable_hash(source_contract()) == c.CONTRACT_SHA


def test_only_target_and_explicit_lineage_change(tmp_path):
    config = extended_config(tmp_path)
    validate_config(config)
    target = extended_contract(config)
    assert c.donor_contract(config, target) == source_contract()
    assert target["optimizer_updates"] == target["policy_optimizer_updates"] == target["critic_optimizer_updates"] == 45000
    assert target["expected_sample_draws"] == 2880000


@pytest.mark.parametrize("field,value", [("source_update", 29999), ("target_updates", 50000),
    ("source_model_sha256", "0" * 64), ("source_manifest_sha256", "0" * 64),
    ("source_contract_sha256", "0" * 64), ("source_config_sha256", "0" * 64),
    ("source_commit", "0" * 40), ("learning_rate_policy", "restart_cosine"),
    ("source_checkpoint", "/other/checkpoints/update_00030000.pt")])
def test_wrong_extension_or_source_rejected(tmp_path, field, value):
    config = extended_config(tmp_path)
    config["training"]["continuation"][field] = value
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize("section,field,value", [("training", "learning_rate", 3e-5),
    ("training", "final_learning_rate", 1e-7), ("training", "warmup_updates", 0),
    ("training", "global_batch_size", 32), ("training", "per_device_batch_size", 16),
    ("training", "seed", 8), ("training", "zero_optimizer", False),
    ("training", "gradient_accumulation_steps", 2), ("training", "sampling", "shuffle"),
    ("data", "validation_fraction", 0), ("data", "seed", 8),
    ("model", "backbone", "frozen"), ("model", "actor_init_policy", "/a/bc_policy"),
    ("critic", "learning_rate", 1e-4), ("critic", "value_learning_rate", 1e-4),
    ("critic", "expectile", .7), ("reward", "discount", .98),
    ("actor_bc", "weight", .2), ("actor_bc", "per_device_batch_size", 2)])
def test_unchanged_math_and_data_hash_is_mandatory(tmp_path, section, field, value):
    config = extended_config(tmp_path)
    config[section][field] = value
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize("field", ["dataset_sha256", "cache_fingerprint", "model_identity_sha256",
    "actor_bc_auxiliary_sha256", "critic_initialization_sha256"])
def test_same_config_with_changed_actual_prepared_inputs_is_rejected(tmp_path, field):
    config = extended_config(tmp_path)
    target = extended_contract(config)
    target[field] = "0" * 64
    with pytest.raises(ValueError):
        c.donor_contract(config, target)


def test_world_size_and_external_paths_remain_restricted(tmp_path):
    config = extended_config(tmp_path)
    target = extended_contract(config)
    target["world_size"] = 1
    with pytest.raises(ValueError, match="two-rank"):
        c.donor_contract(config, target)
    source = Path(config["training"]["continuation"]["source_checkpoint"])
    assert c.check_resume_location(config, source)
    assert c.check_resume_location(config, source.parent / c.SOURCE_GENERATION / "model.pt")
    assert not c.check_resume_location(config, Path(config["output_dir"]) / "checkpoints/update_00030002.pt")
    with pytest.raises(ValueError, match="External resume"):
        c.check_resume_location(config, source.parent / "update_00027500.pt")
    config["training"]["continuation"] = None
    with pytest.raises(ValueError, match="original output_dir"):
        c.check_resume_location(config, source)


def test_actor_lr_is_continuous_terminal_and_never_rewarms(tmp_path):
    config = extended_config(tmp_path)
    old_end = actor_learning_rate(29999, 30000, 3e-6, 3e-7, 1000)
    assert old_end == 3e-7
    for update in (30000, 30001, 32500, 44999):
        ordinary = actor_learning_rate(update, 45000, 3e-6, 3e-7, 1000)
        assert c.actor_rate(config, update, ordinary) == old_end
    with pytest.raises(ValueError, match="warmup"):
        c.actor_rate(config, 0, 3e-9)
    config["training"]["continuation"] = None
    assert c.actor_rate(config, 30000, 1e-6) == 1e-6


class TinyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.2, -.4]))

    def forward(self, value):
        return (self.weight * value).sum()

    def trainable_state_dict(self):
        return {k: v.detach() for k, v in self.named_parameters()}

    def load_trainable_state_dict(self, state):
        self.load_state_dict(state, strict=True)


class TinyCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("q1", "q2", "value", "target_q1", "target_q2"):
            self.register_parameter(name, torch.nn.Parameter(torch.tensor([.1, -.2]),
                requires_grad=not name.startswith("target_")))

    def forward(self, value, stage):
        if stage == "q":
            return ((self.q1 + self.q2) * value).sum()
        return (getattr(self, stage) * value).sum()


def _fixture_worker(root):
    """Real Gloo/ZeRO state transfer, scaled2->6; production constants never change."""
    import torch.distributed as dist
    from post_training.checkpoint import load_checkpoint, save_checkpoint, rng_state
    from post_training.distributed import OptimizerGroup, GlobalBatchSampler, primary_call, wrap, unwrap, rank
    dist.init_process_group("gloo")
    torch.set_num_threads(1)
    device = torch.device("cpu")
    root = Path(root)
    c.SOURCE_UPDATE, c.TARGET_UPDATE = 2, 6
    old_config = source_config()
    old_config["training"]["optimizer_updates"] = 2
    old_config["training"]["warmup_updates"] = 0
    c.CONFIG_SHA = stable_hash(c.comparable_config(old_config))
    old = source_contract()
    old.update(batch_plan(old_config, 2), config_sha256=c.CONFIG_SHA)
    c.CONTRACT_SHA = stable_hash(old)

    def initialize():
        random.seed(91 + rank()); np.random.seed(91 + rank()); torch.manual_seed(91)
        actor = wrap(TinyActor(), device, unused=False)
        critic = wrap(TinyCritic(), device, unused=True)
        opt = {"actor": OptimizerGroup(actor.parameters(), zero=True, lr=3e-7),
            "q": OptimizerGroup([unwrap(critic).q1, unwrap(critic).q2], zero=True, lr=3e-4),
            "value": OptimizerGroup([unwrap(critic).value], zero=True, lr=3e-4)}
        return actor, critic, opt

    def steps(actor, critic, opt, start, stop):
        sampled = []
        native = iter(GlobalBatchSampler(97, 64, rank(), 2, 8, start, stop, "replacement"))
        official = iter(GlobalBatchSampler(171, 2, rank(), 2, 104736, start, stop, "replacement"))
        for update in range(start, stop):
            indices = next(native); demonstration = next(official)
            sampled.append([update + 1, indices, demonstration])
            noise = torch.randn(2) + random.random() + float(np.random.random())
            noise += sum(indices) * 1e-5 + sum(demonstration) * 1e-6
            for stage in ("value", "q"):
                opt[stage].zero_grad()
                loss = critic(noise, stage)
                loss.square().backward(); opt[stage].step()
            opt["actor"].zero_grad(); actor(noise).square().backward(); opt["actor"].step()
            with torch.no_grad():
                unwrap(critic).target_q1.lerp_(unwrap(critic).q1, .005)
                unwrap(critic).target_q2.lerp_(unwrap(critic).q2, .005)
        return sampled

    actor, critic, opt = initialize()
    steps(actor, critic, opt, 0, 2)
    source_alias = root / "resume_input/checkpoints/update_00000002.pt"
    save_checkpoint(source_alias, update=2, actor=actor, critic=critic, optimizers=opt,
        contract=old, metadata={"best_update": 1, "best_metric": .25}, device=device, checkpoint_format="sharded")
    source_model = source_alias.resolve()
    c.SOURCE_GENERATION = source_model.parent.name
    c.SOURCE_SHA = primary_call(lambda: sha256_file(source_model))
    c.MANIFEST_SHA = primary_call(lambda: sha256_file(source_model.with_suffix(".json")))
    immutable = primary_call(lambda: {p.name: sha256_file(p) for p in source_model.parent.iterdir()})

    def run(name, paused):
        config = copy.deepcopy(old_config)
        config["output_dir"] = str(root / "runs" / name)
        config["training"]["optimizer_updates"] = 6
        config["training"]["continuation"] = settings(root)
        validate_config(config)
        current = extended_contract(config, old)
        actor, critic, opt = initialize()
        expected = c.prepare_donor_load(config, source_alias, current)
        start, metadata = load_checkpoint(source_alias, actor=actor, critic=critic, optimizers=opt,
            contract=expected, device=device)
        assert start == 2 and metadata == {"best_update": 1, "best_metric": .25}
        layout = {k: v.local_state_dict() for k, v in opt.items()}
        assert all(int(s["step"]) == 2 for v in layout.values() for state in v["states"]
            for s in state["state"].values())
        assert all(g["lr"] == (3e-7 if k == "actor" else 3e-4) for k, o in opt.items() for g in o.param_groups)
        samples = steps(actor, critic, opt, start, 4 if paused else 6)
        if paused:
            checkpoint = Path(config["output_dir"]) / "checkpoints/update_00000004.pt"
            save_checkpoint(checkpoint, update=4, actor=actor, critic=critic, optimizers=opt,
                contract=current, metadata=metadata, device=device, checkpoint_format="sharded")
            actor, critic, opt = initialize()
            assert not c.check_resume_location(config, checkpoint)
            resumed, metadata = load_checkpoint(checkpoint, actor=actor, critic=critic, optimizers=opt,
                contract=current, device=device)
            samples += steps(actor, critic, opt, resumed, 6)
        result = {"actor": unwrap(actor).trainable_state_dict(), "critic": unwrap(critic).state_dict(),
            "optimizers": {k: o.local_state_dict() for k, o in opt.items()}, "rng": rng_state(device),
            "samples": samples, "update": 6, "sample_draws": 6 * 64, "contract": current}
        torch.save(result, root / f"{name}_rank{rank()}.pt")
    run("reference", False)
    run("resumed", True)
    assert primary_call(lambda: {p.name: sha256_file(p) for p in source_model.parent.iterdir()}) == immutable
    dist.destroy_process_group()


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            assert_same(first, second)
    else:
        assert left == right


def test_two_gloo_zero_ranks_exact_models_three_optimizers_rng_samples_and_resume(tmp_path):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-B", "-m", "torch.distributed.run", "--standalone",
        "--nproc-per-node=2", str(Path(__file__).resolve()), "--worker", str(tmp_path)],
        env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    for rank in (0, 1):
        a = torch.load(tmp_path / f"reference_rank{rank}.pt", weights_only=False)
        b = torch.load(tmp_path / f"resumed_rank{rank}.pt", weights_only=False)
        # Different output_dir is intentionally excluded from both contracts;
        # the frozen source alias and every real training setting are identical.
        assert_same(a, b)


if __name__ == "__main__":
    assert sys.argv[1] == "--worker"
    _fixture_worker(sys.argv[2])
