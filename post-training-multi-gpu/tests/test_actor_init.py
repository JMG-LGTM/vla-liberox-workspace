"""CPU-only BC-to-IQL transfer and exact-resume coverage with real runner steps."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from post_training import training
from post_training.iql.actor_init import inspect_actor_init_policy, initialize_actor_from_policy
from post_training.config import DEFAULTS, validate_config
from post_training.export import export_policy
from post_training.io import sha256_file, stable_hash
from test_policy_export import base, make_policy
from test_training import configuration, install_fake_policy, load_snapshot, runner_dataset


def source_policy(tmp_path, model_base, *, scope="full", method="bc"):
    policy = make_policy(model_base, scope)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.fill_(.25)
    root = tmp_path / f"source-{method}-{scope}"
    export_policy(root, policy.trainable_state_dict(), policy.identity, policy.settings,
                  "d" * 64, 15000, method, "e" * 64 if method == "iql" else None)
    return root, policy


def iql_config(tmp_path, model_root, package, *, updates=2):
    config = configuration(tmp_path, method="iql", updates=updates)
    config["model"].update(checkpoint=str(model_root), backbone="full", actor_init_policy=str(package))
    config["critic"]["warmup_updates"] = 0
    return config


@pytest.mark.parametrize("method,scope,path", [("bc", "full", "package"),
    ("critic", "full", "package"), ("iql", "frozen", "package"),
    ("iql", "full", ""), ("iql", "full", True)])
def test_initialization_config_rejects_unsupported_method_scope_and_path(method, scope, path):
    config = copy.deepcopy(DEFAULTS)
    config["method"] = method
    config["model"].update(backbone=scope, actor_init_policy=path)
    with pytest.raises(ValueError, match="actor_init_policy"):
        validate_config(config)


def test_source_is_full_bc_and_can_relocate_base_without_changing_identity(base, tmp_path):
    root, policy = source_policy(tmp_path, base)
    settings = {**policy.settings, "base_checkpoint": "/relocated/server/base"}
    initialization = inspect_actor_init_policy(root, identity=policy.identity, model_config=settings)
    assert initialization["source_optimizer_updates"] == 15000
    assert initialization["source_dataset_sha256"] == "d" * 64
    assert initialization["initial_iql_actor_updates"] == 0
    assert initialization["optimizer_state_inherited"] is False
    assert initialization["source_base_checkpoint"] != settings["base_checkpoint"]
    target = make_policy(base, "full")
    initialize_actor_from_policy(target, initialization)
    for name, value in policy.trainable_state_dict().items():
        torch.testing.assert_close(target.trainable_state_dict()[name], value, atol=0, rtol=0)


@pytest.mark.parametrize("mutation", ["scope", "algorithm", "identity", "io", "actor_sha"])
def test_source_rejects_wrong_scope_algorithm_identity_io_or_hash(base, tmp_path, mutation):
    root, policy = source_policy(tmp_path, base,
                                 scope="frozen" if mutation == "scope" else "full",
                                 method="iql" if mutation == "algorithm" else "bc")
    expected_identity = copy.deepcopy(policy.identity)
    if mutation == "identity":
        expected_identity["files"]["model.safetensors"] = "1" * 64
    if mutation == "io":
        manifest = yaml.safe_load((root / "policy.yaml").read_text())
        manifest["action_horizon"] = 9
        (root / "policy.yaml").write_text(yaml.safe_dump(manifest))
    if mutation == "actor_sha":
        with (root / "actor.pt").open("ab") as handle:
            handle.write(b"changed")
    with pytest.raises(ValueError):
        inspect_actor_init_policy(root, identity=expected_identity,
                                  model_config={**policy.settings, "backbone": "full"})


def test_full_actor_initialization_rejects_nonfinite_or_missing_tensors(base, tmp_path):
    root, policy = source_policy(tmp_path, base)
    state = policy.trainable_state_dict()
    state[next(iter(state))].fill_(float("nan"))
    torch.save(state, root / "actor.pt")
    manifest = yaml.safe_load((root / "policy.yaml").read_text())
    manifest["component_sha256"]["actor"] = sha256_file(root / "actor.pt")
    (root / "policy.yaml").write_text(yaml.safe_dump(manifest))
    initialization = inspect_actor_init_policy(root, identity=policy.identity, model_config=policy.settings)
    with pytest.raises(ValueError, match="nonfinite"):
        initialize_actor_from_policy(make_policy(base, "full"), initialization)
    state = make_policy(base, "full").trainable_state_dict()
    del state[next(iter(state))]
    torch.save(state, root / "actor.pt")
    manifest["component_sha256"]["actor"] = sha256_file(root / "actor.pt")
    (root / "policy.yaml").write_text(yaml.safe_dump(manifest))
    initialization = inspect_actor_init_policy(root, identity=policy.identity, model_config=policy.settings)
    with pytest.raises(ValueError, match="keys"):
        initialize_actor_from_policy(make_policy(base, "full"), initialization)


@pytest.mark.parametrize("checkpoint_format", ["consolidated", "sharded"])
def test_iql_warmstart_fresh_counters_new_dataset_and_resume_does_not_reinitialize(
        runner_dataset, base, tmp_path, monkeypatch, checkpoint_format):
    model_root = install_fake_policy(monkeypatch, base)
    package, source = source_policy(tmp_path, base)
    config = iql_config(tmp_path, model_root, package)
    config["training"]["checkpoint_format"] = checkpoint_format
    original_step = training.optimization_step
    calls = []

    def stop_first_step(*args, **kwargs):
        actor, optimizers = args[2], args[4]
        assert kwargs["actor_update"] == 0
        for name, value in source.trainable_state_dict().items():
            torch.testing.assert_close(actor.trainable_state_dict()[name], value, atol=0, rtol=0)
        assert all(not state["state"] for optimizer in optimizers.values()
                   for state in optimizer.state_dict()["states"])
        result = original_step(*args, **kwargs)
        calls.append(1)
        (Path(config["output_dir"]) / "stop.request").write_text("pause after one new IQL update\n")
        return result

    monkeypatch.setattr(training, "optimization_step", stop_first_step)
    paused = training.train(config, runner_dataset)
    assert paused["status"] == "STOPPED"
    assert paused["optimizer_updates"] == paused["policy_optimizer_updates"] == 1
    assert paused["critic_optimizer_updates"] == 1 and paused["sample_draws"] == 2
    checkpoint = load_snapshot(paused["checkpoint"])
    initialization = checkpoint["metadata"]["actor_initialization"]
    assert initialization["source_dataset_sha256"] == "d" * 64
    assert checkpoint["metadata"]["dataset_sha256"] != "d" * 64
    assert checkpoint["contract"]["actor_initialization_sha256"] == stable_hash(initialization)
    assert any(not torch.equal(value, source.trainable_state_dict()[name])
               for name, value in checkpoint["actor"].items())

    def forbidden_initialization(*_args, **_kwargs):
        raise AssertionError("Resume reapplied the original BC actor")

    monkeypatch.setattr("post_training.iql.actor_init.initialize_actor_from_policy", forbidden_initialization)

    def inspect_resumed_step(*args, **kwargs):
        assert kwargs["actor_update"] == 1
        for name, value in checkpoint["actor"].items():
            torch.testing.assert_close(args[2].trainable_state_dict()[name], value, atol=0, rtol=0)
        assert all(parameter["step"].item() == 1 for state in args[4]["actor"].state_dict()["states"]
                   for parameter in state["state"].values())
        return original_step(*args, **kwargs)

    monkeypatch.setattr(training, "optimization_step", inspect_resumed_step)
    completed = training.train(config, runner_dataset, resume=paused["checkpoint"])
    assert completed["status"] == "COMPLETED"
    assert completed["optimizer_updates"] == completed["policy_optimizer_updates"] == 2
    assert completed["critic_optimizer_updates"] == 2 and completed["sample_draws"] == 4
    provenance = json.loads((Path(config["output_dir"]) / "provenance.json").read_text())
    assert provenance["actor_initialization"] == initialization
    records = [json.loads(line) for line in (Path(config["output_dir"]) / "metrics.jsonl").read_text().splitlines()]
    for record in records:
        assert record["unweighted_actor_loss"] >= 0
        assert record["advantage_std"] >= 0
        assert 0 <= record["actor_weight_clip_fraction"] <= 1
        assert 0 < record["actor_weight_effective_sample_size"] <= 2.000001
        assert record["global_batch_size"] == 2 and record["gradient_accumulation_steps"] == 2
    assert calls == [1]


def test_resume_rejects_changed_source_even_when_its_manifest_hash_is_updated(
        runner_dataset, base, tmp_path, monkeypatch):
    model_root = install_fake_policy(monkeypatch, base)
    package, _ = source_policy(tmp_path, base)
    config = iql_config(tmp_path, model_root, package)
    original = training.optimization_step

    def pause(*args, **kwargs):
        result = original(*args, **kwargs)
        (Path(config["output_dir"]) / "stop.request").write_text("pause\n")
        return result

    monkeypatch.setattr(training, "optimization_step", pause)
    paused = training.train(config, runner_dataset)
    state = torch.load(package / "actor.pt", weights_only=True)
    state[next(iter(state))].add_(.1)
    torch.save(state, package / "actor.pt")
    manifest = yaml.safe_load((package / "policy.yaml").read_text())
    manifest["component_sha256"]["actor"] = sha256_file(package / "actor.pt")
    (package / "policy.yaml").write_text(yaml.safe_dump(manifest))
    with pytest.raises(RuntimeError, match="Resume contract changed"):
        training.train(config, runner_dataset, resume=paused["checkpoint"])


def test_two_gloo_ranks_warmstart_zero_sharded_resume_is_exact(
        runner_dataset, base, tmp_path):
    from test_iql_all_data_ddp import _WORKER, _assert_identical_state
    package, _ = source_policy(tmp_path, base)
    config = iql_config(tmp_path, base[0], package)
    config["data"]["validation_fraction"] = 0
    config["training"].update(global_batch_size=4, zero_optimizer=True, checkpoint_format="sharded")
    config["distributed"]["backend"] = "gloo"
    config_path, worker_path = tmp_path / "config.yaml", tmp_path / "warmstart_gloo.py"
    config_path.write_text(yaml.safe_dump(config))
    hook = '''
import post_training.iql.actor_init as actor_init
original_initialize = actor_init.initialize_actor_from_policy
initialization_calls = []
def capture_initialization(actor, initialization):
    original_initialize(actor, initialization)
    assert all(torch.equal(value, torch.full_like(value, .25))
               for value in actor.trainable_state_dict().values())
    initialization_calls.append(1)
actor_init.initialize_actor_from_policy = capture_initialization
'''
    worker = _WORKER.replace('observed = {"rank": process_rank', hook + '\nobserved = {"rank": process_rank')
    worker = worker.replace('observed["summary"] = summary',
                            'observed["summary"] = summary\nobserved["initialization_calls"] = len(initialization_calls)')
    worker_path.write_text(worker)
    project = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
        "PYTHONPATH": os.pathsep.join([str(project / "src"), str(project / "tests"),
                                      os.environ.get("PYTHONPATH", "")])}
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               "--nproc_per_node=2", str(worker_path), str(config_path), str(runner_dataset)]
    directory = Path(config["output_dir"])

    def run(resume=None):
        result = subprocess.run(command + ([str(resume)] if resume else []),
            env=environment, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        ranks = [json.loads((directory / f"rank_{rank}_test_result.json").read_text()) for rank in range(2)]
        assert ranks[0]["steps"] == ranks[1]["steps"]
        assert all(r["initialization_calls"] == (0 if resume else 1) for r in ranks)
        assert all(r["summary"]["policy_optimizer_updates"] == 2 for r in ranks)
        return ranks

    run()
    final_path = directory / "checkpoints/update_00000002.pt"
    expected = load_snapshot(final_path)
    expected_shards = [load_snapshot(final_path.resolve().parent / f"rank_{rank:05d}.pt") for rank in range(2)]
    run(directory / "checkpoints/update_00000001.pt")
    actual = load_snapshot(final_path)
    _assert_identical_state(expected["actor"], actual["actor"])
    _assert_identical_state(expected["critic"], actual["critic"])
    for rank, saved in enumerate(expected_shards):
        restored = load_snapshot(final_path.resolve().parent / f"rank_{rank:05d}.pt")
        _assert_identical_state(saved["optimizers"], restored["optimizers"])
        torch.testing.assert_close(saved["rng"]["torch"], restored["rng"]["torch"], atol=0, rtol=0)
