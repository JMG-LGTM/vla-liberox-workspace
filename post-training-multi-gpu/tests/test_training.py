"""Actual CPU runner coverage; policy paths use only a tiny fake OpenPI model."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest
import torch
import yaml

from post_training.config import DEFAULTS, validate_config
from post_training.critic import load_critic
from post_training.data import package_dataset
from post_training.distributed import OptimizerGroup
from post_training.export import validate_policy_package
from post_training.pi05_assets import load_normalization, platform_model_settings
from post_training.policy import Pi05Policy
from post_training import training
from test_data_replay import make_run
from test_policy_export import FakeModel, base  # Reuse the synthetic identity fixture.


@pytest.fixture
def runner_dataset(tmp_path):
    source, output = tmp_path / "source", tmp_path / "dataset"
    runs = ["success_a", "success_b", "success_c", "success_d"]
    for index, run in enumerate(runs):
        make_run(source, run, offset=.1 * index)
    package_dataset(source, output, run_ids=runs, horizon=8)
    return output


def configuration(tmp_path, method="critic", *, updates=3):
    config = copy.deepcopy(DEFAULTS)
    config.update(method=method, output_dir=str(tmp_path / "run"), cache_dir=str(tmp_path / "cache"))
    config["data"].update(validation_fraction=.5)
    config["training"].update(optimizer_updates=updates, global_batch_size=2, per_device_batch_size=1,
        warmup_updates=0, checkpoint_interval=1, log_interval=1, validation_interval=1,
        tensorboard=False, zero_optimizer=False)
    config["critic"].update(image_size=16, random_shift_pad=4,
                            warmup_updates=1 if method == "iql" else 0)
    config["distributed"].update(device="cpu", workers_per_rank=0, pin_memory=False)
    return config


class StochasticFakeModel(FakeModel):
    def forward(self, observation, actions):
        return super().forward(observation, actions + .05 * torch.randn_like(actions))


def install_fake_policy(monkeypatch, model_base):
    root, identity = model_base
    def fake_loader(model_config, device, *, verified_identity):
        assert verified_identity == identity
        settings = platform_model_settings(model_config, identity)
        return Pi05Policy(StochasticFakeModel(), lambda value: {"actions": value["actions"]},
            identity, settings, load_normalization(root), observation_factory=lambda value: value).to(device)
    monkeypatch.setattr("post_training.policy.load_policy", fake_loader)
    return root


def load_snapshot(path):
    # Locally created test checkpoint; resume state includes Python/NumPy RNG tuples.
    return torch.load(path, weights_only=False)


@pytest.mark.parametrize("method", ["critic", "bc", "iql"])
@pytest.mark.parametrize("checkpoint_format", ["consolidated", "sharded"])
def test_runner_stop_and_resume_reproduces_uninterrupted_models(
        runner_dataset, tmp_path, monkeypatch, base, method, checkpoint_format):
    model_root = install_fake_policy(monkeypatch, base)
    config = configuration(tmp_path, method)
    config["training"]["checkpoint_format"] = checkpoint_format
    config["model"]["checkpoint"] = str(model_root)
    config["output_dir"] = str(tmp_path / "uninterrupted")
    full = training.train(config, runner_dataset)
    expected = load_snapshot(full["checkpoint"])
    config["output_dir"] = str(tmp_path / "resumed")
    original = training.optimization_step
    calls = []
    def request_stop_after_one(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(1)
        (Path(config["output_dir"]) / "stop.request").write_text("test stop\n")
        return result
    monkeypatch.setattr(training, "optimization_step", request_stop_after_one)
    paused = training.train(config, runner_dataset)
    assert paused["status"] == "STOPPED"
    assert paused["optimizer_updates"] == len(calls) == 1
    assert paused["policy"] is None
    monkeypatch.setattr(training, "optimization_step", original)
    resumed = training.train(config, runner_dataset, resume=paused["checkpoint"])
    actual = load_snapshot(resumed["checkpoint"])
    assert resumed["status"] == "COMPLETED"
    assert resumed["memory_metrics"]["host_peak_rss_gib_rank0"] > 0
    assert resumed["memory_metrics"]["peak_cuda_memory_gib_max"] == 0
    memory = [json.loads(line) for line in (Path(config["output_dir"]) / "memory.jsonl").read_text().splitlines()]
    assert any(record["stage"] == "after_checkpoint" for record in memory)
    assert memory[-1]["stage"] == "after_export"
    assert not (Path(config["output_dir"]) / "stop.request").exists()
    assert resumed["optimizer_updates"] == full["optimizer_updates"] == (4 if method == "iql" else 3)
    for kind in ("actor", "critic"):
        if expected[kind] is None:
            assert actual[kind] is None
        else:
            assert expected[kind].keys() == actual[kind].keys()
            for name, value in expected[kind].items():
                torch.testing.assert_close(value, actual[kind][name], atol=0, rtol=0)
    records = [json.loads(line) for line in (Path(config["output_dir"]) / "metrics.jsonl").read_text().splitlines()]
    assert [record["optimizer_update"] for record in records] == list(range(1, resumed["optimizer_updates"] + 1))
    assert records[-1]["micro_batch_count"] == resumed["optimizer_updates"] * 2
    if method != "bc":
        model, metadata = load_critic(Path(config["output_dir"]) / "critic_last.pt")
        assert metadata["provenance"]["update"] == resumed["optimizer_updates"]
        assert records[-1]["validation"]["count"] == 4
        assert "spearman_v_remaining_blocks" in records[-1]["validation"]
        assert all(torch.equal(model.state_dict()[key], actual["critic"][key]) for key in actual["critic"])
    if method != "critic":
        payload = validate_policy_package(Path(resumed["policy"]).parent)
        assert payload["algorithm"] == method
        assert payload["training_step"] == 3
        assert records[-1]["policy_optimizer_updates"] == 3
    if method == "iql":
        assert records[0]["policy_optimizer_updates"] == 0
        assert "actor_loss" not in records[0]
        assert all("actor_weight_mean" in record for record in records[1:])


@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGINT], ids=["SIGTERM", "SIGINT"])
def test_real_signal_during_accumulation_saves_complete_update_and_resumes(
        runner_dataset, tmp_path, monkeypatch, stop_signal):
    config = configuration(tmp_path, updates=2)
    config["output_dir"] = str(tmp_path / "reference")
    full = training.train(config, runner_dataset)
    expected_first = load_snapshot(Path(config["output_dir"]) / "checkpoints/update_00000001.pt")
    expected_final = load_snapshot(full["checkpoint"])
    config["output_dir"] = str(tmp_path / "signalled")
    previous_handlers = {kind: signal.getsignal(kind) for kind in (signal.SIGINT, signal.SIGTERM)}
    original_step = training.optimization_step
    gradient_calls = []

    def signal_inside_first_microbatch(*args, **kwargs):
        critic = args[3]
        def on_value_gradient(gradient):
            gradient_calls.append(1)
            if len(gradient_calls) == 1:
                # This hook runs inside optimization_step's first backward,
                # before the second accumulation microbatch or either step.
                assert callable(signal.getsignal(stop_signal))
                assert signal.getsignal(stop_signal) is not previous_handlers[stop_signal]
                os.kill(os.getpid(), stop_signal)
            return gradient
        hook = next(critic.value.parameters()).register_hook(on_value_gradient)
        try:
            return original_step(*args, **kwargs)
        finally:
            hook.remove()

    monkeypatch.setattr(training, "optimization_step", signal_inside_first_microbatch)
    paused = training.train(config, runner_dataset)
    assert paused["status"] == "STOPPED"
    assert paused["optimizer_updates"] == 1
    assert paused["micro_batches"] == 2
    assert len(gradient_calls) == 2  # Both V accumulation microbatches completed.
    assert previous_handlers == {kind: signal.getsignal(kind) for kind in previous_handlers}
    boundary = load_snapshot(paused["checkpoint"])
    assert boundary["accumulation_boundary"] is True
    for name, value in expected_first["critic"].items():
        torch.testing.assert_close(value, boundary["critic"][name], atol=0, rtol=0)
    for optimizer in boundary["optimizers"].values():
        for state in optimizer["states"]:
            assert all(parameter["step"].item() == 1 for parameter in state["state"].values())
    monkeypatch.setattr(training, "optimization_step", original_step)
    resumed = training.train(config, runner_dataset, resume=paused["checkpoint"])
    assert resumed["status"] == "COMPLETED"
    assert resumed["optimizer_updates"] == 2
    actual = load_snapshot(resumed["checkpoint"])
    for name, value in expected_final["critic"].items():
        torch.testing.assert_close(value, actual["critic"][name], atol=0, rtol=0)
    assert previous_handlers == {kind: signal.getsignal(kind) for kind in previous_handlers}


def test_bc_stop_on_final_update_completes_checkpoint_and_exports_policy(
        runner_dataset, tmp_path, monkeypatch, base):
    """SIGTERM at the final optimizer boundary must not strand a finished actor."""
    model_root = install_fake_policy(monkeypatch, base)
    config = configuration(tmp_path, method="bc", updates=2)
    config["model"]["checkpoint"] = str(model_root)
    config["output_dir"] = str(tmp_path / "reference_final")
    reference = training.train(config, runner_dataset)
    expected = load_snapshot(reference["checkpoint"])

    config["output_dir"] = str(tmp_path / "signal_on_final")
    original_step = training.optimization_step
    updates = []

    def signal_after_final_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        updates.append(1)
        if len(updates) == config["training"]["optimizer_updates"]:
            os.kill(os.getpid(), signal.SIGTERM)
        return result

    monkeypatch.setattr(training, "optimization_step", signal_after_final_step)
    completed = training.train(config, runner_dataset)
    assert completed["status"] == "COMPLETED"
    assert completed["optimizer_updates"] == completed["policy_optimizer_updates"] == 2
    assert completed["critic_optimizer_updates"] == 0
    assert len(updates) == 2
    actual = load_snapshot(completed["checkpoint"])
    assert actual["update"] == 2 and actual["accumulation_boundary"] is True
    for name, value in expected["actor"].items():
        torch.testing.assert_close(actual["actor"][name], value, atol=0, rtol=0)
    policy = validate_policy_package(Path(completed["policy"]).parent)
    assert policy["algorithm"] == "bc" and policy["training_step"] == 2
    metrics = [json.loads(line) for line in
               (Path(config["output_dir"]) / "metrics.jsonl").read_text().splitlines()]
    assert [record["optimizer_update"] for record in metrics] == [1, 2]


def test_critic_runner_in_fresh_process_never_imports_policy_or_openpi(runner_dataset, tmp_path):
    config = configuration(tmp_path, updates=1)
    config["model"]["checkpoint"] = "/deliberately/missing/policy"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    script = """
import importlib.abc, json, sys
class NoPolicy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'post_training.policy' or fullname.startswith('openpi'):
            raise AssertionError('critic imported policy: ' + fullname)
sys.meta_path.insert(0, NoPolicy())
from post_training.config import load_config
from post_training.training import train
result = train(load_config(sys.argv[1]), sys.argv[2])
assert result['status'] == 'COMPLETED'
assert 'post_training.policy' not in sys.modules
assert not any(name.startswith('openpi') for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", script, str(path), str(runner_dataset)],
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_nonfinite_loss_with_finite_gradients_never_steps_actor(tmp_path):
    class InvalidActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))
        def forward(self, batch):
            return self.weight.expand(2) + float("inf")
    actor = InvalidActor()
    optimizer = OptimizerGroup(actor.parameters(), zero=False, lr=.1)
    with pytest.raises(FloatingPointError, match="Nonfinite loss"):
        training.optimization_step(configuration(tmp_path, "bc"),
            [{"action_mask": torch.ones(2, 8, dtype=torch.bool)}], actor, None,
            {"actor": optimizer}, torch.device("cpu"), actor_update=0)
    assert actor.weight.item() == 1.
    assert not optimizer.optimizers[0].state


def test_artifact_rollback_restores_best_alias_and_unique_metrics(tmp_path):
    directory = tmp_path / "run"
    (directory / "critics").mkdir(parents=True)
    (directory / "critics/best_update_00000001.pt").write_bytes(b"best-at-checkpoint")
    (directory / "critics/best_update_00000003.pt").write_bytes(b"future-best")
    (directory / "critic_best.pt").write_bytes(b"future-best")
    (directory / "metrics.jsonl").write_text(
        ''.join(json.dumps({"optimizer_update": update}) + '\n' for update in range(1, 4)) + '{"partial"')
    (directory / "stop.request").touch()
    training._restore_run_artifacts(directory, 1, {"best_critic_export": "critics/best_update_00000001.pt"})
    assert (directory / "critic_best.pt").read_bytes() == b"best-at-checkpoint"
    assert (directory / "critics/best_update_00000003.pt").exists()
    assert [json.loads(line)["optimizer_update"] for line in (directory / "metrics.jsonl").read_text().splitlines()] == [1]
    assert not (directory / "stop.request").exists()


@pytest.mark.parametrize("method", ["critic", "bc", "iql"])
def test_completed_run_can_rollback_to_older_checkpoint_without_stale_exports(
        runner_dataset, tmp_path, monkeypatch, base, method):
    config = configuration(tmp_path, method)
    config["model"]["checkpoint"] = str(install_fake_policy(monkeypatch, base))
    result = training.train(config, runner_dataset)
    expected = load_snapshot(result["checkpoint"])
    directory = Path(config["output_dir"])
    earlier = directory / "checkpoints/update_00000001.pt"
    replayed = training.train(config, runner_dataset, resume=earlier)
    actual = load_snapshot(replayed["checkpoint"])
    for kind in ("actor", "critic"):
        if expected[kind] is not None:
            for name, value in expected[kind].items():
                torch.testing.assert_close(value, actual[kind][name], atol=0, rtol=0)
    records = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    assert [record["optimizer_update"] for record in records] == list(range(1, replayed["optimizer_updates"] + 1))
    retired = list((directory / "retired").iterdir())
    assert len(retired) == 1
    assert json.loads((retired[0] / "summary.json").read_text())["status"] == "COMPLETED"
    if method != "critic":
        validate_policy_package(retired[0] / "policy")
        assert validate_policy_package(directory / "policy")["training_step"] == 3
    if method != "bc":
        assert (retired[0] / "critic_last.pt").is_file()
        if actual["metadata"]["best_metric"] is not None:
            _, best = load_critic(directory / "critic_best.pt")
            assert best["provenance"]["update"] == actual["metadata"]["best_update"]
            assert best["provenance"]["validation_metric"] == actual["metadata"]["best_metric"]


def test_dry_run_critic_init_checks_metadata_without_constructing_network(
        runner_dataset, tmp_path, monkeypatch):
    config = configuration(tmp_path, updates=1)
    training.train(config, runner_dataset)
    export = Path(config["output_dir"]) / "critic_last.pt"
    config["critic"]["init_checkpoint"] = str(export)
    config["data"]["normalization"] = str(Path(config["output_dir"]) / "normalization.json")
    import post_training.critic as critic_module
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run must not construct a critic")
    monkeypatch.setattr(critic_module, "PixelIQL", forbidden)
    state = torch.get_rng_state().clone()
    report = training.dry_run(config, runner_dataset, 1)
    assert report["policy_loaded"] is False
    assert torch.equal(torch.get_rng_state(), state)
    malformed = tmp_path / "wrong_normalization.json"
    stats = json.loads(Path(config["data"]["normalization"]).read_text())
    stats["actions"]["q01"][0] -= .1
    malformed.write_text(json.dumps(stats))
    config["data"]["normalization"] = str(malformed)
    with pytest.raises(ValueError, match="normalization"):
        training.dry_run(config, runner_dataset, 1)


@pytest.mark.parametrize("checkpoint_format", ["consolidated", "sharded"])
def test_two_process_torchrun_critic_zero_runner_resume(runner_dataset, tmp_path, checkpoint_format):
    config = configuration(tmp_path, updates=2)
    config["training"].update(global_batch_size=4, zero_optimizer=True, checkpoint_format=checkpoint_format)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    project = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               "--nproc_per_node=2", str(project / "train.py"), "--config", str(path),
               "--data", str(runner_dataset)]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
    initial = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=90)
    assert initial.returncode == 0, initial.stdout + initial.stderr
    directory = Path(config["output_dir"])
    final_path = directory / "checkpoints/update_00000002.pt"
    before = load_snapshot(final_path)
    resumed = subprocess.run(command + ["--resume", str(directory / "checkpoints/update_00000001.pt")],
                              env=environment, capture_output=True, text=True, timeout=90)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    after = load_snapshot(final_path)
    assert before["contract"]["world_size"] == 2
    assert before["contract"]["gradient_accumulation_steps"] == 2
    for name, value in before["critic"].items():
        torch.testing.assert_close(value, after["critic"][name], atol=0, rtol=0)
    assert json.loads((directory / "summary.json").read_text())["status"] == "COMPLETED"


@pytest.mark.parametrize("section,field,value", [("training", "weight_decay", -1.),
    ("training", "tensorboard", "false"), ("critic", "expectile", True),
    ("critic", "tau", "0.005"), ("critic", "grad_clip", float("nan")),
    ("data", "validation_fraction", True)])
def test_config_rejects_ambiguous_types_and_nonfinite_math(tmp_path, section, field, value):
    config = configuration(tmp_path)
    config[section][field] = value
    with pytest.raises(ValueError):
        validate_config(config)
