"""Two real Gloo ranks exercise all-data IQL with only a tiny fake policy."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch
import yaml

from test_policy_export import base
from test_training import configuration, load_snapshot, runner_dataset


_WORKER = r'''
import hashlib
import json
from pathlib import Path
import sys

import torch
from post_training import diagnostics, policy, training
from post_training.config import load_config
from post_training.distributed import rank, unwrap
from post_training.pi05_assets import checkpoint_identity, load_normalization, platform_model_settings
from post_training.policy import Pi05Policy
from test_training import StochasticFakeModel

torch.set_num_threads(1)
config = load_config(sys.argv[1])
model_root = Path(config["model"]["checkpoint"])
identity = checkpoint_identity(model_root)
process_rank = int(__import__("os").environ["RANK"])

def fake_loader(model_config, device, *, verified_identity):
    assert identity == verified_identity
    return Pi05Policy(StochasticFakeModel(), lambda value: {"actions": value["actions"]},
        identity, platform_model_settings(model_config, identity), load_normalization(model_root),
        observation_factory=lambda value: value).to(device)

policy.load_policy = fake_loader
observed = {"rank": process_rank, "diagnostic_calls": [], "steps": []}
original_loader = training._loader
def capture_loader(dataset, *args, **kwargs):
    group = "diagnostic" if kwargs.get("validation") else "train"
    observed[group + "_chunks"] = len(dataset)
    observed[group + "_root_ids"] = sorted({row["root_run_id"] for row in dataset.records})
    assert all(row["split"] == "train" for row in dataset.records)
    return original_loader(dataset, *args, **kwargs)
training._loader = capture_loader

original_evaluate = diagnostics.evaluate_critic
def capture_evaluation(model, loader, *args, **kwargs):
    assert rank() == 0
    result = original_evaluate(model, loader, *args, **kwargs)
    observed["diagnostic_calls"].append({"count": result["count"],
        "spearman": result["spearman_v_remaining_blocks"]})
    return result
diagnostics.evaluate_critic = capture_evaluation

def model_hash(module, *, actor=False):
    digest = hashlib.sha256()
    values = unwrap(module).trainable_state_dict() if actor else unwrap(module).state_dict()
    for key, value in sorted(values.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

original_step = training.optimization_step
def capture_step(config, batches, actor, critic, optimizers, device, *, actor_update):
    assert set(optimizers) == {"actor", "q", "value"}
    assert config["training"]["zero_optimizer"] is True
    result = original_step(config, batches, actor, critic, optimizers, device, actor_update=actor_update)
    observed["steps"].append({"actor": model_hash(actor, actor=True), "critic": model_hash(critic)})
    return result
training.optimization_step = capture_step

summary = training.train(config, sys.argv[2], resume=sys.argv[3] if len(sys.argv) > 3 else None)
assert not torch.distributed.is_initialized()
observed["summary"] = summary
Path(config["output_dir"], f"rank_{process_rank}_test_result.json").write_text(json.dumps(observed))
'''


def _assert_identical_state(expected, actual):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            _assert_identical_state(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual)
        for first, second in zip(expected, actual):
            _assert_identical_state(first, second)
    else:
        assert expected == actual


def test_two_process_all_data_iql_zero_sharded_resume(
        runner_dataset, tmp_path, base):
    config = configuration(tmp_path, "iql", updates=2)
    config["data"]["validation_fraction"] = 0
    config["model"]["checkpoint"] = str(base[0])
    config["critic"]["warmup_updates"] = 0
    config["training"].update(global_batch_size=4, zero_optimizer=True, checkpoint_format="sharded")
    config["distributed"]["backend"] = "gloo"
    config_path, worker_path = tmp_path / "config.yaml", tmp_path / "tiny_iql_worker.py"
    config_path.write_text(yaml.safe_dump(config))
    worker_path.write_text(_WORKER)
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
        records = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
        assert [row["optimizer_update"] for row in records] == [1, 2]
        for row in records:
            assert "validation" not in row
            assert row["train_diagnostic"]["count"] == 8
            assert row["train_diagnostic"]["split"] == "train"
            assert row["train_diagnostic"]["held_out"] is False
            assert 0 < row["actor_weight_mean"] <= config["critic"]["max_weight"]
            assert all(math.isfinite(row[key]) for key in ("actor_loss", "q_loss", "v_loss"))
        for process_rank, rank_result in enumerate(ranks):
            assert rank_result["rank"] == process_rank
            assert rank_result["train_chunks"] == 8
            assert len(rank_result["train_root_ids"]) == 4
            assert rank_result["summary"]["status"] == "COMPLETED"
            assert rank_result["summary"]["policy_optimizer_updates"] == 2
            assert rank_result["summary"]["critic_optimizer_updates"] == 2
            assert rank_result["summary"]["best_validation_spearman"] is None
            assert rank_result["summary"]["best_critic_update"] is None
            assert rank_result["summary"]["last_critic_diagnostic_spearman"] == records[-1]["train_diagnostic"]["spearman_v_remaining_blocks"]
            assert len(rank_result["steps"]) == (1 if resume else 2)
        assert ranks[0]["diagnostic_chunks"] == 8
        assert ranks[0]["diagnostic_root_ids"] == ranks[0]["train_root_ids"]
        assert len(ranks[0]["diagnostic_calls"]) == (1 if resume else 2)
        assert all(call["count"] == 8 for call in ranks[0]["diagnostic_calls"])
        assert ranks[1]["diagnostic_calls"] == []
        assert ranks[0]["steps"] == ranks[1]["steps"]
        assert not (directory / "critic_best.pt").exists()
        return ranks

    initial_ranks = run()
    final_path = directory / "checkpoints/update_00000002.pt"
    expected = load_snapshot(final_path)
    expected_shards = [load_snapshot(final_path.resolve().parent / f"rank_{rank:05d}.pt") for rank in range(2)]
    assert expected["contract"]["world_size"] == 2
    assert expected["contract"]["gradient_accumulation_steps"] == 2
    resumed_ranks = run(directory / "checkpoints/update_00000001.pt")
    actual = load_snapshot(final_path)
    for kind in ("actor", "critic"):
        _assert_identical_state(expected[kind], actual[kind])
    for process_rank, expected_shard in enumerate(expected_shards):
        actual_shard = load_snapshot(final_path.resolve().parent / f"rank_{process_rank:05d}.pt")
        _assert_identical_state(expected_shard["optimizers"], actual_shard["optimizers"])
        torch.testing.assert_close(expected_shard["rng"]["torch"], actual_shard["rng"]["torch"], atol=0, rtol=0)
        assert initial_ranks[process_rank]["steps"][-1] == resumed_ranks[process_rank]["steps"][-1]
