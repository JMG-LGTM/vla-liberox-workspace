"""All-data IQL uses training diagnostics, retaining critic pretraining history."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from post_training import training
from post_training.critic import read_critic_metadata
from post_training.data import inspect_dataset, package_dataset
from post_training.io import sha256_file, stable_hash
from post_training.replay import CachedReplayDataset
from test_policy_export import base
from test_training import configuration, install_fake_policy, load_snapshot, runner_dataset


def assert_all_data_report(report, count):
    assert report["dataset_chunks"] == report["train_chunks"] == count
    assert report["validation_chunks"] == 0
    assert report["critic_evaluation"] == dict(metric_group="train_diagnostic", split="train",
        held_out=False, chunks=count, selects_best_checkpoint=False)


def test_all_data_iql_inspection_and_dry_run_need_no_holdout_or_policy_loading(
        runner_dataset, tmp_path, monkeypatch, base):
    config = configuration(tmp_path, "iql", updates=1)
    config["data"]["validation_fraction"] = 0
    config["model"]["checkpoint"] = str(base[0])
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run must not instantiate a policy or critic")
    monkeypatch.setattr("post_training.policy.load_policy", forbidden)
    monkeypatch.setattr(training, "_new_critic", forbidden)
    info = inspect_dataset(runner_dataset, config)
    assert_all_data_report(info, 8)
    assert {row["split"] for row in info["rows"]} == {"train"}
    report = training.dry_run(config, runner_dataset, 1)
    assert_all_data_report(report, 8)
    assert report["policy_loaded"] is False
    assert report["critic_initialization"] is None
    assert not Path(config["cache_dir"]).exists()
    assert not Path(config["output_dir"]).exists()


@pytest.mark.parametrize("initialize_critic", [False, True], ids=["fresh", "pretrained_best"])
def test_all_data_iql_runs_and_resumes_without_selecting_a_training_best(
        runner_dataset, tmp_path, monkeypatch, base, initialize_critic):
    # Force a finite negative score: diagnostics must never select a best export,
    # even when Spearman would satisfy the held-out checkpoint criterion.
    from post_training import diagnostics
    original_evaluate = diagnostics.evaluate_critic
    def negative_spearman(model, batches, *args, **kwargs):
        result = original_evaluate(model, batches, *args, **kwargs)
        result["spearman_v_remaining_blocks"] = -.85
        return result
    monkeypatch.setattr(diagnostics, "evaluate_critic", negative_spearman)

    config = configuration(tmp_path, "iql", updates=2)
    config["data"]["validation_fraction"] = 0
    config["critic"]["warmup_updates"] = 0
    config["model"]["checkpoint"] = str(install_fake_policy(monkeypatch, base))
    data_root, count, expected_initial, historical = runner_dataset, 8, None, None
    if initialize_critic:
        pretrain = configuration(tmp_path, "critic", updates=1)
        pretrain["output_dir"] = str(tmp_path / "pretrain")
        pretrain["cache_dir"] = str(tmp_path / "pretrain_cache")
        pretrained = training.train(pretrain, runner_dataset)
        assert pretrained["best_validation_spearman"] == -.85
        assert pretrained["best_critic_update"] == 1
        best = Path(pretrain["output_dir"]) / "critic_best.pt"
        historical = read_critic_metadata(best)
        assert historical["provenance"]["critic_evaluation"]["held_out"] is True
        assert historical["provenance"]["validation_metric"] == -.85
        expected_initial = torch.load(best, weights_only=True)["state_dict"]
        config["critic"]["init_checkpoint"] = str(best)
        # The policy dataset can be a subset of critic pretraining's members.
        data_root = tmp_path / "iql_subset"
        package_dataset(tmp_path / "source", data_root,
                        run_ids=["success_a", "success_b"], horizon=8)
        count = 4
        assert_all_data_report(training.dry_run(config, data_root, 1), count)

    loaders = {}
    original_loader = training._loader
    def capture_loader(dataset, *args, **kwargs):
        loaders["diagnostic" if kwargs.get("validation") else "train"] = dataset
        return original_loader(dataset, *args, **kwargs)
    monkeypatch.setattr(training, "_loader", capture_loader)
    original_step = training.optimization_step
    first_step = []
    def check_critic_initialization(*args, **kwargs):
        if not first_step and expected_initial is not None:
            for key, value in args[3].state_dict().items():
                torch.testing.assert_close(value.cpu(), expected_initial[key], atol=0, rtol=0)
        first_step.append(True)
        return original_step(*args, **kwargs)
    monkeypatch.setattr(training, "optimization_step", check_critic_initialization)

    result = training.train(config, data_root)
    assert len(loaders["train"]) == len(loaders["diagnostic"]) == count
    assert loaders["train"].records == loaders["diagnostic"].records
    assert all(row["split"] == "train" for row in loaders["train"].records)
    assert_all_data_report(result, count)
    assert result["policy_optimizer_updates"] == result["critic_optimizer_updates"] == 2
    assert result["best_validation_spearman"] is result["best_critic_update"] is None
    assert result["last_critic_diagnostic_spearman"] == -.85
    directory = Path(config["output_dir"])
    assert not (directory / "critic_best.pt").exists()
    assert not (directory / "critics").exists()
    provenance = json.loads((directory / "provenance.json").read_text())
    assert_all_data_report(provenance, count)
    assert len(CachedReplayDataset(provenance["cache"], split="val")) == 0
    if initialize_critic:
        assert provenance["normalization"] == historical["normalization"]
        initial = provenance["critic_initialization"]
        assert initial["metadata_sha256"] == stable_hash(historical)
        assert initial["artifact_sha256"] == sha256_file(config["critic"]["init_checkpoint"])
        assert initial["pretraining_provenance"] == historical["provenance"]
        assert initial["pretraining_provenance"]["dataset_sha256"] != provenance["dataset_sha256"]
    last = read_critic_metadata(directory / "critic_last.pt")
    assert_all_data_report(last["provenance"], count)
    assert last["provenance"]["validation_metric"] is None
    assert last["provenance"]["diagnostic_metric"] == -.85
    records = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    assert all("validation" not in record for record in records)
    assert all(record["train_diagnostic"]["count"] == count for record in records)
    assert all(record["train_diagnostic"]["split"] == "train" for record in records)
    assert all(record["train_diagnostic"]["held_out"] is False for record in records)
    assert all(record["train_diagnostic"]["selection_direction"] is None for record in records)

    expected = load_snapshot(result["checkpoint"])
    first = directory / "checkpoints/update_00000001.pt"
    replayed = training.train(config, data_root, resume=first)
    actual = load_snapshot(replayed["checkpoint"])
    for kind in ("actor", "critic"):
        for key, value in expected[kind].items():
            torch.testing.assert_close(value, actual[kind][key], atol=0, rtol=0)
    assert replayed["best_validation_spearman"] is replayed["best_critic_update"] is None
    assert not (directory / "critic_best.pt").exists()
    assert actual["metadata"]["last_validation_metric"] is None
    assert actual["metadata"]["last_validation_update"] is None
    assert actual["metadata"]["last_diagnostic_update"] == 2

    if initialize_critic:
        # A path alone cannot identify initial weights. Modifying tensors while
        # retaining identical metadata must invalidate the exact-resume contract.
        payload = torch.load(config["critic"]["init_checkpoint"], weights_only=True)
        first_key = next(iter(payload["state_dict"]))
        payload["state_dict"][first_key] += 1
        torch.save(payload, config["critic"]["init_checkpoint"])
        assert read_critic_metadata(config["critic"]["init_checkpoint"]) == historical
        with pytest.raises(RuntimeError, match="Resume contract changed"):
            training.train(config, data_root, resume=first)
