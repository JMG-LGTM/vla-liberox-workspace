"""Cross-module tests from portable native data through mmap replay to a fake actor."""
from __future__ import annotations

import copy
import json
import shutil
import zipfile

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import default_collate

from post_training import archive_io
from post_training.config import DEFAULTS
from post_training.data import package_dataset, safe_path, sha256_file
from post_training.pi05_assets import (NORM_FILE, OPENPI_COMMIT, SOURCE, SOURCE_REVISION,
                                      platform_model_settings)
from post_training.policy import Pi05Policy
from post_training.replay import CachedReplayDataset, build_replay_cache, normalize


def _native_record(root, name, success):
    directory = root / "projects/libero_x_vla/runs" / name
    episode = directory / "episodes/episode_000"
    episode.mkdir(parents=True)
    run = dict(id=name, status="COMPLETED", error=None, kind="original", task="move bowl",
               task_id="scene23", root_session_id=name, parent_session_id=None,
               historical_observations_path="/unavailable/original/platform/trajectory_observations.npz")
    (directory / "run.json").write_text(json.dumps(run))
    count = 10
    actions = (np.arange(count * 7, dtype=np.float32).reshape(count, 7) - 20) / 31
    state = np.arange((count + 1) * 8, dtype=np.float32).reshape(count + 1, 8) / 19
    done = np.zeros(count, bool)
    if success:
        done[4:9] = True
    np.savez_compressed(episode / "trajectory.npz", env_action=actions, done=done,
                        action_source=np.full(count, "policy"), eef_position=state[:, :3],
                        eef_axis_angle=state[:, 3:6], gripper_qpos=state[:, 6:],
                        metadata_json=json.dumps(dict(session_id=name, stopped_reason="max_steps", error=None)))
    pixels = np.arange((count + 1) * 9 * 11 * 3, dtype=np.uint8).reshape(count + 1, 9, 11, 3)
    np.savez_compressed(episode / "trajectory_observations.npz", agentview_image=pixels, wrist_image=255 - pixels)
    return actions, state, pixels


class _FlowProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.paligemma = nn.Linear(1, 1)
        self.flow_scale = nn.Parameter(torch.ones(()))

    def gradient_checkpointing_enable(self):
        pass

    def forward(self, observation, actions):
        assert actions.dtype == torch.float32 and actions.shape[1:] == (10, 32)
        return actions.square() * self.flow_scale


def test_cached_native_records_enter_policy_without_extra_flip_or_normalization(tmp_path):
    source = tmp_path / "platform-source"
    expected = {name: _native_record(source, name, success) for name, success in (("failed", False), ("succeeded", True))}
    package = tmp_path / "portable"
    package_dataset(source, package, run_ids=list(expected))
    shutil.rmtree(source)  # Native metadata paths must never be consulted by training.
    config = copy.deepcopy(DEFAULTS)
    config.update(method="iql")
    config["data"].update(validation_fraction=.5)
    config["critic"]["image_size"] = 9
    normalization = {key: {"q01": [-2.] * size, "q99": [3.] * size, "codec": "openpi_quantile"}
                     for key, size in (("actions", 7), ("state", 8))}
    cache, _ = build_replay_cache(package, tmp_path / "cache", config, norm_stats=normalization)
    dataset = CachedReplayDataset(cache, split=None)
    samples = [dataset[index] for index in range(len(dataset))]
    assert len(samples) == 4
    for sample in samples:
        actions, state, pixels = expected[sample["run_id"]]
        start, end = sample["start"], sample["end"]
        length = end - start
        np.testing.assert_array_equal(sample["agent_image"], pixels[start, ::-1])
        np.testing.assert_array_equal(sample["wrist_image"], (255 - pixels[start, ::-1]))
        np.testing.assert_array_equal(sample["raw_actions"][:length], actions[start:end])
        np.testing.assert_array_equal(sample["raw_proprio"], state[start])
        np.testing.assert_allclose(sample["next_proprio"], normalize(state[end], normalization["state"]))
        assert sample["action_mask"].sum() == length
        assert torch.count_nonzero(sample["raw_actions"][length:]) == 0
        if sample["run_id"] == "succeeded" and end == 9:
            assert sample["bootstrap_mask"] == 0 and sample["reward"] == 0
        if sample["run_id"] == "failed" and end == 10:
            assert sample["bootstrap_mask"] == 1 and sample["reward"] == -1
    seen = []
    def transform(value):
        seen.append(value)
        return {"actions": np.pad(value["actions"], ((0, 0), (0, 25))).astype(np.float64)}
    identity = {"schema_version": 1, "family": "pi05", "openpi_commit": OPENPI_COMMIT,
                "source": SOURCE, "source_revision": SOURCE_REVISION, "config_name": "pi05_libero",
                "conversion_precision": "bfloat16", "native_action_horizon": 10,
                "native_action_dim": 32, "discrete_state_input": False,
                "files": {"model.safetensors": "a" * 64, NORM_FILE: "b" * 64}}
    settings = platform_model_settings({"checkpoint": "unused-fake-base"}, identity)
    policy = Pi05Policy(_FlowProbe(), transform, identity, settings, normalization,
                        observation_factory=lambda value: value)
    losses = policy(default_collate(samples))
    expected_losses = torch.stack([sample["raw_actions"][sample["action_mask"]].square().mean() for sample in samples])
    torch.testing.assert_close(losses, expected_losses)
    losses.mean().backward()
    assert policy.model.flow_scale.grad is not None
    for sample, transformed in zip(samples, seen):
        np.testing.assert_array_equal(transformed["observation/image"], sample["agent_image"])
        np.testing.assert_array_equal(transformed["observation/wrist_image"], sample["wrist_image"])
        np.testing.assert_array_equal(transformed["actions"][:8], sample["raw_actions"])
        assert not transformed["actions"][8:].any()


def test_crc_recovery_is_bounded_and_non_crc_errors_do_not_retry(tmp_path, monkeypatch):
    path = tmp_path / "recording.npz"
    np.savez_compressed(path, x=np.arange(3))
    digest = sha256_file(path)
    calls = []
    def always_crc(*args, **kwargs):
        calls.append(args[0])
        raise zipfile.BadZipFile("Bad CRC-32 for file 'x.npy'")
    monkeypatch.setattr(np, "load", always_crc)
    with pytest.raises(zipfile.BadZipFile, match="retry failed"):
        archive_io.load_npz_arrays(path, expected_sha256=digest)
    assert len(calls) == 2
    calls.clear()
    def invalid_archive(*args, **kwargs):
        calls.append(args[0])
        raise zipfile.BadZipFile("File is not a zip file")
    monkeypatch.setattr(np, "load", invalid_archive)
    with pytest.raises(zipfile.BadZipFile, match="not a zip"):
        archive_io.load_npz_arrays(path, expected_sha256=digest)
    assert len(calls) == 1


def test_portable_paths_reject_traversal_and_linked_assets(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    (dataset / "linked").symlink_to(outside)
    for relative in ("../outside", str(outside), "linked"):
        with pytest.raises(ValueError):
            safe_path(dataset, relative)
