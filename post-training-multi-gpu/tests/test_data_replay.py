"""CPU checks of portable records, truncation semantics, mmap replay and integrity."""
from copy import deepcopy
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch

from post_training import archive_io
from post_training.data import (add_episode_chunks, confirmed_terminal_step,
    grouped_split, inspect_dataset, iter_unique_replay_chunks, package_dataset,
    sha256_file, verify_dataset)
from post_training.replay import CachedReplayDataset, bootstrap_mask, build_replay_cache, validate_cache


def make_run(root: Path, run_id: str, *, success: bool = True, offset: float = 0):
    directory = root / "projects/libero_x_vla/runs" / run_id
    episode = directory / "episodes/episode_000"
    episode.mkdir(parents=True)
    run = dict(id=run_id, status="COMPLETED", error=None, kind="original", task="place bowl",
               task_id="scene23", root_session_id=run_id, parent_session_id=None)
    (directory / "run.json").write_text(json.dumps(run))
    count = 10
    done = np.zeros(count, dtype=bool)
    if success:
        done[4:9] = True  # Confirmation at 8, one recorded post-success action.
    np.savez_compressed(episode / "trajectory.npz", env_action=np.full((count, 7), offset, np.float32),
        done=done, action_source=np.full(count, "policy"),
        eef_position=np.full((count + 1, 3), offset, np.float32),
        eef_axis_angle=np.full((count + 1, 3), offset, np.float32),
        gripper_qpos=np.full((count + 1, 2), offset, np.float32),
        metadata_json=json.dumps(dict(session_id=run_id, stopped_reason="max_steps", error=None)))
    pixels = np.arange((count + 1) * 8 * 8 * 3, dtype=np.uint8).reshape(count + 1, 8, 8, 3)
    np.savez_compressed(episode / "trajectory_observations.npz", agentview_image=pixels, wrist_image=pixels)
    return directory


@pytest.fixture
def dataset(tmp_path):
    source = tmp_path / "source"
    make_run(source, "success", success=True, offset=.1)
    make_run(source, "failure", success=False, offset=.9)
    output = tmp_path / "dataset"
    package_dataset(source, output, run_ids=["success", "failure"], horizon=4)
    return output


def test_success_confirmation_and_truncation():
    assert confirmed_terminal_step(np.array([0, 1, 1, 1, 1, 0, 1]), 5) is None
    assert confirmed_terminal_step(np.array([0, 1, 1, 1, 1, 1, 0]), 5) == 5
    assert bootstrap_mask(None, 10, 10) == 1
    assert bootstrap_mask(8, 9, 10) == 0
    assert bootstrap_mask(8, 8, 10) == 1


def test_package_portable_integrity_and_chunk_boundaries(dataset, tmp_path):
    manifest = verify_dataset(dataset)
    assert manifest["member_count"] == 2
    assert all(not Path(path).is_absolute() for path in manifest["files"])
    info = inspect_dataset(dataset, {"data": {"action_horizon": 4}})
    assert info["count"] == 6
    assert info["train_count"] == 6
    success = [row for row in info["rows"] if row["run_id"] == "success"]
    assert [(row["start"], row["end"]) for row in success] == [(0, 4), (4, 8), (8, 9)]
    assert [row["reward"] for row in success] == [-1, -1, 0]
    failure = [row for row in info["rows"] if row["run_id"] == "failure"]
    assert failure[-1]["bootstrap_mask"] == 1
    # Move the package; no source directory or absolute platform path is needed.
    relocated = tmp_path / "relocated"
    dataset.rename(relocated)
    assert inspect_dataset(relocated, {"data": {"action_horizon": 4}})["count"] == 6
    trajectory = relocated / manifest["episodes"][0]["trajectory_path"]
    with trajectory.open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_dataset(relocated)


def test_grouped_validation_keeps_every_branch_with_root():
    episodes = [dict(root_run_id="a"), dict(root_run_id="a"), dict(root_run_id="b"), dict(root_run_id="c")]
    first = grouped_split(episodes, .34, 7)
    assert first == grouped_split(list(reversed(episodes)), .34, 7)
    assert list(first.values()).count("val") == 1
    assert grouped_split(episodes, 0, 7) == dict(a="train", b="train", c="train")


def test_takeover_prefix_dedup():
    original = dict(run_id="root", root_run_id="root", kind="original", resume_step=0,
        recorded_action_count=8, terminal_step=None, split="train",
        action_source_segments=[dict(start=0, end=8, action_source="policy")],
        recorded_action_source_segments=[dict(start=0, end=8, action_source="policy")])
    branch = deepcopy(original)
    branch.update(run_id="branch", kind="branch", resume_step=4, terminal_step=7,
        action_source_segments=[dict(start=0, end=4, action_source="policy"), dict(start=4, end=8, action_source="human")],
        recorded_action_source_segments=[dict(start=0, end=4, action_source="policy"), dict(start=4, end=8, action_source="human")])
    add_episode_chunks(original, 4)
    add_episode_chunks(branch, 4)
    rows = list(iter_unique_replay_chunks([original, branch], include_post_success=False))
    assert len(rows) == 3
    assert [(ep["run_id"], number) for ep, number in rows] == [("root", 0), ("root", 1), ("branch", 1)]


def test_cache_mmap_normalization_image_orientation_and_reuse(dataset, tmp_path, monkeypatch):
    config = dict(method="iql", data=dict(action_horizon=4, validation_fraction=.5, seed=7), critic=dict(image_size=6))
    cache, reused = build_replay_cache(dataset, tmp_path / "cache", config)
    assert not reused
    metadata = validate_cache(cache)
    info = inspect_dataset(dataset, config)
    train_runs = {r["run_id"] for r in info["rows"] if r["split"] == "train"}
    norm = metadata["normalization"]
    expected = .1 if "success" in train_runs else .9
    assert np.allclose(norm["actions"]["q01"], expected)
    assert set(norm["provenance"]["root_run_ids"]) == train_runs
    assert build_replay_cache(dataset, tmp_path / "cache", config) == (cache, True)
    replay = CachedReplayDataset(cache, split=None)
    # After cache construction, sampling must never open an NPZ file.
    original_load = np.load
    def no_archives(path, *args, **kwargs):
        assert str(path).endswith(".npy")
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(np, "load", no_archives)
    sample = replay[0]
    assert sample["pixels"].shape == (6, 6, 6)
    assert sample["agent_image"].shape == (8, 8, 3)
    assert sample["pixels"].dtype == torch.uint8
    assert isinstance(replay.arrays["pixels"], np.memmap)
    expected_image = np.arange(11 * 8 * 8 * 3, dtype=np.uint8).reshape(11, 8, 8, 3)[0, ::-1]
    assert np.array_equal(sample["agent_image"].numpy(), expected_image)
    assert not torch.any(sample["actions"][~sample["action_mask"]])
    assert all(isinstance(value, torch.Tensor) for key, value in sample.items() if key in metadata["arrays"])
    batch = next(iter(torch.utils.data.DataLoader(replay, batch_size=2, num_workers=2)))
    assert batch["agent_image"].shape[0] == 2


def test_cache_corruption_is_fatal(dataset, tmp_path):
    cache, _ = build_replay_cache(dataset, tmp_path / "cache", {"method": "bc", "data": {"action_horizon": 4}})
    with (cache / "raw_actions.npy").open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"X")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_cache(cache)


def test_default_critic_resolution_changes_cache_identity(dataset, tmp_path):
    config = dict(method="critic", data=dict(action_horizon=4, validation_fraction=.5, seed=7))
    default_cache, reused = build_replay_cache(dataset, tmp_path / "cache", config)
    assert not reused
    default_metadata = validate_cache(default_cache)
    assert default_metadata["identity"]["critic_image_size"] == 128
    assert CachedReplayDataset(default_cache, split=None)[0]["pixels"].shape == (6, 128, 128)
    # An explicitly requested older resolution cannot reuse the new default cache.
    config["critic"] = dict(image_size=84)
    smaller_cache, reused = build_replay_cache(dataset, tmp_path / "cache", config)
    assert not reused
    assert smaller_cache != default_cache
    assert CachedReplayDataset(smaller_cache, split=None)[0]["pixels"].shape == (6, 84, 84)


def test_crc_retry_is_once_and_frozen_hash_checked(tmp_path, monkeypatch):
    path = tmp_path / "arrays.npz"
    np.savez_compressed(path, x=np.arange(5))
    digest = sha256_file(path)
    original = np.load
    calls = []
    def first_fails(source, *args, **kwargs):
        calls.append(source)
        if len(calls) == 1:
            raise zipfile.BadZipFile("Bad CRC-32 for file 'x.npy'")
        return original(source, *args, **kwargs)
    monkeypatch.setattr(np, "load", first_fails)
    assert np.array_equal(archive_io.load_npz_arrays(path, expected_sha256=digest)["x"], np.arange(5))
    assert len(calls) == 2
    calls.clear()
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        archive_io.load_npz_arrays(path, expected_sha256="0" * 64)
    assert len(calls) == 1


def test_unknown_failure_stop_cannot_bootstrap(tmp_path):
    source = tmp_path / "source"
    directory = make_run(source, "failed", success=False)
    path = directory / "episodes/episode_000/trajectory.npz"
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["metadata_json"] = json.dumps(dict(session_id="failed", stopped_reason="error"))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="Cannot bootstrap"):
        package_dataset(source, tmp_path / "output", run_ids=["failed"])


@pytest.mark.parametrize("field,value", [
    ("termination_rule", "recording_end_terminal_v1"),
    ("reward", dict(source="sparse", failure=-1.0, success=0.0, gamma=.9, discount_per="chunk")),
    ("action_dim", 32), ("proprio_dim", 7), ("member_count", 99),
    ("record_count", 99), ("include_post_success", True), ("success_consecutive_steps", 1),
])
def test_rehashed_manifest_contract_drift_is_rejected(dataset, field, value):
    from post_training.data import _manifest_identity
    path = dataset / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    manifest["dataset_sha256"] = _manifest_identity(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Dataset"):
        verify_dataset(dataset)


def test_rehashed_success_flag_drift_is_rejected(dataset):
    from post_training.data import _manifest_identity
    path = dataset / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["episodes"][0]["success"] = not manifest["episodes"][0]["success"]
    manifest["dataset_sha256"] = _manifest_identity(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Success flag"):
        verify_dataset(dataset)


def test_export_normalization_roundtrip_and_conflicts(dataset, tmp_path):
    from post_training.replay import load_normalization_file
    config = dict(method="critic", data=dict(action_horizon=4, validation_fraction=.5), critic=dict(image_size=6))
    cache, _ = build_replay_cache(dataset, tmp_path / "cache", config)
    stats = validate_cache(cache)["normalization"]
    direct, metadata = tmp_path / "normalization.json", tmp_path / "critic_metadata.json"
    direct.write_text(json.dumps(stats))
    metadata.write_text(json.dumps(dict(schema_version=1, normalization=stats, model_config={})))
    assert load_normalization_file(direct) == load_normalization_file(metadata) == stats
    config["data"]["normalization"] = str(metadata)
    assert build_replay_cache(dataset, tmp_path / "cache", config, norm_stats=stats) == (cache, True)
    assert build_replay_cache(dataset, tmp_path / "cache", config) == (cache, True)
    changed = deepcopy(stats)
    changed["actions"]["q01"][0] -= .1
    with pytest.raises(ValueError, match="Explicit norm_stats differs"):
        build_replay_cache(dataset, tmp_path / "cache", config, norm_stats=changed)
    del config["data"]["normalization"]
    new_cache, reused = build_replay_cache(dataset, tmp_path / "cache", config, norm_stats=changed)
    assert not reused and new_cache != cache
