"""Mixed BC preserves published observations/actions without invented transitions."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from post_training.data import (_manifest_identity, inspect_dataset, package_dataset,
                                sha256_file, verify_dataset)
from post_training.official_bc import (REPOSITORY, REVISION, attach_official_demonstrations)
from post_training.replay import CachedReplayDataset, build_replay_cache, validate_cache
from test_data_replay import make_run
from test_policy_export import base


@pytest.fixture
def mixed(tmp_path):
    source = tmp_path / "source"
    native = make_run(source, "native", success=True)
    pixels = np.zeros((11, 256, 256, 3), np.uint8)
    pixels[:, 0] = 17; pixels[:, -1] = 249
    np.savez_compressed(native / "episodes/episode_000/trajectory_observations.npz",
                        agentview_image=pixels, wrist_image=pixels)
    native_package = tmp_path / "native"
    package_dataset(source, native_package, run_ids=["native"])
    official = tmp_path / "official"
    official.mkdir()
    episodes = []
    for episode, count in [(1719, 9), (1720, 10)]:
        directory = official / ("processed/episode_%06d" % episode)
        directory.mkdir(parents=True)
        states = np.arange(count * 8, dtype=np.float32).reshape(count, 8)
        actions = np.arange(count * 7, dtype=np.float32).reshape(count, 7) / 10 - 1.6
        np.savez_compressed(directory / "numeric.npz", state=states, actions=actions,
                            timestamp=np.arange(count, dtype=np.float32) / 10)
        views = np.zeros((count, 256, 256, 3), np.uint8)
        views[:, 0] = 19; views[:, -1] = 241
        np.savez_compressed(directory / "images.npz", image=views, wrist_image=views)
        np.savez_compressed(directory / "bc_chunks_h8.npz", unused=np.array([0]))
        files = {n: dict(bytes=(directory / n).stat().st_size, sha256=sha256_file(directory / n))
                 for n in ("numeric.npz", "images.npz", "bc_chunks_h8.npz")}
        metadata = dict(episode_index=episode, task_index=264, prompt="place bowl", rows=count,
                        source_sha256=str(episode) * 16, image_orientation="as_published_no_flip", files=files)
        (directory / "metadata.json").write_text(json.dumps(metadata))
        episodes.append(metadata)
    (official / "processed_manifest.json").write_text(json.dumps(dict(
        kind="official_human_bc_intermediate", source_repository=REPOSITORY,
        source_revision=REVISION, episode_count=2, episodes=episodes)))
    for n in ("source_manifest.json", "validation.json"):
        (official / n).write_text("{}")
    result = tmp_path / "mixed"
    attach_official_demonstrations(native_package, official, result)
    return native_package, official, result


def test_bc_membership_tail_masks_and_published_pixels_actions(mixed, tmp_path):
    native, official, root = mixed
    config = dict(method="bc", data=dict(action_horizon=8, validation_fraction=0))
    before = inspect_dataset(native, config)
    info = inspect_dataset(root, config)
    assert (info["native_members"], info["official_demonstrations"], info["total_training_episodes"]) == (1, 2, 3)
    assert info["native_chunks"] == before["count"] == 2
    assert info["official_bc_chunks"] == 4 and info["count"] == 6
    forbidden = {"done", "reward", "bootstrap_mask", "remaining_blocks", "success_terminal", "recording_success"}
    assert all(not forbidden.intersection(row) for row in info["rows"])
    cache, _ = build_replay_cache(root, tmp_path / "cache", config)
    metadata = validate_cache(cache)
    assert not forbidden.intersection(metadata["arrays"])
    replay = CachedReplayDataset(cache)
    assert replay[0]["agent_image"][0, 0, 0].item() == 249  # Native flip happens once.
    index = next(i for i, row in enumerate(replay.records) if row["source_type"] == "official_bc")
    sample = replay[index]
    assert sample["agent_image"][0, 0, 0].item() == 19  # Published official pixels are never flipped.
    assert sample["agent_image"][-1, 0, 0].item() == 241
    assert sample["raw_actions"][0, 0].item() == np.float32(-1.6)  # No clipping/rescaling.
    assert torch.equal(sample["raw_proprio"], torch.arange(8, dtype=torch.float32))
    tail = replay[index + 1]
    assert tail["action_mask"].tolist() == [True] + [False] * 7
    assert not tail["raw_actions"][1:].any()
    assert torch.equal(tail["raw_actions"][0], torch.arange(56, 63, dtype=torch.float32) / 10 - 1.6)
    # Same-key metadata permits truly mixed-source batches, without None/fake labels.
    batch = next(iter(torch.utils.data.DataLoader(replay, batch_size=len(replay))))
    assert batch["raw_actions"].shape == (6, 8, 7)
    assert batch["source_type"] == ["native"] * 2 + ["official_bc"] * 4
    assert verify_dataset(native)["format"] == "liberox-native-post-training"  # Input unchanged.
    assert build_replay_cache(root, tmp_path / "cache", config) == (cache, True)


@pytest.mark.parametrize("method", ["iql", "critic"])
def test_transition_methods_reject_observation_action_only_demonstrations(mixed, tmp_path, method):
    root = mixed[-1]
    with pytest.raises(ValueError, match="BC-only"):
        inspect_dataset(root, {"method": method})
    with pytest.raises(ValueError, match="BC-only"):
        build_replay_cache(root, tmp_path / "cache", {"method": method})


def test_unhashed_or_fabricated_demo_payload_is_rejected(mixed):
    root = mixed[-1]
    path = root / "manifest.json"
    m = json.loads(path.read_text())
    m["bc_demonstrations"][0]["reward"] = 0
    m["dataset_sha256"] = _manifest_identity(m)
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="invent transition"):
        verify_dataset(root)


def test_official_payload_corruption_is_not_silently_accepted(mixed):
    root = mixed[-1]
    manifest = verify_dataset(root)
    path = root / manifest["bc_demonstrations"][0]["numeric_path"]
    with path.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_dataset(root)


def test_mixed_bc_real_runner_checkpoint_export_and_source_pixels(mixed, tmp_path, monkeypatch, base):
    """Run actual BC orchestration/optimizer/checkpoint/export on mixed data, CPU only."""
    from test_training import configuration, install_fake_policy
    from post_training import training
    from post_training.export import validate_policy_package
    from post_training.policy import Pi05Policy
    model_root = install_fake_policy(monkeypatch, base)
    config = configuration(tmp_path, method="bc", updates=2)
    config["model"]["checkpoint"] = str(model_root)
    config["data"]["validation_fraction"] = 0
    config["training"].update(global_batch_size=6, per_device_batch_size=6, sampling="shuffle")
    seen_top_pixels = []
    original = Pi05Policy.forward
    def track_actual_policy_inputs(self, batch):
        seen_top_pixels.extend(batch["agent_image"][:, 0, 0, 0].tolist())
        assert "reward" not in batch and "done" not in batch and "success_terminal" not in batch
        return original(self, batch)
    monkeypatch.setattr(Pi05Policy, "forward", track_actual_policy_inputs)
    result = training.train(config, mixed[-1])
    assert result["status"] == "COMPLETED" and result["policy_optimizer_updates"] == 2
    assert result["critic_optimizer_updates"] == 0
    assert set(seen_top_pixels) == {249, 19} and len(seen_top_pixels) == 12
    assert Path(result["checkpoint"]).is_file()
    package = validate_policy_package(Path(result["policy"]).parent)
    assert package["algorithm"] == "bc" and package["training_step"] == 2
    records = [json.loads(line) for line in (Path(config["output_dir"]) / "metrics.jsonl").read_text().splitlines()]
    assert [row["optimizer_update"] for row in records] == [1, 2]
    assert records[-1]["policy_optimizer_updates"] == 2 and records[-1]["micro_batch_count"] == 2
