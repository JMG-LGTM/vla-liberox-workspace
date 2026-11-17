"""CPU fake-model integration tests; no OpenPI import or real policy allocation."""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
import yaml

from post_training.export import POLICY_FIELDS, compatibility_fields, export_policy, validate_policy_package
from post_training.pi05_assets import (CONTRACT, IDENTITY_FILE, NORM_FILE, OPENPI_COMMIT,
                                      SOURCE, SOURCE_REVISION, checkpoint_identity,
                                      load_normalization, normalize, platform_model_settings,
                                      sha256_file, stable_hash)
from post_training.policy import Pi05Policy, masked_flow_losses


@pytest.fixture
def base(tmp_path):
    root = tmp_path / "base"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"fake model: never load")
    (root / NORM_FILE).parent.mkdir(parents=True)
    (root / NORM_FILE).write_text(json.dumps({"norm_stats": {
        key: {"q01": [-1.] * size, "q99": [1.] * size} for key, size in (("actions", 7), ("state", 8))}}))
    identity = {"schema_version": 1, "family": "pi05", "openpi_commit": OPENPI_COMMIT,
                "source": SOURCE, "source_revision": SOURCE_REVISION, "config_name": "pi05_libero",
                "conversion_precision": "bfloat16", "native_action_horizon": 10,
                "native_action_dim": 32, "discrete_state_input": False,
                "files": {name: sha256_file(root / name) for name in ("model.safetensors", NORM_FILE)}}
    (root / IDENTITY_FILE).write_text(json.dumps(identity))
    return root, identity


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.paligemma = nn.Linear(1, 1)
        self.action_out_proj = nn.Linear(7, 7, bias=False)
        self.checkpointing = False

    def gradient_checkpointing_enable(self):
        self.checkpointing = True

    def forward(self, observation, actions):
        assert actions.dtype == torch.float32
        signal = self.paligemma_with_expert.paligemma(torch.ones(actions.shape[0], 1))
        return (self.action_out_proj(actions) + signal[:, None]).square()


def make_policy(base, scope="frozen"):
    root, identity = base
    settings = platform_model_settings({"checkpoint": str(root), "backbone": scope}, identity)
    return Pi05Policy(FakeModel(), lambda value: {"actions": value["actions"].astype(np.float64)},
                      identity, settings, load_normalization(root), observation_factory=lambda x: x)


def test_identity_hashes_assets_and_pinned_revision_without_torch(base):
    root, identity = base
    assert checkpoint_identity(root) == identity
    result = subprocess.run([sys.executable, "-c", "import sys; from post_training.pi05_assets import checkpoint_identity; "
                             f"checkpoint_identity({str(root)!r}); assert 'torch' not in sys.modules; "
                             "assert 'openpi' not in sys.modules"], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")
                                 + os.pathsep + os.environ.get("PYTHONPATH", "")})
    assert result.returncode == 0, result.stderr
    (root / "model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        checkpoint_identity(root)
    bad = copy.deepcopy(identity)
    bad["source_revision"] = "a" * 40
    (root / IDENTITY_FILE).write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="source_revision"):
        checkpoint_identity(root)


def test_normalization_and_contract(base):
    root, identity = base
    stats = load_normalization(root)
    assert normalize(np.full((2, 7), 2.), stats["actions"]).min() > 1  # no clipping
    settings = platform_model_settings({"checkpoint": str(root)}, identity)
    assert settings["contract"]["io"]["image_flip_vertical"] is True
    assert settings["contract"]["io"]["image_flip_horizontal"] is False
    wrong = copy.deepcopy(CONTRACT)
    wrong["io"]["image_flip_horizontal"] = True
    with pytest.raises(ValueError, match="vertical-only"):
        platform_model_settings({"checkpoint": str(root), "contract": wrong}, identity)


def test_masked_flow_loss_ignores_padding_and_uses_per_sample_mean():
    errors = torch.ones(2, 10, 32, requires_grad=True)
    errors.data[:, :, 7:] = 10000
    errors.data[0, 2:] = 10000
    errors.data[1, 4:] = 10000
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    loss = masked_flow_losses(errors, mask)
    torch.testing.assert_close(loss, torch.ones(2))
    loss.sum().backward()
    assert errors.grad[0, 2:].count_nonzero() == 0
    assert errors.grad[:, :, 7:].count_nonzero() == 0
    with pytest.raises(ValueError, match="nonempty"):
        masked_flow_losses(errors, torch.zeros(2, 4))


@pytest.mark.parametrize("scope", ["frozen", "full"])
def test_actor_forward_scope_and_roundtrip(base, scope):
    policy = make_policy(base, scope)
    batch = {"raw_actions": torch.ones(2, 8, 7), "action_mask": torch.ones(2, 8),
             "raw_proprio": torch.zeros(2, 8), "agent_image": torch.zeros(2, 224, 224, 3, dtype=torch.uint8),
             "wrist_image": torch.zeros(2, 224, 224, 3, dtype=torch.uint8), "prompt": ["task", "task"]}
    policy.eval()
    policy.train()
    backbone = policy.model.paligemma_with_expert.paligemma
    assert backbone.training is (scope == "full")
    assert policy.model.checkpointing
    losses = policy(batch)
    assert losses.shape == (2,)
    losses.mean().backward()
    assert policy.model.action_out_proj.weight.grad is not None
    assert (backbone.weight.grad is None) is (scope == "frozen")
    saved = policy.trainable_state_dict()
    assert all(not key.startswith("model.") for key in saved)
    restored = make_policy(base, scope)
    restored.load_trainable_state_dict(saved)
    for key, value in restored.trainable_state_dict().items():
        torch.testing.assert_close(value, saved[key])
    with pytest.raises(ValueError, match="keys"):
        restored.load_trainable_state_dict({})


@pytest.mark.parametrize("method", ["bc", "iql"])
def test_export_exact_platform_contract_and_rebase(base, tmp_path, method):
    root, identity = base
    policy = make_policy(base)
    output = tmp_path / f"pi05-{method}-test"
    path = export_policy(output, policy.trainable_state_dict(), identity, policy.settings,
                         "d" * 64, 800, method, "e" * 64 if method == "iql" else None,
                         "/evaluation/models/pi05_liberox_torch")
    payload = validate_policy_package(output)
    assert set(payload) == POLICY_FIELDS
    assert payload["policy_id"] == output.name
    assert payload["parent"]["checkpoint"] == payload["base_checkpoint"] == "/evaluation/models/pi05_liberox_torch"
    assert payload["compatibility_sha256"] == stable_hash(compatibility_fields(payload))
    assert payload["parent"]["sha256"] == stable_hash({k: v for k, v in payload["parent"].items() if k != "sha256"})
    assert json.loads((output / IDENTITY_FILE).read_text()) == identity
    assert set(torch.load(output / "actor.pt", weights_only=True)) == set(policy.trainable_state_dict())
    payload["action_horizon"] = 10
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="compatibility fields"):
        validate_policy_package(output)


def test_export_rejects_unsafe_or_incomplete_packages(base, tmp_path):
    _, identity = base
    policy = make_policy(base)
    args = (policy.trainable_state_dict(), identity, policy.settings, "d" * 64, 1)
    with pytest.raises(ValueError, match="reward_sha256"):
        export_policy(tmp_path / "incomplete-iql", *args, "iql")
    with pytest.raises(ValueError, match="directory name"):
        export_policy(tmp_path / "package", *args, "bc", policy_id="different-name")
    assert not (tmp_path / "package").exists()
    output = tmp_path / "valid"
    export_policy(output, *args, "bc")
    (output / "actor.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="component SHA-256"):
        validate_policy_package(output)


@pytest.mark.parametrize("schema_version", [1, 2])
def test_export_cli_checks_checkpoint_and_subtracts_critic_warmup(base, tmp_path, schema_version):
    _, identity = base
    policy = make_policy(base)
    checkpoint = tmp_path / "update_00000012.pt"
    public_checkpoint = checkpoint
    if schema_version == 2:
        generation = tmp_path / ".update_00000012.shards.test"
        generation.mkdir()
        checkpoint = generation / "model.pt"
        public_checkpoint.symlink_to(checkpoint.relative_to(tmp_path))
    torch.save({"schema_version": schema_version, "accumulation_boundary": True, "update": 12,
                "contract": {"critic_warmup_updates": 10}, "actor": policy.trainable_state_dict(),
                "metadata": {"model_identity": identity, "model_settings": policy.settings,
                             "dataset_sha256": "d" * 64, "reward_sha256": "e" * 64, "method": "iql"}}, checkpoint)
    checkpoint.with_suffix(".json").write_text(json.dumps({"sha256": sha256_file(checkpoint)}))
    output = tmp_path / "pi05-iql-cli"
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/export_policy.py"),
               "--checkpoint", str(public_checkpoint), "--output", str(output),
               "--platform-base-checkpoint", "/evaluation/base"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert validate_policy_package(output)["training_step"] == 2
    checkpoint.write_bytes(b"corrupt checkpoint")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and "Checkpoint SHA-256 mismatch" in result.stderr
