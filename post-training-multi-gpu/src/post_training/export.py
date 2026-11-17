"""Platform schema-5 export, matching training.py:_publish_overlay field-for-field.

Artifacts are created outside the running platform. No registry symlinks or
platform configuration files are changed. Base path rebasing happens before
computing the exact original compatibility and parent hashes.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import torch
import yaml

from .io import atomic_json
from .pi05_assets import (IDENTITY_FILE, identity_digest, parent_snapshot,
                          platform_model_settings, sha256_file, stable_hash, validate_identity)

POLICY_FIELDS = {
    "algorithm", "model_config", "policy_id", "label", "base_checkpoint", "stats_key",
    "action_horizon", "action_dim", "proprio_dim", "dataset_sha256", "reward_sha256",
    "training_step", "schema_version", "family", "actor", "base_identity",
    "component_sha256", "compatibility_sha256", "parent",
}


def compatibility_fields(payload: dict) -> dict:
    return {key: payload[key] for key in ("base_checkpoint", "stats_key", "action_horizon",
                                         "action_dim", "proprio_dim")}


def validate_policy_package(directory: str | Path, *, expected_directory_name: str | None = None) -> dict:
    """Check the exact package contract without model instantiation or GPU access."""
    root = Path(directory)
    payload = yaml.safe_load((root / "policy.yaml").read_text())
    if set(payload) != POLICY_FIELDS or payload["schema_version"] != 5 or payload["family"] != "pi05":
        raise ValueError("Invalid platform pi05 policy manifest fields/schema")
    if payload["algorithm"] not in {"bc", "iql"}:
        raise ValueError("Only BC/IQL actor exports are platform policies")
    if payload["policy_id"] != (expected_directory_name or root.name):
        raise ValueError("Platform policy directory name must equal policy_id")
    if (type(payload["training_step"]) is not int or payload["training_step"] < 1
            or not isinstance(payload["label"], str) or not payload["label"].strip()):
        raise ValueError("Platform policy requires completed updates and a nonempty label")
    for key in ("dataset_sha256", "reward_sha256"):
        if key == "reward_sha256" and payload["algorithm"] == "bc" and payload[key] is None:
            continue
        if re.fullmatch(r"[0-9a-f]{64}", str(payload[key])) is None:
            raise ValueError(f"Invalid platform digest: {key}")
    if payload["actor"] != "actor.pt" or payload["base_identity"] != IDENTITY_FILE:
        raise ValueError("Unexpected platform artifact names")
    identity = validate_identity(json.loads((root / IDENTITY_FILE).read_text()))
    settings = platform_model_settings(payload["model_config"], identity)
    if settings != payload["model_config"] or payload["parent"] != parent_snapshot(settings):
        raise ValueError("Parent identity/model settings mismatch")
    if (payload["base_checkpoint"] != settings["base_checkpoint"]
            or payload["stats_key"] != settings["stats_key"]
            or (payload["action_horizon"], payload["action_dim"], payload["proprio_dim"]) != (8, 7, 8)):
        raise ValueError("Platform compatibility fields differ from inherited I/O")
    if payload["compatibility_sha256"] != stable_hash(compatibility_fields(payload)):
        raise ValueError("Platform compatibility hash mismatch")
    expected = {"actor": sha256_file(root / "actor.pt"),
                "base_identity": sha256_file(root / IDENTITY_FILE)}
    if payload["component_sha256"] != expected:
        raise ValueError("Platform component SHA-256 mismatch")
    return payload


def export_policy(output: str | Path, actor_state: dict, identity: dict, model_settings: dict,
                  dataset_sha256: str, training_step: int, method: str,
                  reward_sha256: str | None = None, platform_base_checkpoint: str | None = None,
                  *, provenance: dict | None = None, policy_id: str | None = None) -> Path:
    """Atomically export native actor.pt, immutable base identity and policy.yaml.

    training_step is the number of completed optimizer updates in this repository;
    export_provenance.json records that unit explicitly without extending policy.yaml.
    """
    if method not in {"bc", "iql"}:
        raise ValueError("critic mode has no actor; use its standalone critic checkpoint")
    if type(training_step) is not int or training_step < 1:
        raise ValueError("training_step must be a positive optimizer update count")
    if re.fullmatch(r"[0-9a-f]{64}", dataset_sha256) is None:
        raise ValueError("dataset_sha256 must be a SHA-256 digest")
    if reward_sha256 is not None and re.fullmatch(r"[0-9a-f]{64}", reward_sha256) is None:
        raise ValueError("reward_sha256 must be a SHA-256 digest or null")
    if method == "iql" and reward_sha256 is None:
        raise ValueError("Platform IQL policies require reward_sha256")
    validate_identity(identity)
    if not actor_state or any(not isinstance(key, str) or not isinstance(value, torch.Tensor)
                              for key, value in actor_state.items()):
        raise ValueError("Actor state must be a nonempty native parameter-to-tensor mapping")
    if any(key.startswith(("module.", "model.")) for key in actor_state):
        raise ValueError("Actor parameters must use native OpenPI names, without DDP/wrapper prefixes")
    settings = platform_model_settings(model_settings, identity, checkpoint=platform_base_checkpoint)
    if settings["backbone"] == "frozen" and any(key.startswith("paligemma_with_expert.paligemma.") for key in actor_state):
        raise ValueError("A frozen actor export cannot contain backbone parameters")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing policy package: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        torch.save({name: value.detach().cpu() for name, value in actor_state.items()}, temporary / "actor.pt")
        atomic_json(temporary / IDENTITY_FILE, identity)
        identifier = policy_id or output.name
        if re.fullmatch(r"[A-Za-z0-9_.-]+", identifier) is None or identifier in {".", ".."}:
            raise ValueError("Invalid policy_id")
        if identifier != output.name:
            raise ValueError("Platform policy directory name must equal policy_id")
        compatibility = {"base_checkpoint": settings["base_checkpoint"], "stats_key": settings["stats_key"],
                         "action_horizon": 8, "action_dim": 7, "proprio_dim": 8}
        payload = {"algorithm": method, "model_config": settings, "policy_id": identifier,
                   "label": f"Sparse IQL · step {training_step}" if method == "iql" else f"BC · step {training_step}",
                   **compatibility, "dataset_sha256": dataset_sha256, "reward_sha256": reward_sha256,
                   "training_step": training_step, "schema_version": 5, "family": "pi05",
                   "actor": "actor.pt", "base_identity": IDENTITY_FILE,
                   "component_sha256": {"actor": sha256_file(temporary / "actor.pt"),
                                        "base_identity": sha256_file(temporary / IDENTITY_FILE)},
                   "compatibility_sha256": stable_hash(compatibility), "parent": parent_snapshot(settings)}
        (temporary / "policy.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
        atomic_json(temporary / "export_provenance.json", {
            **(provenance or {}), "schema_version": 1, "algorithm": method,
            "training_step_unit": "optimizer_update", "optimizer_updates": training_step,
            "model_identity_sha256": identity_digest(identity), "dataset_sha256": dataset_sha256,
            "bootstrap_policy": "success_aware_recording_end_v1",
        })
        validate_policy_package(temporary, expected_directory_name=output.name)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output / "policy.yaml"
