"""Verified BC actor warm start, independent of IQL optimizer/resume state.

The original converted base still defines identity and normalization. The source
BC package contributes actor parameters only; a different downstream dataset is
expected. Its complete content identity is bound into the IQL resume contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .export import validate_policy_package
from .io import sha256_file
from .pi05_assets import IDENTITY_FILE, identity_digest, platform_model_settings


def _snapshot(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def inspect_actor_init_policy(directory, *, identity, model_config):
    """Hash/validate a schema-5 full BC package without loading model tensors."""
    if directory is None:
        return None
    root = Path(directory).expanduser().resolve()
    paths = {name: root / name for name in ("policy.yaml", "actor.pt", IDENTITY_FILE)}
    provenance = root / "export_provenance.json"
    if provenance.exists():
        paths[provenance.name] = provenance
    if any(not path.is_file() or path.is_symlink() for path in paths.values()):
        raise ValueError("BC actor initialization requires real package component files")
    before = {name: _snapshot(path) for name, path in paths.items()}
    package = validate_policy_package(root)
    if package["algorithm"] != "bc":
        raise ValueError("Actor initialization source must be a BC policy")
    if package["model_config"]["backbone"] != "full" or model_config["backbone"] != "full":
        raise ValueError("BC actor initialization requires matching full actor scope")
    source_identity = json.loads(paths[IDENTITY_FILE].read_text())
    if source_identity != identity:
        raise ValueError("BC actor initialization base identity/normalization differs")
    settings = platform_model_settings(model_config, identity)
    # Absolute paths can differ between the collection computer and server.
    # Immutable base assets and I/O semantics, rather than a relocated path,
    # certify compatibility. Source inference-step settings are provenance only.
    for key in ("family", "base_id", "base_revision", "stats_key", "contract"):
        if package["model_config"][key] != settings[key]:
            raise ValueError(f"BC actor initialization model I/O differs: {key}")
    files = {name: {"bytes": path.stat().st_size,
                    "sha256": (package["component_sha256"]["actor"] if name == "actor.pt"
                               else package["component_sha256"]["base_identity"] if name == IDENTITY_FILE
                               else sha256_file(path))}
             for name, path in paths.items()}
    if before != {name: _snapshot(path) for name, path in paths.items()}:
        raise ValueError("BC actor initialization package changed during verification")
    return {"schema_version": 1, "kind": "full_bc_actor_warm_start",
            "policy_directory": str(root), "policy_id": package["policy_id"],
            "source_algorithm": "bc", "source_backbone": "full",
            "source_optimizer_updates": package["training_step"],
            "source_dataset_sha256": package["dataset_sha256"],
            "source_base_checkpoint": package["base_checkpoint"],
            "base_identity_sha256": identity_digest(identity), "files": files,
            "optimizer_state_inherited": False, "initial_iql_actor_updates": 0}


def initialize_actor_from_policy(actor, initialization):
    """Copy verified BC tensors into a fresh actor, before DDP/optimizers.

    This is deliberately not called while resuming: the IQL checkpoint owns the
    current actor, optimizer, RNG and counters. Inspection still occurs on resume
    so a changed source fails the saved initialization contract.
    """
    if initialization is None:
        return
    root = Path(initialization["policy_directory"])
    verified = inspect_actor_init_policy(root, identity=actor.identity, model_config=actor.settings)
    if verified != initialization:
        raise ValueError("BC actor initialization package changed after preparation")
    path = root / "actor.pt"
    before = _snapshot(path)
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("BC actor initialization is not a native parameter mapping")
    for name, value in state.items():
        if (not isinstance(name, str) or not isinstance(value, torch.Tensor)
                or not value.is_floating_point() or not torch.isfinite(value).all()):
            raise ValueError("BC actor initialization contains invalid/nonfinite tensors")
    actor.load_trainable_state_dict(state)
    if before != _snapshot(path):
        raise ValueError("BC actor initialization weights changed during loading")
