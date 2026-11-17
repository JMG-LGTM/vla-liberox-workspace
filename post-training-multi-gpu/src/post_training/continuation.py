"""IQL only. One audited full-state IQL continuation; ordinary resume remains exact.

The donor checkpoint is never rewritten. Its original manifest, common payload
and both rank shards are loaded against its original contract. Only the new
run's future checkpoint contract has the extended target and explicit lineage.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .io import sha256_file, stable_hash

SOURCE_UPDATE = 30000
TARGET_UPDATE = 45000
SOURCE_CODE = "215eaaf685bd0afef526511ce3c339a4a359fe4e"
SOURCE_SHA = "ff28666a7e713b0ea8882e929e6634d9f9647ecb552dc4842cc071d60a0b3e7f"
MANIFEST_SHA = "68a29efd6443163e4882c9cd7e35b1bdf28d2bdea1f74328ca8013a9160c1429"
CONTRACT_SHA = "49a4d86940e06d617ed441fad220ae92b3e332256bc8e46f4716f2aa5e2d9bd0"
CONFIG_SHA = "929b7e6fff8f828e948da7b4892b68d310c2fc942c09802768d4032d6942c970"
SOURCE_GENERATION = ".update_00030000.shards.t7ao5tq8"
ACTOR_TERMINAL_LR = 3e-7
MODE = "iql_30000_to_45000_v1"
POLICY = "keep_original_terminal"
FIELDS = {"mode", "source_checkpoint", "source_model_sha256", "source_manifest_sha256",
          "source_contract_sha256", "source_config_sha256", "source_commit",
          "source_update", "target_updates", "learning_rate_policy"}


def comparable_config(config):
    """Preserve pre-extension canonical hashes when the feature is inactive."""
    value = copy.deepcopy(config)
    for key in ("output_dir", "cache_dir"):
        value.pop(key)
    if value["training"].get("continuation") is None:
        value["training"].pop("continuation", None)
    return value


def donor_config(config):
    value = copy.deepcopy(config)
    value["training"].pop("continuation", None)
    value["training"]["optimizer_updates"] = SOURCE_UPDATE
    return value


def validate_continuation(config):
    settings = config["training"].get("continuation")
    if settings is None:
        return
    if not isinstance(settings, dict) or set(settings) != FIELDS:
        raise ValueError("training.continuation requires the exact audited field set")
    expected = {"mode": MODE, "source_model_sha256": SOURCE_SHA,
                "source_manifest_sha256": MANIFEST_SHA, "source_contract_sha256": CONTRACT_SHA,
                "source_config_sha256": CONFIG_SHA, "source_commit": SOURCE_CODE,
                "source_update": SOURCE_UPDATE, "target_updates": TARGET_UPDATE,
                "learning_rate_policy": POLICY}
    if any(type(settings.get(k)) is not type(v) or settings.get(k) != v
           for k, v in expected.items()):
        raise ValueError("Continuation differs from the authorized IQL30000 to45000 source/target/LR policy")
    if (config["method"] != "iql" or config["model"]["backbone"] != "full"
            or config["model"]["actor_init_policy"] is not None
            or config["training"]["optimizer_updates"] != TARGET_UPDATE):
        raise ValueError("Continuation must restore full IQL state to the authorized cumulative target")
    expected_path = (Path(config["output_dir"]).expanduser().absolute().parent.parent
                     / "resume_input/checkpoints" / f"update_{SOURCE_UPDATE:08d}.pt")
    if not isinstance(settings["source_checkpoint"], str) or Path(settings["source_checkpoint"]) != expected_path:
        raise ValueError("Continuation source must be this independent experiment's resume_input/checkpoints alias")
    # All former math, data, initialization, normalization, source/cache and I/O
    # settings remain bound by the original canonical hash. No broad ignore list.
    if stable_hash(comparable_config(donor_config(config))) != CONFIG_SHA:
        raise ValueError("Continuation changed original data/model/auxiliary/optimizer/math configuration")
    if config["training"]["final_learning_rate"] != ACTOR_TERMINAL_LR:
        raise ValueError("Continuation must preserve the original terminal actor learning rate")


def own_run_resume(run_dir, requested):
    return Path(requested).absolute().parent.resolve().parent == Path(run_dir).resolve()


def check_resume_location(config, requested):
    """Allow only own-run checkpoints or the one explicitly frozen donor."""
    if own_run_resume(config["output_dir"], requested):
        return False
    settings = config["training"].get("continuation")
    if settings is None:
        raise ValueError("Resume in the original output_dir so logs and best-checkpoint selection remain connected")
    validate_continuation(config)
    alias = Path(settings["source_checkpoint"]).absolute()
    generation = alias.parent / SOURCE_GENERATION / "model.pt"
    if Path(requested).absolute() not in (alias, generation):
        raise ValueError("External resume is restricted to the exact audited continuation checkpoint")
    return True


def donor_contract(config, current_contract):
    """Reconstruct and bind the entire old contract, including every plan field."""
    from .config import batch_plan
    validate_continuation(config)
    if current_contract["world_size"] != 2:
        raise ValueError("The continuation requires the original two-rank optimizer partition")
    old_config = donor_config(config)
    old = copy.deepcopy(current_contract)
    old.pop("continuation", None)
    old.update(batch_plan(old_config, 2))
    old["config_sha256"] = stable_hash(comparable_config(old_config))
    if stable_hash(old) != CONTRACT_SHA:
        raise ValueError("Continuation donor contract differs in data/cache/model/auxiliary/batch/world/math")
    return old


def prepare_donor_load(config, requested, current_contract):
    """Verify small frozen metadata; native loader then SHA-checks every payload."""
    if not check_resume_location(config, requested):
        raise ValueError("Donor loading is only for the explicit external source")
    contract = donor_contract(config, current_contract)
    path = Path(requested).resolve(strict=True)
    expected = Path(config["training"]["continuation"]["source_checkpoint"]).parent / SOURCE_GENERATION / "model.pt"
    if path != expected.resolve(strict=True) or path.parent.name != SOURCE_GENERATION:
        raise ValueError("Continuation alias resolves to a different immutable generation")
    manifest_path = path.with_suffix(".json")
    if sha256_file(manifest_path) != MANIFEST_SHA:
        raise ValueError("Continuation source manifest SHA-256 differs")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema_version") != 2 or manifest.get("checkpoint_format") != "sharded"
            or manifest.get("update") != SOURCE_UPDATE or manifest.get("world_size") != 2
            or manifest.get("sha256") != SOURCE_SHA or manifest.get("contract") != contract):
        raise ValueError("Continuation source is not the frozen complete IQL30000 checkpoint")
    return contract


def actor_rate(config, actor_update, ordinary_rate):
    if config["training"].get("continuation") is None:
        return ordinary_rate
    if actor_update < SOURCE_UPDATE:
        raise ValueError("Continuation cannot restart actor updates or rerun warmup")
    return ACTOR_TERMINAL_LR
