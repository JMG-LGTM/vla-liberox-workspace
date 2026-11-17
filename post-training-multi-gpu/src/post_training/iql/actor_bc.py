"""Observation/action-only official BC auxiliary; never a critic transition pool."""
from __future__ import annotations

from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from ..archive_io import load_npz_arrays
from ..data import _manifest_identity, grouped_split, safe_path
from ..io import atomic_json, sha256_file, stable_hash
from ..official_bc import (MIXED_FORMAT, CONTRACT, demo_rows, validate_demonstrations,
                          verify_demo_payloads)
from ..replay import CachedReplayDataset, _metadata_hash, validate_cache

FORMAT = "liberox-official-bc-demonstrations"
ACTOR_FIELDS = ("raw_actions", "action_mask", "raw_proprio", "agent_image", "wrist_image")
FORBIDDEN = {"reward", "done", "next_pixels", "next_proprio", "bootstrap_mask", "pixels", "proprio"}


def inspect_actor_bc(root, config, native_info, *, verify=True):
    if config["actor_bc"]["dataset_root"] is None:
        return None
    root = Path(root).expanduser().resolve()
    manifest_path = safe_path(root, "manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema_version") != 1 or manifest.get("format") != FORMAT
            or manifest.get("dataset_sha256") != _manifest_identity(manifest)):
        raise ValueError("Invalid official actor BC package identity/format")
    if any(manifest.get(key) for key in ("episodes", "records", "member_run_ids")):
        raise ValueError("Official actor BC package must contain no native transitions")
    if (manifest.get("action_horizon"), manifest.get("action_dim"), manifest.get("proprio_dim")) != (8, 7, 8):
        raise ValueError("Official actor BC package geometry differs")
    demos = manifest.get("bc_demonstrations", [])
    # Hard-coded for Scene24 L1: exactly the two official LIBERO-X demonstrations of its task
    # (meituan/LIBERO-X episodes 1719 and 1720, task 264). Other scenes need these values made configurable.
    if sorted(ep.get("episode_index") for ep in demos) != [1719, 1720]:
        raise ValueError("Expected exactly the two pinned Scene24 official demonstrations")
    for ep in demos:
        if ep.get("root_run_id") != f"official_meituan_libero_x_episode_{ep['episode_index']:06d}":
            raise ValueError("Official actor BC root identity differs")
    proxy = {**manifest, "format": MIXED_FORMAT, "member_count": 0,
             "member_run_ids": [], "total_training_episode_count": len(demos)}
    validate_demonstrations(proxy)  # Reuse only descriptor validation, not native replay.
    data = config["data"]
    if (manifest.get("split_seed"), manifest.get("validation_fraction"), manifest.get("split_strategy")) != (
            data["seed"], data["validation_fraction"], data["split_strategy"]):
        raise ValueError("Official actor BC split settings differ from native root grouping")
    splits = grouped_split(demos, data["validation_fraction"], data["seed"], data["split_strategy"])
    if set(splits.values()) != {"train"} or manifest.get("root_splits") != splits:
        raise ValueError("Official actor BC contains a held-out validation root")
    native_roots = {ep["root_run_id"] for ep in native_info["episodes"]}
    if native_roots.intersection(splits):
        raise ValueError("Official actor BC root overlaps the native train/validation pool")
    if {ep["prompt"] for ep in demos} != {row["prompt"] for row in native_info["rows"]}:
        raise ValueError("Official actor BC instruction differs from native task")
    if verify:
        for relative, expected in manifest["files"].items():
            path = safe_path(root, relative)
            if not path.is_file() or path.stat().st_size != expected["size"] or sha256_file(path) != expected["sha256"]:
                raise ValueError("Official actor BC payload SHA-256/size differs: " + relative)
        verify_demo_payloads(root, manifest)
    rows = demo_rows(demos, 8, splits)
    return {"dataset_root": str(root), "dataset_sha256": manifest["dataset_sha256"],
        "manifest_sha256": sha256_file(manifest_path), "root_run_ids": sorted(splits),
        "root_splits": splits, "train_chunks": len(rows), "split_seed": data["seed"],
        "validation_fraction": data["validation_fraction"], "split_strategy": data["split_strategy"],
        "per_device_batch_size": config["actor_bc"]["per_device_batch_size"],
        "weight": config["actor_bc"]["weight"], "critic_samples": 0,
        "image_orientation": "as_published_no_flip", "bc_demonstration_contract": deepcopy(CONTRACT)}


def build_actor_bc_cache(root, cache_root, config, native_info):
    info = inspect_actor_bc(root, config, native_info, verify=True)
    if info is None:
        return None
    root, cache_root = Path(root).resolve(), Path(cache_root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    rows = demo_rows(manifest["bc_demonstrations"], 8, info["root_splits"])
    identity = {"kind": "official_bc_actor_auxiliary_only", "dataset_sha256": info["dataset_sha256"],
        "action_horizon": 8, "split_rows_sha256": stable_hash(rows), "orientation": "as_published_no_flip",
        "implementation": {name: sha256_file(Path(__file__).parent.parent / path) for name, path in
            (("actor_bc.py", "iql/actor_bc.py"), ("official_bc.py", "official_bc.py"), ("archive_io.py", "archive_io.py"),
             ("data.py", "data.py"), ("replay.py", "replay.py"))}}
    fingerprint = stable_hash(identity)
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / fingerprint
    with (cache_root / f".{fingerprint}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.is_dir():
            temporary = Path(tempfile.mkdtemp(prefix=f".{fingerprint}.", dir=cache_root))
            arrays = {}
            try:
                by_id = {ep["run_id"]: ep for ep in manifest["bc_demonstrations"]}
                loaded, numeric, images = None, None, None
                for index, row in enumerate(rows):
                    ep = by_id[row["run_id"]]
                    if loaded != row["run_id"]:
                        numeric, images = None, None
                        numeric = load_npz_arrays(safe_path(root, ep["numeric_path"]), keys=("state", "actions"),
                            expected_sha256=manifest["files"][ep["numeric_path"]]["sha256"])
                        images = load_npz_arrays(safe_path(root, ep["observations_path"]), keys=("image", "wrist_image"),
                            expected_sha256=manifest["files"][ep["observations_path"]]["sha256"])
                        loaded = row["run_id"]
                    start, end = row["start"], row["end"]
                    actions = np.zeros((8, 7), np.float32)
                    actions[:end-start] = numeric["actions"][start:end]
                    values = {"raw_actions": actions, "action_mask": np.arange(8) < end-start,
                        "raw_proprio": numeric["state"][start], "agent_image": images["image"][start],
                        "wrist_image": images["wrist_image"][start]}
                    for name, value in values.items():
                        if index == 0:
                            arrays[name] = np.lib.format.open_memmap(temporary / f"{name}.npy", mode="w+",
                                dtype=value.dtype, shape=(len(rows), *value.shape))
                        arrays[name][index] = value
                specs = {}
                for name, value in arrays.items():
                    value.flush()
                    specs[name] = {"shape": list(value.shape), "dtype": str(value.dtype),
                                   "sha256": sha256_file(temporary / f"{name}.npy")}
                metadata = {"schema_version": 1, "fingerprint": fingerprint, "identity": identity,
                    "count": len(rows), "counts": {"train": len(rows)}, "arrays": specs,
                    "records": rows, "root_counts": {"train": 2}, "normalization": None}
                metadata["metadata_sha256"] = _metadata_hash(metadata)
                atomic_json(temporary / "cache.json", metadata)
                validate_cache(temporary, fingerprint)
                os.rename(temporary, target)
            finally:
                arrays.clear()
                if temporary.exists():
                    shutil.rmtree(temporary)
        metadata = validate_cache(target, fingerprint)
        if set(metadata["arrays"]) != set(ACTOR_FIELDS) or any(row["split"] != "train" for row in metadata["records"]):
            raise ValueError("Official actor BC cache contains non-actor or held-out fields")
    return {**info, "cache": str(target), "cache_fingerprint": fingerprint,
            "cache_metadata_sha256": metadata["metadata_sha256"]}


def actor_batch(native, official):
    """Use native prefix plus official tail, preserving the actor batch budget."""
    if FORBIDDEN.intersection(official):
        raise ValueError("Official actor BC batch must not contain critic/transition fields")
    official_count = official["raw_actions"].shape[0]
    native_count = native["raw_actions"].shape[0] - official_count
    if native_count < 1 or official_count < 1:
        raise ValueError("Actor BC batch must include native and official samples")
    prefix = {key: value[:native_count] if isinstance(value, (torch.Tensor, list, tuple)) else value
              for key, value in native.items()}
    if any(prefix[name].shape[1:] != official[name].shape[1:] for name in ACTOR_FIELDS):
        raise ValueError("Native/official raw actor geometry differs; implicit resampling is not permitted")
    combined = {name: torch.cat((prefix[name], official[name]), dim=0) for name in ACTOR_FIELDS}
    combined["prompt"] = list(prefix["prompt"]) + list(official["prompt"])
    combined["actor_source"] = ["native"] * native_count + ["official_bc"] * official_count
    return combined, prefix, native_count
