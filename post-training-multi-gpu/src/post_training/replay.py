"""Immutable mmap replay; archives are decoded once per recording at preprocessing."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .archive_io import load_npz_arrays
from .data import (TERMINATION_RULE, atomic_json, inspect_dataset, safe_path,
                   sha256_file, stable_hash, verify_dataset)


def normalize(values: np.ndarray, stats: dict) -> np.ndarray:
    """OpenPI quantile normalization, preserving values outside [-1, 1]."""
    values = np.asarray(values, dtype=np.float32)
    lower, upper = np.asarray(stats["q01"], np.float32), np.asarray(stats["q99"], np.float32)
    if lower.shape != values.shape[-1:] or upper.shape != lower.shape or not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("Normalization dimensions or finiteness differ from replay")
    if np.any(upper < lower):
        raise ValueError("Invalid quantile ordering")
    return (2.0 * (values - lower) / (upper - lower + 1e-6) - 1.0).astype(np.float32)


def validate_normalization(stats: dict) -> dict:
    """Validate complete exported quantile dictionaries without stripping provenance."""
    if not isinstance(stats, dict):
        raise ValueError("Normalization must be an actions/state quantile dictionary")
    for key, dimension in (("actions", 7), ("state", 8)):
        if not isinstance(stats.get(key), dict) or not {"q01", "q99"} <= stats[key].keys():
            raise ValueError(f"Normalization is missing {key} q01/q99 statistics")
        if stats[key].get("codec", "openpi_quantile") != "openpi_quantile":
            raise ValueError(f"Unsupported {key} normalization codec")
        normalize(np.zeros(dimension, dtype=np.float32), stats[key])
    # Canonical hashing also rejects NaN/Inf in auxiliary metadata.
    stable_hash(stats)
    return stats


def load_normalization_file(path: str | Path) -> dict:
    """Load normalization.json or the normalization field of critic export metadata."""
    payload = json.loads(Path(path).expanduser().read_text())
    if isinstance(payload, dict) and "normalization" in payload and not {"actions", "state"} <= payload.keys():
        payload = payload["normalization"]
    return validate_normalization(payload)


def bootstrap_mask(terminal_step: int | None, end: int, recorded_action_count: int) -> float:
    """Truncation retains its real N+1 next observation; confirmed success is terminal."""
    if end < 1 or end > recorded_action_count:
        raise ValueError("Transition endpoint is outside the native recording")
    return 0.0 if terminal_step is not None and end == terminal_step + 1 else 1.0


def fit_normalization(data_root: Path, manifest: dict, info: dict) -> dict:
    """Fit quantiles on selected train-root actions/states, excluding validation roots."""
    by_id = {ep["run_id"]: ep for ep in info["episodes"]}
    grouped = {}
    for row in info["rows"]:
        if row["split"] == "train":
            grouped.setdefault(row["run_id"], []).append(row)
    action_values, state_values = [], []
    for run_id, rows in grouped.items():
        ep = by_id[run_id]
        official = ep.get("kind") == "official_bc"
        path = ep["numeric_path"] if official else ep["trajectory_path"]
        trajectory = load_npz_arrays(safe_path(data_root, path), expected_sha256=manifest["files"][path]["sha256"])
        state = (trajectory["state"] if official else
                 np.concatenate([trajectory[key] for key in ("eef_position", "eef_axis_angle", "gripper_qpos")], axis=-1))
        indices = sorted({step for row in rows for step in ((row["start"],) if official else (row["start"], row["end"]))})
        state_values.append(state[indices])
        actions = trajectory["actions"] if official else trajectory["env_action"]
        action_values.extend(actions[row["start"]:row["end"]] for row in rows)
    if not action_values:
        raise ValueError("Training split is empty; cannot fit normalization")
    result = {}
    for key, values in (("actions", action_values), ("state", state_values)):
        stacked = np.concatenate(values, axis=0)
        low, high = np.quantile(stacked, [.01, .99], axis=0)
        result[key] = dict(q01=low.tolist(), q99=high.tolist(), codec="openpi_quantile")
    result["provenance"] = dict(source="training_roots_only", dataset_sha256=manifest["dataset_sha256"],
        root_run_ids=sorted({row["root_run_id"] for row in info["rows"] if row["split"] == "train"}))
    return result


def _metadata_hash(metadata: dict) -> str:
    return stable_hash({k: v for k, v in metadata.items() if k != "metadata_sha256"})


def validate_cache(directory: str | Path, fingerprint: str | None = None, *, hashes: bool = True) -> dict:
    directory = Path(directory)
    metadata = json.loads(safe_path(directory, "cache.json").read_text())
    if metadata.get("schema_version") != 1 or metadata.get("metadata_sha256") != _metadata_hash(metadata):
        raise ValueError("Cache metadata identity mismatch")
    if fingerprint is not None and metadata.get("fingerprint") != fingerprint:
        raise ValueError("Cache fingerprint differs from source/config/normalization")
    if metadata.get("fingerprint") != stable_hash(metadata["identity"]):
        raise ValueError("Cache identity hash mismatch")
    if len(metadata["records"]) != metadata["count"] or metadata["count"] < 1:
        raise ValueError("Cache record count mismatch")
    for name, spec in metadata["arrays"].items():
        if not name.isidentifier():
            raise ValueError("Unsafe cache field")
        path = safe_path(directory, f"{name}.npy")
        if hashes and sha256_file(path) != spec["sha256"]:
            raise ValueError(f"Cache SHA-256 mismatch: {name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != spec["shape"] or str(array.dtype) != spec["dtype"] or len(array) != metadata["count"]:
            raise ValueError(f"Cache shape/dtype mismatch: {name}")
    return metadata


def _pixels(agent: np.ndarray, wrist: np.ndarray, size: int) -> np.ndarray:
    value = torch.from_numpy(np.concatenate((agent, wrist), axis=-1)).permute(2, 0, 1)
    value = F.interpolate(value[None].float(), size=(size, size), mode="bilinear", align_corners=False)
    return value[0].clamp(0, 255).to(torch.uint8).numpy()


def build_replay_cache(data_root: str | Path, cache_root: str | Path, config: dict,
                       norm_stats: dict | None = None, *, rebuild: bool = False) -> tuple[Path, bool]:
    """Verify sources and produce an atomic, content-addressed mmap cache.

    Policy samples retain original env actions/proprioception and vertically flipped
    HWC images. Critic samples use OpenPI quantile actions/state, padded zero actions,
    small uint8 RGB views, sparse per-chunk reward and success-aware bootstrap masks.
    Explicit norm_stats are authoritative; if a configured file is also supplied,
    its complete dictionary must agree. Otherwise use the file, then train-only fitting.
    """
    data_root, cache_root = Path(data_root).resolve(), Path(cache_root).resolve()
    manifest = verify_dataset(data_root)
    info = inspect_dataset(data_root, config, verify=False)
    method = config.get("method", "bc")
    transitions = method in {"critic", "iql"}
    policy = method != "critic"
    if method not in {"bc", "iql", "critic"}:
        raise ValueError(f"Unknown training method {method}")
    data_cfg = config.get("data", {})
    if data_cfg.get("normalization"):
        configured_stats = load_normalization_file(data_cfg["normalization"])
        if norm_stats is not None and norm_stats != configured_stats:
            raise ValueError("Explicit norm_stats differs from data.normalization")
        if norm_stats is None:
            norm_stats = configured_stats
    if norm_stats is None:
        norm_stats = fit_normalization(data_root, manifest, info)
    norm_stats = validate_normalization(norm_stats)
    if data_cfg.get("image_flip_vertical", True) is not True or data_cfg.get("image_flip_horizontal", False) is not False:
        raise ValueError("pi05 LIBERO-X requires vertical flip=true and horizontal flip=false")
    size = int(config.get("critic", {}).get("image_size", 128))
    horizon = info["action_horizon"]
    if size < 1 or int(data_cfg.get("action_dim", 7)) != 7 or int(data_cfg.get("proprio_dim", 8)) != 8:
        raise ValueError("Invalid native LIBERO-X dimensions")
    identity = dict(schema_version=1, dataset_sha256=manifest["dataset_sha256"],
        action_horizon=horizon, method=method, normalization=norm_stats,
        critic_image_size=size if transitions else None,
        include_post_success=False, image_flip_vertical=True, image_flip_horizontal=False,
        termination_rule=TERMINATION_RULE, reward=manifest["reward"],
        split_rows_sha256=stable_hash(info["rows"]),
        implementation={name: sha256_file(Path(__file__).with_name(name)) for name in ("replay.py", "data.py", "archive_io.py")})
    mixed_bc = bool(manifest.get("bc_demonstrations"))
    if mixed_bc:
        identity.update(bc_demonstration_contract=manifest["bc_demonstration_contract"],
                        observation_orientation_by_source={"native": "vertical_flip_only", "official_bc": "as_published_no_flip"})
        identity["implementation"]["official_bc.py"] = sha256_file(Path(__file__).with_name("official_bc.py"))
    fingerprint = stable_hash(identity)
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / fingerprint
    with (cache_root / f".{fingerprint}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.is_dir() and not rebuild:
            validate_cache(target, fingerprint)
            return target, True
        temporary = Path(tempfile.mkdtemp(prefix=f".{fingerprint}.", dir=cache_root))
        arrays, records = {}, []
        by_id = {ep["run_id"]: ep for ep in info["episodes"]}
        loaded_id, trajectory, observations, proprio = None, None, None, None
        started = time.monotonic()
        try:
            for index, row in enumerate(info["rows"]):
                ep = by_id[row["run_id"]]
                official = ep.get("kind") == "official_bc"
                if loaded_id != ep["run_id"]:
                    trajectory, observations, proprio = None, None, None
                    numeric_path = ep["numeric_path"] if official else ep["trajectory_path"]
                    trajectory = load_npz_arrays(safe_path(data_root, numeric_path),
                        expected_sha256=manifest["files"][numeric_path]["sha256"])
                    observations = load_npz_arrays(safe_path(data_root, ep["observations_path"]),
                        keys=("image" if official else "agentview_image", "wrist_image"),
                        expected_sha256=manifest["files"][ep["observations_path"]]["sha256"])
                    proprio = (trajectory["state"] if official else
                        np.concatenate([trajectory[key] for key in ("eef_position", "eef_axis_angle", "gripper_qpos")], axis=-1)).astype(np.float32)
                    loaded_id = ep["run_id"]
                start, end = row["start"], row["end"]
                length = end - start
                agent = (observations["image"][start] if official else observations["agentview_image"][start, ::-1]).copy()
                wrist = (observations["wrist_image"][start] if official else observations["wrist_image"][start, ::-1]).copy()
                raw_actions = np.zeros((horizon, 7), np.float32)
                raw_actions[:length] = trajectory["actions" if official else "env_action"][start:end]
                action_mask = np.arange(horizon) < length
                values = dict(raw_proprio=proprio[start], raw_actions=raw_actions,
                    action_mask=action_mask, chunk_length=np.asarray(length, np.int64))
                if not mixed_bc:
                    values.update(remaining_blocks=np.asarray(row["remaining_blocks"], np.int64),
                                  success_terminal=np.asarray(row["success_terminal"], bool))
                if policy:
                    values.update(agent_image=agent, wrist_image=wrist)
                if transitions:
                    normalized_actions = np.zeros_like(raw_actions)
                    normalized_actions[:length] = normalize(raw_actions[:length], norm_stats["actions"])
                    values.update(actions=normalized_actions, proprio=normalize(proprio[start], norm_stats["state"]),
                        next_proprio=normalize(proprio[end], norm_stats["state"]), pixels=_pixels(agent, wrist, size),
                        next_pixels=_pixels(observations["agentview_image"][end, ::-1].copy(), observations["wrist_image"][end, ::-1].copy(), size),
                        reward=np.asarray(row["reward"], np.float32),
                        bootstrap_mask=np.asarray(bootstrap_mask(ep["terminal_step"], end, ep["recorded_action_count"]), np.float32))
                for name, value in values.items():
                    value = np.asarray(value)
                    if index == 0:
                        arrays[name] = np.lib.format.open_memmap(temporary / f"{name}.npy", mode="w+", dtype=value.dtype, shape=(info["count"], *value.shape))
                    if arrays[name].shape[1:] != value.shape:
                        raise ValueError(f"Inconsistent cache shape: {ep['run_id']}/{name}")
                    arrays[name][index] = value
                records.append({key: value for key, value in row.items() if key not in values})
                if index == 0 or (index + 1) % 256 == 0 or index + 1 == info["count"]:
                    print(f"CACHE {index + 1}/{info['count']} chunks elapsed={time.monotonic() - started:.1f}s", flush=True)
            specs = {}
            for name, value in arrays.items():
                value.flush()
                specs[name] = dict(shape=list(value.shape), dtype=str(value.dtype), sha256=sha256_file(temporary / f"{name}.npy"))
            metadata = dict(schema_version=1, fingerprint=fingerprint, identity=identity, count=info["count"],
                counts=info["counts"], arrays=specs, records=records, root_counts=info["root_counts"],
                normalization=norm_stats, termination_rule=TERMINATION_RULE)
            metadata["metadata_sha256"] = _metadata_hash(metadata)
            atomic_json(temporary / "cache.json", metadata)
            validate_cache(temporary, fingerprint)
            if target.exists():
                retired = cache_root / f".{fingerprint}.retired.{os.getpid()}"
                os.replace(target, retired)
                os.replace(temporary, target)
                shutil.rmtree(retired)
            else:
                os.replace(temporary, target)
        finally:
            arrays.clear()
            if temporary.exists():
                shutil.rmtree(temporary)
    return target, False


class CachedReplayDataset(Dataset):
    """Worker-safe lazy mmap reads; no archives or decompression during training."""
    def __init__(self, directory: str | Path, split: str | None = "train"):
        self.directory = Path(directory)
        self.metadata = validate_cache(self.directory, hashes=False)
        split = "val" if split == "validation" else split
        if split not in (None, "train", "val"):
            raise ValueError(f"Unknown replay split: {split}")
        self.indices = [i for i, row in enumerate(self.metadata["records"]) if split is None or row["split"] == split]
        self.records = [self.metadata["records"][i] for i in self.indices]
        self.arrays = None
        self.prompt_groups = {}
        for i, row in enumerate(self.records):
            self.prompt_groups.setdefault(row["prompt"], []).append(i)

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        return {**self.__dict__, "arrays": None}

    def __getitem__(self, index):
        if self.arrays is None:
            self.arrays = {name: np.load(self.directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                           for name in self.metadata["arrays"]}
        global_index = self.indices[index]
        # Copy only the indexed sample: tensors cannot mutate shared file-backed arrays.
        return {**self.records[index], **{name: torch.from_numpy(np.array(array[global_index], copy=True))
                                        for name, array in self.arrays.items()}}
