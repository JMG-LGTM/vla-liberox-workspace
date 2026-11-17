"""GPU-free identity and normalization contract for the pinned LIBERO-X base.

Adapted from vla_rynn_iql/pi05_assets.py at origin/server e27f426. No platform
imports are required. The image contract includes the local vertical-only fix.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
SOURCE = "Adam-YAN/pi05-liberox-base"
SOURCE_REVISION = "4664e4725d2e72c3a8af8b09adfda41eaf74514c"
BASE_ID = "pi05-liberox-base"
STATS_KEY = "meituan/LIBERO-X"
IDENTITY_FILE = "pi05_identity.json"
NORM_FILE = f"assets/{STATS_KEY}/norm_stats.json"
CONTRACT = {
    "io": {"cameras": ["agentview", "robot0_eye_in_hand"], "image_size": 224,
           "image_flip_vertical": True, "image_flip_horizontal": False,
           "proprio": "eef_position_axis_angle_gripper", "proprio_dim": 8,
           "native_action_horizon": 10, "padded_action_dim": 32, "action_dim": 7,
           "replay_horizon": 8, "control_hz": 20, "action_codec": "libero_env_v1"},
    "architecture": {"config_name": "pi05_libero", "discrete_state_input": False},
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value) -> str:
    """Exact platform serialization; changing separators breaks compatibility."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str,
                                     separators=(",", ":")).encode()).hexdigest()


def validate_identity(metadata: dict) -> dict:
    expected = {"schema_version": 1, "family": "pi05", "openpi_commit": OPENPI_COMMIT,
                "source": SOURCE, "source_revision": SOURCE_REVISION,
                "config_name": "pi05_libero", "native_action_horizon": 10,
                "native_action_dim": 32, "discrete_state_input": False}
    if not isinstance(metadata, dict):
        raise ValueError("pi05_identity.json must be an object")
    for key, value in expected.items():
        if metadata.get(key) != value or type(metadata.get(key)) is not type(value):
            raise ValueError(f"Untrusted pi05 checkpoint identity: {key} must be {value!r}")
    if metadata.get("conversion_precision", "float32") not in {"bfloat16", "float32"}:
        raise ValueError("Unsupported conversion_precision")
    files = metadata.get("files")
    if not isinstance(files, dict) or set(files) != {"model.safetensors", NORM_FILE}:
        raise ValueError("Checkpoint identity must bind model.safetensors and LIBERO-X normalization")
    if any(not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
           for digest in files.values()):
        raise ValueError("Checkpoint identity requires SHA-256 hex digests")
    return copy.deepcopy(metadata)


def checkpoint_identity(directory: str | Path, *, hash_file=sha256_file) -> dict:
    """Read and hash every required asset without importing Torch or OpenPI."""
    directory = Path(directory).expanduser().resolve()
    identity = validate_identity(json.loads((directory / IDENTITY_FILE).read_text()))
    for name, digest in identity["files"].items():
        if hash_file(directory / name) != digest:
            raise ValueError(f"pi05 checkpoint SHA-256 mismatch: {name}")
    return identity


def identity_digest(identity: dict) -> str:
    return stable_hash(validate_identity(identity))


def load_normalization(directory: str | Path, *, verify: bool = True) -> dict:
    """Read quantiles only: critic never needs to instantiate the 3.6B policy."""
    if verify:
        checkpoint_identity(directory)
    raw = json.loads((Path(directory).expanduser() / NORM_FILE).read_text())["norm_stats"]
    import numpy as np
    result = {}
    for name, size in (("actions", 7), ("state", 8)):
        lo, hi = (np.asarray(raw[name][key], dtype=np.float64) for key in ("q01", "q99"))
        if (lo.shape != (size,) or hi.shape != (size,) or not np.isfinite(lo).all()
                or not np.isfinite(hi).all() or np.any(hi < lo)):
            raise ValueError(f"Invalid pi05 normalization quantiles: {name}")
        result[name] = {"q01": lo.tolist(), "q99": hi.tolist(), "codec": "openpi_quantile"}
    return result


def normalize(values, stats: dict):
    """OpenPI quantile scaling without clipping or VLA gripper remapping."""
    import numpy as np
    values = np.asarray(values)
    size = values.shape[-1]
    lo, hi = np.asarray(stats["q01"][:size]), np.asarray(stats["q99"][:size])
    if len(lo) != size or len(hi) != size:
        raise ValueError("Normalization dimension mismatch")
    return ((values - lo) / (hi - lo + 1e-6) * 2 - 1).astype(np.float32)


def platform_model_settings(config: dict, identity: dict, *, checkpoint: str | None = None) -> dict:
    backbone = config.get("backbone", "frozen")
    steps = config.get("num_inference_steps", 10)
    if backbone not in {"frozen", "full"}:
        raise ValueError("pi05 backbone must be frozen or full")
    if type(steps) is not int or not 1 <= steps <= 100:
        raise ValueError("num_inference_steps must be in 1..100")
    if "contract" in config and config["contract"] != CONTRACT:
        raise ValueError("The pinned LIBERO-X contract must use vertical-only image flipping")
    path = checkpoint or config.get("checkpoint") or config.get("base_checkpoint")
    if not path:
        raise ValueError("model.checkpoint is required")
    revision = identity_digest(identity)
    if config.get("base_revision") not in (None, revision):
        raise ValueError("Configured model revision differs from verified identity")
    return {"backbone": backbone, "environment": "pi05", "base_revision": revision,
            "num_inference_steps": steps, "family": "pi05", "base_id": BASE_ID,
            "base_checkpoint": str(path), "stats_key": STATS_KEY,
            "contract": copy.deepcopy(CONTRACT)}


def parent_snapshot(settings: dict) -> dict:
    value = {"id": BASE_ID, "family": "pi05", "checkpoint": settings["base_checkpoint"],
             "stats_key": STATS_KEY, "revision": settings["base_revision"], "source": SOURCE,
             "source_revision": SOURCE_REVISION, "contract": copy.deepcopy(CONTRACT)}
    return {**value, "sha256": stable_hash(value)}
