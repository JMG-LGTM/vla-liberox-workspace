"""Explicit observation/action-only official demonstrations for BC, never transitions.

Published LIBERO-X arrays are consumed unchanged: no clipping, re-timing, pose
conversion, gripper remapping or extra image flip. Their physical control-period
and pose conventions are not asserted to equal platform simulator recordings.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import os

import numpy as np

from .archive_io import load_npz_arrays
from .data import (_manifest_identity, _npz_header, atomic_json, safe_path,
                   sha256_file, verify_dataset)

MIXED_FORMAT = "liberox-native-with-bc-demonstrations"
DEMO_FORMAT = "liberox-official-bc-observation-action-v1"
REPOSITORY = "meituan/LIBERO-X"
REVISION = "73053111f932d4dbaee995e3f06c2f42b3ad4adc"
CONTRACT = {
    "schema_version": 1, "method": "bc", "current_observation_only": True,
    "numeric_values": "published_original_no_clipping_or_pose_action_conversion",
    "image_orientation": "as_published_no_flip", "timestamps": "published_original_no_resampling",
    "published_fps": 10, "physical_control_period": "not_verified",
    "state_rotation_and_action_frame": "not_verified_by_published_generator",
    "manipulability": "not_specified_for_official_training_demonstrations",
    "environment_success_done_reward_and_terminal_next_observation": "not_available_not_fabricated",
}


def validate_demonstrations(manifest: dict) -> None:
    demos = manifest.get("bc_demonstrations", [])
    if manifest["format"] != MIXED_FORMAT:
        if demos or "bc_demonstration_contract" in manifest:
            raise ValueError("BC demonstrations require the explicit mixed dataset format")
        return
    if not isinstance(demos, list) or not demos or manifest.get("bc_demonstration_contract") != CONTRACT:
        raise ValueError("Missing or incompatible BC-only demonstration contract")
    if manifest.get("total_training_episode_count") != manifest["member_count"] + len(demos):
        raise ValueError("Mixed BC episode count differs from native plus official membership")
    ids = set(manifest["member_run_ids"])
    for ep in demos:
        if (ep.get("format") != DEMO_FORMAT or ep.get("kind") != "official_bc"
                or ep.get("source") != "official_human_demonstration"
                or ep.get("source_repository") != REPOSITORY or ep.get("source_revision") != REVISION
                or ep.get("observation_orientation") != "as_published_no_flip"
                or ep.get("root_run_id") != ep.get("run_id") or ep.get("split") != "train"
                or type(ep.get("recorded_action_count")) is not int or ep["recorded_action_count"] < 1
                or not isinstance(ep.get("prompt"), str) or not ep["prompt"].strip()):
            raise ValueError("Invalid explicit official BC descriptor")
        if not isinstance(ep.get("run_id"), str) or not ep["run_id"] or ep["run_id"] in ids:
            raise ValueError("Duplicated or invalid official demonstration ID")
        ids.add(ep["run_id"])
        if any(key in ep for key in ("done", "reward", "success", "terminal_step", "stopped_reason", "next_observation", "bootstrap_mask")):
            raise ValueError("BC-only demonstrations must not invent transition/outcome fields")
        for key in ("numeric_path", "observations_path", "metadata_path"):
            if ep.get(key) not in manifest["files"]:
                raise ValueError("Unhashed official BC file: " + key)


def verify_demo_payloads(root: Path, manifest: dict) -> None:
    for ep in manifest.get("bc_demonstrations", []):
        numeric = load_npz_arrays(safe_path(root, ep["numeric_path"]), keys=("state", "actions"),
                                 expected_sha256=manifest["files"][ep["numeric_path"]]["sha256"])
        count = ep["recorded_action_count"]
        for name, shape in (("state", (count, 8)), ("actions", (count, 7))):
            a = numeric[name]
            if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
                raise ValueError("Invalid original official BC " + name)
        for key in ("image", "wrist_image"):
            shape, dtype = _npz_header(safe_path(root, ep["observations_path"]), key)
            if shape != (count, 256, 256, 3) or dtype != np.uint8:
                raise ValueError("Invalid published official RGB observations")
        metadata = json.loads(safe_path(root, ep["metadata_path"]).read_text())
        if any(metadata.get(key) != ep[key] for key in ("episode_index", "task_index", "prompt", "source_sha256")):
            raise ValueError("Official BC metadata differs from its original provenance")
        if metadata.get("rows") != count or metadata.get("image_orientation") != "as_published_no_flip":
            raise ValueError("Official BC original row/orientation contract differs")


def demo_rows(episodes: list[dict], horizon: int, splits: dict) -> list[dict]:
    rows = []
    for ep in episodes:
        for start in range(0, ep["recorded_action_count"], horizon):
            rows.append(dict(run_id=ep["run_id"], root_run_id=ep["root_run_id"], split=splits[ep["root_run_id"]],
                             prompt=ep["prompt"], start=start, end=min(start + horizon, ep["recorded_action_count"]),
                             action_source="official_human", origin_run_id=ep["run_id"], copied_prefix=False,
                             source_type="official_bc"))
    return rows


def attach_official_demonstrations(native_root: str | Path, official_root: str | Path,
                                   output: str | Path) -> dict:
    """Create a NEW portable mixed package; never modify either source package."""
    native_root, official_root, output = (Path(p).resolve() for p in (native_root, official_root, output))
    native = verify_dataset(native_root)
    if native["format"] != "liberox-native-post-training":
        raise ValueError("Attach requires an unmodified native-only package")
    if output.exists() or output.is_relative_to(native_root) or output.is_relative_to(official_root):
        raise ValueError("Output must be a new directory outside both inputs")
    # Hard-coded for Scene24 L1: exactly the two official LIBERO-X demonstrations of its task
    # (meituan/LIBERO-X episodes 1719 and 1720, task 264). Other scenes need these values made configurable.
    prepared = json.loads((official_root / "processed_manifest.json").read_text())
    if (prepared.get("kind") != "official_human_bc_intermediate" or prepared.get("source_repository") != REPOSITORY
            or prepared.get("source_revision") != REVISION or prepared.get("episode_count") != 2
            or [e.get("episode_index") for e in prepared["episodes"]] != [1719, 1720]):
        raise ValueError("Expected the two pinned official Scene24 demonstrations")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + output.name + ".", dir=output.parent))
    try:
        shutil.copytree(native_root, temporary, dirs_exist_ok=True)
        manifest = deepcopy(native)
        manifest.update(format=MIXED_FORMAT, bc_demonstration_contract=deepcopy(CONTRACT), bc_demonstrations=[])
        def copy_original(relative: str, expected: dict | None = None):
            source = safe_path(official_root, relative)
            target_relative = "official_bc/" + relative
            target = safe_path(temporary, target_relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            before = source.stat()
            shutil.copyfile(source, target)
            digest = sha256_file(target)
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or sha256_file(source) != digest:
                raise ValueError("Official source changed while copying: " + relative)
            if expected and (digest != expected["sha256"] or target.stat().st_size != expected["bytes"]):
                raise ValueError("Original processed official SHA mismatch: " + relative)
            manifest["files"][target_relative] = dict(size=target.stat().st_size, sha256=digest)
            return target_relative
        for name in ("source_manifest.json", "processed_manifest.json", "validation.json"):
            copy_original(name)
        for metadata in prepared["episodes"]:
            # Task 264 is the Scene24 L1 task (see the episode check above).
            if metadata["task_index"] != 264 or any(ep["prompt"] != metadata["prompt"] for ep in native["episodes"]):
                raise ValueError("Official Scene24 instruction differs from selected native training paths")
            directory = "processed/episode_%06d" % metadata["episode_index"]
            paths = {}
            for name in ("numeric.npz", "images.npz", "bc_chunks_h8.npz", "metadata.json"):
                paths[name] = copy_original(directory + "/" + name,
                                            None if name == "metadata.json" else metadata["files"][name])
            ident = "official_meituan_libero_x_episode_%06d" % metadata["episode_index"]
            manifest["bc_demonstrations"].append(dict(format=DEMO_FORMAT, run_id=ident, root_run_id=ident,
                kind="official_bc", source="official_human_demonstration", split="train", prompt=metadata["prompt"],
                recorded_action_count=metadata["rows"], episode_index=metadata["episode_index"], task_index=metadata["task_index"],
                source_repository=REPOSITORY, source_revision=REVISION, source_sha256=metadata["source_sha256"],
                observation_orientation="as_published_no_flip", numeric_path=paths["numeric.npz"],
                observations_path=paths["images.npz"], metadata_path=paths["metadata.json"]))
        manifest["total_training_episode_count"] = manifest["member_count"] + len(manifest["bc_demonstrations"])
        card = """# Scene24 BC: native success/takeover paths plus official demonstrations

This portable package contains %d selected native final paths and two official
human demonstrations for the matching Scene24 instruction. Native ancestor
records preserve takeover provenance and do not expand membership. Native paths
use the unchanged five-step success rule and exclude post-success actions.

Official source: meituan/LIBERO-X revision %s, episodes 1719 and 1720, task 264.
Their original numeric arrays, RGB frames, timestamps and provenance files are
preserved byte-for-byte. They provide current observations and future action
chunks for BC only: no environment reward, done, success label or N+1 terminal
observation is invented. This dataset is rejected by IQL and critic methods.

Native LIBERO RGB is flipped vertically once in derived replay. Official RGB is
consumed exactly as published, without an extra flip. Actions/state
are kept as published: no clipping, gripper remapping, pose conversion or
resampling. The official metadata declares 10fps; physical control period and
pose/frame conventions have not been verified against a published generator.
These limits are retained in bc_demonstration_contract, rather than relabeling
official rows as native 20Hz simulator records. The official demonstrations do
not specify a manipulability controller setting. Policy inputs use the pinned
OpenPI pi05_libero transformations and model normalization as usual.

The manifest binds all packaged files by size and SHA256; native member_count
and records retain their original meaning. total_training_episode_count includes
the separate bc_demonstrations. No source input or collection-platform file was
modified during packaging.
""" % (manifest["member_count"], REVISION)
        (temporary / "README.md").write_text(card)
        manifest["files"]["README.md"] = dict(size=(temporary / "README.md").stat().st_size,
                                              sha256=sha256_file(temporary / "README.md"))
        manifest["dataset_sha256"] = _manifest_identity(manifest)
        atomic_json(temporary / "manifest.json", manifest)
        result = verify_dataset(temporary)
        os.replace(temporary, output)
        return result
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
