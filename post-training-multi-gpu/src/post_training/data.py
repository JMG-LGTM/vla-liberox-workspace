"""Portable dataset packaging and replay metadata.

Chunk boundaries and prefix deduplication derive from the local platform data.py;
all I/O, integrity checking, packaging and grouped splits are standalone.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterator
import zipfile

import numpy as np

from .archive_io import load_npz_arrays

SCHEMA_VERSION = 1
TERMINATION_RULE = "success_aware_recording_end_v1"

def replay_end_step(episode: dict[str, Any], include_post_success: bool = True) -> int:
    """Exclusive training endpoint; physical recordings and reward indices stay intact."""
    terminal = episode.get("terminal_step")
    return (int(episode["recorded_action_count"]) if include_post_success or terminal is None
            else int(terminal) + 1)

def confirmed_terminal_step(done: np.ndarray, consecutive_steps: int) -> int | None:
    """Return the action index that confirms a consecutive success streak."""
    if consecutive_steps < 1:
        raise ValueError("consecutive_steps must be positive")
    streak = 0
    for index, value in enumerate(np.asarray(done, dtype=bool)):
        streak = streak + 1 if bool(value) else 0
        if streak >= consecutive_steps:
            return index
    return None

def action_source_segments(sources: list[str], end: int) -> list[dict[str, Any]]:
    """Return a compact partition of ``[0, end)`` by action producer."""
    if not 0 < end <= len(sources):
        raise ValueError(f"Invalid action-source endpoint {end} for {len(sources)} actions")
    segments: list[dict[str, Any]] = []
    start = 0
    current = sources[0]
    for index in range(1, end):
        if sources[index] == current:
            continue
        segments.append({"start": start, "end": index, "action_source": current})
        start, current = index, sources[index]
    segments.append({"start": start, "end": end, "action_source": current})
    return segments

def build_semi_mdp_chunks(
    *,
    first: int,
    end: int,
    horizon: int,
    source_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build variable-duration transitions without crossing controller boundaries.

    The fixed horizon is only a maximum. Action-source changes and branch resume
    points are hard boundaries, so a takeover can end the final policy chunk at
    its actual executed length instead of pretending that the full horizon ran.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not 0 <= first < end:
        raise ValueError(f"Invalid replay interval [{first}, {end})")
    boundaries = {first, end}
    segment_lookup: list[dict[str, Any]] = []
    for segment in source_segments:
        segment_start = max(first, int(segment["start"]))
        segment_end = min(end, int(segment["end"]))
        if segment_start >= segment_end:
            continue
        source = str(segment["action_source"])
        segment_lookup.append({
            "start": segment_start, "end": segment_end, "action_source": source,
        })
        # Each controller/source segment owns its own horizon grid. A source
        # change therefore starts a new macro-action sequence.
        boundaries.update(range(segment_start, segment_end, horizon))
        boundaries.add(segment_end)
    covered = sum(segment["end"] - segment["start"] for segment in segment_lookup)
    if covered != end - first:
        raise ValueError(
            f"Action-source segments cover {covered} actions, expected {end - first}"
        )

    ordered = sorted(boundaries)
    chunks: list[dict[str, Any]] = []
    segment_index = 0
    for start, stop in zip(ordered, ordered[1:]):
        while segment_lookup[segment_index]["end"] <= start:
            segment_index += 1
        segment = segment_lookup[segment_index]
        if not (segment["start"] <= start < stop <= segment["end"]):
            raise ValueError(f"Chunk [{start}, {stop}) crosses an action-source boundary")
        chunks.append({
            "start": start,
            "length": stop - start,
            "end": stop,
            "action_source": segment["action_source"],
            "transition_type": str(segment["action_source"]),
            "interrupted": False,
            "copied_prefix": False,
        })
    return chunks

def replay_chunks(episode: dict[str, Any]) -> list[dict[str, Any]]:
    """Return full-recording replay without changing prepared/cache identities.

    Older schema-4 manifests kept the post-success tail in evaluation_chunks
    and annotated its rewards at those same indices. Reuse that complete list
    in memory, while retaining the manifest and its existing reward hash.
    """
    recorded_end = int(episode["recorded_action_count"])
    prepared = episode["chunks"]
    complete = bool(prepared) and int(prepared[-1]["end"]) == recorded_end
    chunks = prepared if complete else episode.get("evaluation_chunks", [])
    run_id = episode["run_id"]
    error = (
        f"Run {run_id} lacks aligned full-recording replay chunks; "
        "rerun preparation and reward materialization for the complete recording"
    )
    expected_start = 0
    for chunk in chunks:
        start, end, length = int(chunk["start"]), int(chunk["end"]), int(chunk["length"])
        if start != expected_start or length <= 0 or end - start != length:
            raise ValueError(error)
        expected_start = end
    if expected_start != recorded_end or not chunks:
        raise ValueError(error)
    if not complete:
        if len(prepared) > len(chunks) or any(
            any(original[key] != full[key] for key in ("start", "end", "length", "action_source"))
            for original, full in zip(prepared, chunks)
        ):
            raise ValueError(error)
        return [
            {**chunk, "transition_type": chunk["action_source"]}
            if chunk.get("transition_type") == "post_terminal_evaluation" else chunk
            for chunk in chunks
        ]
    return chunks

def replay_chunk_key(episode: dict[str, Any], chunk: dict[str, Any]) -> tuple[str, int, int, str]:
    origin = chunk.get("origin_run_id") or (episode["root_run_id"]
        if chunk.get("copied_prefix", False) else episode["run_id"])
    return str(origin), int(chunk["start"]), int(chunk["end"]), str(chunk["action_source"])

def iter_unique_replay_chunks(
    episodes: list[dict[str, Any]], *, split: str | None = None,
    include_post_success: bool = True,
) -> Iterator[tuple[dict[str, Any], int]]:
    """Select the training interval, retaining full reward indices and de-duplicating prefixes."""
    full_chunks = {str(episode["run_id"]): replay_chunks(episode) for episode in episodes}
    for episode in episodes:
        end = replay_end_step(episode, include_post_success)
        if not any(int(chunk["end"]) == end for chunk in full_chunks[str(episode["run_id"])]):
            raise ValueError(f"Run {episode['run_id']} lacks a chunk at the confirmed success boundary; rerun Prepare")
    copied_prefix_keys = {
        replay_chunk_key(episode, chunk)
        for episode in episodes
        for chunk in full_chunks[str(episode["run_id"])]
        if bool(chunk.get("copied_prefix", False))
    }
    seen_copied_prefixes: set[tuple[str, int, int, str]] = set()
    produced = {replay_chunk_key(episode, chunk)
        for episode in episodes if split is None or episode["split"] == split
        for chunk in full_chunks[str(episode["run_id"])]
        if int(chunk["end"]) <= replay_end_step(episode, include_post_success)
        and not chunk.get("copied_prefix", False)}
    ordered = sorted(
        episodes,
        key=lambda item: (item.get("kind") == "branch", str(item["run_id"])),
    )
    for episode in ordered:
        if split is not None and episode["split"] != split:
            continue
        for chunk_index, chunk in enumerate(full_chunks[str(episode["run_id"])]):
            if int(chunk["end"]) > replay_end_step(episode, include_post_success):
                continue
            key = replay_chunk_key(episode, chunk)
            if chunk.get("copied_prefix", False) and key in produced:
                continue
            if key in copied_prefix_keys:
                if key in seen_copied_prefixes:
                    continue
                seen_copied_prefixes.add(key)
            yield episode, chunk_index

def add_episode_chunks(episode: dict[str, Any], horizon: int) -> None:
    """Describe the complete recording identically for Prepare and inheritance."""
    resume_step = int(episode["resume_step"] or 0) if episode["kind"] == "branch" else 0
    success_end = (int(episode["terminal_step"]) + 1 if episode["terminal_step"] is not None
                   else int(episode["recorded_action_count"]))
    segments = episode.get("trajectory_segments")
    raw_handoffs = episode.get("handoff_steps", [])
    if not isinstance(raw_handoffs, list) or any(type(step) is not int or not 0 <= step <= int(episode["recorded_action_count"]) for step in raw_handoffs):
        raise ValueError("handoff_steps must contain integer recording boundaries")
    handoffs = set(raw_handoffs) | ({resume_step} if resume_step else set())
    if segments is not None:
        if not isinstance(segments, list):
            raise ValueError("trajectory_segments must be a list")
        cursor = 0
        for segment in segments:
            if (not isinstance(segment, dict) or type(segment.get("start")) is not int or type(segment.get("end")) is not int
                    or segment["start"] != cursor or not cursor < segment["end"] <= int(episode["recorded_action_count"])
                    or not isinstance(segment.get("origin_run_id"), str) or not segment["origin_run_id"]
                    or not isinstance(segment.get("policy_instruction"), str) or not segment["policy_instruction"].strip()):
                raise ValueError(f"Invalid instruction/origin segments: {episode['run_id']}")
            cursor = segment["end"]
        if cursor != int(episode["recorded_action_count"]):
            raise ValueError("Instruction/origin segments do not cover the recording")
    boundaries = handoffs | ({int(s["start"]) for s in segments} if segments else set())
    def split_sources(sources):
        result = []
        for source in sources:
            cuts = sorted({int(source["start"]), int(source["end"])} |
                          {point for point in boundaries if source["start"] < point < source["end"]})
            result.extend({**source, "start": start, "end": end} for start, end in zip(cuts, cuts[1:]))
        return result
    chunks = build_semi_mdp_chunks(first=0, end=success_end, horizon=horizon,
                                   source_segments=split_sources(episode["action_source_segments"]))
    if success_end < int(episode["recorded_action_count"]):
        chunks.extend(build_semi_mdp_chunks(first=success_end,
            end=int(episode["recorded_action_count"]), horizon=horizon,
            source_segments=split_sources(episode["recorded_action_source_segments"])))
    for chunk in chunks:
        if segments:
            segment = next(s for s in segments if s["start"] <= chunk["start"] < s["end"])
            chunk.update(origin_run_id=segment["origin_run_id"], policy_instruction=segment["policy_instruction"])
        if chunk["end"] in handoffs and chunk["length"] < horizon:
            chunk["interrupted"] = True
    if episode["kind"] == "branch":
        for chunk in chunks:
            if int(chunk["end"]) > resume_step:
                continue
            chunk["copied_prefix"] = True
            if int(chunk["end"]) == resume_step and int(chunk["length"]) < horizon:
                chunk["transition_type"] = str(chunk["action_source"]) + "_interrupted"
                chunk["interrupted"] = True
            else:
                chunk["transition_type"] = str(chunk["action_source"]) + "_prefix"
    episode["chunks"] = chunks
    episode["evaluation_chunks"] = [
        {**chunk, "transition_type": "post_terminal_evaluation"}
        if int(chunk["start"]) >= success_end else dict(chunk) for chunk in chunks]
    episode["reward_boundaries"] = sorted({
        value for chunk in episode["evaluation_chunks"] for value in (chunk["start"], chunk["end"])})

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def safe_path(root: Path, relative: str) -> Path:
    """Resolve a portable manifest path; reject traversal and symbolic links."""
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise ValueError(f"Unsafe relative path: {relative!r}")
    path = root / value
    if any((root / Path(*value.parts[:i])).is_symlink() for i in range(1, len(value.parts) + 1)):
        raise ValueError(f"Symbolic link is forbidden: {relative}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes dataset: {relative}")
    return path


def _npz_header(path: Path, key: str) -> tuple[tuple, np.dtype]:
    with zipfile.ZipFile(path) as archive, archive.open(f"{key}.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"Unsupported NPY version {version}")
        if dtype.hasobject or archive.getinfo(f"{key}.npy").file_size != stream.tell() + int(np.prod(shape)) * dtype.itemsize:
            raise ValueError(f"Invalid NPY payload: {path}:{key}")
        return shape, dtype


def describe_run(run_path: Path, *, horizon: int = 8, consecutive: int = 5,
                 expected_hashes: dict | None = None) -> dict:
    """Validate native N+1 recordings without decompressing image payloads."""
    run = json.loads(run_path.read_text())
    if run.get("status") != "COMPLETED" or run.get("error"):
        raise ValueError(f"Incomplete/error recording: {run_path}")
    run_id = str(run["id"])
    directory = run_path.parent / "episodes" / "episode_000"
    trajectory, observations = directory / "trajectory.npz", directory / "trajectory_observations.npz"
    arrays = load_npz_arrays(trajectory, expected_sha256=(expected_hashes or {}).get("trajectory"))
    count = len(arrays["env_action"])
    required_shapes = {"env_action": (count, 7), "done": (count,), "action_source": (count,),
                       "eef_position": (count + 1, 3), "eef_axis_angle": (count + 1, 3), "gripper_qpos": (count + 1, 2)}
    if count < 1:
        raise ValueError(f"Empty recording: {run_id}")
    for key, shape in required_shapes.items():
        if arrays[key].shape != shape:
            raise ValueError(f"Invalid {key} shape in {run_id}: {arrays[key].shape} != {shape}")
        if arrays[key].dtype.kind in "fc" and not np.isfinite(arrays[key]).all():
            raise ValueError(f"Nonfinite {key} in {run_id}")
    for key in ("agentview_image", "wrist_image"):
        shape, dtype = _npz_header(observations, key)
        if len(shape) != 4 or shape[0] != count + 1 or shape[-1] != 3 or min(shape[1:3]) < 1 or dtype != np.dtype("uint8"):
            raise ValueError(f"Invalid native N+1 RGB {key} in {run_id}")
    metadata = json.loads(str(arrays["metadata_json"].item())) if "metadata_json" in arrays else {}
    if metadata.get("error") or metadata.get("session_id", run_id) != run_id:
        raise ValueError(f"Invalid recording metadata: {run_id}")
    terminal = confirmed_terminal_step(arrays["done"], consecutive)
    reason = metadata.get("stopped_reason")
    if terminal is None and reason not in {"max_steps", "user_stop", "controller_stop"}:
        raise ValueError(f"Cannot bootstrap unknown/error stop {reason!r}: {run_id}")
    kind = str(run.get("kind", "original"))
    resume = int(run.get("resume_step") or 0) if kind == "branch" else 0
    if not 0 <= resume < count:
        raise ValueError(f"Invalid resume_step: {run_id}")
    sources = [str(v) for v in arrays["action_source"]]
    if not set(sources) <= {"policy", "human", "policy_requery"}:
        raise ValueError(f"Unsupported action producer: {run_id}")
    prompt = run.get("task")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Missing task instruction: {run_id}")
    episode = dict(run_id=run_id, root_run_id=str(run.get("root_session_id") or run_id),
        parent_run_id=run.get("parent_session_id"), kind=kind, resume_step=resume,
        task_id=run.get("task_id", ""), prompt=prompt.strip(), recorded_action_count=count,
        terminal_step=terminal, stopped_reason=reason, success=terminal is not None,
        action_source_segments=action_source_segments(sources, count),
        recorded_action_source_segments=action_source_segments(sources, count),
        handoff_steps=metadata.get("handoff_steps", run.get("handoff_steps", [])),
        source="takeover_chain" if kind == "branch" else ("autonomous_success" if terminal is not None else "autonomous_failure"),
        observation_orientation="libero_raw")
    segments = metadata.get("trajectory_segments", run.get("trajectory_segments"))
    if segments is not None:
        episode["trajectory_segments"] = segments
    add_episode_chunks(episode, horizon)
    return episode


def _manifest_identity(manifest: dict) -> str:
    return stable_hash({k: v for k, v in manifest.items() if k != "dataset_sha256"})


def validate_manifest_contract(manifest: dict) -> None:
    """Reject incompatible metadata before hashing potentially large payloads."""
    fixed = {
        "termination_rule": TERMINATION_RULE,
        "reward": dict(source="sparse", failure=-1.0, success=0.0, gamma=.99, discount_per="chunk"),
        "action_dim": 7, "proprio_dim": 8,
        "include_post_success": False, "success_consecutive_steps": 5,
    }
    for key, expected in fixed.items():
        value = manifest.get(key)
        if value != expected or (type(expected) in (int, bool) and type(value) is not type(expected)):
            raise ValueError(f"Dataset {key} differs from the supported contract")
    if any(type(manifest["reward"][key]) not in (int, float) for key in ("failure", "success", "gamma")):
        raise ValueError("Dataset reward values must be numeric")
    if type(manifest.get("action_horizon")) is not int or manifest["action_horizon"] < 1:
        raise ValueError("Dataset action_horizon must be positive")
    try:
        episodes, records, members = manifest["episodes"], manifest["records"], manifest["member_run_ids"]
        episode_ids = [ep["run_id"] for ep in episodes]
        record_ids = [record["run_id"] for record in records]
        for name, ids, count in (("member", members, manifest["member_count"]),
                                 ("record", record_ids, manifest["record_count"])):
            if not ids or type(count) is not int or count != len(ids) or len(set(ids)) != len(ids):
                raise ValueError(f"Dataset {name} count or IDs are inconsistent")
            if any(not isinstance(run_id, str) or not run_id for run_id in ids):
                raise ValueError(f"Invalid {name} run ID")
        if len(episode_ids) != len(members) or set(episode_ids) != set(members):
            raise ValueError("Episode/member membership differs")
        if not set(members) <= set(record_ids):
            raise ValueError("A selected member lacks its complete physical record")
        by_id = {record["run_id"]: record for record in records}
        for record in records:
            if record["selected"] is not (record["run_id"] in members):
                raise ValueError("Physical record selection flag differs from membership")
            if record["root_run_id"] not in by_id or (record["parent_run_id"] is not None and record["parent_run_id"] not in by_id):
                raise ValueError("Physical ancestor record is missing")
        for ep in episodes:
            record = by_id[ep["run_id"]]
            if any(ep[key] != record[key] for key in ("root_run_id", "parent_run_id", "source_manifest")):
                raise ValueError("Selected episode differs from its physical record")
            count, terminal = ep["recorded_action_count"], ep["terminal_step"]
            if type(count) is not int or count < 1:
                raise ValueError("Invalid recorded action count")
            if terminal is not None and (type(terminal) is not int or not 4 <= terminal < count):
                raise ValueError("Invalid five-step success confirmation index")
            if ep["success"] is not (terminal is not None):
                raise ValueError("Success flag disagrees with confirmation index")
            if terminal is None and ep["stopped_reason"] not in {"max_steps", "user_stop", "controller_stop"}:
                raise ValueError("Failure bootstrap requires a recognized native truncation")
            if ep["observation_orientation"] != "libero_raw":
                raise ValueError("Dataset requires native LIBERO RGB orientation")
            if ep["split"] not in {"train", "val", "validation"}:
                raise ValueError("Unknown packaged split")
            expected_source = ("takeover_chain" if ep["kind"] == "branch" else
                               "autonomous_success" if terminal is not None else "autonomous_failure")
            if ep["kind"] not in {"original", "branch"} or ep["source"] != expected_source:
                raise ValueError("Episode source category differs from its outcome/kind")
        if manifest["composition"] != dict(Counter(ep["source"] for ep in episodes)):
            raise ValueError("Dataset composition differs from selected episodes")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing or invalid dataset membership metadata: {exc}") from exc


def verify_dataset(root: str | Path) -> dict:
    """Verify every packaged byte and manifest identity before data is consumed."""
    root = Path(root).resolve()
    manifest = json.loads(safe_path(root, "manifest.json").read_text())
    from .official_bc import MIXED_FORMAT, validate_demonstrations, verify_demo_payloads
    if (type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("format") not in {"liberox-native-post-training", MIXED_FORMAT}):
        raise ValueError("Unsupported portable dataset format/version")
    if manifest.get("dataset_sha256") != _manifest_identity(manifest):
        raise ValueError("Dataset manifest SHA-256 identity mismatch")
    validate_manifest_contract(manifest)
    validate_demonstrations(manifest)
    for relative, expected in manifest["files"].items():
        path = safe_path(root, relative)
        if not path.is_file() or path.stat().st_size != expected["size"] or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"Dataset SHA-256/size mismatch: {relative}")
    if not manifest["episodes"] or len({ep["run_id"] for ep in manifest["episodes"]}) != len(manifest["episodes"]):
        raise ValueError("Empty or duplicated selected members")
    for ep in manifest["episodes"]:
        for key in ("source_manifest", "trajectory_path", "observations_path"):
            if ep[key] not in manifest["files"]:
                raise ValueError(f"Unhashed episode file: {ep[key]}")
            safe_path(root, ep[key])
    verify_demo_payloads(root, manifest)
    return manifest


def grouped_split(episodes: list[dict], fraction: float, seed: int,
                  strategy: str = "hash_rank_count") -> dict[str, str]:
    """Split complete roots, keeping parents, branches and copied prefixes together."""
    if not 0 <= fraction < 1:
        raise ValueError("validation_fraction must be in [0,1)")
    roots = sorted({str(ep["root_run_id"]) for ep in episodes})
    if strategy == "haizhou_hash_threshold":
        # Match Haizhou _split exactly: threshold each root, no fixed-count rounding.
        splits = {root: ("val" if int.from_bytes(hashlib.sha256(f"{seed}:{root}".encode()).digest()[:8], "big")
                        / float(2**64) < fraction else "train") for root in roots}
        if splits and all(value == "val" for value in splits.values()):
            # Upstream's all-validation fallback uses the first selected root.
            splits[str(episodes[0]["root_run_id"])] = "train"
        return splits
    if strategy != "hash_rank_count":
        raise ValueError("Unknown root split strategy")
    ranked = sorted(roots, key=lambda root: hashlib.sha256(f"{seed}:{root}".encode()).digest())
    if fraction and len(roots) < 2:
        raise ValueError("Root-grouped validation requires at least two roots")
    count = max(1, min(len(roots) - 1, round(len(roots) * fraction))) if fraction else 0
    val = set(ranked[:count])
    return {root: "val" if root in val else "train" for root in roots}


def inspect_dataset(root: str | Path, config: dict | None = None, *, verify: bool = True) -> dict:
    """Build exact lightweight chunk rows for dry-run, without image decompression."""
    root, config = Path(root), config or {}
    manifest = verify_dataset(root) if verify else json.loads((root / "manifest.json").read_text())
    from .official_bc import demo_rows
    demonstrations = deepcopy(manifest.get("bc_demonstrations", []))
    if demonstrations and config.get("method", "bc") != "bc":
        raise ValueError("Observation/action-only official demonstrations are BC-only; IQL/critic require genuine transitions")
    cfg = config.get("data", {})
    if cfg.get("include_post_success", False) is not False:
        raise ValueError("Version 1 requires include_post_success=false")
    if cfg.get("success_consecutive_steps", 5) != 5:
        raise ValueError("Success confirmation must use five consecutive steps")
    horizon = int(cfg.get("action_horizon", 8))
    episodes = deepcopy(manifest["episodes"])
    splits = grouped_split(episodes + demonstrations, float(cfg.get("validation_fraction", 0)),
                           int(cfg.get("seed", config.get("seed", 7))), cfg.get("split_strategy", "hash_rank_count"))
    for ep in episodes:
        ep["split"] = splits[ep["root_run_id"]]
        if horizon != manifest.get("action_horizon", 8):
            add_episode_chunks(ep, horizon)
    rows = []
    for ep, index in iter_unique_replay_chunks(episodes, include_post_success=False):
        chunk = replay_chunks(ep)[index]
        end = replay_end_step(ep, False)
        remaining = sum(int(c["end"]) > int(chunk["end"]) and int(c["end"]) <= end for c in ep["chunks"]) if ep["success"] else -1
        rows.append(dict(run_id=ep["run_id"], root_run_id=ep["root_run_id"], split=ep["split"],
            prompt=chunk.get("policy_instruction", ep["prompt"]), start=int(chunk["start"]), end=int(chunk["end"]),
            action_source=chunk["action_source"], origin_run_id=replay_chunk_key(ep, chunk)[0],
            remaining_blocks=remaining, success_terminal=bool(ep["success"] and int(chunk["end"]) == end),
            recording_success=bool(ep["success"]), copied_prefix=bool(chunk.get("copied_prefix")),
            bootstrap_mask=0.0 if ep["success"] and int(chunk["end"]) == end else 1.0,
            reward=0.0 if ep["success"] and int(chunk["end"]) > ep["terminal_step"] else -1.0))
    native_count = len(rows)
    if demonstrations:
        # Mixed BC batches share only truthful actor metadata. There are no fake
        # terminal/reward/success entries for demonstrations and no unused target
        # transition arrays. Native replay boundaries and sample order are retained.
        actor_keys = ("run_id", "root_run_id", "split", "prompt", "start", "end", "action_source", "origin_run_id", "copied_prefix")
        rows = [{**{key: row[key] for key in actor_keys}, "source_type": "native"} for row in rows]
        rows.extend(demo_rows(demonstrations, horizon, splits))
        for ep in demonstrations:
            ep["split"] = splits[ep["root_run_id"]]
        episodes.extend(demonstrations)
    train_count = sum(row["split"] == "train" for row in rows)
    val_count = sum(row["split"] == "val" for row in rows)
    has_critic = config.get("method", "bc") in {"critic", "iql"}
    held_out = has_critic and bool(cfg.get("validation_fraction", 0))
    evaluation = dict(metric_group=("validation" if held_out else "train_diagnostic") if has_critic else None,
                      split=("val" if held_out else "train") if has_critic else None,
                      held_out=held_out, chunks=(val_count if held_out else train_count) if has_critic else 0,
                      selects_best_checkpoint=held_out)
    return dict(dataset_sha256=manifest["dataset_sha256"], episodes=episodes, rows=rows,
                count=len(rows), train_count=train_count, val_count=val_count,
                dataset_chunks=len(rows), train_chunks=train_count, validation_chunks=val_count,
                critic_evaluation=evaluation, counts=dict(Counter(row["split"] for row in rows)),
                root_counts=dict(Counter(splits.values())), action_horizon=horizon,
                termination_rule=TERMINATION_RULE,
                native_members=manifest["member_count"], official_demonstrations=len(demonstrations),
                total_training_episodes=len(episodes), native_chunks=native_count,
                official_bc_chunks=len(rows) - native_count)


def package_dataset(dataset_root: str | Path, output: str | Path, *, dataset_id: str | None = None,
                    run_ids: list[str] | None = None, project: str = "libero_x_vla", horizon: int = 8) -> dict:
    """Copy complete immutable native records plus ancestors into a portable directory.

    Ancestors are provenance records only; they never expand the selected training set.
    The destination must not already exist. No platform source is changed.
    """
    source_root, output = Path(dataset_root).resolve(), Path(output).resolve()
    if bool(dataset_id) == bool(run_ids):
        raise ValueError("Provide exactly one dataset ID/path or nonempty run ID list")
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing dataset: {output}")
    project_root = source_root / "projects" / project
    frozen = None
    if dataset_id:
        candidate = Path(dataset_id)
        source_manifest = candidate if candidate.is_file() else project_root / "datasets" / dataset_id / "dataset.json"
        frozen = json.loads(source_manifest.read_text())
        selected = [str(member["run_id"]) for member in frozen["members"]]
        expected = {str(m["run_id"]): m["artifacts"] for m in frozen["members"]}
        dataset_name = frozen.get("id", source_manifest.parent.name)
    else:
        selected, expected, dataset_name = list(dict.fromkeys(run_ids)), {}, "explicit-runs"
    locations, runs = {}, {}
    for path in (project_root / "runs").rglob("run.json"):
        run = json.loads(path.read_text())
        run_id = str(run["id"])
        if run_id in locations:
            raise ValueError(f"Duplicate source run ID: {run_id}")
        locations[run_id], runs[run_id] = path, run
    all_ids = set(selected)
    pending = list(selected)
    while pending:
        run_id = pending.pop()
        if run_id not in runs:
            raise ValueError(f"Missing selected run/ancestor: {run_id}")
        run = runs[run_id]
        ancestors = [run.get("parent_session_id"), run.get("root_session_id")]
        ancestors.extend(s.get("origin_run_id") for s in run.get("trajectory_segments", []))
        for ancestor in ancestors:
            if ancestor and ancestor not in all_ids:
                all_ids.add(ancestor)
                pending.append(ancestor)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        files, episodes, records = {}, [], []
        for number, run_id in enumerate(sorted(all_ids), 1):
            path = locations[run_id]
            if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
                raise ValueError("Unsafe run ID")
            for name, relative in (("manifest", "run.json"), ("trajectory", "episodes/episode_000/trajectory.npz"), ("observations", "episodes/episode_000/trajectory_observations.npz")):
                if run_id in expected and sha256_file(path.parent / relative) != expected[run_id][name]["sha256"]:
                    raise ValueError(f"Frozen source SHA-256 mismatch: {run_id}/{name}")
            ep = describe_run(path, horizon=horizon, expected_hashes={k: v["sha256"] for k, v in expected.get(run_id, {}).items()})
            destination = temporary / "runs" / run_id
            for native in sorted(path.parent.rglob("*")):
                if native.is_symlink():
                    raise ValueError(f"Refusing source symlink: {native}")
                if not native.is_file():
                    continue
                relative = Path("runs") / run_id / native.relative_to(path.parent)
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                before = native.stat()
                shutil.copyfile(native, target)
                digest = sha256_file(target)
                after = native.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or sha256_file(native) != digest:
                    raise ValueError(f"Source changed while packaging: {native}")
                native_relative = native.relative_to(path.parent).as_posix()
                frozen_key = {"run.json": "manifest", "episodes/episode_000/trajectory.npz": "trajectory",
                              "episodes/episode_000/trajectory_observations.npz": "observations"}.get(native_relative)
                if run_id in expected and frozen_key and digest != expected[run_id][frozen_key]["sha256"]:
                    raise ValueError(f"Copied bytes differ from frozen source: {run_id}/{frozen_key}")
                files[relative.as_posix()] = dict(size=target.stat().st_size, sha256=digest)
            # Bind descriptors to the copied native files, including explicit run-list packages.
            ep = describe_run(destination / "run.json", horizon=horizon)
            for key, relative in (("source_manifest", "run.json"), ("trajectory_path", "episodes/episode_000/trajectory.npz"), ("observations_path", "episodes/episode_000/trajectory_observations.npz")):
                ep[key] = (Path("runs") / run_id / relative).as_posix()
            ep["split"] = "train"
            records.append(dict(run_id=run_id, selected=run_id in selected, parent_run_id=ep["parent_run_id"], root_run_id=ep["root_run_id"], source_manifest=ep["source_manifest"]))
            if run_id in selected:
                episodes.append(ep)
            print(f"PACKAGE {number}/{len(all_ids)} {run_id} selected={run_id in selected}", flush=True)
        card = f"""---
license: other
license_name: source-permissions-required
task_categories:
- robotics
---
# LIBERO-X SCENE23 L2 post-training records

Source selection: `{dataset_name}`. Contains {len(selected)} selected complete recordings
and {len(all_ids) - len(selected)} additional ancestor records. Ancestors preserve takeover
provenance and are excluded from training unless explicitly selected in the manifest.

Selected composition: {json.dumps(dict(Counter(ep['source'] for ep in episodes)), sort_keys=True)}.
These are platform autonomous policy recordings and human takeover chains, not human subjects data.

`manifest.json` has portable relative paths, per-file SHA-256, tasks/instructions,
root/parent IDs, success confirmation steps, stop reasons and replay chunk boundaries.
`runs/<id>/` contains complete original run files; historical absolute paths inside
original run metadata are provenance only and are never followed by training.
Trajectory NPZ contains executed OSC_POSE actions and N+1 proprioception; observation
NPZ contains N+1 raw uint8 RGB frames for agent and wrist cameras. During training
images are flipped vertically only. Preserve original source licensing and permissions.

Five consecutive success signals confirm success. Training excludes post-success
samples, uses chunk reward -1 (success-confirming chunk 0), gamma 0.99 once per chunk,
and `success_aware_recording_end_v1`: truncated failures bootstrap their real N+1 state.
Root-grouped validation is selected reproducibly by the training configuration;
packaged split is all-train for faithful BC50 reproduction.
"""
        (temporary / "README.md").write_text(card)
        files["README.md"] = dict(size=(temporary / "README.md").stat().st_size, sha256=sha256_file(temporary / "README.md"))
        manifest = dict(schema_version=SCHEMA_VERSION, format="liberox-native-post-training", dataset_id=dataset_name,
            source_dataset_sha256=frozen.get("dataset_sha256") if frozen else None,
            member_run_ids=selected, member_count=len(selected), record_count=len(records), records=records,
            include_post_success=False, success_consecutive_steps=5, action_horizon=horizon,
            action_dim=7, proprio_dim=8, termination_rule=TERMINATION_RULE,
            reward=dict(source="sparse", failure=-1.0, success=0.0, gamma=.99, discount_per="chunk"),
            composition=dict(Counter(ep["source"] for ep in episodes)), episodes=episodes, files=files)
        manifest["dataset_sha256"] = _manifest_identity(manifest)
        atomic_json(temporary / "manifest.json", manifest)
        verified = verify_dataset(temporary)
        os.replace(temporary, output)
        return verified
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
