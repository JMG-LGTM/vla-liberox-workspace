"""Atomic optimizer-boundary checkpoints, including each rank's random state."""
from __future__ import annotations

import os
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .distributed import active, all_rank_call, primary_call, rank, unwrap, world_size
from .io import atomic_json, sha256_file


def rng_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None}


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def _cpu_state(state):
    if isinstance(state, torch.Tensor):
        return state.detach().cpu()
    if isinstance(state, dict):
        return {key: _cpu_state(value) for key, value in state.items()}
    if isinstance(state, (list, tuple)):
        return type(state)(_cpu_state(value) for value in state)
    return state


def save_checkpoint(path, *, update, actor, critic, optimizers, contract, metadata, device,
                    checkpoint_format="consolidated"):
    if checkpoint_format == "sharded":
        return _save_sharded(path, update=update, actor=actor, critic=critic,
                             optimizers=optimizers, contract=contract, metadata=metadata, device=device)
    if checkpoint_format != "consolidated":
        raise ValueError("Unknown checkpoint format")
    path = Path(path)
    states = [None] * world_size()
    local_rng = rng_state(device)
    if active():
        dist.all_gather_object(states, local_rng)
    else:
        states[0] = local_rng
    for optimizer in optimizers.values():
        optimizer.consolidate()

    def save():
        state = {"schema_version": 1, "update": update, "accumulation_boundary": True,
                 "contract": contract, "metadata": metadata, "rng": states,
                 "actor": unwrap(actor).trainable_state_dict() if actor is not None else None,
                 "critic": unwrap(critic).state_dict() if critic is not None else None,
                 "optimizers": {name: opt.state_dict() for name, opt in optimizers.items()}}
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                torch.save(_cpu_state(state), handle)
                handle.flush()
                os.fsync(handle.fileno())
            digest = sha256_file(temporary)
            os.replace(temporary, path)
            atomic_json(path.with_suffix(".json"), {"schema_version": 1, "sha256": digest,
                        "update": update, "world_size": world_size(), "contract": contract})
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return str(path)
    return Path(primary_call(save))


def load_checkpoint(path, *, actor, critic, optimizers, contract, device):
    # A sharded checkpoint's public symlink is the single atomic commit point.
    # Resolve before reading its sidecar so overwrites cannot mix generations.
    requested = Path(path)
    path = Path(primary_call(lambda: str(requested.resolve())))
    def check():
        metadata = json.loads(path.with_suffix(".json").read_text())
        if sha256_file(path) != metadata["sha256"]:
            raise ValueError("Checkpoint SHA-256 mismatch")
        if metadata["contract"] != contract:
            raise ValueError("Resume contract changed (data/model/config/batch/world size)")
        return metadata
    manifest = primary_call(check)
    if manifest.get("schema_version") == 2:
        return _load_sharded(path, manifest, actor=actor, critic=critic,
                             optimizers=optimizers, contract=contract, device=device)
    # Optimizer and RNG objects require pickle: load only your own trusted files.
    state = all_rank_call(lambda: torch.load(path, map_location="cpu", weights_only=False))
    if state.get("schema_version") != 1 or not state.get("accumulation_boundary") or state["contract"] != contract:
        raise ValueError("Invalid checkpoint or incompatible resume contract")
    if len(state["rng"]) != world_size():
        raise ValueError("Exact resume requires the same world size")
    if actor is not None:
        unwrap(actor).load_trainable_state_dict(state["actor"])
    if critic is not None:
        unwrap(critic).load_state_dict(state["critic"], strict=True)
    if set(state["optimizers"]) != set(optimizers):
        raise ValueError("Checkpoint optimizer set differs")
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(state["optimizers"][name])
    restore_rng(state["rng"][rank()], device)
    return state["update"], state["metadata"]


def _atomic_torch_save(path, state):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(_cpu_state(state), handle)
            handle.flush()
            os.fsync(handle.fileno())
        digest = sha256_file(temporary)
        os.replace(temporary, path)
        return {"file": path.name, "sha256": digest, "bytes": path.stat().st_size}
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_symlink(target, path):
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    os.unlink(temporary)
    try:
        os.symlink(target, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _save_sharded(path, *, update, actor, critic, optimizers, contract, metadata, device):
    path = Path(path).absolute()
    def prepare():
        path.parent.mkdir(parents=True, exist_ok=True)
        # Keep immutable generations: a failed rewrite leaves the old alias valid.
        return tempfile.mkdtemp(prefix=f".{path.stem}.shards.", dir=path.parent)
    generation = Path(primary_call(prepare))

    def save_local():
        state = {"schema_version": 2, "rank": rank(), "world_size": world_size(),
                 "update": update, "contract": contract, "rng": rng_state(device),
                 "optimizers": {name: optimizer.local_state_dict() for name, optimizer in optimizers.items()}}
        info = _atomic_torch_save(generation / f"rank_{rank():05d}.pt", state)
        return {"rank": rank(), **info}
    local_info = all_rank_call(save_local)
    shards = [None] * world_size()
    if active():
        dist.all_gather_object(shards, local_info)
    else:
        shards[0] = local_info

    def commit():
        common = {"schema_version": 2, "checkpoint_format": "sharded", "update": update,
                  "accumulation_boundary": True, "contract": contract, "metadata": metadata,
                  "actor": unwrap(actor).trainable_state_dict() if actor is not None else None,
                  "critic": unwrap(critic).state_dict() if critic is not None else None}
        model = _atomic_torch_save(generation / "model.pt", common)
        atomic_json(generation / "model.json", {"schema_version": 2, "checkpoint_format": "sharded",
                    "sha256": model["sha256"], "update": update, "world_size": world_size(),
                    "contract": contract, "shards": shards})
        # Flush directory entries before publishing this complete generation.
        descriptor = os.open(generation, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _atomic_symlink(str((generation / "model.json").relative_to(path.parent)), path.with_suffix(".json"))
        _atomic_symlink(str((generation / "model.pt").relative_to(path.parent)), path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return str(path)
    return Path(primary_call(commit))


def _load_sharded(path, manifest, *, actor, critic, optimizers, contract, device):
    def validate_manifest():
        shards = manifest.get("shards", [])
        if (manifest.get("checkpoint_format") != "sharded" or manifest.get("world_size") != world_size()
                or len(shards) != world_size() or [item.get("rank") for item in shards] != list(range(world_size()))):
            raise ValueError("Sharded checkpoint requires one complete shard per rank and the same world size")
        for item in shards:
            if item.get("file") != f"rank_{item['rank']:05d}.pt":
                raise ValueError("Invalid checkpoint shard filename")
    primary_call(validate_manifest)

    def load_local():
        info = manifest["shards"][rank()]
        shard_path = path.parent / info["file"]
        if shard_path.stat().st_size != info["bytes"] or sha256_file(shard_path) != info["sha256"]:
            raise ValueError("Checkpoint shard SHA-256 mismatch")
        common = torch.load(path, map_location="cpu", weights_only=False)
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        if (common.get("schema_version") != 2 or not common.get("accumulation_boundary")
                or common.get("contract") != contract or common.get("update") != manifest["update"]
                or shard.get("schema_version") != 2 or shard.get("rank") != rank()
                or shard.get("world_size") != world_size() or shard.get("contract") != contract
                or shard.get("update") != common["update"]):
            raise ValueError("Invalid sharded checkpoint or incompatible resume contract")
        if set(shard["optimizers"]) != set(optimizers):
            raise ValueError("Checkpoint optimizer set differs")
        if actor is not None:
            unwrap(actor).load_trainable_state_dict(common["actor"])
        if critic is not None:
            unwrap(critic).load_state_dict(common["critic"], strict=True)
        for name, optimizer in optimizers.items():
            optimizer.load_local_state_dict(shard["optimizers"][name])
        return common, shard["rng"]
    common, rng = all_rank_call(load_local)
    all_rank_call(lambda: restore_rng(rng, device))
    return common["update"], common["metadata"]
