"""Shared optimizer-step runner for BC, IQL and critic-only training, on one node (CPU test support).

What differs between the methods is in algorithms.py; IQL-only components are in iql/.
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import signal
import shutil
import subprocess
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, save_checkpoint, rng_state, restore_rng
from .algorithms import build_algorithm, critic_model_config
from .config import actor_learning_rate, batch_plan, validate_config
from .distributed import (GlobalBatchSampler, OptimizerGroup, active, all_rank_call, initialize,
                          primary_call, rank, reduce_metrics, unwrap, world_size, wrap)
from .io import atomic_json, sha256_file, stable_hash


def code_revision():
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _split_report(info):
    return {key: info[key] for key in ("dataset_chunks", "train_chunks", "validation_chunks",
                                      "root_counts", "critic_evaluation")}


def _critic_initialization(config, metadata):
    """Keep pretraining validation as historical evidence, separate from this run."""
    return {"path": str(Path(config["critic"]["init_checkpoint"]).expanduser().resolve()),
            "artifact_sha256": sha256_file(Path(config["critic"]["init_checkpoint"]).expanduser()),
            "metadata_sha256": stable_hash(metadata),
            "pretraining_provenance": metadata["provenance"]}


def dry_run(config, data_root, world):
    """Read/verify only: no process group, cache, model creation or CUDA context."""
    from .data import inspect_dataset
    validate_config(config)
    plan = batch_plan(config, world)
    from .preflight import environment_report
    environment = environment_report(config, world)
    data = inspect_dataset(data_root, config, verify=True)
    # IQL only (None for BC): official-demonstration auxiliary, BC-initialized actor, pretrained critic.
    from .iql.actor_bc import inspect_actor_bc
    actor_bc_auxiliary = inspect_actor_bc(config["actor_bc"]["dataset_root"], config, data)
    if not data["train_count"]:
        raise ValueError("No training samples")
    if data["critic_evaluation"]["held_out"] and not data["val_count"]:
        raise ValueError("No held-out samples; increase validation_fraction")
    if config["training"]["bc_validation"] and not data["val_count"]:
        raise ValueError("BC validation requires a nonempty held-out split")
    if config["training"]["sampling"] == "shuffle" and data["train_count"] < plan["global_micro_batch_size"]:
        raise ValueError("Global micro batch exceeds training split")
    identity = None
    if config["method"] != "critic":
        from .pi05_assets import checkpoint_identity, platform_model_settings
        identity = checkpoint_identity(config["model"]["checkpoint"])
        platform_model_settings(config["model"], identity)
    from .iql.actor_init import inspect_actor_init_policy
    actor_initialization = inspect_actor_init_policy(config["model"]["actor_init_policy"],
        identity=identity, model_config=config["model"])
    initialization = None
    if config["critic"]["init_checkpoint"]:
        from .iql.critic import default_input_spec, read_critic_metadata
        from .replay import load_normalization_file
        normalization = (load_normalization_file(config["data"]["normalization"])
                         if config["data"]["normalization"] else None)
        metadata = read_critic_metadata(config["critic"]["init_checkpoint"],
                    expected_model_config=critic_model_config(config),
                    expected_input_spec=default_input_spec(critic_model_config(config), config["critic"]["image_size"]),
                    expected_normalization=normalization,
                    expected_reward_definition=config["reward"],
                    expected_termination_rule=config["reward"]["termination_rule"])
        initialization = _critic_initialization(config, metadata)
    elif config["data"]["normalization"]:
        from .replay import load_normalization_file
        load_normalization_file(config["data"]["normalization"])
    return {"dry_run": True, "method": config["method"], **plan,
            "critic_implementation": "custom_pytorch_pixel_iql_v1" if config["method"] != "bc" else None,
            "uses_author_official_idql": False,
            "environment": environment,
            "dataset_sha256": data["dataset_sha256"], "chunks": data["count"],
            **_split_report(data), "critic_initialization": initialization,
            "actor_initialization": actor_initialization,
            "actor_bc_auxiliary": actor_bc_auxiliary,
            "model_identity": identity, "policy_loaded": False,
            "reward_definition": config["reward"], "output_dir": config["output_dir"],
            "sampling": config["training"]["sampling"],
            "note": "Optimizer updates count complete accumulation boundaries; IQL includes extra critic warmup."}


def _sample_counters(plan, update, *, actor, critic):
    actor_updates = max(0, update - plan["critic_warmup_updates"]) if actor else 0
    actor_batch = plan.get("global_actor_batch_size", plan["global_batch_size"]) if actor else 0
    native_batch = plan.get("global_native_actor_batch_size", actor_batch) if actor else 0
    official_batch = plan.get("global_official_bc_batch_size", 0) if actor else 0
    critic_batch = plan["global_batch_size"] if critic else 0
    return {"policy_sample_draws": actor_updates * actor_batch,
        "native_actor_sample_draws": actor_updates * native_batch,
        "official_bc_sample_draws": actor_updates * official_batch,
        "critic_sample_draws": update * critic_batch,
        "global_actor_batch_size": actor_batch, "global_native_actor_batch_size": native_batch,
        "global_official_bc_batch_size": official_batch, "global_critic_batch_size": critic_batch}


def _loader(dataset, config, *, sampler=None, validation=False):
    cfg = config["distributed"]
    kwargs = {"num_workers": cfg["workers_per_rank"],
              "pin_memory": cfg["pin_memory"] and cfg["device"] == "cuda",
              "generator": torch.Generator().manual_seed(config["training"]["seed"] + 100 + rank())}
    if cfg["workers_per_rank"]:
        kwargs.update(multiprocessing_context="spawn", persistent_workers=True,
                      prefetch_factor=cfg["prefetch_factor"])
    if sampler is not None:
        kwargs["batch_sampler"] = sampler
    else:
        batch_size = (config["training"]["validation_batch_size"] if validation else None)
        kwargs.update(batch_size=batch_size or config["training"]["per_device_batch_size"], shuffle=False)
    return DataLoader(dataset, **kwargs)


def _tensor_batch(batch, device):
    keys = {"pixels", "next_pixels", "proprio", "next_proprio", "actions", "action_mask",
            "reward", "bootstrap_mask", "chunk_length"}
    return {key: value.to(device, non_blocking=True) for key, value in batch.items() if key in keys}


def evaluate_actor(actor, loader, device, *, seed):
    """Full held-out, sample-mean flow objective with an isolated fixed RNG stream.

    Pi05Policy applies the original per-sample real-action mask before returning
    losses. The pinned OpenPI forward preprocesses with train=True, so this is a
    fixed-seed Monte Carlo estimate of the original training objective, including
    its image augmentation, flow noise and flow time; it is not rollout success.
    Rank zero calls the unwrapped module while peers wait in primary_call.
    """
    actor = unwrap(actor)
    previous_rng = rng_state(device)
    generator = getattr(loader, "generator", None)
    loader_rng = generator.get_state() if generator is not None else None
    modes = [(module, module.training) for module in actor.modules()]
    total, count = 0.0, 0
    try:
        random.seed(seed)
        np.random.seed(seed)
        # torch.manual_seed also changes other visible GPUs' generators. Each
        # rank owns one device; touch only the CPU and assigned CUDA stream.
        torch.random.default_generator.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        if generator is not None:
            generator.manual_seed(seed + 100)
        actor.eval()
        with torch.no_grad():
            for batch in loader:
                losses = actor(batch)
                if losses.ndim != 1 or losses.shape[0] != batch["action_mask"].shape[0]:
                    raise ValueError("Policy must return one validation loss per sample")
                if not bool(torch.isfinite(losses).all()):
                    raise FloatingPointError("Nonfinite held-out actor loss")
                total += float(losses.detach().double().sum().cpu())
                count += losses.numel()
        if not count:
            raise ValueError("BC validation loader is empty")
        return {"val_actor_loss": total / count, "samples": count, "split": "val", "held_out": True,
                "seed": seed, "aggregation": "mean_of_per_sample_masked_flow_losses",
                "objective": "fixed_seed_mc_original_training_flow_objective",
                "includes_original_openpi_training_augmentation": True}
    finally:
        # Preserve mixed frozen/trainable submodule modes exactly, even on error.
        for module, mode in modes:
            module.training = mode
        restore_rng(previous_rng, device)
        if loader_rng is not None:
            generator.set_state(loader_rng)


def _finite_step(optimizer, parameters, clip, device):
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip)
    invalid = torch.tensor(int(not torch.isfinite(norm)), device=device)
    if active():
        dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
    if bool(invalid):
        raise FloatingPointError("Nonfinite gradient; no optimizer step was taken")
    optimizer.step()
    optimizer.zero_grad()
    return float(norm)


def _finite_loss(loss, device):
    """Reject a nonfinite objective on every rank before any backward/step."""
    invalid = (~torch.isfinite(loss.detach()).all()).to(device=device, dtype=torch.int32)
    if active():
        dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
    if bool(invalid):
        raise FloatingPointError("Nonfinite loss; this optimizer stage was not updated")


def optimization_step(config, batches, actor, critic, optimizers, device, *, actor_update,
                      official_bc_batches=None):
    """One effective batch: the method's critic update (IQL), then the shared actor update.

    Method-specific parts live in algorithms.py; the order V -> Q -> target -> actor is kept.
    """
    algorithm = build_algorithm(config)
    accumulation = len(batches)
    if accumulation < 1:
        raise ValueError("An optimizer update requires at least one microbatch")
    actor_inputs = algorithm.actor_inputs(batches, official_bc_batches, actor_update)
    metrics = algorithm.update_critic(batches, critic, optimizers, device)
    if actor is not None and actor_update is not None:
        cfg = config["training"]
        optimizer = optimizers["actor"]
        lr = actor_learning_rate(actor_update, cfg["optimizer_updates"], cfg["learning_rate"],
                                 cfg["final_learning_rate"], cfg["warmup_updates"])
        from .continuation import actor_rate
        lr = actor_rate(config, actor_update, lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad()
        record = {"actor_losses": [], "unweighted_losses": [], "weights_mean": []}
        for index, (inputs, native_batch, native_count) in enumerate(actor_inputs):
            scores = algorithm.actor_scores(critic, native_batch, device)
            context = actor.no_sync() if hasattr(actor, "no_sync") and index + 1 < accumulation else nullcontext()
            with context:
                losses = actor(inputs)
                if losses.ndim != 1 or losses.shape[0] != batches[index]["action_mask"].shape[0]:
                    raise ValueError("Policy must return one scalar loss per sample")
                loss = algorithm.actor_loss(losses, native_count, scores, record)
                _finite_loss(loss, device)
                (loss / accumulation).backward()
            record["actor_losses"].append(float(loss.detach()))
            record["unweighted_losses"].append(float(losses[:native_count].detach().mean()))
            record["weights_mean"].append(float(scores["weights"].mean()) if scores is not None else 1.0)
        parameters = [parameter for parameter in actor.parameters() if parameter.requires_grad]
        metrics["actor_grad_norm"] = _finite_step(optimizer, parameters, cfg["grad_clip"], device)
        metrics.update(actor_loss=sum(record["actor_losses"]) / accumulation,
                       unweighted_actor_loss=sum(record["unweighted_losses"]) / accumulation, learning_rate=lr,
                       actor_weight_mean=sum(record["weights_mean"]) / accumulation)
        metrics.update(algorithm.actor_metrics(record, accumulation))
    return metrics


def train(config, data_root, *, resume=None):
    validate_config(config)
    device = initialize(config)
    stop = {"requested": False}
    def request_stop(*_):
        stop["requested"] = True
    handlers = {kind: signal.signal(kind, request_stop) for kind in (signal.SIGINT, signal.SIGTERM)}
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        return _train(config, Path(data_root), device, resume, lambda: stop["requested"])
    finally:
        for kind, handler in handlers.items():
            signal.signal(kind, handler)
        torch.use_deterministic_algorithms(deterministic)
        if active():
            dist.destroy_process_group()


def _atomic_copy(source, destination):
    """Publish a complete immutable export through a replaceable convenience path."""
    source, destination = Path(source), Path(destination)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
            shutil.copyfileobj(input_file, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _restore_run_artifacts(run_dir, update, metadata):
    """Roll back logs and the best-export alias to a checkpoint's exact boundary.

    Versioned best exports are retained. Future metrics and partially written
    final lines are discarded so a crash/restart never creates duplicate steps.
    """
    run_dir = Path(run_dir)
    best = metadata.get("best_critic_export")
    source = None
    if best is not None:
        source = (run_dir / best).resolve()
        if not source.is_relative_to(run_dir) or not source.is_file():
            raise ValueError("Checkpoint best critic export is missing or outside output_dir")
    metrics = run_dir / "metrics.jsonl"
    kept = []
    if metrics.exists():
        lines = metrics.read_text().splitlines()
        previous = 0
        for index, line in enumerate(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    break
                raise ValueError("Invalid metrics before the final incomplete line") from None
            step = record.get("optimizer_update")
            if type(step) is not int or step <= previous:
                raise ValueError("Metrics contain duplicate or unordered optimizer updates")
            previous = step
            if step <= update:
                kept.append(line)
    validation_kept = []
    validation_metrics = run_dir / "validation_metrics.jsonl"
    if validation_metrics.exists():
        lines = validation_metrics.read_text().splitlines()
        previous = 0
        for index, line in enumerate(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    break
                raise ValueError("Invalid validation metrics before the final incomplete line") from None
            step = record.get("optimizer_update")
            if type(step) is not int or step <= previous:
                raise ValueError("Validation metrics contain duplicate or unordered optimizer updates")
            previous = step
            if step <= update:
                validation_kept.append(line)
    # Preserve completed policy exports under the original basename: policy_id
    # remains "policy" and its package continues to pass the platform checker.
    retired_paths = [run_dir / name for name in ("policy", "critic_best.pt", "critic_last.pt",
        "summary.json", "provenance.json", "config.yaml", "metrics.jsonl", "validation_metrics.jsonl", "memory.jsonl") if (run_dir / name).exists()]
    if retired_paths:
        retired = run_dir / "retired"
        retired.mkdir(exist_ok=True)
        destination = Path(tempfile.mkdtemp(prefix=f"resume_{update:08d}_", dir=retired))
        for artifact in retired_paths:
            os.replace(artifact, destination / artifact.name)
    if source is not None:
        _atomic_copy(source, run_dir / "critic_best.pt")
    if kept:
        descriptor, temporary = tempfile.mkstemp(prefix=".metrics.", dir=run_dir)
        try:
            with os.fdopen(descriptor, "w") as output:
                output.write("".join(line + "\n" for line in kept))
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, metrics)
        finally:
            Path(temporary).unlink(missing_ok=True)
    if validation_kept:
        descriptor, temporary = tempfile.mkstemp(prefix=".validation_metrics.", dir=run_dir)
        try:
            with os.fdopen(descriptor, "w") as output:
                output.write("".join(line + "\n" for line in validation_kept))
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, validation_metrics)
        finally:
            Path(temporary).unlink(missing_ok=True)
    # Explicit --resume is the user's request to continue after a prior stop.
    (run_dir / "stop.request").unlink(missing_ok=True)


def _best_export_name(run_dir, update):
    """Keep earlier best snapshots intact when replaying an older checkpoint."""
    stem = f"critics/best_update_{update:08d}"
    name, revision = stem + ".pt", 0
    while (Path(run_dir) / name).exists():
        revision += 1
        name = f"{stem}_revision_{revision:03d}.pt"
    return name


def _train(config, data_root, device, resume, stopping):
    from .continuation import (check_resume_location, comparable_config, prepare_donor_load,
                               SOURCE_UPDATE)
    from .iql.critic import default_input_spec, export_critic, read_critic_metadata
    from .data import inspect_dataset
    from .diagnostics import evaluate_critic
    from .replay import CachedReplayDataset, build_replay_cache, load_normalization_file

    plan = batch_plan(config, world_size())
    cfg = config["training"]
    run_dir = Path(config["output_dir"]).expanduser().resolve()

    def prepare():
        if resume:
            check_resume_location(config, resume)
        elif run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError("output_dir is not empty; use a fresh directory or --resume")
        elif config["training"].get("continuation") is not None:
            raise ValueError("Continuation requires the complete bound checkpoint through --resume")
        info = inspect_dataset(data_root, config, verify=True)
        if not info["train_count"] or ((info["critic_evaluation"]["held_out"] or cfg["bc_validation"]) and not info["val_count"]):
            raise ValueError("Training/validation split is empty")
        identity = None
        if config["method"] != "critic":
            from .pi05_assets import checkpoint_identity
            identity = checkpoint_identity(config["model"]["checkpoint"])
        # IQL only (None for BC): BC-initialized actor, pretrained critic, official-demonstration cache.
        from .iql.actor_init import inspect_actor_init_policy
        actor_initialization = inspect_actor_init_policy(config["model"]["actor_init_policy"],
            identity=identity, model_config=config["model"])
        normalization, initialization = None, None
        if config["critic"]["init_checkpoint"]:
            metadata = read_critic_metadata(config["critic"]["init_checkpoint"],
                expected_model_config=critic_model_config(config),
                expected_input_spec=default_input_spec(critic_model_config(config), config["critic"]["image_size"]),
                expected_reward_definition=config["reward"],
                expected_termination_rule=config["reward"]["termination_rule"])
            normalization = metadata["normalization"]
            initialization = _critic_initialization(config, metadata)
            if config["data"]["normalization"]:
                explicit = load_normalization_file(config["data"]["normalization"])
                if explicit != normalization:
                    raise ValueError("Explicit normalization differs from critic initialization")
        print("Preparing verified replay cache (CPU); no per-sample NPZ reads during training", flush=True)
        cache, reused = build_replay_cache(data_root, Path(config["cache_dir"]), config, norm_stats=normalization)
        from .iql.actor_bc import build_actor_bc_cache
        auxiliary = build_actor_bc_cache(config["actor_bc"]["dataset_root"],
            Path(config["cache_dir"]) / "official_actor_bc", config, info)
        result = {"cache": str(cache), "cache_reused": reused, "dataset_sha256": info["dataset_sha256"],
                "train_count": info["train_count"], "val_count": info["val_count"], "model_identity": identity,
                **_split_report(info), "critic_initialization": initialization,
                "actor_initialization": actor_initialization, "actor_bc_auxiliary": auxiliary}
        return result

    prepared = primary_call(prepare)
    dataset = CachedReplayDataset(prepared["cache"], split="train")
    evaluation_spec = prepared["critic_evaluation"]
    validation = (CachedReplayDataset(prepared["cache"], split=evaluation_spec["split"])
                  if evaluation_spec["split"] else None)
    if len(dataset) != prepared["train_chunks"] or (
            validation is not None and len(validation) != evaluation_spec["chunks"]):
        raise ValueError("Cached replay counts differ from the inspected training/evaluation splits")
    if cfg["bc_validation"]:
        validation = CachedReplayDataset(prepared["cache"], split="val")
        if len(validation) != prepared["validation_chunks"]:
            raise ValueError("Cached held-out actor split count differs from inspection")
    normalization = dataset.metadata["normalization"]
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed(cfg["seed"])
    torch.use_deterministic_algorithms(cfg["deterministic"])
    algorithm = build_algorithm(config)
    actor, critic, optimizers = None, None, {}
    if algorithm.trains_actor:
        from .policy import load_policy
        actor = load_policy(config["model"], device, verified_identity=prepared["model_identity"])
        # IQL only: warm start the actor from a trained BC policy package.
        if not resume and prepared["actor_initialization"] is not None:
            from .iql.actor_init import initialize_actor_from_policy
            all_rank_call(lambda: initialize_actor_from_policy(actor, prepared["actor_initialization"]))
        actor = wrap(actor, device, unused=True, move_inputs=False,
                     gradient_as_bucket_view=config["distributed"]["gradient_as_bucket_view"])
        optimizers["actor"] = OptimizerGroup(actor.parameters(), zero=cfg["zero_optimizer"],
            lr=cfg["learning_rate"], betas=(.9, .95), eps=1e-8, weight_decay=cfg["weight_decay"],
            foreach=cfg["adamw_foreach"])
    # IQL / critic-only: PixelIQL critic and its Q and V optimizers (algorithms.py). BC: none.
    critic, critic_optimizers, critic_optimizer_class = algorithm.build_critic(device, normalization)
    optimizers.update(critic_optimizers)

    effective = copy.deepcopy(config)
    effective["training"]["gradient_accumulation_steps"] = plan["gradient_accumulation_steps"]
    comparable = comparable_config(effective)
    # Conservative exact resume: retain schedule, world size, sampling and all math settings.
    contract = {"config_sha256": stable_hash(comparable), "dataset_sha256": prepared["dataset_sha256"],
                "cache_fingerprint": dataset.metadata["fingerprint"], "world_size": world_size(),
                "model_identity_sha256": stable_hash(prepared["model_identity"]),
                "critic_initialization_sha256": stable_hash(prepared["critic_initialization"]),
                # The next IQL-only fields are None / 0 for BC.
                "actor_initialization_sha256": stable_hash(prepared["actor_initialization"]),
                "actor_bc_auxiliary_sha256": stable_hash(prepared["actor_bc_auxiliary"]),
                "actor_bc_dataset_sha256": (prepared["actor_bc_auxiliary"]["dataset_sha256"]
                                            if prepared["actor_bc_auxiliary"] is not None else None),
                "actor_bc_weight": config["actor_bc"]["weight"] if prepared["actor_bc_auxiliary"] is not None else None,
                "actor_bc_per_device_batch_size": (config["actor_bc"]["per_device_batch_size"]
                                                   if prepared["actor_bc_auxiliary"] is not None else 0),
                "termination_rule": config["reward"]["termination_rule"], **plan}
    if cfg.get("continuation") is not None:
        contract["continuation"] = copy.deepcopy(cfg["continuation"])
    metadata = {"method": config["method"], "model_identity": prepared["model_identity"],
                "actor_initialization": prepared["actor_initialization"],
                "actor_bc_auxiliary": prepared["actor_bc_auxiliary"],
                "model_settings": unwrap(actor).settings if actor is not None else None,
                "dataset_sha256": prepared["dataset_sha256"], "reward_sha256": stable_hash(config["reward"]),
                "best_metric": None, "best_update": None, "best_critic_export": None,
                "last_validation_metric": None, "last_validation_update": None,
                "last_bc_validation_loss": None, "last_bc_validation_update": None,
                "last_diagnostic_metric": None, "last_diagnostic_update": None}
    start = 0
    if resume:
        external_donor = check_resume_location(config, resume)
        load_contract = (primary_call(lambda: prepare_donor_load(config, resume, contract))
                         if external_donor else contract)
        start, metadata = load_checkpoint(resume, actor=actor, critic=critic, optimizers=optimizers,
                                         contract=load_contract, device=device)
        if external_donor and start != SOURCE_UPDATE:
            raise ValueError("Continuation donor did not restore update30000")
    else:
        random.seed(cfg["seed"] + rank())
        np.random.seed(cfg["seed"] + rank())
        torch.manual_seed(cfg["seed"] + rank())
        if device.type == "cuda":
            torch.cuda.manual_seed(cfg["seed"] + rank())
    if start >= plan["optimizer_updates"]:
        raise ValueError("Checkpoint already reached the configured training length")
    if resume:
        primary_call(lambda: _restore_run_artifacts(run_dir, start, metadata))
    provenance = {"schema_version": 1, "code": code_revision(), **prepared, **plan,
                  "critic_implementation": "custom_pytorch_pixel_iql_v1" if critic is not None else None,
                  "uses_author_official_idql": False,
                  "method": config["method"], "termination_rule": config["reward"]["termination_rule"],
                  "reward_definition": config["reward"], "normalization": normalization,
                  "cache_fingerprint": dataset.metadata["fingerprint"], "device": str(device),
                  "resume_checkpoint": str(resume) if resume else None,
                  "critic_optimizer": ({"optimizer": config["critic"]["optimizer"],
                      "optimizer_class": critic_optimizer_class.__module__ + "." + critic_optimizer_class.__qualname__,
                      "q_learning_rate": optimizers["q"].param_groups[0]["lr"],
                      "value_learning_rate": optimizers["value"].param_groups[0]["lr"],
                      "weight_decay": optimizers["q"].param_groups[0]["weight_decay"]} if critic is not None else None),
                  "model_parameters": unwrap(actor).parameter_counts() if actor is not None else None,
                  "sample_count_unit": "draws_including_repeated_samples", "contract": contract}
    if cfg.get("continuation") is not None:
        provenance["continuation"] = {**copy.deepcopy(cfg["continuation"]),
            "resumed_update": start, "actor_learning_rate": cfg["final_learning_rate"],
            "restored_actor_critic_optimizers_rng": True,
            "source_checkpoint_rewritten": False}
    provenance["actor_evaluation"] = {"enabled": cfg["bc_validation"], "held_out": cfg["bc_validation"],
        "split": "val" if cfg["bc_validation"] else None, "interval": cfg["validation_interval"],
        "seed": cfg["validation_seed"], "batch_size": cfg["validation_batch_size"] or cfg["per_device_batch_size"],
        "objective": "fixed_seed_mc_original_training_flow_objective"}
    provenance["sample_draws_definition"] = "main_native_replay_draws_including_critic_warmup; see separate policy/native_actor/official_bc/critic counters"
    provenance["actor_objective"] = ({"kind": "native_iql_plus_official_bc",
        "formula": "mean(native_flow_loss * IQL_weight) + lambda * mean(official_flow_loss)",
        "lambda": config["actor_bc"]["weight"], "single_forward_and_backward": True,
        "native_samples": "prefix_of_each_native_critic_batch", "official_sampler": {
            "kind": "global_step_rank_reconstructible_replacement", "seed": cfg["seed"] + 104729,
            "start_unit": "completed_actor_microbatches", "prefetch_state_required": False},
        "native_transition_dataset_sha256": prepared["dataset_sha256"],
        "official_bc_dataset_sha256": prepared["actor_bc_auxiliary"]["dataset_sha256"],
        **_sample_counters(plan, 0, actor=True, critic=True)}
        if prepared["actor_bc_auxiliary"] is not None else {"kind": "pure_" + config["method"]})
    def write_setup():
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(effective, sort_keys=False))
        atomic_json(run_dir / "provenance.json", provenance)
        atomic_json(run_dir / "normalization.json", normalization)
    primary_call(write_setup)
    sampler = GlobalBatchSampler(len(dataset), plan["global_micro_batch_size"], rank(), world_size(),
        cfg["seed"] + 1, start * plan["gradient_accumulation_steps"], plan["micro_batches"], cfg["sampling"])
    loader = _loader(dataset, config, sampler=sampler)
    validation_loader = (_loader(validation, config, validation=True)
                         if rank() == 0 and (critic is not None or cfg["bc_validation"]) else None)
    iterator = iter(loader)
    # IQL only: sampler of the official-demonstration auxiliary (iql/actor_bc.py).
    official_iterator = None
    if prepared["actor_bc_auxiliary"] is not None:
        official = CachedReplayDataset(prepared["actor_bc_auxiliary"]["cache"], split="train")
        official_sampler = GlobalBatchSampler(len(official), config["actor_bc"]["per_device_batch_size"] * world_size(),
            rank(), world_size(), cfg["seed"] + 104729,
            max(0, start - plan["critic_warmup_updates"]) * plan["gradient_accumulation_steps"],
            cfg["optimizer_updates"] * plan["gradient_accumulation_steps"], "replacement")
        official_loader = _loader(official, config, sampler=official_sampler)
        official_iterator = iter(official_loader)
    writer = None
    def start_writer():
        if cfg["tensorboard"]:
            from torch.utils.tensorboard import SummaryWriter
            return SummaryWriter(str(run_dir / "tensorboard"), purge_step=start + 1 if resume else None)
        return None
    # A writer cannot be pickled; only communicate creation failures.
    def create_writer():
        nonlocal writer
        writer = start_writer()
    primary_call(create_writer)
    if rank() == 0:
        print("TRAIN " + json.dumps({"method": config["method"], **plan}, sort_keys=True), flush=True)
    started = time.perf_counter()
    checkpoint, stopped = None, False
    last_update = start
    memory_peaks = {}

    def record_memory(stage, update):
        from .monitoring import memory_metrics
        snapshot = memory_metrics(device)
        for key, value in snapshot.items():
            memory_peaks[key] = max(memory_peaks.get(key, 0.0), value)
        def write():
            with (run_dir / "memory.jsonl").open("a") as handle:
                handle.write(json.dumps({"stage": stage, "optimizer_update": update, **snapshot}) + "\n")
            if writer is not None:
                for key, value in snapshot.items():
                    writer.add_scalar("memory/" + key, value, update)
        primary_call(write)

    def export_current_critic(name, update):
        return export_critic(run_dir / name, unwrap(critic),
            input_spec=default_input_spec(unwrap(critic), config["critic"]["image_size"]),
            normalization=normalization, reward_definition=config["reward"],
            termination_rule=config["reward"]["termination_rule"],
            provenance={"dataset_sha256": prepared["dataset_sha256"], "update": update,
                        "code": provenance["code"], **_split_report(prepared),
                        "critic_initialization": prepared["critic_initialization"],
                        "critic_optimizer": provenance["critic_optimizer"],
                        "diagnostic_metric": metadata["last_diagnostic_metric"]
                        if metadata["last_diagnostic_update"] == update else None,
                        "validation_metric": metadata["last_validation_metric"]
                        if evaluation_spec["held_out"] and metadata["last_validation_update"] == update else None})

    try:
        record_memory("before_updates", start)
        for update in range(start, plan["optimizer_updates"]):
            begin = time.perf_counter()
            batches = [next(iterator) for _ in range(plan["gradient_accumulation_steps"])]
            actor_update = update - plan["critic_warmup_updates"]
            additional = {}
            if official_iterator is not None and actor_update >= 0:
                additional["official_bc_batches"] = [next(official_iterator) for _ in range(plan["gradient_accumulation_steps"])]
            data_wait = time.perf_counter() - begin
            metrics = optimization_step(config, batches, actor, critic, optimizers, device,
                actor_update=actor_update if actor is not None and actor_update >= 0 else None, **additional)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            duration = time.perf_counter() - begin
            metrics["data_wait_seconds"] = data_wait
            metrics = reduce_metrics(metrics, device)
            # Equal per-rank/per-microbatch sample counts make reduced raw
            # moments exact for the complete effective batch, rather than an
            # average of per-rank standard deviations or effective sizes.
            if "advantage_second_moment" in metrics:
                second = metrics.pop("advantage_second_moment")
                metrics["advantage_std"] = math.sqrt(max(0.0, second - metrics["advantage_mean"] ** 2))
                weight_second = metrics.pop("actor_weight_second_moment")
                weight_batch_size = plan.get("global_native_actor_batch_size", plan["global_batch_size"])
                metrics["actor_weight_effective_sample_size"] = (
                    min(weight_batch_size,
                        weight_batch_size * metrics["actor_weight_mean"] ** 2 / weight_second)
                    if weight_second > 0 else 0.0)
            slow = torch.tensor([duration, torch.cuda.max_memory_allocated(device) / 1024 ** 3
                                 if device.type == "cuda" else 0.0], device=device, dtype=torch.float64)
            if active():
                dist.all_reduce(slow, op=dist.ReduceOp.MAX)
            last_update = update + 1
            metrics.update(optimizer_update=last_update, policy_optimizer_updates=max(0, last_update - plan["critic_warmup_updates"]) if actor else 0,
                critic_optimizer_updates=last_update if critic else 0,
                micro_batch_count=last_update * plan["gradient_accumulation_steps"],
                sample_draws=last_update * plan["global_batch_size"],
                global_batch_size=plan["global_batch_size"], per_device_batch_size=plan["per_device_batch_size"],
                gradient_accumulation_steps=plan["gradient_accumulation_steps"],
                update_seconds=float(slow[0]), samples_per_second=plan["global_batch_size"] / max(float(slow[0]), 1e-12),
                peak_cuda_memory_gib=float(slow[1]), elapsed_seconds=time.perf_counter() - started)
            metrics.update(_sample_counters(plan, last_update, actor=actor is not None, critic=critic is not None))
            stop_requested = torch.tensor(int(stopping() or (run_dir / "stop.request").exists()), device=device)
            if active():
                dist.all_reduce(stop_requested, op=dist.ReduceOp.MAX)
            stopped = bool(stop_requested) and last_update < plan["optimizer_updates"]
            final = last_update == plan["optimizer_updates"] or stopped
            # IQL / critic-only: held-out critic diagnostics and best-critic export.
            if critic is not None and (last_update % cfg["validation_interval"] == 0 or final):
                def validate():
                    result = evaluate_critic(unwrap(critic), validation_loader, device,
                        beta=config["critic"]["beta"], max_weight=config["critic"]["max_weight"])
                    score = result["spearman_v_remaining_blocks"]
                    result.update(split=evaluation_spec["split"], held_out=evaluation_spec["held_out"],
                                  selects_best_checkpoint=evaluation_spec["selects_best_checkpoint"])
                    if not evaluation_spec["held_out"]:
                        result["selection_direction"] = None
                    improved = bool(evaluation_spec["selects_best_checkpoint"] and result["finite"] and score is not None and
                                    (metadata["best_metric"] is None or score < metadata["best_metric"]))
                    return {"diagnostics": result, "improved": improved, "score": score}
                evaluation = primary_call(validate)
                metrics[evaluation_spec["metric_group"]] = evaluation["diagnostics"]
                metadata.update(last_diagnostic_metric=evaluation["score"], last_diagnostic_update=last_update)
                if evaluation_spec["held_out"]:
                    metadata.update(last_validation_metric=evaluation["score"], last_validation_update=last_update)
                if evaluation["improved"]:
                    name = primary_call(lambda: _best_export_name(run_dir, last_update))
                    metadata.update(best_metric=evaluation["score"], best_update=last_update,
                                    best_critic_export=name)
                    def export_best():
                        export_current_critic(name, last_update)
                        _atomic_copy(run_dir / name, run_dir / "critic_best.pt")
                    primary_call(export_best)
            # BC: held-out flow loss on the validation roots.
            if cfg["bc_validation"] and (last_update % cfg["validation_interval"] == 0 or final):
                evaluation = primary_call(lambda: evaluate_actor(actor, validation_loader, device,
                                                                  seed=cfg["validation_seed"]))
                evaluation.update(optimizer_update=last_update)
                metrics["val_actor_loss"] = evaluation["val_actor_loss"]
                metadata.update(last_bc_validation_loss=evaluation["val_actor_loss"],
                                last_bc_validation_update=last_update)
                def write_actor_validation():
                    with (run_dir / "validation_metrics.jsonl").open("a") as handle:
                        handle.write(json.dumps(evaluation, allow_nan=False) + "\n")
                    if writer is not None:
                        writer.add_scalar("validation/actor_loss", evaluation["val_actor_loss"], last_update)
                primary_call(write_actor_validation)
            def write_metrics():
                with (run_dir / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(metrics, allow_nan=False) + "\n")
                if writer is not None:
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            writer.add_scalar(key, value, last_update)
                    for group in ("validation", "train_diagnostic"):
                        for key, value in metrics.get(group, {}).items():
                            if isinstance(value, (int, float)):
                                writer.add_scalar(group + "/" + key, value, last_update)
                if last_update == start + 1 or last_update % cfg["log_interval"] == 0 or final:
                    print(f"UPDATE {last_update}/{plan['optimizer_updates']} micro={metrics['micro_batch_count']} "
                          f"actor={metrics['policy_optimizer_updates']} critic={metrics['critic_optimizer_updates']} "
                          f"samples/s={metrics['samples_per_second']:.2f} wait={metrics['data_wait_seconds']:.3f}s "
                          f"peak_GiB={metrics['peak_cuda_memory_gib']:.2f}", flush=True)
            primary_call(write_metrics)
            if last_update % cfg["checkpoint_interval"] == 0 or final:
                record_memory("before_checkpoint", last_update)
                checkpoint = save_checkpoint(run_dir / "checkpoints" / f"update_{last_update:08d}.pt",
                    update=last_update, actor=actor, critic=critic, optimizers=optimizers, contract=contract,
                    metadata=metadata, device=device, checkpoint_format=cfg["checkpoint_format"])
                record_memory("after_checkpoint", last_update)
            if stopped:
                break
        if critic is not None:
            primary_call(lambda: export_current_critic("critic_last.pt", last_update))
        policy = None
        if actor is not None and not stopped:
            from .export import export_policy
            policy = primary_call(lambda: str(export_policy(run_dir / "policy", unwrap(actor).trainable_state_dict(),
                prepared["model_identity"], unwrap(actor).settings, prepared["dataset_sha256"],
                max(0, last_update - plan["critic_warmup_updates"]), config["method"],
                metadata["reward_sha256"] if critic is not None else None, provenance=provenance)))
        record_memory("after_export", last_update)
        summary = {"status": "STOPPED" if stopped else "COMPLETED", "optimizer_updates": last_update,
                   "policy_optimizer_updates": max(0, last_update - plan["critic_warmup_updates"]) if actor else 0,
                   "critic_optimizer_updates": last_update if critic else 0,
                   "micro_batches": last_update * plan["gradient_accumulation_steps"],
                   "sample_draws": last_update * plan["global_batch_size"],
                   "checkpoint": str(checkpoint), "policy": policy, "best_validation_spearman": metadata["best_metric"],
                   "best_critic_update": metadata["best_update"],
                   **_split_report(prepared),
                   "last_critic_diagnostic_spearman": metadata["last_diagnostic_metric"],
                   "last_bc_validation_loss": metadata.get("last_bc_validation_loss"),
                   "last_bc_validation_update": metadata.get("last_bc_validation_update"),
                   "critic_initialization": prepared["critic_initialization"],
                   "actor_initialization": prepared["actor_initialization"],
                   "actor_bc_auxiliary": prepared["actor_bc_auxiliary"],
                   **_sample_counters(plan, last_update, actor=actor is not None, critic=critic is not None),
                   "memory_metrics": memory_peaks, "checkpoint_format": cfg["checkpoint_format"],
                   "elapsed_seconds": time.perf_counter() - started}
        primary_call(lambda: atomic_json(run_dir / "summary.json", summary))
        return summary
    finally:
        if writer is not None:
            writer.close()
