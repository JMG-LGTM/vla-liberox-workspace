"""Strict configuration; all lengths and intervals count optimizer updates."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import yaml


DEFAULTS = {
    "schema_version": 1,
    "method": "bc",
    "output_dir": "outputs/run",
    "cache_dir": "cache",
    "model": {"checkpoint": "models/pi05_liberox_torch", "backbone": "frozen",
              "dtype": "bfloat16", "num_inference_steps": 10,
              "vision_gradient_checkpointing": False, "actor_init_policy": None},
    "actor_bc": {"dataset_root": None, "weight": 0.1, "per_device_batch_size": 1},
    "data": {"action_horizon": 8, "action_dim": 7, "proprio_dim": 8,
             "include_post_success": False, "image_flip_vertical": True,
             "image_flip_horizontal": False, "validation_fraction": 0.2, "seed": 7,
             "normalization": None, "split_strategy": "hash_rank_count"},
    "training": {"global_batch_size": 16, "per_device_batch_size": 2,
                 "gradient_accumulation_steps": "auto", "optimizer_updates": 800,
                 "seed": 7, "learning_rate": 3e-5, "final_learning_rate": 3e-6,
                 "warmup_updates": 80, "weight_decay": 1e-10, "grad_clip": 1.0,
                 "checkpoint_interval": 100, "checkpoint_format": "consolidated", "log_interval": 10,
                 "validation_interval": 500, "bc_validation": False,
                 "validation_batch_size": None, "validation_seed": 17007, "tensorboard": True,
                 "zero_optimizer": True, "adamw_foreach": None,
                 "sampling": "shuffle", "deterministic": False, "continuation": None},
    "critic": {"image_size": 128, "learning_rate": 3e-4, "value_learning_rate": 3e-4,
               "optimizer": "adamw", "weight_decay": 0.01, "grad_clip": 10.0, "expectile": 0.8, "tau": 0.005,
               "beta": 3.0, "max_weight": 20.0, "random_shift_pad": 4,
               "warmup_updates": 0, "init_checkpoint": None},
    "reward": {"type": "sparse", "step_reward": -1.0, "success_reward": 0.0,
               "discount": 0.99, "discount_unit": "chunk", "normalize": False,
               "success_consecutive_steps": 5,
               "termination_rule": "success_aware_recording_end_v1"},
    "distributed": {"device": "cuda", "backend": "auto", "timeout_seconds": 1800,
                    "workers_per_rank": 2, "prefetch_factor": 2, "pin_memory": True,
                    "gradient_as_bucket_view": False},
}


def _merge(default, override, prefix=""):
    if not isinstance(override, dict):
        raise ValueError(f"{prefix or 'config'} must be a mapping")
    result = copy.deepcopy(default)
    unknown = set(override) - set(default)
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(prefix + str(k) for k in sorted(unknown))}")
    for key, value in override.items():
        result[key] = _merge(default[key], value, prefix + key + ".") if isinstance(default[key], dict) else value
    return result


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def load_config(path, *, method=None):
    raw = yaml.safe_load(Path(path).read_text())
    config = _merge(DEFAULTS, raw if raw is not None else {})
    if method:
        config["method"] = method
    validate_config(config)
    return config


def validate_config(c):
    if type(c["schema_version"]) is not int or c["schema_version"] != 1 or c["method"] not in {"bc", "iql", "critic"}:
        raise ValueError("Expected schema_version=1 and method bc|iql|critic")
    for section, field in (("model", "vision_gradient_checkpointing"),
                           ("training", "tensorboard"), ("training", "bc_validation"), ("training", "zero_optimizer"),
                           ("training", "deterministic"), ("distributed", "pin_memory"),
                           ("distributed", "gradient_as_bucket_view")):
        if type(c[section][field]) is not bool:
            raise ValueError(f"{section}.{field} must be a boolean")
    if c["training"]["adamw_foreach"] is not None and type(c["training"]["adamw_foreach"]) is not bool:
        raise ValueError("training.adamw_foreach must be a boolean or null")
    for field in ("output_dir", "cache_dir"):
        if not isinstance(c[field], str) or not c[field].strip():
            raise ValueError(f"{field} must be a nonempty path string")
    if c["model"]["backbone"] not in {"frozen", "full"}:
        raise ValueError("model.backbone must be frozen or full")
    actor_init = c["model"]["actor_init_policy"]
    if actor_init is not None:
        if not isinstance(actor_init, str) or not actor_init.strip():
            raise ValueError("model.actor_init_policy must be a nonempty policy package path or null")
        if c["method"] != "iql" or c["model"]["backbone"] != "full":
            raise ValueError("model.actor_init_policy requires full IQL training")
    if c["model"]["vision_gradient_checkpointing"] and c["model"]["backbone"] != "full":
        raise ValueError("model.vision_gradient_checkpointing requires full backbone training")
    if c["model"]["dtype"] not in {"float32", "bfloat16"}:
        raise ValueError("model.dtype must be float32 or bfloat16")
    auxiliary = c["actor_bc"]
    _integer(auxiliary["per_device_batch_size"], "actor_bc.per_device_batch_size")
    if auxiliary["dataset_root"] is not None:
        if not isinstance(auxiliary["dataset_root"], str) or not auxiliary["dataset_root"].strip():
            raise ValueError("actor_bc.dataset_root must be a nonempty path or null")
        if c["method"] != "iql":
            raise ValueError("actor_bc auxiliary is only supported for IQL")
        if c["training"]["gradient_accumulation_steps"] not in {1, "auto"}:
            raise ValueError("actor_bc uses one effective actor batch, without gradient accumulation")
        if auxiliary["per_device_batch_size"] >= c["training"]["per_device_batch_size"]:
            raise ValueError("actor_bc must leave at least one native actor sample per device")
    if (type(auxiliary["weight"]) not in (int, float) or not math.isfinite(auxiliary["weight"])
            or auxiliary["weight"] <= 0):
        raise ValueError("actor_bc.weight must be finite and positive")
    _integer(c["model"]["num_inference_steps"], "model.num_inference_steps")
    for section, fields in {
        "training": ["global_batch_size", "per_device_batch_size", "optimizer_updates", "checkpoint_interval", "log_interval", "validation_interval"],
        "data": ["action_horizon", "action_dim", "proprio_dim"],
        "critic": ["image_size"], "distributed": ["timeout_seconds", "prefetch_factor"],
    }.items():
        for field in fields:
            _integer(c[section][field], f"{section}.{field}")
    for section, field in [("training", "seed"), ("training", "warmup_updates"), ("data", "seed"),
                           ("training", "validation_seed"),
                           ("critic", "warmup_updates"), ("critic", "random_shift_pad"),
                           ("distributed", "workers_per_rank")]:
        _integer(c[section][field], f"{section}.{field}", 0)
    accumulation = c["training"]["gradient_accumulation_steps"]
    if accumulation != "auto":
        _integer(accumulation, "training.gradient_accumulation_steps")
    if c["training"]["warmup_updates"] > c["training"]["optimizer_updates"]:
        raise ValueError("warmup_updates cannot exceed optimizer_updates")
    fraction = c["data"]["validation_fraction"]
    if type(fraction) not in (float, int) or not math.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError("validation_fraction must be in [0,1)")
    if c["method"] == "critic" and fraction <= 0:
        raise ValueError("critic requires a root-grouped validation split")
    if c["data"]["split_strategy"] not in {"hash_rank_count", "haizhou_hash_threshold"}:
        raise ValueError("Unknown data.split_strategy")
    if c["training"]["validation_batch_size"] is not None:
        _integer(c["training"]["validation_batch_size"], "training.validation_batch_size")
    if c["training"]["bc_validation"] and (c["method"] != "bc" or fraction <= 0):
        raise ValueError("training.bc_validation requires BC and a held-out validation split")
    if c["data"]["include_post_success"] is not False:
        raise ValueError("Version 1 requires include_post_success=false")
    if c["data"]["image_flip_vertical"] is not True or c["data"]["image_flip_horizontal"] is not False:
        raise ValueError("LIBERO-X pi0.5 requires vertical flip and no horizontal mirror")
    if c["reward"] != DEFAULTS["reward"]:
        raise ValueError("Version 1 implements only the documented sparse reward contract")
    if c["data"]["action_horizon"] != 8 or c["data"]["action_dim"] != 7 or c["data"]["proprio_dim"] != 8:
        raise ValueError("Version 1 pi0.5 LIBERO-X contract requires horizon=8, actions=7, proprio=8")
    if c["distributed"]["device"] not in {"cpu", "cuda"} or c["distributed"]["backend"] not in {"auto", "gloo", "nccl"}:
        raise ValueError("Invalid distributed device/backend")
    if c["distributed"]["device"] == "cpu" and c["distributed"]["backend"] == "nccl":
        raise ValueError("NCCL requires CUDA")
    if c["training"]["sampling"] not in {"shuffle", "replacement"}:
        raise ValueError("training.sampling must be shuffle or replacement")
    if c["training"]["checkpoint_format"] not in {"consolidated", "sharded"}:
        raise ValueError("training.checkpoint_format must be consolidated or sharded")
    if not isinstance(c["critic"]["optimizer"], str) or c["critic"]["optimizer"] not in {"adamw", "adam"}:
        raise ValueError("critic.optimizer must be adamw or adam")
    for section, field in [("training", "learning_rate"), ("training", "final_learning_rate"),
                           ("training", "grad_clip"), ("critic", "learning_rate"),
                           ("critic", "value_learning_rate"), ("critic", "beta"), ("critic", "max_weight"),
                           ("critic", "grad_clip")]:
        value = c[section][field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{section}.{field} must be finite and positive")
    for field in ("expectile", "tau"):
        value = c["critic"][field]
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not 0 < value <= 1 or field == "expectile" and value == 1):
            raise ValueError("Invalid critic expectile/tau")
    for section in ("training", "critic"):
        value = c[section]["weight_decay"]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{section}.weight_decay must be finite and nonnegative")
    if c["method"] == "bc" and c["critic"]["init_checkpoint"] is not None:
        raise ValueError("BC does not use critic.init_checkpoint")
    if c["method"] != "iql" and c["critic"]["warmup_updates"]:
        raise ValueError("critic.warmup_updates only applies to IQL")
    from .continuation import validate_continuation
    validate_continuation(c)


def batch_plan(c, world_size):
    _integer(world_size, "world_size")
    if world_size > 8:
        raise ValueError("This runner supports one node with 1-8 processes")
    t = c["training"]
    micro = t["per_device_batch_size"] * world_size
    if t["global_batch_size"] % micro:
        raise ValueError("global_batch_size must divide exactly by per_device_batch_size * world_size")
    accumulation = t["global_batch_size"] // micro
    if t["gradient_accumulation_steps"] not in {"auto", accumulation}:
        raise ValueError("global_batch_size != per_device_batch_size * world_size * gradient_accumulation_steps")
    warmup = c["critic"]["warmup_updates"] if c["method"] == "iql" else 0
    updates = t["optimizer_updates"] + warmup
    plan = {"world_size": world_size, "global_batch_size": t["global_batch_size"],
            "per_device_batch_size": t["per_device_batch_size"], "global_micro_batch_size": micro,
            "gradient_accumulation_steps": accumulation, "optimizer_updates": updates,
            "policy_optimizer_updates": t["optimizer_updates"] if c["method"] != "critic" else 0,
            "critic_optimizer_updates": updates if c["method"] != "bc" else 0,
            "critic_warmup_updates": warmup, "micro_batches": updates * accumulation,
            "expected_sample_draws": updates * t["global_batch_size"]}
    if c["actor_bc"]["dataset_root"] is not None:
        if accumulation != 1:
            raise ValueError("actor_bc requires gradient_accumulation_steps=1 on the selected world size")
        official = c["actor_bc"]["per_device_batch_size"] * world_size
        plan.update(global_actor_batch_size=t["global_batch_size"],
            global_native_actor_batch_size=t["global_batch_size"] - official,
            global_official_bc_batch_size=official, global_critic_batch_size=t["global_batch_size"])
    return plan


def actor_learning_rate(update, total, peak, final, warmup):
    """Zero-based update index; last update reaches final learning rate."""
    if update < warmup:
        return peak * (update + 1) / max(warmup, 1)
    progress = (update - warmup) / max(total - warmup - 1, 1)
    return final + (peak - final) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
