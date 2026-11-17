"""Small standalone OpenPI adapter, derived from server e27f426 pi05.py.

The upstream model remains an explicitly pinned external dependency. Raw cache
images are already oriented vertically; this module never flips them a second time.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .pi05_assets import (CONTRACT, IDENTITY_FILE, OPENPI_COMMIT, checkpoint_identity,
                          load_normalization, parent_snapshot, platform_model_settings,
                          sha256_file, validate_identity)


def verify_runtime() -> Path:
    """Validate pinned source and every Transformers replacement, without GPU use."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    try:
        import openpi
        import transformers
    except ImportError as exc:
        raise RuntimeError("Run scripts/setup_pi05.py in a dedicated Python 3.11 environment") from exc
    source = Path(openpi.__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if (commit != OPENPI_COMMIT or transformers.__version__ != "4.53.2"
            or torch.__version__.split("+")[0] != "2.7.1" or torch.version.cuda != "12.8"):
        raise RuntimeError(f"Requires OpenPI {OPENPI_COMMIT}, patched Transformers 4.53.2, Torch 2.7.1+cu128")
    changes = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--", "src",
                                       "examples/convert_jax_model_to_pytorch.py"], text=True)
    if changes.strip():
        raise RuntimeError("Pinned OpenPI source contains modifications; refusing changed model behavior")
    replacements = source / "src/openpi/models_pytorch/transformers_replace"
    target = Path(transformers.__file__).resolve().parent
    files = list(replacements.rglob("*.py"))
    if not files:
        raise RuntimeError("OpenPI Transformers replacements are absent")
    for replacement in files:
        installed = target / replacement.relative_to(replacements)
        if not installed.is_file() or sha256_file(installed) != sha256_file(replacement):
            raise RuntimeError("OpenPI Transformers patches are missing/modified; rerun setup_pi05.py")
    return source


def masked_flow_losses(errors: torch.Tensor, mask: torch.Tensor, action_dim: int = 7) -> torch.Tensor:
    """Mean per sample over valid timesteps and the seven real motor coordinates."""
    if (errors.ndim != 3 or mask.ndim != 2 or errors.shape[0] != mask.shape[0]
            or not 0 < mask.shape[1] <= errors.shape[1] or not 0 < action_dim <= errors.shape[2]):
        raise ValueError("Invalid flow error/action mask dimensions")
    if not torch.all((mask == 0) | (mask == 1)) or not mask.bool().any(dim=1).all():
        raise ValueError("Action mask must be binary with a nonempty horizon per sample")
    valid = mask.to(errors.device, dtype=errors.dtype)
    return (errors[:, :mask.shape[1], :action_dim] * valid[..., None]).sum((1, 2)) / (valid.sum(1) * action_dim)


class Pi05Policy(nn.Module):
    """DDP forward owns CPU transforms and returns differentiable [batch] losses."""

    def __init__(self, model: nn.Module, input_transform, identity: dict, settings: dict,
                 normalization: dict, *, observation_factory=None, vision_gradient_checkpointing=False):
        super().__init__()
        self.model = model
        self.input_transform = input_transform
        self.identity = validate_identity(identity)
        self.settings = settings
        self.parent = parent_snapshot(settings)
        self.action_stats = normalization["actions"]
        self.proprio_stats = normalization["state"]
        self.stats_key = settings["stats_key"]
        self._observation_factory = observation_factory
        model.requires_grad_(True)
        if settings["backbone"] == "frozen":
            model.paligemma_with_expert.paligemma.requires_grad_(False)
        model.gradient_checkpointing_enable()
        if vision_gradient_checkpointing:
            if settings["backbone"] != "full":
                raise ValueError("Vision layer checkpointing requires full backbone training")
            # OpenPI checkpoints the complete image embedding call and sets a
            # top-level vision flag. HF 4.53 SigLIP checks flags on each layer;
            # its public API recursively enables those finer-grained checkpoints.
            vision = model.paligemma_with_expert.paligemma.vision_tower
            vision.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.vision_gradient_checkpointing = vision_gradient_checkpointing
        self.train()

    def train(self, mode: bool = True):
        super().train(mode)
        # Calling DDP/module.train() must never reactivate dropout in the frozen VLM.
        if self.settings["backbone"] == "frozen":
            self.model.paligemma_with_expert.paligemma.eval()
        return self

    def forward(self, batch: dict) -> torch.Tensor:
        from torch.utils.data import default_collate
        from torch.utils._pytree import tree_map

        factory = self._observation_factory
        if factory is None:
            from openpi.models.model import Observation
            factory = Observation.from_dict
        device = next(self.model.parameters()).device
        raw_actions = batch.get("actor_raw_actions", batch["raw_actions"])
        mask = batch.get("actor_action_mask", batch["action_mask"])
        if (raw_actions.ndim != 3 or raw_actions.shape[2] != 7
                or raw_actions.shape[1] > 10 or tuple(mask.shape) != tuple(raw_actions.shape[:2])):
            raise ValueError("pi05 actor expects raw actions [B,H<=10,7] and matching mask")
        def cpu_numpy(value):
            if isinstance(value, torch.Tensor):
                if value.device.type != "cpu":
                    raise ValueError("Raw replay fields must stay on CPU for OpenPI transforms")
                return value.detach().numpy()
            return np.asarray(value)
        samples = []
        if len(batch["prompt"]) != raw_actions.shape[0]:
            raise ValueError("Prompt count does not match action batch")
        for index, prompt in enumerate(batch["prompt"]):
            actions = np.zeros((10, 7), dtype=np.float32)
            actions[:raw_actions.shape[1]] = cpu_numpy(raw_actions[index])
            samples.append(self.input_transform({
                "observation/image": cpu_numpy(batch["agent_image"][index]),
                "observation/wrist_image": cpu_numpy(batch["wrist_image"][index]),
                "observation/state": cpu_numpy(batch["raw_proprio"][index]),
                "actions": actions, "prompt": prompt,
            }))
        inputs = tree_map(lambda value: value.to(device, non_blocking=True)
                          if isinstance(value, torch.Tensor) else value, default_collate(samples))
        inputs["actions"] = inputs["actions"].float()
        errors = self.model(factory(inputs), inputs["actions"])
        return masked_flow_losses(errors, mask)

    def trainable_state_dict(self) -> dict:
        """Use native upstream parameter names, as required by the platform loader."""
        return {name: value.detach().cpu() for name, value in self.model.named_parameters() if value.requires_grad}

    def load_trainable_state_dict(self, state: dict) -> None:
        params = {name: value for name, value in self.model.named_parameters() if value.requires_grad}
        if state.keys() != params.keys():
            raise ValueError("Actor parameter keys do not match frozen/full scope")
        for name, value in state.items():
            if value.shape != params[name].shape:
                raise ValueError(f"Actor parameter shape mismatch: {name}")
        with torch.no_grad():
            for name, value in state.items():
                params[name].copy_(value)

    def parameter_counts(self) -> dict:
        return {"total": sum(p.numel() for p in self.model.parameters()),
                "trainable": sum(p.numel() for p in self.model.parameters() if p.requires_grad)}


def load_policy(model_config: dict, device="cpu", dtype: str | None = None,
                *, verified_identity: dict | None = None) -> Pi05Policy:
    """Load only for BC/IQL. Rank zero may broadcast a freshly verified identity."""
    root = Path(model_config.get("checkpoint") or model_config.get("base_checkpoint", "")).expanduser().resolve()
    identity = checkpoint_identity(root) if verified_identity is None else validate_identity(verified_identity)
    if verified_identity is not None and json.loads((root / IDENTITY_FILE).read_text()) != identity:
        raise ValueError("Base identity changed after rank-zero verification")
    settings = platform_model_settings(model_config, identity)
    verify_runtime()
    from openpi import transforms
    from openpi.training import config as configs, checkpoints
    cfg = configs.get_config(identity["config_name"])
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, repo_id=settings["stats_key"]),
                              model=dataclasses.replace(cfg.model, dtype=dtype or model_config.get("dtype", "bfloat16"),
                              pytorch_compile_mode=None, action_horizon=10, action_dim=32, discrete_state_input=False))
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    if f"assets/{data.asset_id}/norm_stats.json" not in identity["files"]:
        raise ValueError("OpenPI selected normalization not bound to the model identity")
    stats = checkpoints.load_norm_stats(root / "assets", data.asset_id)
    model = cfg.model.load_pytorch(cfg, str(root / "model.safetensors"))
    inputs = transforms.compose([*data.data_transforms.inputs,
                                 transforms.Normalize(stats, use_quantiles=data.use_quantile_norm),
                                 *data.model_transforms.inputs])
    policy = Pi05Policy(model, inputs, identity, settings, load_normalization(root, verify=False),
                       vision_gradient_checkpointing=model_config.get("vision_gradient_checkpointing", False))
    return policy.to(device)
