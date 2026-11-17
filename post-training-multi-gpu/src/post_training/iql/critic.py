"""Standalone pixel IQL networks, objectives, augmentation and portable exports.

Adapted from the platform PixelIQL at origin/server (e27f426) and the local
standalone Q/V experiment. This module never imports a policy or OpenPI.
Optimizers, DDP, accumulation and checkpoint scheduling belong to the runner.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


ARCHITECTURE = "pixel_iql_cnn_v1"
EXPORT_SCHEMA_VERSION = 1


def _positive_int(value: int, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _channel_pixels(pixels: torch.Tensor) -> torch.Tensor:
    if pixels.ndim == 5:
        if pixels.shape[2] != 3:
            raise ValueError("Five-dimensional pixels must have shape [B, cameras, 3, H, W]")
        pixels = pixels.flatten(1, 2)
    if pixels.ndim != 4 or any(size < 1 for size in pixels.shape):
        raise ValueError("pixels must have nonempty shape [B, cameras*3, H, W]")
    return pixels


class PixelEncoder(nn.Module):
    """The small two-camera CNN used in the standalone critic experiments."""

    def __init__(self, proprio_dim: int = 8, output_dim: int = 256,
                 pixel_channels: int = 6):
        super().__init__()
        self.pixel_channels = _positive_int(pixel_channels, "pixel_channels")
        self.proprio_dim = _positive_int(proprio_dim, "proprio_dim")
        channels = (pixel_channels, 32, 64, 128, 256)
        layers: list[nn.Module] = []
        for input_channels, output_channels in zip(channels, channels[1:]):
            layers.extend((nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1),
                           nn.GroupNorm(8, output_channels), nn.SiLU()))
        self.vision = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
        self.proprio = nn.Sequential(nn.Linear(proprio_dim, 64), nn.LayerNorm(64), nn.SiLU())
        self.output = nn.Sequential(nn.Linear(256 + 64, output_dim),
                                    nn.LayerNorm(output_dim), nn.SiLU())

    def forward(self, pixels: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        pixels = _channel_pixels(pixels)
        if pixels.shape[1] != self.pixel_channels:
            raise ValueError(f"Expected {self.pixel_channels} pixel channels, got {pixels.shape[1]}")
        if proprio.shape != (pixels.shape[0], self.proprio_dim):
            raise ValueError(f"proprio must have shape [B, {self.proprio_dim}]")
        # Both uint8 replay and floating point augmentation use the 0..255 domain.
        visual = self.vision(pixels.float() / 255.0).flatten(1)
        return self.output(torch.cat((visual, self.proprio(proprio.float())), dim=-1))


class QNetwork(nn.Module):
    def __init__(self, horizon: int = 8, action_dim: int = 7, proprio_dim: int = 8,
                 pixel_channels: int = 6):
        super().__init__()
        self.horizon = _positive_int(horizon, "horizon")
        self.action_dim = _positive_int(action_dim, "action_dim")
        self.encoder = PixelEncoder(proprio_dim, pixel_channels=pixel_channels)
        self.head = nn.Sequential(nn.Linear(256 + horizon * action_dim * 2, 512),
                                  nn.LayerNorm(512), nn.SiLU(), nn.Linear(512, 256),
                                  nn.SiLU(), nn.Linear(256, 1))

    def forward(self, pixels: torch.Tensor, proprio: torch.Tensor,
                actions: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        if actions.shape != (pixels.shape[0], self.horizon, self.action_dim):
            raise ValueError(f"actions must have shape [B, {self.horizon}, {self.action_dim}]")
        if action_mask.shape != actions.shape[:2]:
            raise ValueError("action_mask must have shape [B, horizon]")
        mask = action_mask.to(dtype=torch.bool).unsqueeze(-1).expand_as(actions)
        # where also masks arbitrary padded NaN values instead of propagating NaN*0.
        masked = torch.where(mask, actions.float(), torch.zeros_like(actions, dtype=torch.float32))
        action_features = torch.cat((masked.flatten(1), mask.float().flatten(1)), dim=-1)
        return self.head(torch.cat((self.encoder(pixels, proprio), action_features), dim=-1)).squeeze(-1)


class ValueNetwork(nn.Module):
    def __init__(self, proprio_dim: int = 8, pixel_channels: int = 6):
        super().__init__()
        self.encoder = PixelEncoder(proprio_dim, pixel_channels=pixel_channels)
        self.head = nn.Sequential(nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 1))

    def forward(self, pixels: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(pixels, proprio)).squeeze(-1)


def expectile_loss(diff: torch.Tensor, expectile: float,
                   reduction: str = "none") -> torch.Tensor:
    """Asymmetric squared error; default reduction preserves each sample."""
    if not math.isfinite(expectile) or not 0 < expectile < 1:
        raise ValueError("expectile must lie strictly between 0 and 1")
    loss = torch.where(diff > 0, expectile, 1.0 - expectile) * diff.float().square()
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError("reduction must be none, mean or sum")


def chunk_bellman_target(reward: torch.Tensor, next_value: torch.Tensor,
                         bootstrap_mask: torch.Tensor, discount: float = .99) -> torch.Tensor:
    """One discount per action block, independent of its primitive-step length.

    bootstrap_mask=0 only for confirmed success. Recording-limit failures keep
    mask=1, including their final block; the replay builder owns that rule.
    """
    if not math.isfinite(discount) or not 0 < discount <= 1:
        raise ValueError("discount must lie in (0, 1]")
    if reward.shape != next_value.shape or bootstrap_mask.shape != next_value.shape:
        raise ValueError("reward, next_value and bootstrap_mask must have matching shapes")
    return reward.float() + discount * bootstrap_mask.float() * next_value.float()


def advantage_weights(advantage: torch.Tensor, beta: float = 3.,
                      maximum: float = 20.) -> torch.Tensor:
    """Unnormalized IQL weights, clipped in log space to avoid exp overflow."""
    if not math.isfinite(beta) or beta <= 0 or not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("beta and maximum must be finite and positive")
    return torch.exp((beta * advantage.float()).clamp(max=math.log(maximum))).clamp(max=maximum)


class PixelIQL(nn.Module):
    """Q1/Q2/V plus target Q, without an actor or embedded optimizers.

    For the reference update order call forward(stage='value'), step the value
    optimizer, then forward(stage='q') and step the Q optimizer. Only after that
    call update_targets() and weights() for the policy update. Accumulate each
    stage over the same microbatches. Wrap this complete module with DDP using
    find_unused_parameters=True when alternating these stages.
    """

    def __init__(self, *, horizon: int = 8, action_dim: int = 7, proprio_dim: int = 8,
                 pixel_channels: int = 6, discount: float = .99,
                 expectile: float = .8, tau: float = .005):
        super().__init__()
        if not math.isfinite(discount) or not 0 < discount <= 1:
            raise ValueError("discount must lie in (0, 1]")
        if not math.isfinite(expectile) or not 0 < expectile < 1:
            raise ValueError("expectile must lie strictly between 0 and 1")
        if not math.isfinite(tau) or not 0 < tau <= 1:
            raise ValueError("tau must lie in (0, 1]")
        self.model_config = {"horizon": horizon, "action_dim": action_dim,
                             "proprio_dim": proprio_dim, "pixel_channels": pixel_channels,
                             "discount": float(discount), "expectile": float(expectile),
                             "tau": float(tau)}
        self.q1 = QNetwork(horizon, action_dim, proprio_dim, pixel_channels)
        self.q2 = QNetwork(horizon, action_dim, proprio_dim, pixel_channels)
        self.target_q1 = copy.deepcopy(self.q1).requires_grad_(False)
        self.target_q2 = copy.deepcopy(self.q2).requires_grad_(False)
        self.value = ValueNetwork(proprio_dim, pixel_channels)
        self.discount, self.expectile, self.tau = float(discount), float(expectile), float(tau)

    def forward(self, batch: Mapping[str, torch.Tensor], stage: str = "both", *,
                beta: float = 3., max_weight: float = 20.) -> dict[str, torch.Tensor]:
        """Return per-sample losses; only the selected stage creates gradients.

        'both' is a joint objective using the pre-update V for Q targets. Use
        separate 'value' and 'q' calls for reference-compatible training.
        """
        if stage not in {"value", "q", "both"}:
            raise ValueError("stage must be value, q or both")
        result: dict[str, torch.Tensor] = {}
        arguments = (batch["pixels"], batch["proprio"], batch["actions"], batch["action_mask"])
        if stage in {"value", "both"}:
            with torch.no_grad():
                target_q = torch.minimum(self.target_q1(*arguments), self.target_q2(*arguments))
            value = self.value(batch["pixels"], batch["proprio"])
            result.update(v_loss=expectile_loss(target_q - value, self.expectile),
                          v=value.detach(), target_q=target_q)
        if stage in {"q", "both"}:
            with torch.no_grad():
                next_value = self.value(batch["next_pixels"], batch["next_proprio"])
                target = chunk_bellman_target(batch["reward"], next_value,
                                              batch["bootstrap_mask"], self.discount)
            q1, q2 = self.q1(*arguments), self.q2(*arguments)
            result.update(q_loss=.5 * ((q1 - target).square() + (q2 - target).square()),
                          q1=q1.detach(), q2=q2.detach(),
                          q=torch.minimum(q1, q2).detach(), target=target)
        if stage == "both":
            result["advantage"] = result["q"] - result["v"]
            result["weights"] = advantage_weights(result["advantage"], beta, max_weight)
        return result

    @torch.no_grad()
    def scores(self, batch: Mapping[str, torch.Tensor], *, beta: float = 3.,
               max_weight: float = 20.) -> dict[str, torch.Tensor]:
        """Score unaugmented cached actions using the online networks."""
        arguments = (batch["pixels"], batch["proprio"], batch["actions"], batch["action_mask"])
        q1, q2 = self.q1(*arguments), self.q2(*arguments)
        q = torch.minimum(q1, q2)
        value = self.value(batch["pixels"], batch["proprio"])
        advantage = q - value
        return {"q1": q1, "q2": q2, "q": q, "v": value, "advantage": advantage,
                "weights": advantage_weights(advantage, beta, max_weight)}

    @torch.no_grad()
    def advantage(self, batch: Mapping[str, torch.Tensor], target: bool = False) -> torch.Tensor:
        arguments = (batch["pixels"], batch["proprio"], batch["actions"], batch["action_mask"])
        q1 = (self.target_q1 if target else self.q1)(*arguments)
        q2 = (self.target_q2 if target else self.q2)(*arguments)
        return torch.minimum(q1, q2) - self.value(batch["pixels"], batch["proprio"])

    @torch.no_grad()
    def weights(self, batch: Mapping[str, torch.Tensor], *, beta: float = 3.,
                max_weight: float = 20.) -> torch.Tensor:
        return advantage_weights(self.advantage(batch), beta, max_weight)

    @torch.no_grad()
    def update_targets(self) -> None:
        for source, target in ((self.q1, self.target_q1), (self.q2, self.target_q2)):
            for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
                target_parameter.lerp_(source_parameter, self.tau)

    soft_update = update_targets


def random_shift_pixels(pixels: torch.Tensor, pad: int = 4, *,
                        generator: torch.Generator | None = None) -> torch.Tensor:
    """Replicate-pad and crop each RGB camera independently, without flipping.

    A separate CPU generator permits deterministic continuation independently
    of the replay sampler and CPU/CUDA placement. Save its state in checkpoints.
    """
    original_shape = pixels.shape
    pixels = _channel_pixels(pixels)
    if pixels.shape[1] % 3:
        raise ValueError("pixels must contain whole RGB cameras")
    if pixels.dtype != torch.uint8 and not pixels.is_floating_point():
        raise ValueError("pixels must be uint8 or floating point in the 0..255 domain")
    if type(pad) is not int or pad < 0:
        raise ValueError("pad must be a nonnegative integer")
    if generator is not None and generator.device.type != "cpu":
        raise ValueError("augmentation generator must be a CPU torch.Generator")
    if pad == 0:
        return pixels.clone().reshape(original_shape)
    batch, channels, height, width = pixels.shape
    cameras = channels // 3
    offsets = torch.randint(0, 2 * pad + 1, (batch, cameras, 2),
                            generator=generator, device="cpu").to(pixels.device)
    y = offsets[..., 0, None, None] + torch.arange(height, device=pixels.device)[None, None, :, None]
    x = offsets[..., 1, None, None] + torch.arange(width, device=pixels.device)[None, None, None, :]
    positions = (y * (width + 2 * pad) + x).reshape(batch, cameras, 1, height * width)
    positions = positions.expand(batch, cameras, 3, height * width)
    padded = F.pad(pixels, (pad, pad, pad, pad), mode="replicate")
    shifted = padded.reshape(batch, cameras, 3, -1).gather(-1, positions)
    return shifted.reshape(original_shape)


def augment_batch(batch: Mapping[str, torch.Tensor], pad: int = 4, *,
                  generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """Shallow-copy a batch, shifting current and next critic images only."""
    result = dict(batch)
    result["pixels"] = random_shift_pixels(batch["pixels"], pad, generator=generator)
    result["next_pixels"] = random_shift_pixels(batch["next_pixels"], pad, generator=generator)
    return result


def default_input_spec(model: PixelIQL | Mapping[str, Any], image_size: int | None = None) -> dict[str, Any]:
    """Describe inputs from either a model or its config, without allocating one."""
    config = model if isinstance(model, Mapping) else model.model_config
    return {"pixels": {"layout": "BCHW", "channels": config["pixel_channels"],
                        "height": image_size, "width": image_size, "dtype": "uint8",
                        "range": [0, 255], "camera_order": ["agentview", "eye_in_hand"],
                        "flip_vertical": True, "flip_horizontal": False},
            "proprio_dim": config["proprio_dim"], "action_dim": config["action_dim"],
            "action_horizon": config["horizon"], "action_mask": "BH_boolean",
            "architecture": ARCHITECTURE}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_metadata(metadata: dict[str, Any]) -> None:
    required = {"schema_version", "kind", "architecture", "model_config", "input_spec",
                "normalization", "reward_definition", "termination_rule", "provenance"}
    if set(metadata) != required:
        raise ValueError("Critic export metadata fields differ from schema")
    if (metadata["schema_version"] != EXPORT_SCHEMA_VERSION or metadata["kind"] != "standalone_critic"
            or metadata["architecture"] != ARCHITECTURE):
        raise ValueError("Unsupported critic export schema or architecture")
    for name in ("model_config", "input_spec", "normalization", "reward_definition"):
        if not isinstance(metadata[name], dict) or not metadata[name]:
            raise ValueError(f"Critic export requires a nonempty {name} mapping")
    if not isinstance(metadata["termination_rule"], str) or not metadata["termination_rule"]:
        raise ValueError("Critic export requires a termination rule version")
    config = metadata["model_config"]
    if set(config) != {"horizon", "action_dim", "proprio_dim", "pixel_channels", "discount", "expectile", "tau"}:
        raise ValueError("Critic model configuration fields differ from schema")
    for name in ("horizon", "action_dim", "proprio_dim", "pixel_channels"):
        _positive_int(config[name], name)
    for name in ("discount", "expectile", "tau"):
        value = config[name]
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not 0 < value <= 1 or name == "expectile" and value == 1):
            raise ValueError(f"Invalid critic export {name}")
    _canonical(metadata)


def export_critic(path: str | Path, model: PixelIQL, *, input_spec: dict[str, Any],
                  normalization: dict[str, Any], reward_definition: dict[str, Any],
                  termination_rule: str, provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    """Atomically export CPU state_dict and JSON-compatible interpretation metadata.

    Normalization must include actual statistics and the transform definition,
    not an absolute path into another machine. No policy object is serialized.
    """
    metadata = {"schema_version": EXPORT_SCHEMA_VERSION, "kind": "standalone_critic",
                "architecture": ARCHITECTURE, "model_config": copy.deepcopy(model.model_config),
                "input_spec": input_spec, "normalization": normalization,
                "reward_definition": reward_definition, "termination_rule": termination_rule,
                "provenance": provenance or {}}
    _validate_metadata(metadata)
    metadata = json.loads(_canonical(metadata))
    payload = {"metadata": metadata,
               "metadata_sha256": hashlib.sha256(_canonical(metadata).encode()).hexdigest(),
               "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return metadata


def _read_export(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict) or set(payload) != {"metadata", "metadata_sha256", "state_dict"}:
        raise ValueError("Not a portable critic export")
    metadata = payload["metadata"]
    _validate_metadata(metadata)
    if hashlib.sha256(_canonical(metadata).encode()).hexdigest() != payload["metadata_sha256"]:
        raise ValueError("Critic export metadata checksum mismatch")
    return payload


def _check_expected(metadata: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, value in expected.items():
        if value is not None and _canonical(value) != _canonical(metadata[key]):
            raise ValueError(f"Critic export {key} differs from the requested training inputs")


def read_critic_metadata(path: str | Path, *, expected_input_spec: dict[str, Any] | None = None,
                         expected_normalization: dict[str, Any] | None = None,
                         expected_reward_definition: dict[str, Any] | None = None,
                         expected_termination_rule: str | None = None,
                         expected_model_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Inspect a tensor-safe, CPU-mapped export without constructing any network."""
    metadata = _read_export(path)["metadata"]
    _check_expected(metadata, {"input_spec": expected_input_spec, "normalization": expected_normalization,
        "reward_definition": expected_reward_definition, "termination_rule": expected_termination_rule,
        "model_config": expected_model_config})
    return copy.deepcopy(metadata)


def load_critic(path: str | Path, model: PixelIQL | None = None, *,
                expected_input_spec: dict[str, Any] | None = None,
                expected_normalization: dict[str, Any] | None = None,
                expected_reward_definition: dict[str, Any] | None = None,
                expected_termination_rule: str | None = None) -> tuple[PixelIQL, dict[str, Any]]:
    """Load tensors without arbitrary pickle code and reject incompatible inputs.

    Fresh standalone inference may omit expected fields to inspect the export.
    IQL initialization must supply all four expected metadata fields from its
    verified replay/config; dimensions alone cannot certify compatibility.
    """
    payload = _read_export(path)
    metadata = payload["metadata"]
    _check_expected(metadata, {"input_spec": expected_input_spec, "normalization": expected_normalization,
        "reward_definition": expected_reward_definition, "termination_rule": expected_termination_rule})
    if model is None:
        model = PixelIQL(**metadata["model_config"])
    elif _canonical(model.model_config) != _canonical(metadata["model_config"]):
        raise ValueError("Critic export model dimensions or hyperparameters differ")
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, metadata
