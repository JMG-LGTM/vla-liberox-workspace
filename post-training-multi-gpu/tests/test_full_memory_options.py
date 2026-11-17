"""CPU evidence for opt-in bucket gradients, scalar AdamW and HF vision GC."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from post_training.config import DEFAULTS, validate_config
from post_training.pi05_assets import load_normalization, platform_model_settings
from post_training.policy import Pi05Policy
from full_memory_fakes import TinyVisionFakeModel
from test_official_bc import mixed
from test_policy_export import base


@pytest.mark.parametrize("section,field,value", [
    ("training", "adamw_foreach", 0), ("training", "adamw_foreach", "false"),
    ("distributed", "gradient_as_bucket_view", None), ("model", "vision_gradient_checkpointing", 1)])
def test_memory_options_reject_ambiguous_types(section, field, value):
    config = copy.deepcopy(DEFAULTS)
    config[section][field] = value
    with pytest.raises(ValueError, match=field):
        validate_config(config)


def test_memory_defaults_preserve_legacy_behavior():
    assert DEFAULTS["training"]["adamw_foreach"] is None
    assert DEFAULTS["distributed"]["gradient_as_bucket_view"] is False
    assert DEFAULTS["model"]["vision_gradient_checkpointing"] is False
    config = copy.deepcopy(DEFAULTS)
    config["model"]["vision_gradient_checkpointing"] = True
    with pytest.raises(ValueError, match="requires full"):
        validate_config(config)


def test_real_hf_vision_layer_gc_retains_full_loss_and_gradients(base):
    root, identity = base
    torch.manual_seed(10)
    reference_model = TinyVisionFakeModel()
    candidate_model = copy.deepcopy(reference_model)
    settings = platform_model_settings(dict(checkpoint=str(root), backbone="full"), identity)
    def actor(model, gc):
        return Pi05Policy(model, lambda value: {"actions": value["actions"]}, identity,
            settings, load_normalization(root), observation_factory=lambda value: value,
            vision_gradient_checkpointing=gc)
    reference, candidate = actor(reference_model, False), actor(candidate_model, True)
    layers = candidate.model.paligemma_with_expert.paligemma.vision_tower.vision_model.encoder.layers
    assert all(layer.gradient_checkpointing is True and layer.training for layer in layers)
    assert all(parameter.requires_grad for parameter in candidate.parameters())
    batch = dict(raw_actions=torch.linspace(-.5, .5, 2 * 8 * 7).reshape(2, 8, 7),
        action_mask=torch.tensor([[True] * 8, [True] * 3 + [False] * 5]),
        agent_image=torch.zeros(2, 28, 28, 3, dtype=torch.uint8),
        wrist_image=torch.zeros(2, 28, 28, 3, dtype=torch.uint8),
        raw_proprio=torch.zeros(2, 8), prompt=["place bowl"] * 2)
    torch.manual_seed(20)
    first = reference(batch)
    first.mean().backward()
    torch.manual_seed(20)
    second = candidate(batch)
    second.mean().backward()
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    gradients = dict(reference.named_parameters())
    nonzero_vision = 0
    for name, parameter in candidate.named_parameters():
        expected = gradients[name]
        assert (parameter.grad is None) == (expected.grad is None)
        if parameter.grad is not None:
            torch.testing.assert_close(expected.grad, parameter.grad, atol=0, rtol=0)
            if ".vision_tower." in name and parameter.grad.abs().sum().item() > 0:
                nonzero_vision += 1
    assert nonzero_vision > 0


def test_scalar_adamw_matches_foreach_for_same_gradients():
    first = torch.nn.Parameter(torch.linspace(-1, 1, 17))
    second = torch.nn.Parameter(first.detach().clone())
    scalar = torch.optim.AdamW([first], lr=3e-5, betas=(.9, .95), eps=1e-8, foreach=False)
    foreach = torch.optim.AdamW([second], lr=3e-5, betas=(.9, .95), eps=1e-8, foreach=True)
    for step in range(3):
        gradient = torch.arange(17, dtype=torch.float32) * (step + 1) / 10
        first.grad, second.grad = gradient.clone(), gradient.clone()
        scalar.step(); foreach.step()
    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_two_rank_full_vision_gc_bucket_views_scalar_adamw_sharded_resume(
        mixed, tmp_path, base, monkeypatch):
    import test_haizhou_bc_validation as checks
    original = checks.configuration
    def full_configuration(*args, **kwargs):
        config = original(*args, **kwargs)
        config["model"].update(backbone="full", vision_gradient_checkpointing=True)
        config["training"]["adamw_foreach"] = False
        config["distributed"]["gradient_as_bucket_view"] = True
        return config
    monkeypatch.setattr(checks, "configuration", full_configuration)
    checks.test_two_process_bc_heldout_zero_sharded_resume(mixed, tmp_path, base)
    for rank in range(2):
        observed = json.loads((tmp_path / "run" / f"rank_{rank}_bc_validation.json").read_text())
        assert observed["memory_options"]["bucket_view"] is True
        assert observed["memory_options"]["foreach"] is False
        assert observed["memory_options"]["vision_gc"] is True
        assert observed["gradient_view_checks"] and all(observed["gradient_view_checks"])
