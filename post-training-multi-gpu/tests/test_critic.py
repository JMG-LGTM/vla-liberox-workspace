"""CPU checks for the actual IQL equations, update order and export boundary."""
from __future__ import annotations

import json

import pytest
import torch

from post_training.iql.critic import (PixelIQL, QNetwork, advantage_weights, augment_batch,
                                 chunk_bellman_target, default_input_spec, expectile_loss,
                                 export_critic, load_critic, random_shift_pixels)
from post_training.diagnostics import (average_ranks, critic_diagnostics, evaluate_critic,
                                      spearman, weight_statistics)


def batch(size=3):
    generator = torch.Generator().manual_seed(54)
    return {"pixels": torch.randint(0, 256, (size, 6, 16, 16), dtype=torch.uint8, generator=generator),
            "next_pixels": torch.randint(0, 256, (size, 6, 16, 16), dtype=torch.uint8, generator=generator),
            "proprio": torch.randn(size, 8, generator=generator),
            "next_proprio": torch.randn(size, 8, generator=generator),
            "actions": torch.randn(size, 2, 7, generator=generator),
            "action_mask": torch.ones(size, 2, dtype=torch.bool),
            "reward": torch.full((size,), -1.), "bootstrap_mask": torch.ones(size),
            "remaining_blocks": torch.arange(size - 1, -1, -1),
            "success_terminal": torch.arange(size) == size - 1}


def test_loss_equations_masks_and_stable_advantage_weights():
    assert torch.allclose(expectile_loss(torch.tensor([-2., 0., 2.]), .7), torch.tensor([1.2, 0., 2.8]))
    # Failure recording limit keeps bootstrapping; confirmed success ends at 0.
    target = chunk_bellman_target(torch.tensor([-1., 0.]), torch.tensor([-5., -5.]),
                                  torch.tensor([1., 0.]), .99)
    assert torch.allclose(target, torch.tensor([-5.95, 0.]))
    weights = advantage_weights(torch.tensor([-10000., 0., 10000.]), 3., 20.)
    assert torch.equal(weights, torch.tensor([0., 1., 20.]))
    assert torch.isfinite(weights).all()
    with pytest.raises(ValueError, match="matching shapes"):
        chunk_bellman_target(torch.ones(2, 1), torch.ones(2), torch.ones(2))


def test_padded_actions_do_not_change_q_even_for_nan_padding():
    model = QNetwork(horizon=2)
    data = batch()
    data["action_mask"][:, 1] = False
    before = model(data["pixels"], data["proprio"], data["actions"], data["action_mask"])
    data["actions"][:, 1] = float("nan")
    after = model(data["pixels"], data["proprio"], data["actions"], data["action_mask"])
    assert torch.equal(before, after)
    assert torch.isfinite(after).all()


def test_staged_losses_use_updated_v_and_only_backpropagate_selected_networks():
    torch.manual_seed(11)
    model, data = PixelIQL(horizon=2), batch()
    target_before = model(data, stage="q")["target"].clone()
    result = model(data, stage="value")
    assert result["v_loss"].shape == (3,)
    result["v_loss"].mean().backward()
    assert all(parameter.grad is not None for parameter in model.value.parameters())
    assert all(parameter.grad is None for parameter in model.q1.parameters())
    assert all(parameter.grad is None for parameter in model.target_q1.parameters())
    optimizer = torch.optim.SGD(model.value.parameters(), lr=.05)
    optimizer.step()
    model.zero_grad(set_to_none=True)
    result = model(data, stage="q")
    assert not torch.equal(result["target"], target_before)
    result["q_loss"].mean().backward()
    assert all(parameter.grad is not None for parameter in model.q1.parameters())
    assert all(parameter.grad is not None for parameter in model.q2.parameters())
    assert all(parameter.grad is None for parameter in model.value.parameters())
    assert all(parameter.grad is None for parameter in model.target_q1.parameters())
    joint = model(data)
    assert not joint["weights"].requires_grad
    assert joint["q_loss"].shape == joint["v_loss"].shape == (3,)


def test_polyak_updates_targets_once_without_training_them():
    model = PixelIQL(horizon=2, tau=.25)
    before = next(model.target_q1.parameters()).clone()
    with torch.no_grad():
        next(model.q1.parameters()).add_(4.)
    model.update_targets()
    assert torch.allclose(next(model.target_q1.parameters()), before + 1., atol=1e-7)
    assert all(not parameter.requires_grad for parameter in model.target_q1.parameters())


def test_augmentation_matches_independent_replicate_crops_and_restores_rng():
    pixels = torch.arange(2 * 6 * 7 * 9).remainder(255).to(torch.uint8).reshape(2, 6, 7, 9)
    original = pixels.clone()
    state = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(17)
    saved = generator.get_state()
    shifted = random_shift_pixels(pixels, pad=4, generator=generator)
    generator.set_state(saved)
    offsets = torch.randint(0, 9, (2, 2, 2), generator=generator)
    padded = torch.nn.functional.pad(pixels, (4, 4, 4, 4), mode="replicate")
    for sample in range(2):
        for camera in range(2):
            y, x = offsets[sample, camera].tolist()
            expected = padded[sample, camera * 3:(camera + 1) * 3, y:y + 7, x:x + 9]
            assert torch.equal(shifted[sample, camera * 3:(camera + 1) * 3], expected)
    assert torch.equal(pixels, original)
    assert torch.equal(torch.get_rng_state(), state)
    generator.set_state(saved)
    assert torch.equal(random_shift_pixels(pixels, generator=generator), shifted)
    assert torch.equal(random_shift_pixels(pixels, pad=0), pixels)
    views = pixels.reshape(2, 2, 3, 7, 9)
    assert random_shift_pixels(views).shape == views.shape
    data = batch()
    augmented = augment_batch(data, generator=torch.Generator().manual_seed(3))
    assert augmented["actions"] is data["actions"]
    assert augmented["pixels"] is not data["pixels"]


def test_diagnostics_ties_constant_failure_exclusion_and_weight_mass():
    assert average_ranks([3, 1, 1]).tolist() == [3., 1.5, 1.5]
    assert spearman([3., 1., 1.], [1., 3., 3.]) == pytest.approx(-1.)
    assert spearman([0., 0.], [1., 2.]) is None
    assert spearman([float("nan"), 0.], [1., 2.]) is None
    report = critic_diagnostics([-3., -2., -1., 100.], [2., 1., 0., -1.],
                                [.01, 1., 20., 1.], [False, False, True, False])
    assert report["spearman_v_remaining_blocks"] == pytest.approx(-1.)
    assert report["success_progress_count"] == 3
    assert report["clipped_fraction"] == .25
    assert report["below_0_1_fraction"] == .25
    assert report["success_terminal_weight_share"] == pytest.approx(20 / 22.01)
    assert report["ess_fraction"] == pytest.approx(22.01 ** 2 / (.01 ** 2 + 1 + 400 + 1) / 4)
    json.dumps(report, allow_nan=False)
    zero = weight_statistics([0., 0.], [0, 1])
    assert zero["ess_fraction"] == 0.
    assert zero["success_terminal_weight_share"] is None
    invalid = critic_diagnostics([0., float("nan")], [1., 0.], [1., float("inf")], [0, 1])
    assert not invalid["finite"]
    assert invalid["ess_fraction"] is None
    json.dumps(invalid, allow_nan=False)


def export_metadata(model):
    return {"input_spec": default_input_spec(model, image_size=16),
            "normalization": {"kind": "openpi_quantile", "actions": {"q01": [-1.] * 7, "q99": [1.] * 7},
                              "proprio": {"q01": [-1.] * 8, "q99": [1.] * 8}},
            "reward_definition": {"kind": "sparse", "block_reward": -1., "success_reward": 0.,
                                  "gamma": .99, "discount_per": "block", "normalized": False},
            "termination_rule": "success_aware_recording_end_v1"}


def test_portable_export_roundtrip_and_reject_incompatible_metadata(tmp_path):
    model, data = PixelIQL(horizon=2), batch()
    expected = model.scores(data)
    path = tmp_path / "critic.pt"
    metadata = export_metadata(model)
    export_critic(path, model, **metadata)
    loaded, saved = load_critic(path, **{f"expected_{key}": value for key, value in metadata.items()})
    assert saved["kind"] == "standalone_critic"
    assert all(torch.equal(expected[key], loaded.scores(data)[key]) for key in expected)
    for key in metadata:
        changed = "wrong_version" if key == "termination_rule" else {"incompatible": True}
        with pytest.raises(ValueError, match=key):
            load_critic(path, **{f"expected_{key}": changed})
    with pytest.raises(ValueError, match="dimensions or hyperparameters"):
        load_critic(path, model=PixelIQL(horizon=3))
    payload = torch.load(path, weights_only=True)
    payload["metadata"]["termination_rule"] = "tampered"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="checksum"):
        load_critic(path)


def test_evaluation_uses_online_scores_preserves_modes_and_rng():
    model, data = PixelIQL(horizon=2), batch()
    model.train()
    model.target_q1.eval()
    modes = {name: item.training for name, item in model.named_modules()}
    state = torch.get_rng_state().clone()
    report = evaluate_critic(model, [data])
    assert report["count"] == 3
    assert report["finite"]
    assert modes == {name: item.training for name, item in model.named_modules()}
    assert torch.equal(state, torch.get_rng_state())
    json.dumps(report, allow_nan=False)
    empty = evaluate_critic(model, [])
    assert empty["count"] == 0
    assert empty["spearman_v_remaining_blocks"] is None
