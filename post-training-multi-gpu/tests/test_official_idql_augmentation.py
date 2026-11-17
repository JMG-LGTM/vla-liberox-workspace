"""Author DrQ crops, read-only diagnostics and full augmentation-RNG restore.

All execution is CPU-only. Small explicit encoders keep these behavioral tests
cheap; they do not claim real A100 throughput or task rollout performance.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import os
from pathlib import Path
import sys

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
if "jax" not in sys.modules:
    flags = [x for x in os.environ.get("XLA_FLAGS", "").split()
             if not x.startswith("--xla_force_host_platform_device_count=")]
    os.environ["XLA_FLAGS"] = " ".join(flags + ["--xla_force_host_platform_device_count=2"])
pytest.importorskip("gym")
jax = pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")
from flax import serialization
import jax.numpy as jnp
import numpy as np

from post_training.official_idql import core, upstream
from post_training.official_idql.checkpoint import load_checkpoint, save_checkpoint

CAMERAS = ("agent_pixels", "wrist_pixels")


@pytest.fixture(scope="module", autouse=True)
def cpu_only():
    assert all(device.platform == "cpu" for device in jax.devices())


def pixels(*, seed=17, pad=4, axis_name=None):
    return core.create_agent(
        seed, {key: np.zeros((16, 16, 3), np.uint8) for key in CAMERAS}
        | {"state": np.zeros(8, np.float32)}, np.zeros(64, np.float32),
        observation_kind="pixels", pixel_keys=CAMERAS,
        hidden_dims=(16, 16), encoder_features=(4, 4), encoder_filters=(3, 3),
        encoder_strides=(2, 1), latent_dim=5, augmentation_pad=pad,
        axis_name=axis_name,
    )


def batch(size=8):
    ramp = ((np.arange(size * 16 * 16 * 3).reshape(size, 16, 16, 3) * 7) % 251).astype(np.uint8)
    observations = {key: ramp.copy() for key in CAMERAS}
    observations["state"] = np.linspace(-.7, .7, size * 8, dtype=np.float32).reshape(size, 8)
    actions = np.linspace(-.8, .8, size * 64, dtype=np.float32).reshape(size, 64)
    actions[:, -8:] = 1
    return dict(observations=observations, next_observations=copy.deepcopy(observations),
                actions=actions, rewards=np.full(size, -1, np.float32), masks=np.ones(size, np.float32))


def same_tree(left, right):
    a, sa = jax.tree_util.tree_flatten(left)
    b, sb = jax.tree_util.tree_flatten(right)
    assert sa == sb
    for x, y in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_actual_upstream_crop_each_view_and_time_without_mutating_input(monkeypatch):
    module = upstream.load_augmentations()
    source = Path(inspect.getsourcefile(module.batched_random_crop))
    expected = upstream.verify_upstream()["files"]["jaxrl5/agents/drq/augmentations.py"]["sha256"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    assert core._AUGMENTATIONS is module
    agent, data = pixels(), batch()
    original = copy.deepcopy(data)
    calls = []
    author_crop = module.batched_random_crop
    def record(key, observations, pixel_key, padding=4):
        calls.append((pixel_key, padding))
        return author_crop(key, observations, pixel_key, padding)
    monkeypatch.setattr(module, "batched_random_crop", record)
    advanced, augmented = core._augment_training_batch(agent, data)
    assert calls == [(key, 4) for _ in range(2) for key in CAMERAS]
    same_tree(data, original)

    # Independent author-function reference with the declared four-key stream.
    expected_batch = dict(data)
    rng = agent.rng
    for field in ("observations", "next_observations"):
        observations = core._observations(agent, original[field])
        for key in CAMERAS:
            crop_key, rng = jax.random.split(rng)
            observations = author_crop(crop_key, observations, key, padding=4)
        expected_batch[field] = observations
    same_tree(augmented, expected_batch)
    np.testing.assert_array_equal(advanced.rng, rng)
    assert not np.array_equal(agent.rng, advanced.rng)
    views = [np.asarray(augmented[field][key]) for field in ("observations", "next_observations") for key in CAMERAS]
    for index, view in enumerate(views):
        assert view.dtype == np.uint8 and view.shape == (8, 16, 16, 3, 1)
        assert not np.array_equal(view[..., 0], original["observations"][CAMERAS[0]])
        for other in views[:index]:
            assert not np.array_equal(view, other)
    same_tree(augmented["actions"], data["actions"])
    same_tree(augmented["observations"]["state"], data["observations"]["state"])


def test_update_uses_one_common_augmented_batch_for_original_v_then_q():
    agent, data = pixels(), batch(4)
    advanced, augmented = core._augment_training_batch(agent, data)
    expected, expected_info = core.update(advanced.replace(augmentation_pad=0), augmented)
    actual, info = core.update(agent, data)
    for name in ("critic", "value", "target_critic"):
        left = serialization.to_state_dict(getattr(actual, name))
        right = serialization.to_state_dict(getattr(expected, name))
        for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
            np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-5)
    np.testing.assert_array_equal(actual.rng, advanced.rng)
    for key in ("value_loss", "critic_loss"):
        np.testing.assert_allclose(info[key], expected_info[key], atol=1e-6, rtol=1e-5)
    assert int(actual.critic.step) == int(actual.value.step) == int(actual.step) == 1
    assert type(actual).update_v is upstream.load_upstream().DDPMIQLLearner.update_v
    assert type(actual).update_q is upstream.load_upstream().DDPMIQLLearner.update_q


def test_disabled_crop_and_vector_path_never_consume_rng():
    agent, data = pixels(pad=0), batch(4)
    after, prepared = core._augment_training_batch(agent, data)
    np.testing.assert_array_equal(after.rng, agent.rng)
    for field in ("observations", "next_observations"):
        for camera in CAMERAS:
            np.testing.assert_array_equal(prepared[field][camera][..., 0], data[field][camera])
    updated, _ = core.update(agent, data)
    np.testing.assert_array_equal(updated.rng, agent.rng)

    vector = core.create_agent(17, np.zeros(8, np.float32), np.zeros(64, np.float32), hidden_dims=(16, 16))
    vector_batch = dict(data, observations=data["observations"]["state"], next_observations=data["next_observations"]["state"])
    initial_rng = np.asarray(vector.rng).copy()
    for _ in range(3):
        vector, _ = core.update(vector, vector_batch)
        np.testing.assert_array_equal(vector.rng, initial_rng)
    with pytest.raises(ValueError, match="pixels"):
        core.create_agent(17, np.zeros(8), np.zeros(64), augmentation_pad=4)
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="augmentation_pad"):
            pixels(pad=bad)


def test_validation_and_scoring_do_not_augment_or_change_any_state(monkeypatch):
    agent, data = pixels(), batch(4)
    before = serialization.to_bytes(agent)
    def forbidden(*args, **kwargs):
        raise AssertionError("Held-out/scoring inputs must not be augmented")
    monkeypatch.setattr(core._AUGMENTATIONS, "batched_random_crop", forbidden)
    outputs = core.evaluate_outputs(agent, data)
    losses = core.evaluate_losses(agent, data)
    repeated = core.evaluate_outputs(agent, data)
    same_tree(outputs, repeated)
    assert set(outputs) == {"q1", "q2", "q_min", "target_q1", "target_q2", "target_q_min", "v"}
    assert all(np.asarray(value).shape == (4,) for value in outputs.values())
    observations = core._observations(agent, data["observations"])
    online = agent.critic.apply_fn({"params": agent.critic.params}, observations, data["actions"])
    target = agent.target_critic.apply_fn({"params": agent.target_critic.params}, observations, data["actions"])
    values = agent.value.apply_fn({"params": agent.value.params}, observations)
    for key, expected in (("q1", online[0]), ("q2", online[1]), ("target_q1", target[0]),
                          ("target_q2", target[1]), ("v", values)):
        np.testing.assert_allclose(outputs[key], expected, atol=1e-6, rtol=1e-5)
    candidates = np.repeat(data["actions"][:, None], 3, axis=1)
    scores = core.score_actions(agent, data["observations"], candidates)
    np.testing.assert_allclose(scores["q_min"], np.asarray(outputs["target_q_min"])[:, None].repeat(3, 1), atol=1e-6, rtol=1e-5)
    assert all(np.isfinite(np.asarray(value)).all() for value in losses.values())
    assert serialization.to_bytes(agent) == before


def test_pmap_uses_different_rank_crops_for_identical_input():
    if jax.local_device_count() != 2:
        pytest.skip("Run this file in a fresh process with two virtual CPU devices")
    agent, data = pixels(axis_name="devices"), batch(4)
    replicated = jax.device_put_replicated(agent, jax.local_devices())
    identical = jax.tree_util.tree_map(lambda value: np.stack([value, value]), data)
    after, augmented = jax.pmap(core._augment_training_batch, axis_name="devices")(replicated, identical)
    assert not np.array_equal(after.rng[0], after.rng[1])
    for field in ("observations", "next_observations"):
        for camera in CAMERAS:
            assert not np.array_equal(augmented[field][camera][0], augmented[field][camera][1])
    same_tree(identical, jax.tree_util.tree_map(lambda value: np.stack([value, value]), data))


def test_full_checkpoint_restores_each_augmented_rank_stream_exactly(tmp_path):
    if jax.local_device_count() != 2:
        pytest.skip("Run this file in a fresh process with two virtual CPU devices")
    agent = pixels(axis_name="devices")
    state = jax.device_put_replicated(agent, jax.local_devices())
    update = jax.pmap(core.update, axis_name="devices")
    source = batch(12)
    generator = np.random.default_rng(42)
    def draw(rng):
        indices = rng.integers(0, 12, 8)
        return jax.tree_util.tree_map(lambda values: values[indices].reshape(2, 4, *values.shape[1:]), source)
    for _ in range(2):
        state, _ = update(state, draw(generator))
    assert not np.array_equal(state.rng[0], state.rng[1])
    contract = {"backend": "official_idql_jax", "upstream_commit": upstream.UPSTREAM_COMMIT,
                "augmentation_pad": 4, "device_count": 2}
    path = save_checkpoint(tmp_path / "update_00000002", state, contract,
                           {"rng_state": generator.bit_generator.state, "completed_updates": 2, "draws": 16})
    template = jax.device_put_replicated(pixels(axis_name="devices"), jax.local_devices())
    restored, sampling, _ = load_checkpoint(path, template, contract)
    same_tree(serialization.to_state_dict(state), serialization.to_state_dict(restored))
    restored_generator = np.random.default_rng()
    restored_generator.bit_generator.state = sampling["rng_state"]
    for _ in range(3):
        uninterrupted_batch = draw(generator)
        resumed_batch = draw(restored_generator)
        same_tree(uninterrupted_batch, resumed_batch)
        state, left_metrics = update(state, uninterrupted_batch)
        restored, right_metrics = update(restored, resumed_batch)
        same_tree(serialization.to_state_dict(state), serialization.to_state_dict(restored))
        same_tree(left_metrics, right_metrics)
        for model in (state.critic.params, state.value.params, state.target_critic.params):
            for leaf in jax.tree_util.tree_leaves(model):
                np.testing.assert_array_equal(leaf[0], leaf[1])
    assert np.asarray(state.step).tolist() == [5, 5]
