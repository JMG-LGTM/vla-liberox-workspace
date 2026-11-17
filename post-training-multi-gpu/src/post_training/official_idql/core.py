"""Author IDQL Q/V functions with explicit multi-device and input adapters.

``vector`` uses the official simple learner's exact MLP construction and key
split. ``pixels`` explicitly combines author DrQ PixelMultiplexer/D4PG modules
with the same Q/V update functions; it is a domain adaptation, not the paper's
state-benchmark network. No Torch critic or preference objective is used here.
"""

from __future__ import annotations

from functools import partial
from numbers import Integral, Real
from typing import Mapping, Sequence

from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .upstream import load_augmentations, load_networks, load_pixel_encoder, load_upstream


_OFFICIAL = load_upstream()
_NETWORKS = load_networks()
_AUGMENTATIONS = load_augmentations()


class SynchronizedTrainState(TrainState):
    """Only the gradient collective is added; Adam itself remains Optax Adam."""

    axis_name: str | None = struct.field(pytree_node=False, default=None)

    def apply_gradients(self, *, grads, **kwargs):
        if self.axis_name is not None:
            grads = jax.lax.pmean(grads, axis_name=self.axis_name)
        return super().apply_gradients(grads=grads, **kwargs)


class CriticAgent(struct.PyTreeNode):
    critic: SynchronizedTrainState
    target_critic: TrainState
    value: SynchronizedTrainState
    rng: jax.Array
    discount: float
    tau: float
    critic_hyperparam: float
    step: jax.Array
    axis_name: str | None = struct.field(pytree_node=False)
    observation_kind: str = struct.field(pytree_node=False)
    pixel_keys: tuple[str, ...] = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)
    augmentation_pad: int = struct.field(pytree_node=False)

    # These are the *same function objects* read from unmodified author source.
    update_v = _OFFICIAL.DDPMIQLLearner.update_v
    update_q = _OFFICIAL.DDPMIQLLearner.update_q


def _observations(agent_or_kind, observations, pixel_keys=()):
    kind = agent_or_kind if isinstance(agent_or_kind, str) else agent_or_kind.observation_kind
    if not isinstance(agent_or_kind, str):
        pixel_keys = agent_or_kind.pixel_keys
    if kind == "vector":
        return jnp.asarray(observations, dtype=jnp.float32)
    if not isinstance(observations, Mapping):
        raise TypeError("pixels observations must contain explicitly named image/state fields")
    result = {key: jnp.asarray(value) for key, value in observations.items()}
    for key in pixel_keys:
        image = result[key]
        # Upstream PixelMultiplexer expects (..., H, W, C, stack).  Public
        # adapters use (..., H, W, C); add only the singleton stack dimension.
        if image.ndim == 4:
            image = image[..., None]
        if image.ndim != 5:
            raise ValueError(f"Batched pixel field {key!r} must be BHWC or BHWC-stack")
        result[key] = image
    if "state" in result:
        result["state"] = result["state"].astype(jnp.float32)
    return FrozenDict(result)


def create_agent(
    seed: int,
    observation_sample,
    action_sample,
    *,
    observation_kind: str = "vector",
    hidden_dims: Sequence[int] = (256, 256),
    critic_lr: float = 3e-4,
    value_lr: float = 3e-4,
    expectile: float = 0.7,
    discount: float = 0.99,
    tau: float = 0.005,
    axis_name: str | None = None,
    pixel_keys: Sequence[str] = ("pixels",),
    encoder_features: Sequence[int] = (32, 32, 32, 32),
    encoder_filters: Sequence[int] = (3, 3, 3, 3),
    encoder_strides: Sequence[int] = (2, 1, 1, 1),
    latent_dim: int = 50,
    augmentation_pad: int = 0,
) -> CriticAgent:
    """Initialize Q/V from unbatched samples, matching official split-four keys.

    The unused actor key is deliberately retained in the split so vector
    initialization is directly comparable to ``DDPMIQLLearner.create``. There
    is no actor state and no policy update in this adapter.
    """
    if observation_kind not in {"vector", "pixels"}:
        raise ValueError("observation_kind must explicitly be 'vector' or 'pixels'")
    if isinstance(augmentation_pad, (bool, np.bool_)) or not isinstance(augmentation_pad, Integral) or augmentation_pad < 0:
        raise ValueError("augmentation_pad must be a nonnegative integer")
    augmentation_pad = int(augmentation_pad)
    if observation_kind != "pixels" and augmentation_pad:
        raise ValueError("Random crop augmentation requires explicit pixels mode")
    scalars = (critic_lr, value_lr, expectile, discount, tau)
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) or not np.isfinite(x) for x in scalars):
        raise ValueError("IDQL scalar parameters must be finite real numbers")
    if not 0 < expectile < 1 or not 0 < discount <= 1 or not 0 < tau <= 1:
        raise ValueError("Invalid expectile, discount or target tau")
    if critic_lr <= 0 or value_lr <= 0:
        raise ValueError("Learning rates must be positive")
    def positive_dimensions(name, values):
        values = tuple(values)
        if not values or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) or x <= 0 for x in values):
            raise ValueError(f"{name} must contain positive integers")
        return tuple(int(x) for x in values)
    hidden_dims = positive_dimensions("hidden_dims", hidden_dims)
    encoder_features = positive_dimensions("encoder_features", encoder_features)
    encoder_filters = positive_dimensions("encoder_filters", encoder_filters)
    encoder_strides = positive_dimensions("encoder_strides", encoder_strides)
    if len({len(encoder_features), len(encoder_filters), len(encoder_strides)}) != 1:
        raise ValueError("Encoder feature/filter/stride arrays must have equal lengths")
    latent_dim = positive_dimensions("latent_dim", (latent_dim,))[0]
    if axis_name is not None and (not isinstance(axis_name, str) or not axis_name):
        raise ValueError("axis_name must be None or a nonempty string")
    action_sample = np.asarray(action_sample, dtype=np.float32)
    if action_sample.ndim != 1 or not action_sample.size or not np.isfinite(action_sample).all():
        raise ValueError("action_sample must be one finite, unbatched vector")
    pixel_keys = tuple(pixel_keys) if observation_kind == "pixels" else ()
    if observation_kind == "vector":
        observation_sample = np.asarray(observation_sample, dtype=np.float32)
        if observation_sample.ndim != 1 or not observation_sample.size or not np.isfinite(observation_sample).all():
            raise ValueError("vector observation_sample must be one finite vector")
    else:
        if not pixel_keys or not isinstance(observation_sample, Mapping):
            raise ValueError("pixels mode requires a mapping and at least one pixel key")
        if len(set(pixel_keys)) != len(pixel_keys) or any(not isinstance(key, str) or not key or key == "state" for key in pixel_keys):
            raise ValueError("Pixel keys must be distinct nonempty image names")
        if set(observation_sample) != set(pixel_keys) | ({"state"} if "state" in observation_sample else set()):
            raise ValueError("Pixel observations contain missing or undeclared fields")
        for key in pixel_keys:
            image = np.asarray(observation_sample[key])
            if image.ndim not in {3, 4} or image.dtype != np.uint8 or any(x <= 0 for x in image.shape):
                raise ValueError("pixel samples must be uint8 HWC or HWC-stack")
        if "state" in observation_sample:
            state = np.asarray(observation_sample["state"])
            if state.ndim != 1 or not state.size or not np.isfinite(state).all():
                raise ValueError("pixel proprio state must be one finite vector")

    rng = jax.random.PRNGKey(seed)
    rng, _actor_key, critic_key, value_key = jax.random.split(rng, 4)
    observations = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), observation_sample)
    observations = _observations(observation_kind, observations, pixel_keys)
    actions = jnp.expand_dims(action_sample, 0)

    q_base = partial(_NETWORKS.MLP, hidden_dims=tuple(hidden_dims), activate_final=True)
    q_network = partial(_NETWORKS.StateActionValue, base_cls=q_base)
    q_ensemble = partial(_NETWORKS.Ensemble, net_cls=q_network, num=2)
    v_base = partial(
        _NETWORKS.MLP, hidden_dims=tuple(hidden_dims), use_layer_norm=True, activate_final=True
    )
    v_network = partial(_NETWORKS.StateValue, base_cls=v_base)

    if observation_kind == "vector":
        critic_def = q_ensemble()
        value_def = v_network()
    else:
        encoder = partial(
            load_pixel_encoder(), features=tuple(encoder_features),
            filters=tuple(encoder_filters), strides=tuple(encoder_strides), padding="VALID",
        )
        critic_def = _NETWORKS.PixelMultiplexer(
            encoder_cls=encoder, network_cls=q_ensemble, latent_dim=latent_dim, pixel_keys=pixel_keys
        )
        value_def = _NETWORKS.PixelMultiplexer(
            encoder_cls=encoder, network_cls=v_network, latent_dim=latent_dim, pixel_keys=pixel_keys
        )

    critic_params = critic_def.init(critic_key, observations, actions)["params"]
    value_params = value_def.init(value_key, observations)["params"]
    critic = SynchronizedTrainState.create(
        apply_fn=critic_def.apply, params=critic_params, tx=optax.adam(critic_lr), axis_name=axis_name
    )
    value = SynchronizedTrainState.create(
        apply_fn=value_def.apply, params=value_params, tx=optax.adam(value_lr), axis_name=axis_name
    )
    target = TrainState.create(
        apply_fn=critic_def.apply, params=critic_params,
        tx=optax.GradientTransformation(lambda _: None, lambda _: None),
    )
    return CriticAgent(
        critic=critic, target_critic=target, value=value, rng=rng,
        discount=discount, tau=tau, critic_hyperparam=expectile,
        step=jnp.asarray(0, dtype=jnp.int32), axis_name=axis_name,
        observation_kind=observation_kind, pixel_keys=pixel_keys, action_dim=action_sample.shape[0],
        augmentation_pad=augmentation_pad,
    )


def _augment_training_batch(agent: CriticAgent, batch):
    """Return a fresh batch with author DrQ crops and its advanced agent RNG.

    Current/next images and every camera each receive a separate key. Under
    pmap, fold the device index into the saved RNG before those splits so equal
    observations on different devices do not receive identical augmentations.
    Each enabled update saves the final key; checkpoint resume reproduces the
    local stream exactly. Pad zero never splits or otherwise changes the RNG.
    """
    prepared = dict(batch)
    for field in ("observations", "next_observations"):
        prepared[field] = _observations(agent, batch[field])
    if not agent.augmentation_pad:
        return agent, prepared
    rng = agent.rng
    if agent.axis_name is not None:
        rng = jax.random.fold_in(rng, jax.lax.axis_index(agent.axis_name))
    for field in ("observations", "next_observations"):
        observations = prepared[field]
        for pixel_key in agent.pixel_keys:
            key, rng = jax.random.split(rng)
            observations = _AUGMENTATIONS.batched_random_crop(
                key, observations, pixel_key, padding=agent.augmentation_pad
            )
        prepared[field] = observations
    return agent.replace(rng=rng), prepared


@jax.jit
def update(agent: CriticAgent, batch):
    """Run original V, then original Q/target update; synchronize before Adam."""
    agent, prepared = _augment_training_batch(agent, batch)
    agent, v_info = agent.update_v(prepared)
    agent, q_info = agent.update_q(prepared)
    agent = agent.replace(step=agent.step + 1)
    info = {**v_info, **q_info}
    if agent.axis_name is not None:
        info = jax.lax.pmean(info, axis_name=agent.axis_name)
    leaves = jax.tree_util.tree_leaves((
        agent.critic.params, agent.value.params, agent.target_critic.params,
        agent.critic.opt_state, agent.value.opt_state, info,
    ))
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]))
    if agent.axis_name is not None:
        finite = jax.lax.pmin(finite.astype(jnp.int32), axis_name=agent.axis_name).astype(jnp.bool_)
    info = {**info, "finite": finite, "updates": agent.step}
    return agent, info


@jax.jit
def evaluate_outputs(agent: CriticAgent, batch):
    """Per-sample online Q, target Q and V on original, unaugmented inputs.

    This read-only diagnostic never consumes the training RNG, even when the
    agent was created with random crops enabled. All arrays have shape [B].
    """
    observations = _observations(agent, batch["observations"])
    actions = batch["actions"]
    online = agent.critic.apply_fn({"params": agent.critic.params}, observations, actions)
    targets = agent.target_critic.apply_fn(
        {"params": agent.target_critic.params}, observations, actions
    )
    values = agent.value.apply_fn({"params": agent.value.params}, observations)
    return {
        "q1": online[0], "q2": online[1], "q_min": online.min(axis=0),
        "target_q1": targets[0], "target_q2": targets[1],
        "target_q_min": targets.min(axis=0), "v": values,
    }


@jax.jit
def evaluate_losses(agent: CriticAgent, batch):
    """Read-only current-state losses, without a hypothetical validation update.

    The TD target uses the current, unchanged V.  This differs deliberately
    from calling upstream ``critic_update`` and discarding its returned state,
    which would first take a temporary V optimizer step on held-out samples.
    No RNG, parameter, target or Adam state is changed; per-device held-out
    means must be weighted by their actual sample counts by the caller.
    """
    observations = _observations(agent, batch["observations"])
    next_observations = _observations(agent, batch["next_observations"])
    actions = batch["actions"]
    target_qs = agent.target_critic.apply_fn(
        {"params": agent.target_critic.params}, observations, actions
    )
    values = agent.value.apply_fn({"params": agent.value.params}, observations)
    next_values = agent.value.apply_fn({"params": agent.value.params}, next_observations)
    td_target = batch["rewards"] + agent.discount * batch["masks"] * next_values
    qs = agent.critic.apply_fn({"params": agent.critic.params}, observations, actions)
    head_mse = jnp.mean((qs - td_target) ** 2, axis=-1)
    return {
        "critic_loss": head_mse.mean(),
        "value_loss": _OFFICIAL.expectile_loss(target_qs.min(axis=0) - values, agent.critic_hyperparam).mean(),
        "q1_mse": head_mse[0],
        "q2_mse": head_mse[1],
    }


@jax.jit
def score_actions(agent: CriticAgent, observations, actions):
    """Score externally generated candidates using min of target Q heads.

    Accept a single observation with [N,A] candidates, or batched observations
    with [B,N,A] candidates. Raw candidates must already use the explicit action
    adapter contract. This function does not sample or alter the frozen actor.
    """
    actions = jnp.asarray(actions, dtype=jnp.float32)
    if actions.ndim not in (2, 3) or actions.shape[-1] != agent.action_dim:
        raise ValueError("Candidates must have shape [N,A] or [B,N,A]")
    single = actions.ndim == 2
    if single:
        actions = actions[None]
        observations = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), observations)
    batch_size, count, width = actions.shape
    if count == 0:
        raise ValueError("At least one candidate is required")
    observations = _observations(agent, observations)
    if any(x.shape[0] != batch_size for x in jax.tree_util.tree_leaves(observations)):
        raise ValueError("Observation and candidate batch sizes differ")
    observations = jax.tree_util.tree_map(lambda x: jnp.repeat(x, count, axis=0), observations)
    qs = agent.target_critic.apply_fn(
        {"params": agent.target_critic.params}, observations, actions.reshape(batch_size * count, width)
    ).reshape(2, batch_size, count)
    result = {"q1": qs[0], "q2": qs[1], "q_min": qs.min(axis=0)}
    return jax.tree_util.tree_map(lambda x: x[0], result) if single else result


@jax.jit
def greedy_actions(agent: CriticAgent, observations, actions):
    scores = score_actions(agent, observations, actions)
    indices = jnp.argmax(scores["q_min"], axis=-1)
    actions = jnp.asarray(actions)
    if actions.ndim == 2:
        selected = actions[indices]
    else:
        selected = actions[jnp.arange(actions.shape[0]), indices]
    return selected, indices, scores


create = create_agent
update_critic = update
