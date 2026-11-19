"""Aggregate complete BC-normalized action chunks and decode only after selection."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


WEIGHTED = 'weighted'
RANDOM_MEAN = 'random_mean'
MEAN_MODES = (WEIGHTED, RANDOM_MEAN)
Q_MODES = ('greedy', WEIGHTED)


def validate_options(pick, count, elites, temperature, steps):
    if type(count) is not int or count < 1 or type(steps) is not int or steps < 1:
        raise ValueError('Candidate count and flow steps must be positive integers')
    if pick in MEAN_MODES:
        if type(elites) is not int or not 1 <= elites <= count:
            raise ValueError('Elite count must be between one and candidate count')
        if pick == WEIGHTED or temperature is not None:
            if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
                raise ValueError('A finite positive softmax temperature is required')
    elif pick not in ('greedy', 'random') or elites is not None or temperature is not None:
        raise ValueError('Invalid selector or aggregation options')


def digest(array):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@dataclass
class PlanningPool:
    normalized: np.ndarray
    raw: np.ndarray
    noise: np.ndarray
    state: np.ndarray


def elite_weights(count, scores, elites, temperature, rng):
    """Float32 top-K softmax or a uniform sample without replacement."""
    import torch
    if scores is None:
        indices = torch.from_numpy(rng.choice(count, size=elites, replace=False))
        weights = torch.full((elites,), 1.0 / elites, dtype=torch.float32)
    else:
        values = torch.as_tensor(scores, dtype=torch.float32)
        if values.shape != (count,) or not torch.isfinite(values).all():
            raise ValueError('Q returned invalid candidate scores')
        top, indices = values.topk(elites)
        weights = ((top - top.max()) / temperature).softmax(dim=0)
    return indices, weights


def aggregate(normalized, scores, elites, temperature, rng):
    import torch
    candidates = torch.from_numpy(normalized)
    indices, weights = elite_weights(len(candidates), scores, elites, temperature, rng)
    chunk = (candidates[indices] * weights[:, None, None]).sum(dim=0)
    return chunk.numpy(), indices.tolist(), weights.tolist()


def sample_prefix_once(model, observation, noise, steps):
    """OpenPI π₀.₅ inference with the identical single-draw denoising kernels for every candidate.

    Image/text tokens depend only on the observation. The expert reads their KV cache with
    use_cache=False, so all candidate trajectories can share it without updating the prefix.
    """
    import torch
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    images, image_masks, tokens, token_masks, state = model._preprocess_observation(observation, train=False)
    embeddings, padding, attention = model.embed_prefix(images, image_masks, tokens, token_masks)
    mask = model._prepare_attention_masks_4d(make_att_2d_masks(padding, attention))
    positions = torch.cumsum(padding, dim=1) - 1
    language = model.paligemma_with_expert
    language.paligemma.language_model.config._attn_implementation = 'eager'
    _, cache = language.forward(attention_mask=mask, position_ids=positions, past_key_values=None,
                                inputs_embeds=[embeddings, None], use_cache=True)
    dt = torch.tensor(-1.0 / steps, device=noise.device, dtype=torch.float32)
    time = torch.tensor(1.0, device=noise.device, dtype=torch.float32)
    times = []
    while time >= -dt / 2:
        times.append(time.expand(1))
        time = time + dt
    outputs = []
    for draw in noise.split(1):
        for t in times:
            draw = draw + dt * model.denoise_step(state, padding, cache, draw, t)
        outputs.append(draw)
    return torch.cat(outputs)
