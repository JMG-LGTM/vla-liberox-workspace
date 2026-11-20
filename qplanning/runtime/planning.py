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


def validate_score_tolerance(tolerance):
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('The offline score tolerance must be a finite nonnegative number')


class PlanningParityError(ValueError):
    def __init__(self, evidence):
        self.evidence = evidence
        location = {key: evidence[key] for key in ('fixture_row', 'seed', 'query', 'score_max_abs') if key in evidence}
        super().__init__(f"{evidence['stage']} failed {location}")


def compare_scores(online, offline, elites, tolerance, *, rank_singleton=True):
    """Compare same-state candidate scores; ranking diagnostics do not gate acceptance."""
    import numpy as np
    import torch

    validate_score_tolerance(tolerance)
    online = np.asarray(online, dtype=np.float64)
    offline = np.asarray(offline, dtype=np.float64)
    if online.ndim != 1 or online.size == 0 or online.shape != offline.shape:
        raise ValueError('Expected two aligned, nonempty score vectors')
    if elites is not None and (type(elites) is not int or elites < 1):
        raise ValueError('Elite count must be positive')

    def json_scores(values):
        return [float(x) if np.isfinite(x) else None for x in values]

    result = dict(tolerance=tolerance, online_scores=json_scores(online), offline_scores=json_scores(offline),
                  mismatched_score_indices=np.flatnonzero(online != offline).tolist(),
                  nonfinite_online_indices=np.flatnonzero(~np.isfinite(online)).tolist(),
                  nonfinite_offline_indices=np.flatnonzero(~np.isfinite(offline)).tolist(),
                  within_tolerance=False, score_max_abs=None, score_mean_abs=None,
                  online_top1=None, offline_top1=None, top1_same=None, elite_count=elites,
                  online_topk=None, offline_topk=None, topk_overlap=None, spearman=None)
    if result['nonfinite_online_indices'] or result['nonfinite_offline_indices']:
        return result
    with np.errstate(over='ignore'):
        difference = np.abs(online - offline)
    if not np.isfinite(difference).all():
        return result

    maximum = float(difference.max())
    top1 = int(online.argmax())
    other_top1 = int(offline.argmax())
    # Scaling before the mean avoids overflow for large finite differences.
    result.update(within_tolerance=maximum <= tolerance, score_max_abs=maximum,
                  score_mean_abs=float(maximum * np.mean(difference / maximum)) if maximum else 0.)
    if len(online) == 1 and not rank_singleton:
        return result
    result.update(online_top1=top1, offline_top1=other_top1, top1_same=top1 == other_top1)
    if elites is not None and elites <= len(online):
        # Match the planner's CPU top-K operator, including ties at its cutoff.
        top = torch.from_numpy(online).topk(elites).indices.tolist()
        other = torch.from_numpy(offline).topk(elites).indices.tolist()
        result.update(online_topk=top, offline_topk=other, topk_overlap=len(set(top).intersection(other)))

    def centered_midranks(scores):
        _, group, counts = np.unique(scores, return_inverse=True, return_counts=True)
        ranks = (counts.cumsum() - (counts - 1) * .5)[group]
        return ranks - ranks.mean()

    left, right = centered_midranks(online), centered_midranks(offline)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator:
        result['spearman'] = float(np.clip(np.dot(left, right) / denominator, -1., 1.))
    return result


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
