"""Finite, JSON-safe critic diagnostics with no policy or SciPy dependency."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
from typing import Any

import numpy as np
import torch


def _array(values: Any) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("Diagnostic inputs must be one-dimensional")
    return array


def average_ranks(values: Any) -> np.ndarray:
    """One-based average ranks, including ties."""
    values = _array(values)
    if not np.isfinite(values).all():
        raise ValueError("Cannot rank nonfinite values")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def spearman(x: Any, y: Any) -> float | None:
    """Return None for undefined correlation, including constant predictions."""
    x, y = _array(x), _array(y)
    if x.shape != y.shape:
        raise ValueError("Spearman inputs must have matching shapes")
    if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    x, y = average_ranks(x), average_ranks(y)
    x, y = x - x.mean(), y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.clip(np.dot(x, y) / denominator, -1, 1)) if denominator else None


def distribution(values: Any) -> dict[str, Any]:
    """Describe finite values while retaining an explicit nonfinite count."""
    values = _array(values)
    finite = values[np.isfinite(values)]
    result: dict[str, Any] = {"count": len(values), "finite_count": len(finite),
                              "nonfinite_count": len(values) - len(finite)}
    if len(finite):
        names = ("min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max")
        result.update(zip(names, map(float, np.quantile(finite, [0, .01, .05, .25, .5, .75, .95, .99, 1]))))
        result.update(mean=float(finite.mean()), std=float(finite.std()))
    return result


def weight_statistics(weights: Any, success_terminal: Any, *,
                      max_weight: float = 20.) -> dict[str, Any]:
    """Compute capped-weight fractions, ESS/N, and success-terminal mass.

    The terminal indicator is explicit: a truncated recording is not a
    successful final block. No normalization is applied to training weights.
    ESS uses max-scaled weights so large finite inputs cannot overflow.
    """
    if not math.isfinite(max_weight) or max_weight <= 0:
        raise ValueError("max_weight must be finite and positive")
    weights, terminal = _array(weights), _array(success_terminal)
    if weights.shape != terminal.shape:
        raise ValueError("weights and success_terminal must have matching shapes")
    if not np.isin(terminal, [0, 1]).all():
        raise ValueError("success_terminal must contain boolean indicators")
    count = len(weights)
    valid = bool(count) and bool(np.isfinite(weights).all() and (weights >= 0).all())
    result: dict[str, Any] = {"count": count, "finite": valid,
                              "success_terminal_count": int(terminal.sum()),
                              "weights": distribution(weights), "clipped_fraction": None,
                              "below_0_1_fraction": None, "below_0_01_fraction": None,
                              "ess": None, "ess_fraction": None,
                              "success_terminal_weight_share": None}
    if not valid:
        return result
    result.update(clipped_fraction=float(np.mean(weights >= max_weight)),
                  below_0_1_fraction=float(np.mean(weights < .1)),
                  below_0_01_fraction=float(np.mean(weights < .01)))
    maximum = float(weights.max())
    if maximum == 0:
        result.update(ess=0., ess_fraction=0.)
        return result
    scaled = weights / maximum
    total = float(scaled.sum())
    ess = total * total / float(np.dot(scaled, scaled))
    result.update(ess=ess, ess_fraction=ess / count,
                  success_terminal_weight_share=float(scaled[terminal.astype(bool)].sum() / total))
    return result


def critic_diagnostics(values: Any, remaining_blocks: Any, weights: Any,
                       success_terminal: Any, *, max_weight: float = 20.,
                       trajectory_ids: Sequence[Any] | None = None) -> dict[str, Any]:
    """Summarize each evaluated block once; failed paths use remaining=-1.

    Spearman is measured only on confirmed-success paths. More negative is
    better for V versus blocks remaining. This is a progress diagnostic, not
    evidence that Q ranks different actions in the same state correctly.
    """
    values, remaining = _array(values), _array(remaining_blocks)
    weights, terminal = _array(weights), _array(success_terminal)
    if not values.shape == remaining.shape == weights.shape == terminal.shape:
        raise ValueError("Critic diagnostic inputs must have matching shapes")
    if not np.isfinite(remaining).all():
        raise ValueError("remaining_blocks must be finite; use -1 for failed paths")
    success = remaining >= 0
    statistics = weight_statistics(weights, terminal, max_weight=max_weight)
    result: dict[str, Any] = {**statistics, "value_distribution": distribution(values),
                              "success_progress_count": int(success.sum()),
                              "spearman_v_remaining_blocks": spearman(values[success], remaining[success]),
                              "finite": bool(len(values)) and bool(np.isfinite(values).all()) and statistics["finite"],
                              "progress_definition": "confirmed-success paths only; final successful block has remaining_blocks=0",
                              "clipped_fraction_definition": "fraction of weights at the configured cap",
                              "selection_direction": "minimize_spearman_v_remaining_blocks"}
    if trajectory_ids is not None:
        if len(trajectory_ids) != len(values):
            raise ValueError("trajectory_ids must match the number of values")
        ids = np.asarray([str(value) for value in trajectory_ids])
        per_trajectory = []
        for trajectory in sorted(set(ids[success].tolist())):
            selected = success & (ids == trajectory)
            per_trajectory.append({"trajectory_id": trajectory, "count": int(selected.sum()),
                                   "spearman": spearman(values[selected], remaining[selected])})
        correlations = [item["spearman"] for item in per_trajectory]
        result["per_trajectory"] = per_trajectory
        result["median_per_trajectory_spearman"] = (float(np.median(correlations))
            if correlations and all(value is not None for value in correlations) else None)
    return result


def summarize_rows(rows: Sequence[Mapping[str, Any]], *,
                   max_weight: float = 20.) -> dict[str, Any]:
    """Convenience adapter for cached/scored rows and historical key aliases."""
    def field(row: Mapping[str, Any], primary: str, alternate: str) -> Any:
        return row[primary] if primary in row else row[alternate]
    result = critic_diagnostics([row["v"] for row in rows],
        [field(row, "remaining_blocks", "remaining_chunks") for row in rows],
        [field(row, "weights", "w") for row in rows],
        [field(row, "success_terminal", "terminal_success") for row in rows],
        max_weight=max_weight,
        trajectory_ids=([row["run_id"] for row in rows] if all("run_id" in row for row in rows) else None))
    for key in ("q1", "q2", "q", "advantage"):
        if rows and all(key in row for row in rows):
            result[f"{key}_distribution"] = distribution([row[key] for row in rows])
    return result


@torch.no_grad()
def evaluate_critic(model: Any, batches: Iterable[Mapping[str, Any]],
                    device: str | torch.device = "cpu", *, beta: float = 3.,
                    max_weight: float = 20.) -> dict[str, Any]:
    """Evaluate unaugmented diagnostics and per-transition critic losses.

    Pass the unwrapped critic on rank zero and a complete evaluation loader.
    The runner labels held-out validation versus training diagnostics and only
    permits held-out validation to select a best checkpoint.
    Q losses use the current online V Bellman target; the expectile V loss
    uses the minimum of the two target Q networks, as in training. These are
    self-consistency losses, not candidate-action ranking accuracy. Model modes
    and global/loader RNG state are restored, including when evaluation fails.
    This function performs no distributed collectives; the runner coordinates
    ranks around validation and checkpoint selection.
    """
    from .checkpoint import rng_state, restore_rng

    if hasattr(model, "module"):
        model = model.module
    device = torch.device(device)
    modes = {module: module.training for module in model.modules()}
    previous_rng = rng_state(device)
    generator = getattr(batches, "generator", None)
    loader_rng = generator.get_state() if generator is not None else None
    collected: dict[str, list[torch.Tensor]] = {
        key: [] for key in ("q1", "q2", "q", "v", "advantage", "weights",
                           "remaining_blocks", "success_terminal")}
    trajectories: list[Any] = []
    all_have_trajectories = True
    loss_sums = {key: 0. for key in ("q1_mse", "q2_mse", "q_loss", "v_expectile_loss",
                                    "q_abs_disagreement", "target_mean")}
    loss_count = 0
    model.eval()
    try:
        for original in batches:
            batch = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                     for key, value in original.items()}
            scores = model.scores(batch, beta=beta, max_weight=max_weight)
            for key, value in scores.items():
                collected[key].append(value.detach().cpu())
            # Use the actual staged training equations at one fixed snapshot.
            # No optimizer or EMA update is performed by this forward call.
            losses = model(batch, stage="both", beta=beta, max_weight=max_weight)
            target = losses["target"]
            if not isinstance(target, torch.Tensor) or target.ndim != 1:
                raise ValueError("Critic loss target must be a per-transition vector")
            count = len(target)
            for key in ("q1", "q2", "v_loss"):
                value = losses[key]
                if not isinstance(value, torch.Tensor) or value.shape != target.shape:
                    raise ValueError("Critic loss outputs must match the per-transition target")
            q1_mse = (losses["q1"] - target).square()
            q2_mse = (losses["q2"] - target).square()
            per_transition = {"q1_mse": q1_mse, "q2_mse": q2_mse,
                "q_loss": .5 * (q1_mse + q2_mse), "v_expectile_loss": losses["v_loss"],
                "q_abs_disagreement": (losses["q1"] - losses["q2"]).abs(),
                "target_mean": target}
            if len(scores["v"]) != count:
                raise ValueError("Critic diagnostics and loss transition counts differ")
            for key, value in per_transition.items():
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError("Nonfinite held-out critic statistic: " + key)
                total = loss_sums[key] + float(value.detach().double().sum().cpu())
                if not math.isfinite(total):
                    raise FloatingPointError("Nonfinite held-out critic aggregate: " + key)
                loss_sums[key] = total
            loss_count += count
            for primary, alternate in (("remaining_blocks", "remaining_chunks"),
                                       ("success_terminal", "terminal_success")):
                if primary not in original and alternate not in original:
                    raise ValueError(f"Validation batch is missing {primary}")
                source = original[primary] if primary in original else original[alternate]
                collected[primary].append(torch.as_tensor(source).detach().cpu())
            if "run_id" in original:
                trajectories.extend(original["run_id"])
            else:
                all_have_trajectories = False
    finally:
        for module, training in modes.items():
            module.training = training
        restore_rng(previous_rng, device)
        if loader_rng is not None:
            generator.set_state(loader_rng)
    merged = {key: torch.cat(parts).numpy() if parts else np.empty(0)
              for key, parts in collected.items()}
    result = critic_diagnostics(merged["v"], merged["remaining_blocks"], merged["weights"],
                                merged["success_terminal"], max_weight=max_weight,
                                trajectory_ids=trajectories if all_have_trajectories else None)
    for key in ("q1", "q2", "q", "advantage"):
        result[f"{key}_distribution"] = distribution(merged[key])
    result["beta"] = float(beta)
    result["max_weight"] = float(max_weight)
    result["losses"] = {"count": loss_count,
        **{key: total / loss_count if loss_count else None for key, total in loss_sums.items()},
        "aggregation": "mean_per_transition_without_augmentation",
        "q_target_definition": "reward + discount * bootstrap_mask * online_V(next_observation)",
        "v_target_definition": "min(target_Q1, target_Q2)(observation, action)"}
    return result
