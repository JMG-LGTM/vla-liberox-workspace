#!/usr/bin/env python3
"""Shared, UI-agnostic control loop for LIBERO evaluation and intervention."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Sequence

import numpy as np

from trajectory_utils import TrajectoryRecorder


PolicyQuery = Callable[[dict[str, Any], int], tuple[dict[str, Any], Sequence[np.ndarray]]]
ManualQuery = Callable[[int], np.ndarray | None]
ActionTransform = Callable[[np.ndarray], np.ndarray]
TransitionCallback = Callable[[dict[str, Any], int, bool], None]
StopPredicate = Callable[[], bool]


def success_confirm_steps() -> int:
    """Consecutive ``done`` values that confirm success (system-wide setting).

    Defined once in the training package's ``configs/runtime.yaml``. ``success``
    on a run record means "done at least once" and is kept as a diagnostic.
    """
    from vla_rynn_iql.config import success_consecutive_steps
    return success_consecutive_steps()


def confirmed_success_step(dones: Sequence[Any], required: int | None = None) -> int | None:
    """Return the action index that completes the first confirming done streak."""
    required = success_confirm_steps() if required is None else required
    if required < 1:
        raise ValueError("required must be positive")
    streak = 0
    for index, value in enumerate(np.asarray(dones, dtype=bool).reshape(-1)):
        streak = streak + 1 if value else 0
        if streak >= required:
            return index
    return None


@dataclass(frozen=True)
class ControlLoopResult:
    success: bool
    executed_steps: int
    policy_queries: int
    stopped_reason: str
    control_step_times: list[float]
    final_observation: dict[str, Any]


def _validated_action(value: Any, name: str) -> np.ndarray:
    action = np.asarray(value, dtype=np.float32)
    if action.shape != (7,):
        raise ValueError(f"{name} has shape {action.shape}; expected (7,)")
    if not np.isfinite(action).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return action


def run_control_loop(
    *,
    env: Any,
    recorder: TrajectoryRecorder,
    initial_observation: dict[str, Any],
    target_action_count: int,
    rate_limiter: Any,
    action_source: str,
    open_loop_steps: int = 1,
    policy_query: PolicyQuery | None = None,
    policy_action_transform: ActionTransform | None = None,
    manual_query: ManualQuery | None = None,
    stop_requested: StopPredicate | None = None,
    on_transition: TransitionCallback | None = None,
    stop_on_success: bool = False,
    stop_after_success: bool = False,
    horizon_reason: str = "horizon",
) -> ControlLoopResult:
    """Advance an environment while preserving recorder and timing semantics.

    Exactly one action source must be provided. A stop request is observed at
    safe control boundaries and again after a synchronous policy call returns,
    so CUDA execution is never interrupted.

    ``stop_after_success`` permits two seconds of additional simulated control
    after the first native success in this invocation. Historical prefix success
    still contributes to the result, but does not start a resumed branch's timer.
    An immediate ``stop_on_success`` retains its existing precedence. Horizon
    and user stop remain the upper bounds on the additional recording.
    """
    if target_action_count < recorder.action_count:
        raise ValueError("target_action_count precedes the recorded trajectory prefix")
    if type(open_loop_steps) is not int or open_loop_steps < 1:
        raise ValueError("open_loop_steps must be a positive integer")
    if (policy_query is None) == (manual_query is None):
        raise ValueError("Provide exactly one of policy_query or manual_query")
    if policy_query is not None and policy_action_transform is None:
        raise ValueError("Policy control requires policy_action_transform")
    success_delay_steps = 0
    if stop_after_success:
        control_hz = float(recorder.control_hz)
        if not math.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("stop_after_success requires finite positive control_hz")
        success_delay_steps = math.ceil(2.0 * control_hz)

    observation = initial_observation
    queue: deque[np.ndarray] = deque()
    policy_queries = 0
    control_step_times: list[float] = []
    success = any(recorder.dones)
    stopped_reason = horizon_reason
    start_action_count = recorder.action_count
    success_stop_at: int | None = None

    while recorder.action_count < target_action_count:
        if stop_requested is not None and stop_requested():
            stopped_reason = "user_stop"
            break
        if success_stop_at is not None and recorder.action_count >= success_stop_at:
            stopped_reason = "success_after_delay"
            break

        if policy_query is not None:
            if not queue:
                observation, chunk_values = policy_query(observation, recorder.action_count)
                chunk = [
                    _validated_action(value, f"policy action {index}")
                    for index, value in enumerate(chunk_values)
                ]
                if not chunk:
                    raise RuntimeError("Policy returned an empty action chunk")
                if len(chunk) < open_loop_steps:
                    raise ValueError(f"Policy returned {len(chunk)} actions; execution requires {open_loop_steps}")
                recorder.record_inference(recorder.action_count, np.stack(chunk, axis=0))
                queue.extend(chunk[:open_loop_steps])
                policy_queries += 1
            if stop_requested is not None and stop_requested():
                stopped_reason = "user_stop"
                break
            raw_action = queue.popleft()
            assert policy_action_transform is not None
            env_action = _validated_action(
                policy_action_transform(raw_action),
                "environment action",
            )
        else:
            assert manual_query is not None
            # Human input is sampled at the control boundary. Sampling before
            # the rate limiter can add almost one full control period of avoidable
            # latency when the controller exposes a continuously updated state.
            rate_limiter.wait_before_step()
            if stop_requested is not None and stop_requested():
                stopped_reason = "user_stop"
                break
            value = manual_query(recorder.action_count)
            if value is None:
                stopped_reason = "controller_stop"
                break
            raw_action = _validated_action(value, "manual action")
            env_action = raw_action.copy()
            control_step_times.append(time.monotonic())

        if policy_query is not None:
            control_step_times.append(rate_limiter.wait_before_step())
        observation, reward, done, _ = env.step(env_action.tolist())
        success = success or bool(done)
        recorder.record_transition(
            env,
            observation,
            raw_action,
            env_action,
            reward,
            done,
            action_source=action_source,
        )
        if done and stop_after_success and success_stop_at is None:
            success_stop_at = recorder.action_count + success_delay_steps
        if on_transition is not None:
            on_transition(observation, recorder.action_count, success)
        if done and stop_on_success:
            stopped_reason = "success"
            break

    return ControlLoopResult(
        success=success,
        executed_steps=recorder.action_count - start_action_count,
        policy_queries=policy_queries,
        stopped_reason=stopped_reason,
        control_step_times=control_step_times,
        final_observation=observation,
    )
