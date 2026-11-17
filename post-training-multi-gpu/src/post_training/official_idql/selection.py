"""Deterministic held-out selection among retained full checkpoints only."""
from __future__ import annotations

import copy
import math


def initial_selection(thresholds):
    return dict(schema_version=1, status='NO_QUALIFIED_SCORER',
        policy=dict(thresholds=copy.deepcopy(thresholds),
            population='retained_full_checkpoints_with_same_update_heldout_diagnostics',
            order=['lowest_global_spearman', 'lowest_path_median_spearman',
                   'lowest_heldout_td_loss', 'earliest_update'],
            fallback_to_last=False), best_qualified_scorer=None,
        best_progress_candidate=None, retained_candidates=[])


def _rank(candidate):
    metrics = candidate['metrics']
    return (metrics['spearman_v_remaining_blocks'], metrics['median_per_trajectory_spearman'],
            metrics['critic_loss'], candidate['optimizer_update'])


def consider_checkpoint(selection, validation, *, update, checkpoint, scorer,
                        diagnostics_path, thresholds):
    """Record the boundary's evidence; missing/failed gates never select best.

    A progress-only candidate is kept for analysis with its explicit failed
    gates. It is not an evaluation-ready scorer and is never used as fallback.
    The caller publishes this decision only after the named artifacts exist.
    """
    result = copy.deepcopy(selection)
    if result['policy']['thresholds'] != thresholds:
        raise ValueError('Selection thresholds changed during the run')
    if type(update) is not int or update < 1:
        raise ValueError('Checkpoint selection requires a positive real update')
    if any(row['optimizer_update'] == update for row in result['retained_candidates']):
        raise ValueError('Checkpoint boundary already considered')
    if validation.get('held_out') is not True or validation.get('updated') is not False:
        raise ValueError('Checkpoint selection requires read-only held-out evidence')
    if type(validation.get('optimizer_update')) is not int or validation['optimizer_update'] != update:
        raise ValueError('Held-out diagnostics and checkpoint update differ')
    report = validation.get('diagnostics') or {}
    gates = copy.deepcopy(report.get('gates') or {})
    metrics = {key: report.get(key) for key in (
        'spearman_v_remaining_blocks', 'median_per_trajectory_spearman',
        'ess_fraction', 'success_terminal_weight_share')}
    metrics.update(critic_loss=validation.get('critic_loss'), value_loss=validation.get('value_loss'),
        human_higher_fraction=(report.get('human_comparison') or {}).get('human_higher_fraction'),
        same_state_higher_fraction=(report.get('same_state_action_ranking') or {}).get('human_higher_fraction'))
    def valid(key, low, high):
        value = metrics[key]
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and low <= value <= high
    finite_rank = (valid('spearman_v_remaining_blocks', -1, 1)
        and valid('median_per_trajectory_spearman', -1, 1) and valid('critic_loss', 0, float('inf')))
    progress = (finite_rank and report.get('finite') is True and gates.get('progress_passed') is True
        and metrics['spearman_v_remaining_blocks'] <= thresholds['rho_max']
        and metrics['median_per_trajectory_spearman'] <= thresholds['trajectory_median_max'])
    metric_gates = (progress
        and valid('ess_fraction', 0, 1) and metrics['ess_fraction'] >= thresholds['ess_fraction_min']
        and valid('success_terminal_weight_share', 0, 1)
        and metrics['success_terminal_weight_share'] <= thresholds['success_terminal_weight_share_max']
        and valid('human_higher_fraction', 0, 1) and metrics['human_higher_fraction'] > thresholds['human_higher_fraction_min']
        and valid('same_state_higher_fraction', 0, 1)
        and metrics['same_state_higher_fraction'] > thresholds['same_state_higher_fraction_min'])
    required = ('progress_passed', 'human_passed', 'same_state_passed', 'ess_passed', 'terminal_mass_passed')
    qualified = (metric_gates and gates.get('eligible_for_idql_evaluation') is True
                 and all(gates.get(key) is True for key in required)
                 and not gates.get('failures') and not gates.get('unavailable'))
    candidate = dict(optimizer_update=update, checkpoint=str(checkpoint), scorer=str(scorer),
        diagnostics_path=str(diagnostics_path), metrics=metrics, gates=gates,
        status='QUALIFIED' if qualified else 'PROGRESS_ONLY' if progress else 'REJECTED')
    result['retained_candidates'].append(candidate)
    for key, eligible in (('best_qualified_scorer', qualified), ('best_progress_candidate', progress)):
        previous = result[key]
        if eligible and (previous is None or _rank(candidate) < _rank(previous)):
            result[key] = copy.deepcopy(candidate)
    result['status'] = 'QUALIFIED_SCORER_SELECTED' if result['best_qualified_scorer'] else 'NO_QUALIFIED_SCORER'
    return result
