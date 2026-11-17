from copy import deepcopy

import pytest

from post_training.official_idql.selection import initial_selection, consider_checkpoint

THRESHOLDS = dict(rho_max=-.5, trajectory_median_max=-.3, human_higher_fraction_min=.5,
                  same_state_higher_fraction_min=.5, ess_fraction_min=.1,
                  success_terminal_weight_share_max=.5)


def evidence(rho=-.8, median=-.7, *, human=True, same_state=True, loss=1.):
    return dict(held_out=True, updated=False, critic_loss=loss, value_loss=.4,
        diagnostics=dict(finite=True,spearman_v_remaining_blocks=rho, median_per_trajectory_spearman=median,
            ess_fraction=.8, success_terminal_weight_share=.1,
            human_comparison=dict(human_higher_fraction=.75 if human else .45),
            same_state_action_ranking=dict(human_higher_fraction=.8 if same_state else None),
            gates=dict(progress_passed=True, human_passed=human, same_state_passed=same_state,
                ess_passed=True, terminal_mass_passed=True,
                eligible_for_idql_evaluation=human and same_state,
                failures=[] if human and same_state else ['human_or_same_state_unavailable'])))


def add(state, row, step):
    row = {**row, 'optimizer_update':row.get('optimizer_update',step)}
    return consider_checkpoint(state, row, update=step, checkpoint=f'checkpoints/u{step}',
        scorer=f'scorers/u{step}', diagnostics_path=f'validation/u{step}.json', thresholds=THRESHOLDS)


def test_lower_td_loss_or_later_step_cannot_override_failed_action_ranking():
    state = initial_selection(THRESHOLDS)
    state = add(state, evidence(rho=-.85, human=False, loss=.001), 2500)
    assert state['best_qualified_scorer'] is None
    assert state['best_progress_candidate']['optimizer_update'] == 2500
    state = add(state, evidence(rho=-.7, loss=1.), 5000)
    state = add(state, evidence(rho=-.95, same_state=False, loss=.00001), 7500)
    assert state['best_qualified_scorer']['optimizer_update'] == 5000
    assert state['best_progress_candidate']['optimizer_update'] == 7500
    assert state['policy']['fallback_to_last'] is False


def test_deterministic_rank_uses_progress_then_path_median_then_loss_then_earliest():
    state = initial_selection(THRESHOLDS)
    for step, row in [(2500, evidence(-.7, -.6, loss=.1)), (5000, evidence(-.8, -.6, loss=.2)),
                      (7500, evidence(-.8, -.7, loss=.3)), (10000, evidence(-.8, -.7, loss=.2)),
                      (12500, evidence(-.8, -.7, loss=.2))]:
        state = add(state, row, step)
    assert state['best_qualified_scorer']['optimizer_update'] == 10000
    assert state['status'] == 'QUALIFIED_SCORER_SELECTED'


def test_missing_diagnostics_and_inconsistent_gate_flags_do_not_pass():
    state = initial_selection(THRESHOLDS)
    state = add(state, dict(held_out=True, updated=False, critic_loss=.01, value_loss=.01), 2500)
    wrong = evidence(human=False)
    wrong['diagnostics']['gates']['eligible_for_idql_evaluation'] = True
    state = add(state, wrong, 5000)
    assert state['best_qualified_scorer'] is None


@pytest.mark.parametrize('key,value', [('spearman_v_remaining_blocks',.9),
    ('median_per_trajectory_spearman',.8), ('ess_fraction',None), ('ess_fraction',.01),
    ('success_terminal_weight_share',.9), ('human_higher_fraction',None),
    ('same_state_higher_fraction',None), ('same_state_higher_fraction',.5)])
def test_raw_evidence_must_pass_even_if_boolean_gate_labels_claim_success(key,value):
    row = evidence()
    if key == 'human_higher_fraction':row['diagnostics']['human_comparison'][key] = value
    elif key == 'same_state_higher_fraction':row['diagnostics']['same_state_action_ranking']['human_higher_fraction'] = value
    else:row['diagnostics'][key] = value
    state = add(initial_selection(THRESHOLDS), row, 2500)
    assert state['best_qualified_scorer'] is None


def test_resume_preserves_prior_selection_and_rejects_redefinition_or_duplicate():
    state = add(initial_selection(THRESHOLDS), evidence(), 2500)
    saved = deepcopy(state)
    continued = add(deepcopy(saved), evidence(-.9), 5000)
    assert saved == state and saved['best_qualified_scorer']['optimizer_update'] == 2500
    assert continued['best_qualified_scorer']['optimizer_update'] == 5000
    with pytest.raises(ValueError, match='already considered'):
        add(continued, evidence(-.99), 5000)
    other = deepcopy(THRESHOLDS); other['rho_max'] = -.1
    with pytest.raises(ValueError, match='thresholds changed'):
        consider_checkpoint(state, evidence(), update=7500, checkpoint='c', scorer='s',
                            diagnostics_path='v', thresholds=other)
    with pytest.raises(ValueError, match='held-out'):
        add(state, {**evidence(), 'held_out': False}, 7500)
    with pytest.raises(ValueError, match='update differ'):
        add(state, {**evidence(), 'optimizer_update':7500}, 10000)
