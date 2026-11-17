"""Read-only held-out critic diagnostics and pre-registered qualification gates.

The capped exponential weights below are hypothetical actor diagnostics. They
are not the official critic objective and are never applied to the frozen actor.
Same-root segment comparisons are explicitly different-state heuristics; genuine
same-state ranking is a separate, independently verified measurement.
"""
from __future__ import annotations

import math
import numpy as np

from ..diagnostics import distribution, spearman, weight_statistics

DEFAULT_THRESHOLDS = dict(rho_max=-.5, trajectory_median_max=-.3,
    human_higher_fraction_min=.5, same_state_higher_fraction_min=.5,
    ess_fraction_min=.1, success_terminal_weight_share_max=.5)


def _require(condition, message):
    if not condition:
        raise ValueError('Official IDQL diagnostics: ' + message)


def unavailable_same_state(reason='no independently verified same-state action package configured'):
    return dict(available=False, status='UNAVAILABLE', reason=reason, pair_count=0,
                root_count=0, human_higher_fraction=None, evidence_verified=False, held_out=False)


def classify_gates(report, thresholds=None):
    """Missing/undefined/partial evidence fails closed; no success-rate claim."""
    threshold = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    _require(set(threshold) == set(DEFAULT_THRESHOLDS), 'unknown threshold')
    _require(all(type(x) in (int,float) and math.isfinite(x) for x in threshold.values()), 'nonfinite threshold')
    for key,value in threshold.items():
        _require(-1 <= value < 0 if key in {'rho_max','trajectory_median_max'} else 0 <= value <= 1,
                 'threshold outside metric range: ' + key)
    failures, unavailable = [], []
    coverage = report['coverage']['complete_heldout'] and report.get('finite') is True
    if report.get('finite') is not True:
        unavailable.append('finite_outputs_and_weights')
    if not coverage:
        unavailable.append('complete_heldout_coverage')
    def check(name, value, predicate, available=True):
        if (not available or not isinstance(value,(int,float,np.integer,np.floating))
                or isinstance(value,(bool,np.bool_)) or not math.isfinite(value)):
            unavailable.append(name)
            return False
        passed = bool(predicate(value))
        if not passed:
            failures.append(name)
        return passed and coverage
    progress_available = report.get('progress_scope') == 'complete_successful_paths' and report.get('complete_success_path_coverage', False)
    rho = check('spearman_v_remaining_blocks', report.get('spearman_v_remaining_blocks'),
                lambda x:x <= threshold['rho_max'], progress_available)
    perpath = check('median_per_trajectory_spearman', report.get('median_per_trajectory_spearman'),
                   lambda x:x <= threshold['trajectory_median_max'], progress_available)
    human_report = report['human_comparison']
    human = check('human_higher_fraction', human_report.get('human_higher_fraction'),
        lambda x:0 <= x <= 1 and x > threshold['human_higher_fraction_min'],
        human_report.get('matched_root_count',0)>0 and not human_report.get('unmatched_human_roots'))
    pair = report['same_state_action_ranking']
    same = check('true_same_state_human_higher_fraction', pair.get('human_higher_fraction'),
        lambda x:0 <= x <= 1 and x > threshold['same_state_higher_fraction_min'], pair.get('available') is True
        and pair.get('evidence_verified') is True and pair.get('held_out') is True
        and pair.get('pair_count',0)>0)
    ess = check('ess_fraction', report.get('ess_fraction'), lambda x:0 <= x <= 1 and x >= threshold['ess_fraction_min'])
    terminal = check('success_terminal_weight_share', report.get('success_terminal_weight_share'),
        lambda x:0 <= x <= 1 and x <= threshold['success_terminal_weight_share_max'], report.get('success_terminal_count',0)>0)
    return dict(thresholds=threshold, progress_passed=rho and perpath, human_passed=human,
        same_state_passed=same, ess_passed=ess, terminal_mass_passed=terminal,
        eligible_for_idql_evaluation=bool(rho and perpath and human and same and ess and terminal),
        failures=failures, unavailable=unavailable,
        interpretation='pre-registered held-out critic checks; not proof of policy success or improvement over BC')


def summarize_diagnostics(rows, outputs, *, beta=3., max_weight=20., thresholds=None,
                          same_state_result=None, expected_cache_indices=None):
    """Summarize unaugmented outputs in exactly the supplied replay row order.

    Global progress counts unique intervals once, using the median remaining
    distance across their successful-path memberships. Per-path progress restores
    every verified shared-prefix membership. Undefined paths are retained and
    prevent a defined median; they are never silently discarded.
    """
    rows = list(rows)
    _require(math.isfinite(beta) and beta>0 and math.isfinite(max_weight) and max_weight>0, 'invalid weight diagnostic parameters')
    n = len(rows)
    values = {key:np.asarray(outputs[key], dtype=np.float64) for key in ('q1','q2','target_q1','target_q2','v')}
    _require(all(x.shape == (n,) and np.isfinite(x).all() for x in values.values()), 'outputs require finite per-sample vectors')
    indices = [r['cache_index'] for r in rows]
    _require(all(type(x) is int and x >= 0 for x in indices), 'invalid cache index')
    expected = None if expected_cache_indices is None else [int(x) for x in expected_cache_indices]
    unique = len(indices) == len(set(indices))
    complete = bool(n) and unique and expected is not None and len(expected)==len(set(expected)) \
        and len(indices)==len(expected) and set(indices)==set(expected) \
        and all(r.get('split') in {'val','validation'} for r in rows)
    roots = sorted({r['root_run_id'] for r in rows})
    coverage = dict(count=n, unique_count=len(set(indices)), expected_count=None if expected is None else len(expected),
        complete_heldout=bool(complete), duplicate_cache_indices=not unique,
        root_count=len(roots), root_run_ids=roots, definition='exact unique held-out cache roster, no replacement samples')
    scoped = bool(n) and all(r.get('progress_scope') == 'complete_successful_paths' for r in rows)
    scopes = {r.get('membership_sha256') for r in rows} if scoped else set()
    _require(not scoped or len(scopes)==1 and None not in scopes, 'inconsistent membership identity')
    path_values, remaining, terminal = {}, [], []
    expected_paths = rows[0].get('expected_success_paths', {}) if scoped else {}
    if scoped:
        _require(all(r.get('expected_success_paths') == expected_paths for r in rows), 'inconsistent path roster')
    for index,row in enumerate(rows):
        if scoped:
            members = row['success_memberships']
            remaining.append(float(np.median([m['remaining_blocks'] for m in members])) if members else -1.)
            terminal.append(any(m['success_terminal'] for m in members))
            for member in members:
                _require(member['root_run_id'] == row['root_run_id'] and member['run_id'] in expected_paths,
                         'successful membership root/roster differs')
                path_values.setdefault(member['run_id'], []).append((row['cache_index'], values['v'][index], member['remaining_blocks']))
        else:
            remaining.append(float(row.get('remaining_blocks',-1)))
            terminal.append(bool(row.get('success_terminal',False)))
    complete_paths = scoped and all(run in path_values and len(path_values[run])==len(path['cache_indices'])
        and set(x[0] for x in path_values[run])==set(path['cache_indices']) for run,path in expected_paths.items())
    perpath=[]
    for run,path in sorted(expected_paths.items()):
        entries=path_values.get(run,[])
        full = len(entries)==len(path['cache_indices']) and set(x[0] for x in entries)==set(path['cache_indices'])
        corr=spearman([x[1] for x in entries],[x[2] for x in entries]) if full else None
        perpath.append(dict(trajectory_id=run, root_run_id=path['root_run_id'], count=len(entries),
            expected_count=len(path['cache_indices']), complete=full, spearman=corr))
    correlations=[x['spearman'] for x in perpath]
    remaining=np.asarray(remaining,float); success=remaining>=0
    with np.errstate(over='ignore',invalid='ignore'):
        advantage=np.minimum(values['q1'],values['q2'])-values['v']
    _require(np.isfinite(advantage).all(), 'derived advantage must be finite')
    with np.errstate(under='ignore',over='ignore'):
        weights=np.exp(np.minimum(beta*advantage,math.log(max_weight)))
    stats=weight_statistics(weights,terminal,max_weight=max_weight)
    comparisons=[];unmatched=[]
    for root in roots:
        human=[i for i,r in enumerate(rows) if r['root_run_id']==root and r.get('action_source')=='human']
        onset=min((rows[i]['start'] for i in human),default=None)
        failure=[i for i,r in enumerate(rows) if r['root_run_id']==root and r.get('run_id')==root
            and r.get('action_source')=='policy' and r.get('recording_success') is False
            and onset is not None and r['start']>=onset]
        if not human or not failure:
            if human:
                unmatched.append(root)
            continue
        a,b=float(np.median(advantage[human])),float(np.median(advantage[failure]))
        comparisons.append(dict(root_run_id=root,human_count=len(human),failure_policy_count=len(failure),
            first_retained_human_start=int(onset),human_median_advantage=a,failure_median_advantage=b,
            human_higher=bool(a>b),tie=bool(a==b)))
    pair = dict(same_state_result) if same_state_result is not None else unavailable_same_state()
    if pair.get('available'):
        _require(set(pair.get('root_run_ids',[])) <= set(roots), 'pair result includes non-held-out roots')
    report=dict(schema_version=1,coverage=coverage,**stats,
        progress_scope='complete_successful_paths' if scoped else 'retained_unique_trajectory_intervals',
        complete_success_path_coverage=bool(complete_paths),
        spearman_v_remaining_blocks=spearman(values['v'][success],remaining[success]),
        median_per_trajectory_spearman=float(np.median(correlations)) if correlations and all(x is not None for x in correlations) else None,
        success_progress_count=int(success.sum()),per_trajectory=perpath,
        defined_trajectory_count=sum(x is not None for x in correlations),
        undefined_trajectory_count=sum(x is None for x in correlations),
        progress_definition='confirmed-success full paths; unique global row uses median successful-membership remaining; terminal has remaining=0',
        weight_definition='hypothetical capped exp(beta*(online_min_Q-V)); diagnostic only, never actor updates or argmax selection',
        beta=float(beta),max_weight=float(max_weight),
        human_comparison=dict(available=bool(comparisons), human_higher_fraction=float(np.mean([x['human_higher'] for x in comparisons])) if comparisons else None,
            matched_root_count=len(comparisons),unmatched_human_roots=unmatched,
            complete_human_root_coverage=bool(comparisons) and not unmatched,
            tie_count=sum(x['tie'] for x in comparisons),per_root=comparisons,
            definition='same-root different-state segment median online advantage; original failed-policy retained blocks after first retained human start; heuristic, not causal same-state evidence'),
        same_state_action_ranking=pair)
    for key,value in values.items():
        report[key+'_distribution']=distribution(value)
    report['advantage_distribution']=distribution(advantage)
    report['gates']=classify_gates(report,thresholds)
    return report
