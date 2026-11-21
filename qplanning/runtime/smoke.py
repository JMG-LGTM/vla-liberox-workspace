#!/usr/bin/env python3
"""GPU smoke for a configured reference scorer and the current candidate-selection policy."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import numpy as np
from .planning import MEAN_MODES, Q_MODES, PlanningParityError
from .contract import atomic_json, gpu_lock, require, validate_contract, validate_feature_tolerance
from .reference_probe import (check_limits, load_spec, offline_scores, require_matrix_pass, run_reference,
                              score_errors, score_matrix, validate_reference)


def backend_flags(torch):
    return [torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(), torch.backends.cuda.matmul.allow_tf32,
            torch.get_float32_matmul_precision()]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--runtime-contract', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--observation', type=Path, required=True, help='Policy-oriented observation with the configured camera views')
    p.add_argument('--reference-proof', type=Path, help='Reuse a GPU reference report for the same sealed archive assets')
    p.add_argument('--train-frames', type=Path, help='Cached feature fixtures for greedy smoke')
    p.add_argument('--q-fixture', type=Path, help='Optional raw candidate fixture for the current critic; defaults to configured reference inputs')
    p.add_argument('--trial-seeds', type=int, nargs='+', required=True)
    p.add_argument('--prompt', required=True)
    a = p.parse_args(argv)
    require(not a.output.exists(), 'preserve real probe attempts')
    c = validate_contract(a.runtime_contract)
    with gpu_lock(c):
        return smoke_locked(a,c)


def check_single_selection(runtime, c, observation, trials, remaining):
    import torch
    from openpi.policies.policy import Policy
    horizon = runtime.config.horizon
    baseline = Policy(runtime.components.model, transforms=[runtime.components.input_transform],
                      output_transforms=[runtime.components.output_transform], is_pytorch=True,
                      pytorch_device='cuda:0', sample_kwargs={'num_steps': runtime.flow_steps})
    diagnostics, checks = [], []
    original_sample = runtime.components.model.sample_noise
    for trial in trials:
        baseline_noise = []

        def capture_noise(shape, device):
            noise = original_sample(shape, device)
            baseline_noise.append(noise.detach().float().cpu().numpy().copy())
            return noise
        runtime.components.model.sample_noise = capture_noise
        try:
            with torch.random.fork_rng(devices=[runtime.device.index or 0]):
                torch.manual_seed(trial)
                torch.cuda.manual_seed_all(trial)
                expected = [np.clip(np.asarray(baseline.infer(observation)['actions'], dtype=np.float32), -1., 1.) for _ in remaining]
        finally:
            runtime.components.model.sample_noise = original_sample
        require(len(baseline_noise) == len(remaining), 'BC default stream not observed')
        runtime.reset_seed(trial)
        reference_rng = np.random.default_rng(np.random.SeedSequence([c['selection_seed'],trial]))
        captured = []
        original_pool = runtime.generate_pool

        def capture_pool(obs):
            actions, noise = original_pool(obs)
            captured.append((actions.copy(), noise.copy()))
            return actions, noise
        runtime.generate_pool = capture_pool
        try:
            for query, length in enumerate(remaining):
                selected, d = runtime.infer(observation, remaining_steps=length)
                pool, noise = captured[-1]
                index = d['selected_index']
                require(np.array_equal(noise[0], baseline_noise[query][0]), 'anchor noise differs from the BC stream')
                require(np.array_equal(pool[0], expected[query]), 'anchor actions differ from original BC')
                count = c['candidate_count']
                require(pool.shape == (count, *expected[query].shape) and len({n.tobytes() for n in noise}) == count, 'candidate pool differs')
                slot_errors=[]
                for slot in range(count):
                    ref=baseline.infer(observation,noise=noise[slot:slot+1])['actions']
                    ref=np.clip(np.asarray(ref,dtype=np.float32),-1.,1.)
                    error=float(np.abs(pool[slot]-ref).max())
                    require(np.array_equal(pool[slot],ref),f'BC candidate {slot} differs under the same noise: {error}')
                    slot_errors.append(error)
                expected_index=int(np.argmax(d['scores'])) if c['pick']=='greedy' else int(reference_rng.integers(count))
                require(np.array_equal(selected, pool[index]) and index == expected_index, 'selection differs')
                require(np.abs(pool).max() <= 1. and d['remaining_horizon_mask'] == min(horizon, length), 'clip or horizon differs')
                diagnostics.append(d)
                checks.append(dict(seed=trial, query=query, noise_exact=True, actions_exact=True, per_slot_max_abs=slot_errors))
        finally:
            runtime.generate_pool = original_pool
    return diagnostics, checks


def fixture_score_parity(online, offline, shape, contract, spec):
    """Check the current critic separately from the frozen historical reference."""
    if 'score_max_abs_tolerance' not in contract:
        report=score_errors(online,offline)
        check_limits(report,dict(spec,score_shape=list(shape)))
        return report

    report=score_matrix(online,offline,shape,contract.get('elite_count'),contract['score_max_abs_tolerance'],
                        rank_singleton=not contract.get('include_archive_in_score_tolerance'))
    require_matrix_pass(report,'fixture_offline_scores')
    return report


def training_feature_parity(scorer, frames, view_fields, tolerance):
    """Compare local RGB encodings to the cached training features, retaining each frame's measurement."""
    validate_feature_tolerance(tolerance)
    cached = frames['features']
    count = len(cached)
    require(count > 0 and view_fields and all(len(frames[k]) == count for k in view_fields),
            'expected aligned RGB views and cached training features')
    require(all(k not in frames or len(frames[k]) == count for k in ('root_id','frame')),
            'training feature metadata is misaligned')
    rows = []
    for index, original in enumerate(cached):
        row = dict(index=index, root_id=str(frames['root_id'][index]) if 'root_id' in frames else None,
                   frame=int(frames['frame'][index]) if 'frame' in frames else None,
                   max_abs_error=None, mean_abs_error=None, within_tolerance=False)
        rows.append(row)
        try:
            online = scorer.features(*(frames[k][index] for k in view_fields))[0].float().cpu().numpy()
            require(online.shape == original.shape and original.size > 0, 'training feature shape differs')
            with np.errstate(invalid='ignore',over='ignore'):
                delta = np.abs(online - original.astype(np.float32))
            row['elements'] = int(delta.size)
            if np.isfinite(delta).all():
                maximum = float(delta.max())
                row.update(max_abs_error=maximum, mean_abs_error=float(delta.mean(dtype=np.float64)), within_tolerance=maximum <= tolerance)
            else:
                row['error'] = 'nonfinite training or local feature'
        except (ValueError,RuntimeError,FloatingPointError) as exc:
            row['error'] = str(exc)
            break

    complete = len(rows) == count and all(r['max_abs_error'] is not None for r in rows)
    failed = [r['index'] for r in rows if not r['within_tolerance']]
    return dict(tolerance=tolerance, within_tolerance=complete and not failed, frame_count=count,
                frames_compared=sum(r['max_abs_error'] is not None for r in rows), failed_frames=failed, per_frame=rows,
                max_abs_error=max(r['max_abs_error'] for r in rows) if complete else None,
                mean_abs_error=_mean_feature_error(rows) if complete else None)


def _mean_feature_error(rows):
    return math.fsum(r['mean_abs_error'] * r['elements'] for r in rows) / sum(r['elements'] for r in rows)


def validate_training_feature_parity(report, tolerance, scorer_parity):
    validate_feature_tolerance(tolerance)
    require(isinstance(report,dict) and report.get('within_tolerance') is True and report.get('tolerance') == tolerance,
            'training feature evidence missing or bound to a different tolerance')
    rows = report['per_frame']
    require(rows and len(rows) == report['frame_count'] == report['frames_compared'] == scorer_parity['training_feature_frames']
            and report['failed_frames'] == [], 'incomplete training feature evidence')
    for index,r in enumerate(rows):
        maximum, mean = r['max_abs_error'], r['mean_abs_error']
        require(r['index'] == index and r['within_tolerance'] is True and not r.get('error')
                and type(r['elements']) is int and r['elements'] > 0
                and type(maximum) in (int,float) and type(mean) in (int,float) and 0 <= mean <= maximum <= tolerance,
                'training feature frame exceeded tolerance or has invalid statistics')
    require(report['max_abs_error'] == max(r['max_abs_error'] for r in rows) == scorer_parity['feature_max_abs_error']
            and report['mean_abs_error'] == _mean_feature_error(rows), 'training feature summary differs from its frames')


def smoke_locked(a,c, *, on_fixture_scores=None):
    progress={}
    try:
        return _smoke_locked(a,c,progress,on_fixture_scores=on_fixture_scores)
    except (ValueError,RuntimeError,FloatingPointError) as exc:
        stage=progress.pop('stage','smoke_setup')
        failure=exc.evidence if isinstance(exc,PlanningParityError) else dict(stage=stage,error=str(exc))
        if failure.get('stage')=='fixture_offline_scores' and 'score_matrix' in failure:
            progress['scorer_parity']=failure['score_matrix']
        atomic_json(a.output,dict(status='FAIL_QPLANNING_GPU_SMOKE',pick=c['pick'],composite_sha256=c['composite_sha256'],
                                  failure=failure,**progress))
        raise


def _smoke_locked(a,c,progress, *, on_fixture_scores):
    seed = (c['critic'] or {}).get('seed')
    import torch
    from .runtime_core import q_numerics
    archived = None
    if c['pick'] in Q_MODES:
        progress['stage']='archive_reference_scores'
        require(a.train_frames is not None,'greedy smoke requires cached feature fixtures')
        spec=load_spec(c)
        if getattr(a,'reference_proof',None):
            archived=json.loads(Path(a.reference_proof).read_text())
        else:
            with q_numerics(), torch.random.fork_rng(devices=[0]):
                archived=run_reference(spec,torch.device('cuda:0'))
        progress['reference_gpu']=archived
        validate_reference(archived,spec)

    progress['stage']='load_runtime'
    import sys
    workspace = Path(c['workspace'])
    terminal = workspace/'liberox-vla-adapter-terminal'
    sys.path[:0] = [str(terminal),str(terminal/'scripts')]
    from backend.app.policies.snapshots import catalog_from_snapshot
    raw = c['snapshot']
    entry = catalog_from_snapshot(raw,a.output.parent/'.empty-policy-registry').select(raw['policy_id'])
    request = dict(runtime_contract=str(a.runtime_contract), policy_id=entry.policy_id, content_sha256=entry.content_sha256,
                   base_revision=entry.base_revision,runtime_composite_sha256=c['composite_sha256'],model=entry.settings,
                   actor=str(entry.actor) if entry.actor else None,actor_sha256=entry.component_sha256.get('actor'),parent=entry.parent)
    from .worker import load_runtime
    runtime, _ = load_runtime(request, device='cuda:0')
    require(runtime.device.type == 'cuda' and torch.cuda.is_initialized(), 'GPU inference required')
    flags = backend_flags(torch)
    scorer = runtime.scorer

    parity=None
    if c['pick'] in Q_MODES:
        progress['stage']='training_feature_parity'
        with np.load(a.train_frames) as z:
            frames = {k: z[k] for k in z.files}
        view_fields=spec['view_fields']
        feature_tolerance=c['training_feature_max_abs_tolerance'] if 'training_feature_max_abs_tolerance' in c else spec['feature_atol']
        feature_report=training_feature_parity(scorer,frames,view_fields,feature_tolerance)
        progress['training_feature_parity']=feature_report
        if not feature_report['within_tolerance']:
            raise PlanningParityError(dict(stage='training_feature_parity',first_failed_frame=feature_report['failed_frames'][0],
                                           max_abs_error=feature_report['max_abs_error'],tolerance=feature_tolerance))

        progress['stage']='fixture_offline_scores'
        fixture_path=a.q_fixture or spec['assets']['fixture']['path']
        with np.load(fixture_path,allow_pickle=False) as z:
            fixture={k:z[k] for k in (*view_fields,spec['candidate_field'])}
        actions=fixture[spec['candidate_field']]
        online=[]
        for index,(*views,cand) in enumerate(zip(*(fixture[k] for k in view_fields),actions,strict=True)):
            try:
                online.append(scorer.score(scorer.features(*views),cand,runtime.config.horizon))
            except (ValueError,RuntimeError,FloatingPointError) as exc:
                raise PlanningParityError(dict(stage='fixture_runtime_scores',fixture_row=index,error=str(exc),
                                               completed_online_scores=[row.tolist() for row in online])) from exc
        online=np.stack(online)
        if on_fixture_scores is not None:
            progress['stage']='training_fixture_scores'
            on_fixture_scores(online.copy())
            progress['stage']='fixture_offline_scores'
        try:
            with q_numerics():
                offline=offline_scores(scorer.net,scorer.dino,fixture,spec,scorer.device,scorer.mean,scorer.std,runtime.config)
        except (ValueError,RuntimeError,FloatingPointError) as exc:
            raise PlanningParityError(dict(stage='fixture_offline_scores',error=str(exc),
                                           completed_online_scores=online.tolist())) from exc
        parity=fixture_score_parity(online,offline,actions.shape[:2],c,spec)
        progress['scorer_parity']=parity
        progress['stage']='masked_tail_invariance'
        tail=actions[0]
        horizon=runtime.config.horizon
        if horizon>1:
            masked_length=horizon-1
            masked=tail.copy()
            masked[:,masked_length:]=np.clip(masked[:,masked_length:]+.5,-1.,1.)
            features=scorer.features(*(fixture[k][0] for k in view_fields))
            require(np.array_equal(scorer.score(features,tail,masked_length),scorer.score(features,masked,masked_length)),
                    'masked tail actions changed Q')
        progress['stage']='scoring_backend_flags'
        require(backend_flags(torch)==flags,'Q scoring changed process-wide attention flags')
        parity.update(checkpoint_sha256=c['critic']['sha256'],training_feature_frames=feature_report['frame_count'],
                      feature_max_abs_error=feature_report['max_abs_error'])

    progress['stage']='live_candidate_checks'
    with np.load(a.observation, allow_pickle=False) as data:
        observation = {k: data[k] for k in data.files}
    observation['prompt'] = a.prompt
    steps, horizon = runtime.config.max_steps, runtime.config.horizon
    remaining = [steps-i*horizon for i in range(3) if steps>i*horizon]+[min(steps,max(1,horizon-1))]
    if runtime.pick in MEAN_MODES:
        from .planning_smoke import check_planning
        diagnostics, checks = check_planning(runtime, observation, a.trial_seeds, remaining,
                                             score_max_abs_tolerance=c.get('score_max_abs_tolerance',0.),
                                             rank_singleton=not c.get('include_archive_in_score_tolerance'))
    else:
        diagnostics, checks = check_single_selection(runtime, c, observation, a.trial_seeds, remaining)
    progress.update(bc_checks=checks,candidate_diagnostics=diagnostics,stage='inference_backend_flags')
    require(backend_flags(torch) == flags, 'attention flags changed during inference')
    progress['stage']='frozen_actor'
    require(all(not q.requires_grad for q in runtime.components.model.parameters()), 'actor is trainable')
    proof = dict(status='PASS_QPLANNING_GPU_SMOKE', observed_at=datetime.now(timezone.utc).isoformat(),
                 critic_seed=seed, composite_sha256=c['composite_sha256'], attention_flags=flags,
                 scorer_parity=parity, reference_gpu=archived, candidate_policy_id=c['candidate_policy_id'], pick=c['pick'],
                 bc_checks=checks, simulator_rollout_executed=False,
                 candidate_diagnostics=diagnostics if runtime.pick in MEAN_MODES else None,
                 latency_ms=dict(total=float(np.median([d['total_ms'] for d in diagnostics])),
                                 candidates=float(np.median([d['candidate_generation_ms'] for d in diagnostics])),
                                 features=float(np.median([d['feature_ms'] for d in diagnostics])),
                                 q=float(np.median([d['critic_selection_ms'] for d in diagnostics]))),
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    if 'training_feature_parity' in progress:
        proof['training_feature_parity']=progress['training_feature_parity']
    atomic_json(a.output, proof)
    print(json.dumps({k: v for k, v in proof.items() if k not in ('bc_checks','candidate_diagnostics')}))


if __name__ == '__main__':
    main()
