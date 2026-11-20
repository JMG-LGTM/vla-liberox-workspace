"""Compare each planned candidate to a separate native single-draw policy inference."""
from __future__ import annotations
import json
import numpy as np
import torch

from .contract import require
from .planning import compare_scores, digest
from .runtime_core import derive_seed, q_numerics


class PlanningParityError(ValueError):
    def __init__(self, evidence):
        self.evidence=evidence
        super().__init__(json.dumps(evidence,allow_nan=False))


def check_planning(runtime, observation, trials, remaining, *, score_max_abs_tolerance=0.0):
    from openpi.policies.policy import Policy
    from ..c_score import candidate_scores
    from ..features import encode

    captured = []
    def decode(value):
        captured.append(np.array(value['actions'],copy=True))
        return runtime.components.output_transform(value)
    baseline = Policy(runtime.components.model, transforms=[runtime.components.input_transform], output_transforms=[decode],
                      is_pytorch=True, pytorch_device=str(runtime.device), sample_kwargs={'num_steps':runtime.flow_steps})
    count, elites, horizon = runtime.candidate_count, runtime.elite_count, runtime.config.horizon
    generate = runtime.generate_planning_pool
    sampled = []
    def capture_pool(obs):
        pool = generate(obs)
        sampled.append(pool)
        return pool

    diagnostics, checks = [], []
    runtime.generate_planning_pool = capture_pool
    try:
        for trial in trials:
            runtime.reset_seed(trial)
            selector = np.random.default_rng(np.random.SeedSequence([runtime.selection_seed,trial]))
            generator = torch.Generator(device=runtime.device).manual_seed(derive_seed('pool',trial,runtime.noise_namespace))
            for query,left in enumerate(remaining):
                actions, d = runtime.infer(observation, remaining_steps=left)
                pool = sampled.pop()
                reference_noise = torch.normal(0., 1., size=pool.noise.shape, dtype=torch.float32,
                                               device=runtime.device, generator=generator).cpu().numpy()
                require(np.array_equal(reference_noise,pool.noise), 'candidate RNG differs from its seeded stream')
                errors, normalized_errors, mismatches = [], [], []
                for slot in range(count):
                    native=baseline.infer(observation,noise=reference_noise[slot:slot+1])
                    raw=np.asarray(native['actions'])
                    normalized=captured.pop()
                    normalized_errors.append(float(np.max(np.abs(normalized-pool.normalized[slot]))))
                    errors.append(float(np.max(np.abs(raw-pool.raw[slot]))))
                    if not (normalized.dtype==pool.normalized.dtype and raw.dtype==pool.raw.dtype and
                            np.array_equal(normalized,pool.normalized[slot]) and np.array_equal(raw,pool.raw[slot])):
                        mismatches.append(slot)
                sample_evidence=dict(seed=trial,query=query,candidates_compared=count,matched_count=count-len(mismatches),
                                     mismatched_indices=mismatches,per_slot_max_abs=errors,
                                     per_slot_normalized_max_abs=normalized_errors)
                if mismatches:
                    raise PlanningParityError(dict(stage='native_candidate_parity',**sample_evidence))
                comparison=None
                if runtime.scorer is not None:
                    raw_views=np.stack([observation[k][::-1] for k in runtime.policy_camera_keys])[None]
                    scorer=runtime.scorer
                    with q_numerics():
                        features=encode(scorer.dino,raw_views,scorer.device,runtime.config)
                        offline=candidate_scores(scorer.net,features,pool.raw[None,:,:horizon],scorer.device,scorer.mean,scorer.std)[0]
                    scores=np.asarray(d['scores'],dtype=np.float32)
                    comparison=compare_scores(scores,offline,elites,score_max_abs_tolerance)
                    if not comparison['within_tolerance']:
                        raise PlanningParityError(dict(stage='offline_candidate_scores',**comparison,**sample_evidence))
                    values, idx = torch.from_numpy(scores).topk(elites)
                    weights = torch.softmax((values-values.max())/runtime.temperature, dim=0)
                else:
                    require(d['scores'] is None, 'random aggregation used Q')
                    idx = torch.from_numpy(selector.choice(count,size=elites,replace=False))
                    weights = torch.full((elites,),1./elites, dtype=torch.float32)
                normalized=(torch.from_numpy(pool.normalized)[idx] * weights.view(-1,1,1)).sum(0).numpy()
                decoded=runtime.components.output_transform(dict(state=pool.state.copy(),actions=normalized.copy()))
                expected=np.clip(np.asarray(decoded['actions'],dtype=np.float32),-1.,1.)
                if not (idx.tolist()==d['selected_indices'] and weights.tolist()==d['weights'] and
                        digest(normalized)==d['normalized_action_sha256'] and np.array_equal(expected,actions)):
                    raise PlanningParityError(dict(stage='recorded_score_aggregation',expected_indices=idx.tolist(),
                                                   actual_indices=d['selected_indices'],expected_weights=weights.tolist(),actual_weights=d['weights'],
                                                   normalized_expected_sha256=digest(normalized),normalized_actual_sha256=d['normalized_action_sha256'],
                                                   action_max_abs=float(np.abs(expected-actions).max()),
                                                   mismatched_action_coordinates=np.argwhere(expected!=actions).tolist(),score_comparison=comparison,**sample_evidence))
                checks.append(dict(noise_exact=True,actions_exact=True,normalized_exact=True,raw_exact=True,aggregation_exact=True,
                                   selected_indices=idx.tolist(),weights=weights.tolist(),**sample_evidence,
                                   score_comparison=comparison,
                                   offline_scores_exact=(comparison['score_max_abs']==0.0 if comparison is not None else None)))
                diagnostics.append(d)
    finally:
        runtime.generate_planning_pool=generate
    return diagnostics, checks
