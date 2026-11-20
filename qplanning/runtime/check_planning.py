"""CPU regression for the averaged planners, using OpenPI's native sampler on a tiny actor."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from ..config import DEFAULT_CONFIG
from .planning import aggregate, compare_scores, validate_score_tolerance
from .planning_smoke import PlanningParityError, check_planning
from .runtime_core import CandidateRuntime
from .check_runtime import Scorer


class TinyActor(torch.nn.Module):
    """Exercise the real OpenPI sample_actions control flow without model weights."""
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(action_horizon=DEFAULT_CONFIG.horizon+2, action_dim=DEFAULT_CONFIG.action_dim+3)
        self.prefix_calls=0
        config=SimpleNamespace(_attn_implementation=None)
        self.paligemma_with_expert=SimpleNamespace(paligemma=SimpleNamespace(language_model=SimpleNamespace(config=config)),
                                                   forward=self.prefix_forward)

    def sample_actions(self,*args,**kwargs):
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        return PI0Pytorch.sample_actions(self,*args,**kwargs)

    def _preprocess_observation(self,observation,*,train):
        assert not train
        return None,None,observation.state,None,observation.state

    def embed_prefix(self,images,image_masks,tokens,token_masks):
        self.prefix_calls+=1
        mask=torch.ones_like(tokens,dtype=torch.bool)
        return tokens,mask,mask

    def _prepare_attention_masks_4d(self,mask):
        return mask[:,None]

    def prefix_forward(self,*,inputs_embeds,use_cache,**kwargs):
        assert use_cache
        return None,inputs_embeds[0].sum()

    def denoise_step(self,state,mask,cache,draw,t):
        assert len(draw)==1
        return .125*draw + t[:,None,None] - cache


def check_score_evidence():
    from .contract import verify_score_comparison
    from .smoke import fixture_score_parity
    tolerance=5e-4
    limits=dict(score_max_abs_tolerance=tolerance,elite_count=2,candidate_count=3)
    def fixture(v):
        online=np.zeros((1,len(v)))
        return fixture_score_parity(online,np.array([v]),online.shape,limits,{})

    record=fixture([0.,tolerance,0.])['score_comparisons'][0]
    assert record['score_max_abs']==tolerance and record['within_tolerance'] and record['spearman'] is None
    verify_score_comparison(record,limits,3)
    try:
        verify_score_comparison(dict(record,score_max_abs=0.),limits,3)
    except ValueError:
        pass
    else:
        raise AssertionError('A modified maximum score error passed verification')
    for value in (np.nextafter(tolerance,np.inf),np.nan,np.inf,-np.inf):
        try:
            fixture([0.,value])
        except PlanningParityError as exc:
            evidence=exc.evidence
            assert evidence['stage']=='fixture_offline_scores' and not evidence['within_tolerance']
            json.dumps(evidence,allow_nan=False)
        else:
            raise AssertionError('An over-limit or nonfinite score passed')
    exact=compare_scores([1,2],[1,2],2,0.)
    assert exact['within_tolerance'] and exact['score_max_abs']==0 and exact['top1_same']
    assert not compare_scores([0],[1e-12],1,0.)['within_tolerance']
    assert not compare_scores([np.inf],[np.inf],1,tolerance)['within_tolerance']
    for bad in (-1.,np.nan,np.inf,True,'0.1'):
        try:
            validate_score_tolerance(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid score tolerance accepted')
    tied=compare_scores([1,1,2],[2,1,1],2,1.)
    assert np.isclose(tied['spearman'],-.5)
    assert not tied['top1_same'] and tied['topk_overlap']==len(set(tied['online_topk']).intersection(tied['offline_topk']))
    singleton=fixture([0.])['score_comparisons'][0]
    assert singleton['topk_overlap'] is None and singleton['online_topk'] is None and singleton['spearman'] is None


def check_runtime_scores(runtime, observation):
    # Close score vectors have opposite ordering. The hand calculation must
    # reconstruct the action using the runtime side even when no elites overlap.
    count=runtime.candidate_count
    online=np.linspace(0.,.0002,count,dtype=np.float32)
    offline=online[::-1].copy()
    runtime.scorer.net=runtime.scorer.dino=runtime.scorer.mean=runtime.scorer.std=None
    with patch.object(runtime.scorer,'score',return_value=online), patch('qplanning.features.encode'), \
            patch('qplanning.c_score.candidate_scores',return_value=offline[None]):
        queries,samples=check_planning(runtime,observation,[1],[8],score_max_abs_tolerance=.0005)
        sample=samples[0]
        record=sample['score_comparison']
        assert sample['aggregation_exact'] and not sample['offline_scores_exact']
        assert record['within_tolerance'] and not record['top1_same'] and record['topk_overlap']==0
        assert np.isclose(record['spearman'],-1.) and sample['selected_indices']==record['online_topk']
        from .contract import validate_smoke
        from .smoke import fixture_score_parity
        contract=dict(composite_sha256='cpu-proof',candidate_policy_id='fixture',pick='weighted',elite_count=runtime.elite_count,
                      candidate_count=count,critic={'sha256':'fixture-critic'},score_max_abs_tolerance=.0005)
        fixture=fixture_score_parity(online[None],offline[None],(1,count),contract,{})
        proof=dict(status='PASS_QPLANNING_GPU_SMOKE',composite_sha256='cpu-proof',candidate_policy_id='fixture',pick='weighted',
                   bc_checks=samples,candidate_diagnostics=queries,scorer_parity=dict(fixture,checkpoint_sha256='fixture-critic'),
                   reference_gpu={})
        with patch('qplanning.runtime.reference_probe.load_spec'),patch('qplanning.runtime.reference_probe.validate_reference'):
            validate_smoke(proof,contract)
            try:
                validate_smoke(dict(proof,candidate_diagnostics=[dict(queries[0],scores=offline.tolist())]),contract)
            except ValueError:
                pass
            else:
                raise AssertionError('Score evidence for a different query passed')
        try:
            check_planning(runtime,observation,[1],[8])
        except PlanningParityError as exc:
            assert exc.evidence['stage']=='offline_candidate_scores' and not exc.evidence['within_tolerance']
        else:
            raise AssertionError('The default did not require exact scores')


def check():
    check_score_evidence()
    rng=np.random.default_rng(37)
    norm=rng.normal(size=(5,7,3)).astype('float32')
    scores=np.array([-.3,.1,.2,.4,.3],dtype=np.float32)
    best,idx,w=aggregate(norm,scores,1,.75,rng)
    assert idx==[3] and w==[1.] and np.array_equal(best,norm[scores.argmax()])
    flat, idx,w=aggregate(norm,scores,3,1e25,rng)
    assert idx==[3,4,2] and np.allclose(flat,norm[idx].mean(0),atol=2e-7,rtol=0.)
    assert np.isclose(sum(w),1.)
    result,idx,w=aggregate(norm,scores,3,.02,rng)
    assert np.isclose(sum(w),1.) and w[0]>w[1]>w[2] and result.shape==norm.shape[1:]
    actor=TinyActor()
    c=DEFAULT_CONFIG
    components=SimpleNamespace(model=actor,
        input_transform=lambda obs:dict(state=obs['observation/state'],image={},image_mask={}),
        output_transform=lambda x:dict(actions=x['actions'][:,:c.action_dim].astype(np.float64)**2 + .03*x['state'][0]))
    obs=dict(prompt='CPU fixture',**{'observation/state':np.array([.2],np.float32)},
             **{k:np.full((3,3,3),i,np.uint8) for i,k in enumerate(c.policy_camera_keys)})
    scorer=Scorer(c)
    counts=[]
    oldscore=scorer.score
    def score(features,candidates,length):
        assert candidates.shape==(7,c.horizon,c.action_dim) and length==c.horizon
        counts.append((candidates.copy(),length))
        return oldscore(features,candidates,length).astype(np.float32)
    scorer.score=score
    common=dict(device='cpu',candidate_count=7,elite_count=3,temperature=.9,flow_steps=4,selection_seed=37)
    q=CandidateRuntime(components,scorer,pick='weighted',**common)
    r=CandidateRuntime(components,None,pick='random_mean',**common)
    for runtime in (q,r):
        runtime.reset_seed(2)
    before=actor.prefix_calls
    pool=q.generate_planning_pool(obs)
    assert actor.prefix_calls==before+1
    q.reset_seed(2)
    a,qa=q.infer(obs,remaining_steps=2)
    b,rb=r.infer(obs,remaining_steps=2)
    assert actor.prefix_calls==before+3 and len(counts)==1
    assert pool.raw.dtype==np.float64 and np.array_equal(counts[0][0],pool.raw[:,:c.horizon]) and np.max(counts[0][0])>1.
    assert qa['noise_sha256']==rb['noise_sha256'] and qa['normalized_pool_sha256']==rb['normalized_pool_sha256']
    assert qa['observation_sha256']==rb['observation_sha256'] and qa['candidate_pool_sha256']==rb['candidate_pool_sha256']
    idx=qa['selected_indices']
    assert np.isclose(sum(qa['weights']),1.) and len(set(idx))==3
    weighted=torch.tensor(qa['weights'])[:,None,None]
    combined=(weighted*torch.from_numpy(pool.normalized[idx])).sum(0).numpy()
    decoded=np.clip(components.output_transform(dict(actions=combined,state=pool.state))['actions'].astype(np.float32),-1.,1.)
    assert np.array_equal(a,decoded) and len(a)==actor.config.action_horizon and qa['scored_horizon']==c.horizon
    wrong=(weighted*torch.from_numpy(np.clip(pool.raw[idx],-1.,1.))).sum(0).numpy()
    assert not np.allclose(a,wrong)
    assert rb['scores'] is None and set(rb['weights'])=={np.float32(1/3)}
    assert rb['selected_indices']==np.random.default_rng(np.random.SeedSequence([37,2])).choice(7,size=3,replace=False).tolist()
    # Run the same native single-draw Policy.infer cross-check that gates a real evaluation.
    diagnostics,checks=check_planning(r,obs,[2],[c.max_steps,1])
    assert len(checks)==2 and all(x['per_slot_max_abs']==[0.]*7 and x['normalized_exact'] and x['aggregation_exact'] for x in checks)
    generate=r.generate_planning_pool
    def damaged_pool(observation):
        p=generate(observation)
        p.raw[0,0,0]+=.5
        p.raw[-1,0,0]+=.25
        return p
    r.generate_planning_pool=damaged_pool
    try:
        check_planning(r,obs,[2],[1])
    except PlanningParityError as exc:
        e=exc.evidence
        assert e['candidates_compared']==7 and e['matched_count']==5 and e['mismatched_indices']==[0,6]
        assert e['per_slot_normalized_max_abs']==[0.]*7 and len(e['per_slot_max_abs'])==7
    else:
        raise AssertionError('accepted altered candidate slots')
    check_runtime_scores(q,obs)
    assert not torch.cuda.is_initialized()
    return dict(device='cpu',one_prefix_per_pool=True,native_normalized_and_raw_actions_exact=True,
                complete_normalized_chunk_averaged=True,score_horizon_fixed=True,top1_exact=True,large_temperature_mean=True,
                inclusive_score_tolerance=True,tie_aware_ranking=True,aggregation_uses_recorded_scores=True,
                equal_candidate_streams=True,selection_without_replacement=True)


if __name__=='__main__': print(json.dumps(check()))
