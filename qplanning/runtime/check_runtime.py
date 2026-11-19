"""Exercise the real candidate and selection paths on CPU with a deterministic synthetic actor."""
from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from ..config import DEFAULT_CONFIG, QConfig
from .contract import DEFAULT_NOISE_NAMESPACE, gpu_lock, validate_effective, validate_smoke
from .runtime_core import CandidateRuntime, QScorer, derive_seed


class Actor:
    def __init__(self, q_config):
        self.config=SimpleNamespace(action_horizon=q_config.horizon+2, action_dim=32)

    def sample_actions(self, device, observation, *, noise, num_steps):
        assert noise.device.type == 'cpu' and num_steps == 10
        return noise + observation['state'][:, None, :1]


class OriginalPolicy:
    def __init__(self, action_dim):
        self.action_dim=action_dim

    def infer(self, observation, *, noise):
        actions = noise[0] + observation['state'][0]
        return {'actions':actions[:, :self.action_dim]}


class Scorer:
    device = torch.device('cpu')
    identity = {'seed': 1, 'training_step': 2, 'sha256': 'test-double'}

    def __init__(self, config):
        self.config=config

    def features(self, *views):
        assert len(views) == len(self.config.camera_keys)
        assert [int(image.flat[0]) for image in views] == list(range(len(views)))

    def score(self, features, candidates, length):
        assert candidates.shape[1:] == (self.config.horizon,self.config.action_dim)
        assert 0 < length <= self.config.horizon
        return np.linspace(0., 1., len(candidates))


def make_runtime(count, mode, selector_seed, config=DEFAULT_CONFIG, namespace=DEFAULT_NOISE_NAMESPACE):
    original=OriginalPolicy(config.action_dim)
    components=SimpleNamespace(model=Actor(config), input_transform=lambda obs: {'state':obs['state']},
                               output_transform=lambda value:{'actions':value['actions'][:,:config.action_dim]})
    return CandidateRuntime(components,Scorer(config) if mode=='greedy' else None,device='cpu',observation_factory=lambda t:t,
                            bc_policy=original,candidate_count=count,pick=mode,selection_seed=selector_seed,
                            config=config,noise_namespace=namespace)


def make_observation(cfg):
    return {'state':np.array([.125],np.float32),
            **{k:np.full((2,2,3),i,np.uint8) for i,k in enumerate(cfg.policy_camera_keys)}}


def check_guards():
    snap=dict(policy_id='synthetic-bc',family='pi05',model_config={'num_inference_steps':10})
    protocol=dict(max_steps=DEFAULT_CONFIG.max_steps,open_loop_steps=DEFAULT_CONFIG.horizon,control_hz=20,
                  realtime=True,disabled_policy_cameras=[],manipulability_enabled=False)
    c=dict(q_config=DEFAULT_CONFIG.to_dict(),protocol=protocol,candidate_policy_id='synthetic-bc',continuation_policy_id='synthetic-bc',snapshot=snap,
           schedule_sha256=['half0','half1'],composite_sha256='test-contract',pick='random',candidate_count=4)
    proof=dict(status='PASS_QPLANNING_GPU_SMOKE',composite_sha256='test-contract',candidate_policy_id='synthetic-bc',
               pick='random',bc_checks=[dict(noise_exact=True,actions_exact=True,per_slot_max_abs=[0.0]*4)])
    validate_smoke(proof,c)
    try:
        validate_smoke(dict(proof,composite_sha256='stale'),c)
    except ValueError:
        pass
    else:
        raise AssertionError('stale smoke evidence accepted')
    effective=dict(config=protocol,policy_snapshot=snap,schedule_sha256='half1')
    validate_effective(effective,c)
    try:
        validate_effective(effective,dict(c,continuation_policy_id='different-actor'))
    except ValueError:
        pass
    else:
        raise AssertionError('a different continuation actor was silently accepted')
    # Verify exclusion on an isolated fixture; the real platform lock is never opened in this CPU test.
    with TemporaryDirectory() as tmp:
        path=Path(tmp)/'dataset-root/projects/libero_x_vla/.gpu-task.lock'
        path.parent.mkdir(parents=True)
        path.touch()
        with gpu_lock({'workspace':tmp}):
            try:
                with gpu_lock({'workspace':tmp}):
                    raise AssertionError('second locker admitted')
            except RuntimeError:
                pass
        with gpu_lock({'workspace':tmp}):
            pass


def check_scorer_layout(config):
    from ..critic import Critic
    class PatchGrid(torch.nn.Module):
        def forward(self, pixel_values):
            patches=pixel_values.mean(dim=(1,2,3))[:,None,None].expand(-1,257,16)
            return SimpleNamespace(last_hidden_state=patches)
    with TemporaryDirectory() as directory:
        files=Path(directory)
        model=Critic(layers=1,dim=16,heads=2,config=config)
        torch.save(dict(model=model.state_dict(),q_config=config.to_dict()),files/'q.pt')
        stats=dict(mean=[0.]*config.action_dim,std=[1.]*config.action_dim)
        (files/'stats.json').write_text(json.dumps(stats))
        identity=dict(path=str(files/'q.pt'),action_stats=str(files/'stats.json'),dino_path=directory,
                      layers=1,heads=2,dim=16)
        with patch('transformers.Dinov2Model.from_pretrained',return_value=PatchGrid()):
            scorer=QScorer(identity,'cpu')
            views=make_observation(config)
            feat=scorer.features(*(views[k] for k in config.policy_camera_keys))
            assert feat.shape==(1,len(config.camera_keys),256,16)
            a=np.random.default_rng(36).standard_normal(size=(4,config.horizon,config.action_dim)).astype(np.float32)
            length=config.horizon-1
            scores=scorer.score(feat,a,length)
            assert scores.shape==(4,) and np.isfinite(scores).all()
            a[:,length:]+=15.
            np.testing.assert_array_equal(scores,scorer.score(feat,a,length))


def check():
    cfg=DEFAULT_CONFIG
    observation=make_observation(cfg)
    check_guards()
    # Pin the default extra-candidate stream and compare three successive pools.
    expected_seed=322725624670834750
    trial_seed=3
    assert derive_seed('extra3',trial_seed)==expected_seed
    default_runtime=make_runtime(4,'greedy',37)
    default_runtime.reset_seed(trial_seed)
    reference_generator=torch.Generator(device='cpu').manual_seed(expected_seed)
    for _ in range(3):
        _,noise=default_runtime.generate_pool(observation)
        ref=torch.normal(0.,1.,(3,default_runtime.native_horizon,default_runtime.noise_dim),generator=reference_generator)
        assert noise[1:].tobytes()==ref.numpy().tobytes()
    custom=QConfig(max_steps=17,horizon=3,action_dim=5,camera_keys=('detail','wide','top'),
                   platform_camera_keys=('detail_sensor','wide_sensor','top_sensor'),
                   policy_camera_keys=('detail_input','wide_input','top_input'))
    check_scorer_layout(custom)
    alt_obs=make_observation(custom)
    alt=make_runtime(3,'greedy',31,custom,namespace='synthetic-lab')
    alt_random=make_runtime(3,'random',31,custom,namespace='synthetic-lab')
    alt_default=make_runtime(3,'greedy',31,custom)
    for rt in (alt,alt_random,alt_default):
        rt.reset_seed(trial_seed)
    a,greedy_trace=alt.infer(alt_obs,remaining_steps=2)
    _,random_trace=alt_random.infer(alt_obs,remaining_steps=2)
    _,default_trace=alt_default.infer(alt_obs,remaining_steps=2)
    assert greedy_trace['candidate_pool_sha256']==random_trace['candidate_pool_sha256']
    assert greedy_trace['noise_sha256']==random_trace['noise_sha256']!=default_trace['noise_sha256']
    assert a.shape==(alt.native_horizon,custom.action_dim) and greedy_trace['remaining_horizon_mask']==2
    selections=[]
    for n in (1,4,7):
        greedy=make_runtime(n,'greedy',37)
        random=make_runtime(n,'random',37)
        other=make_runtime(n,'random',38)
        for trial in (3,5,3):
            for rt in (greedy,random,other):
                rt.reset_seed(trial)
            expected=np.random.default_rng(np.random.SeedSequence([37,trial]))
            expected_other=np.random.default_rng(np.random.SeedSequence([38,trial]))
            # Explicitly compare every batched slot to original single-policy inference at identical noise.
            pool,noise=greedy.generate_pool(observation)
            for i in range(n):
                reference=greedy.bc_policy.infer(observation,noise=noise[i:i+1])['actions']
                np.testing.assert_array_equal(pool[i],np.clip(reference,-1,1))
            greedy.reset_seed(trial)
            for step in range(24):
                a,d=greedy.infer(observation,remaining_steps=cfg.max_steps-cfg.horizon*step)
                _,r=random.infer(observation,remaining_steps=cfg.max_steps-cfg.horizon*step)
                _,s=other.infer(observation,remaining_steps=cfg.max_steps-cfg.horizon*step)
                assert d['candidate_pool_sha256']==r['candidate_pool_sha256']==s['candidate_pool_sha256']
                assert d['noise_sha256']==r['noise_sha256']==s['noise_sha256']
                assert a.shape==(greedy.native_horizon,cfg.action_dim) and d['selected_index']==n-1 and r['scores'] is None
                assert r['selected_index']==expected.integers(n) and s['selected_index']==expected_other.integers(n)
                if n==4:
                    selections.append(r['selected_index'])
    assert set(selections)==set(range(4)) and not torch.cuda.is_initialized()
    return dict(passed=True,device='cpu',candidate_counts=[1,4,7],selection_seed_replay=True,
                mock_all_slots_same_noise_exact=True,identical_candidate_streams=True,smoke_and_lock_guards=True,custom_layout=True,custom_scorer_mask_and_views=True,default_noise_bytes_exact=True,stage0_real_actor_used=False)


if __name__=='__main__':
    print(json.dumps(check()))
