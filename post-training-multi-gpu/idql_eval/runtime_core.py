"""IDQL action selection at evaluation time: the frozen BC proposes 4 candidates, the Q scorer picks one.

This is the file to edit to change how candidates are drawn or chosen:
- CandidateRuntime.generate_pool: candidate 0 is the original BC sample (same noise stream as plain BC),
  candidates 1-3 are drawn in one batch from independent noise.
- CandidateRuntime.infer: scores the 4 candidates with min(Q1, Q2) of the target critic over the 8 executed
  steps and executes the argmax (greedy).
Q scoring itself is post_training/official_idql/scoring.py (score_chunks), the same code as training.


The provider has already oriented both images once, exactly as the BC provider.
No actor/critic artifacts or GPU context are created when importing this module.
"""
from __future__ import annotations
import copy
import hashlib
import json
import time
import numpy as np
import torch
from torch.utils._pytree import tree_map


SAMPLING = {'recipe':'bc_anchor_single1_plus_independent3_v1',
    'anchor_forward':'original_openpi_policy_infer_batch1','extra_forward':'batch3',
    'anchor_noise':'torch.normal_float32_shape1x10x32_default_bc_seed_stream',
    'extra_noise':'independent_cuda_generator','selection_noise':'independent_cpu_generator',
    'seed_derivation':'sha256_bc78-n4-v1_domain_seed_first8_big_mask63',
    'extra_domain':'extra3','selection_domain':'selector'}


def derive_seed(domain, seed):
    if domain not in ('extra3','selector') or type(seed) is not int:
        raise ValueError('Invalid independent noise seed domain')
    return int.from_bytes(hashlib.sha256(f'bc78-n4-v1:{domain}:{seed}'.encode()).digest()[:8],'big') & ((1<<63)-1)


def observation_sha256(observation):
    value=hashlib.sha256()
    for key in ('observation/image','observation/wrist_image','observation/state','prompt'):
        value.update(key.encode()+b'\0')
        if key=='prompt':value.update(str(observation[key]).encode())
        else:
            array=np.asarray(observation[key]);value.update(json.dumps([array.dtype.str,list(array.shape)],separators=(',',':')).encode()+b'\0')
            value.update(np.ascontiguousarray(array).tobytes())
        value.update(b'\0')
    return value.hexdigest()


class CandidateRuntime:
    def __init__(self, components, critic, normalization, *, expected_step, device='cuda:0', observation_factory=None, bc_policy=None, scorer_sha256=None):
        from post_training.replay import validate_normalization
        self.components, self.critic = components, critic
        self.normalization = validate_normalization(normalization)
        from post_training.io import stable_hash
        metadata = critic.metadata
        if (metadata['backend'] != 'official_idql_jax' or metadata['network'] != 'target_twin_q_min'
                or metadata['step'] != expected_step or stable_hash(metadata['normalization']) != stable_hash(self.normalization)):
            raise ValueError('configured twin-Q scorer step/normalization required')
        if not isinstance(scorer_sha256,str) or len(scorer_sha256)!=64 or any(c not in '0123456789abcdef' for c in scorer_sha256):
            raise ValueError('Verified target-Q payload SHA256 is required')
        self.scorer_sha256 = scorer_sha256
        self.critic_update = metadata['step']
        self.device = torch.device(device)
        self.observation_factory = observation_factory
        self.bc_policy = bc_policy
        self._seed = None
        self._pool_queries = 0

    def reset_seed(self, seed):
        if type(seed) is not int or not 0 <= seed < 1<<63:
            raise ValueError('Trial seed must be a nonnegative63-bit integer')
        self._seed=seed;self._pool_queries=0
        self.bc_generator=torch.Generator(device=self.device).manual_seed(seed)
        self.extra_generator=torch.Generator(device=self.device).manual_seed(derive_seed('extra3',seed))
        self.selector_generator=torch.Generator(device='cpu').manual_seed(derive_seed('selector',seed))
        return {'anchor_seed':seed,'extra_seed':derive_seed('extra3',seed),'selector_seed':derive_seed('selector',seed)}

    def _original_bc_policy(self):
        if self.bc_policy is None:
            from openpi.policies.policy import Policy
            self.bc_policy=Policy(self.components.model,transforms=[self.components.input_transform],
                output_transforms=[self.components.output_transform],is_pytorch=True,
                pytorch_device=str(self.device),sample_kwargs={'num_steps':10})
        return self.bc_policy

    @torch.inference_mode()
    def generate_pool(self, observation):
        if self._seed is None:raise ValueError('reset_seed is required before generating a BC-anchored pool')
        devices=[] if self.device.type=='cpu' else [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        # Same torch.normal API/dtype/shape as pinned OpenPI sample_noise. Extra
        # candidates and selection never consume the BC anchor stream.
        with torch.random.fork_rng(devices=devices):
            anchor=torch.normal(mean=0.,std=1.,size=(1,10,32),dtype=torch.float32,device=self.device,generator=self.bc_generator)
            extra=torch.normal(mean=0.,std=1.,size=(3,10,32),dtype=torch.float32,device=self.device,generator=self.extra_generator)
            output=self._original_bc_policy().infer(copy.deepcopy(observation),noise=anchor.detach().cpu().numpy())
            first=np.asarray(output['actions'],dtype=np.float32)
            if first.shape!=(10,7) or not np.isfinite(first).all():raise ValueError('Invalid original BC anchor output')
            first=np.clip(first,-1.,1.)
            others,extra_cpu=self.generate(observation,count=3,noise=extra)
        candidates=np.concatenate((first[None],others),axis=0)
        noise=np.concatenate((anchor.detach().float().cpu().numpy(),extra_cpu),axis=0)
        self._pool_queries+=1
        return candidates,noise

    @torch.inference_mode()
    def generate(self, observation, *, count=4, noise=None):
        """Use the exact BC transforms, one model call for all independent candidates."""
        inputs = self.components.input_transform(copy.deepcopy(observation))
        tensors = tree_map(lambda x: torch.from_numpy(np.array(x)).to(self.device)[None].expand(count, *np.shape(x)), inputs)
        factory = self.observation_factory
        if factory is None:
            from openpi.models.model import Observation
            factory = Observation.from_dict
        model = self.components.model
        shape = (count, 10, 32)
        noise = model.sample_noise(shape, self.device) if noise is None else noise.to(self.device)
        if noise.shape != shape or not torch.isfinite(noise).all(): raise ValueError('Invalid independent flow noise')
        output = model.sample_actions(self.device, factory(tensors), noise=noise, num_steps=10)
        if output.shape != shape or not torch.isfinite(output).all(): raise ValueError('Invalid batched actor output')
        candidates = []
        for index in range(count):
            decoded = self.components.output_transform(dict(
                state=tensors['state'][index].detach().cpu().numpy().copy(),
                actions=output[index].detach().cpu().numpy().copy()))
            actions = np.asarray(decoded['actions'], dtype=np.float32)
            if actions.shape != (10, 7) or not np.isfinite(actions).all(): raise ValueError('Invalid decoded BC action candidate')
            candidates.append(np.clip(actions, -1., 1.))
        noise_cpu = noise.detach().float().cpu().numpy()
        return np.stack(candidates), noise_cpu

    @torch.inference_mode()
    def infer(self, observation, *, remaining_steps, mode='greedy', generator=None):
        from post_training.replay import normalize, _pixels
        if type(remaining_steps) is not int or not 1 <= remaining_steps <= 950:
            raise ValueError('Remaining executed action count must be in1..950')
        if mode != 'greedy' or generator is not None:
            raise ValueError('This official scorer runtime requires fixed greedy target-min selection')
        started = time.perf_counter()
        candidates, noise = self.generate_pool(observation)
        generated = time.perf_counter()
        length = min(8, remaining_steps)
        state = np.asarray(observation['observation/state'], dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all(): raise ValueError('Invalid raw critic state')
        # Provider has applied exactly the original BC vertical orientation.
        pixels = _pixels(observation['observation/image'], observation['observation/wrist_image'], 128).transpose(1,2,0)
        scorer_observation = {'agent_pixels':np.array(pixels[...,:3],copy=True),
            'wrist_pixels':np.array(pixels[...,3:],copy=True),
            'state':normalize(state,self.normalization['state'])}
        masks = np.broadcast_to(np.arange(8)<length,(4,8)).copy()
        # The actual export owns normalization and the masked56+mask8 codec.
        # Score the same clipped raw actions that will be executed, only for
        # their common actual H8/H6 horizon. No second normalization or flip.
        result = self.critic.score_chunks(scorer_observation,candidates[:,:8],masks,actions_normalized=False)
        q1,q2,scores = [np.asarray(result[key],dtype=np.float32) for key in ('q1','q2','q_min')]
        if any(x.shape!=(4,) or not np.isfinite(x).all() for x in (q1,q2,scores)):
            raise ValueError('Official CPU scorer returned invalid target-Q vectors')
        index = int(np.argmax(scores))
        selected = candidates[index].copy()
        selected_matches = np.array_equal(selected[:length],candidates[index,:length])
        min_matches = np.array_equal(scores,np.minimum(q1,q2))
        if not (selected_matches and min_matches): raise ValueError('Official IDQL selection proof failed')
        finished = time.perf_counter()
        diagnostic = dict(candidate_count=4,candidate_shape=list(candidates.shape),noise_shape=list(noise.shape),
            flowsteps=10,target_reduction='min',selection_mode=mode,selected_index=index,
            q1=q1.tolist(),q2=q2.tolist(),scores=scores.tolist(),remaining_horizon_mask=length,
            target_min_matches=True,selected_index_matches_argmax=True,selected_index_obeys_arm=True,argmax_index=index,
            scorer_backend='official_idql_jax',scorer_device='cpu',scorer_sha256=self.scorer_sha256,critic_update=self.critic_update,
            critic_training_step=self.critic_update,
            sampling_recipe=SAMPLING['recipe'],pool_query_index=self._pool_queries,
            anchor_seed=self._seed,extra_seed=derive_seed('extra3',self._seed),selector_seed=derive_seed('selector',self._seed),
            slot0_original_policy_infer=True,anchor_batch_size=1,extra_batch_size=3,independent_noise_streams=True,
            observation_sha256=observation_sha256(observation),
            anchor_noise_sha256=hashlib.sha256(noise[:1].tobytes()).hexdigest(),
            extra_noise_sha256=hashlib.sha256(noise[1:].tobytes()).hexdigest(),
            per_candidate_noise_sha256=[hashlib.sha256(row.tobytes()).hexdigest() for row in noise],
            candidate_pool_sha256=hashlib.sha256(candidates.tobytes()).hexdigest(),
            per_candidate_action_sha256=[hashlib.sha256(row.tobytes()).hexdigest() for row in candidates],
            selected_action_matches_candidate=selected_matches,scored_action_is_executed_clipped=True,
            input_images_already_oriented_once=True,critic_image_size=128,
            noise_sha256=hashlib.sha256(noise.tobytes()).hexdigest(),
            noise_candidates_distinct=len({row.tobytes() for row in noise})==4,
            candidate_pairwise_l2=[float(np.linalg.norm(candidates[i]-candidates[j])) for i in range(4) for j in range(i)],
            candidate_generation_ms=1000*(generated-started),critic_selection_ms=1000*(finished-generated),
            total_ms=1000*(finished-started),device=str(self.device))
        return selected,diagnostic
