"""A frozen actor generates an anchor and independent candidates for Q or uniform selection.

The provider has already oriented the configured images once, exactly as for BC.
"""
from __future__ import annotations
import contextlib
import copy
import hashlib
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils._pytree import tree_map

from ..config import DEFAULT_CONFIG, resolve_model_config
from .contract import DEFAULT_NOISE_NAMESPACE

def derive_seed(domain, seed, noise_namespace=DEFAULT_NOISE_NAMESPACE):
    if not isinstance(noise_namespace, str) or not noise_namespace.strip() or domain != 'extra3' or type(seed) is not int:
        raise ValueError('Invalid independent noise stream')
    label = f'{noise_namespace}:{domain}:{seed}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(label).digest()[:8], 'big') & ((1 << 63)-1)


@contextlib.contextmanager
def q_numerics():
    """Offline numerics for DINO and Q only. OpenPI turns on TF32 process-wide; BC keeps it."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def image_hash(*images):
    h = hashlib.sha256()
    for image in images:
        h.update(np.ascontiguousarray(image, dtype=np.uint8).tobytes())
    return h.hexdigest()


class QScorer:
    """DINOv2 features and Q, using the training patch order and cache precision."""

    def __init__(self, critic, device, config=None):
        from transformers import Dinov2Model
        from .q_model import Critic
        self.device = torch.device(device)
        self.identity = critic
        stats = json.loads(open(critic['action_stats']).read())
        self.mean, self.std = np.array(stats['mean']), np.array(stats['std'])
        state = torch.load(critic['path'], map_location='cpu', weights_only=True, mmap=True)
        self.config = resolve_model_config(state, config)
        if (self.mean.shape != (self.config.action_dim,) or self.std.shape != self.mean.shape
                or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or not (self.std > 0).all()):
            raise ValueError('Action normalization does not match the Q action space')
        with torch.device('meta'):
            self.net = Critic(layers=critic['layers'], dim=critic['dim'], heads=critic['heads'], config=self.config)
        self.net.load_state_dict(state.get('model',state), strict=True, assign=True)
        self.net.requires_grad_(False).eval().to(self.device)
        self.dino = Dinov2Model.from_pretrained(critic['dino_path'], local_files_only=True)
        self.dino.requires_grad_(False).eval().to(self.device)

    @torch.inference_mode()
    def features(self, *views):
        """Inputs are policy-oriented; Q was trained on the stored raw orientation."""
        if len(views) != len(self.config.camera_keys):
            raise ValueError('Q feature inputs do not match configured cameras')
        images = np.stack([np.asarray(image)[::-1] for image in views])[None]
        x = torch.from_numpy(np.ascontiguousarray(images)).to(self.device).flatten(0, 1).permute(0, 3, 1, 2).float().div_(255)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False, antialias=True)
        mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
        std = x.new_tensor([.229, .224, .225])[None, :, None, None]
        with q_numerics():
            tokens = self.dino(pixel_values=(x-mean)/std).last_hidden_state[:, 1:, :]
        y = tokens.unsqueeze(0).half()
        if not torch.isfinite(y).all():
            raise ValueError('nonfinite DINO features')
        return y

    @torch.inference_mode()
    def score(self, features, candidates, length):
        raw = np.asarray(candidates)
        horizon = self.config.horizon
        if raw.ndim != 3 or raw.shape[1:] != (horizon, self.config.action_dim) or not 1 <= length <= horizon:
            raise ValueError('Q candidate action shape or valid length does not match the configured chunk')
        mask = np.broadcast_to(np.arange(horizon) < length, raw.shape[:2]).copy()
        actions = np.where(mask[..., None], (raw-self.mean)/self.std, 0).astype('float32')
        f = features.repeat(len(raw), 1, 1, 1).float()
        a = torch.from_numpy(actions).to(self.device)
        m = torch.from_numpy(mask).to(self.device)
        from .q_model import expectation
        with q_numerics(), torch.autocast(self.device.type, dtype=torch.bfloat16):
            q = expectation(self.net(f, a, m))
        q = q.float().cpu().numpy()
        if not np.isfinite(q).all() or q.min() < -.01 or q.max() > 1.01:
            raise ValueError('invalid Q value')
        return q


class CandidateRuntime:
    def __init__(self, components, scorer, *, device='cuda:0', observation_factory=None, bc_policy=None,
                 candidate_count=4, pick='greedy', selection_seed=0, noise_namespace=DEFAULT_NOISE_NAMESPACE,
                 config=DEFAULT_CONFIG, flow_steps=10):
        if type(candidate_count) is not int or candidate_count < 1:
            raise ValueError('candidate_count must be positive')
        if pick not in ('greedy', 'random') or (pick == 'greedy' and scorer is None):
            raise ValueError('greedy selection requires a Q scorer')
        if type(selection_seed) is not int or not 0 <= selection_seed < (1 << 63):
            raise ValueError('selection_seed must be a nonnegative 63-bit integer')
        self.candidate_count, self.pick, self.selection_seed = candidate_count, pick, selection_seed
        self.config = config
        self.policy_camera_keys = config.policy_camera_keys
        self.noise_namespace = noise_namespace
        self.flow_steps = flow_steps
        self.native_horizon = components.model.config.action_horizon
        self.noise_dim = components.model.config.action_dim
        if self.native_horizon < config.horizon or self.noise_dim < config.action_dim:
            raise ValueError('The actor has insufficient horizon or action width for the configured Q chunks')
        self.components, self.scorer = components, scorer
        self.device = torch.device(device)
        self.observation_factory = observation_factory
        self.bc_policy = bc_policy
        self._seed = None
        self._pool_queries = 0

    def reset_seed(self, seed):
        if type(seed) is not int or not 0 <= seed < 1 << 63:
            raise ValueError('Trial seed must be a nonnegative 63-bit integer')
        self.selection_rng = np.random.default_rng(np.random.SeedSequence([self.selection_seed, seed]))
        self._seed = seed
        self._pool_queries = 0
        self.bc_generator = torch.Generator(device=self.device).manual_seed(seed)
        self.extra_seed = derive_seed('extra3', seed, self.noise_namespace)
        self.extra_generator = torch.Generator(device=self.device).manual_seed(self.extra_seed)
        return {'anchor_seed': seed, 'extra_seed': self.extra_seed}

    def _original_bc_policy(self):
        if self.bc_policy is None:
            from openpi.policies.policy import Policy
            self.bc_policy = Policy(self.components.model, transforms=[self.components.input_transform],
                                    output_transforms=[self.components.output_transform], is_pytorch=True,
                                    pytorch_device=str(self.device), sample_kwargs={'num_steps': self.flow_steps})
        return self.bc_policy

    @torch.inference_mode()
    def generate_pool(self, observation):
        if self._seed is None:
            raise ValueError('reset_seed is required before generating a BC-anchored pool')
        devices = [] if self.device.type == 'cpu' else [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices):
            anchor = torch.normal(mean=0., std=1., size=(1, self.native_horizon, self.noise_dim), dtype=torch.float32, device=self.device, generator=self.bc_generator)

            output = self._original_bc_policy().infer(copy.deepcopy(observation), noise=anchor.detach().cpu().numpy())
            first = np.asarray(output['actions'], dtype=np.float32)
            if first.shape != (self.native_horizon, self.config.action_dim) or not np.isfinite(first).all():
                raise ValueError('Invalid original BC anchor output')
            first = np.clip(first, -1., 1.)
            candidates, noise = first[None], anchor.detach().float().cpu().numpy()
            extra_count = self.candidate_count - 1
            if extra_count:
                extra = torch.normal(mean=0., std=1., size=(extra_count, self.native_horizon, self.noise_dim), dtype=torch.float32,
                                     device=self.device, generator=self.extra_generator)
                others, extra_cpu = self.generate(observation, count=extra_count, noise=extra)
                candidates = np.concatenate((candidates, others), axis=0)
                noise = np.concatenate((noise, extra_cpu), axis=0)
        self._pool_queries += 1
        return candidates, noise

    @torch.inference_mode()
    def generate(self, observation, *, count=4, noise=None):
        inputs = self.components.input_transform(copy.deepcopy(observation))
        tensors = tree_map(lambda x: torch.from_numpy(np.array(x)).to(self.device)[None].expand(count, *np.shape(x)), inputs)
        factory = self.observation_factory
        if factory is None:
            from openpi.models.model import Observation
            factory = Observation.from_dict
        model = self.components.model
        shape = (count, self.native_horizon, self.noise_dim)
        noise = model.sample_noise(shape, self.device) if noise is None else noise.to(self.device)
        if noise.shape != shape or not torch.isfinite(noise).all():
            raise ValueError('Invalid independent flow noise')
        output = model.sample_actions(self.device, factory(tensors), noise=noise, num_steps=self.flow_steps)
        if output.shape != shape or not torch.isfinite(output).all():
            raise ValueError('Invalid batched actor output')
        candidates = []
        for index in range(count):
            decoded = self.components.output_transform(dict(
                state=tensors['state'][index].detach().cpu().numpy().copy(),
                actions=output[index].detach().cpu().numpy().copy()))
            actions = np.asarray(decoded['actions'], dtype=np.float32)
            if actions.shape != (self.native_horizon, self.config.action_dim) or not np.isfinite(actions).all():
                raise ValueError('Invalid decoded BC action candidate')
            candidates.append(np.clip(actions, -1., 1.))
        return np.stack(candidates), noise.detach().float().cpu().numpy()

    def infer(self, observation, *, remaining_steps):
        if type(remaining_steps) is not int or not 1 <= remaining_steps <= self.config.max_steps:
            raise ValueError('Remaining action count is outside the configured episode length')
        started = time.perf_counter()
        candidates, noise = self.generate_pool(observation)
        generated = time.perf_counter()
        horizon = self.config.horizon
        length = min(horizon, remaining_steps)
        views = [observation[key] for key in self.policy_camera_keys]
        count = self.candidate_count
        if self.pick == 'greedy':
            features = self.scorer.features(*views)
            encoded = time.perf_counter()
            scores = self.scorer.score(features, candidates[:, :horizon], length)
            if scores.shape != (count,):
                raise ValueError('Q returned an invalid score vector')
            index = int(np.argmax(scores))
        else:
            encoded = generated
            scores = None
            index = int(self.selection_rng.integers(count))
        selected = candidates[index].copy()
        finished = time.perf_counter()
        diagnostic = dict(candidate_count=count, candidate_shape=list(candidates.shape), noise_shape=list(noise.shape), flowsteps=self.flow_steps,
                          selection_mode=self.pick, selection_seed=self.selection_seed, selected_index=index,
                          scores=scores.tolist() if scores is not None else None,
                          score_range=float(scores.max()-scores.min()) if scores is not None else None, remaining_horizon_mask=length,
                          selected_index_matches_argmax=True if scores is not None else None,
                          selected_action_matches_candidate=bool(np.array_equal(selected, candidates[index])),
                          scored_action_is_executed_clipped=True if scores is not None else None,
                          scorer_backend='torch_qplanning_offline_q' if scores is not None else None,
                          scorer_device=str(self.scorer.device) if scores is not None else None,
                          critic_seed=self.scorer.identity['seed'] if scores is not None else None,
                          critic_training_step=self.scorer.identity['training_step'] if scores is not None else None,
                          scorer_sha256=self.scorer.identity['sha256'] if scores is not None else None,
                          sampling_recipe=f'bc_anchor_single1_plus_independent{count-1}_v1', slot0_original_policy_infer=True,
                          pool_query_index=self._pool_queries, anchor_seed=self._seed, extra_seed=self.extra_seed, noise_namespace=self.noise_namespace,
                          q_input_sha256=image_hash(*(np.asarray(img)[::-1] for img in views)),
                          candidate_pool_sha256=hashlib.sha256(candidates.tobytes()).hexdigest(),
                          noise_sha256=hashlib.sha256(noise.tobytes()).hexdigest(),
                          noise_candidates_distinct=len({row.tobytes() for row in noise}) == count,
                          candidate_pairwise_l2=[float(np.linalg.norm(candidates[i]-candidates[j])) for i in range(count) for j in range(i)],
                          candidate_generation_ms=1000*(generated-started), feature_ms=1000*(encoded-generated),
                          critic_selection_ms=1000*(finished-encoded), total_ms=1000*(finished-started), device=str(self.device))
        return selected, diagnostic
