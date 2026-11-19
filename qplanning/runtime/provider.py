"""Per-evaluation opt-in provider; ordinary policy IDs keep the original worker."""
from __future__ import annotations
import base64
import io
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import numpy as np

from backend.app.policies.pi05 import Pi05PolicyProvider as OriginalProvider
from backend.app.policies.registry import model_module
from ..config import QConfig
from .planning import MEAN_MODES, Q_MODES, digest, elite_weights
from .contract import HERE, algorithm, require, validate_contract

BACKEND = 'torch_qplanning_offline_q'


class QPlanningProvider(OriginalProvider):
    def __init__(self, runtime, eval_config, catalog, *, contract_path, trace_path):
        super().__init__(runtime, eval_config, catalog)
        self.contract_path, self.trace_path = Path(contract_path), Path(trace_path)
        self._q_contract = None
        self._seeded, self._queries, self._trial = False, 0, -1

    def load(self, open_loop_steps, policy_id=None, *, expected_content_sha256=None):
        value = validate_contract(self.contract_path)
        policy_id = value['continuation_policy_id'] if policy_id is None else policy_id
        if policy_id != value['continuation_policy_id']:
            if self._q_contract is not None:
                self.unload()
            return super().load(open_loop_steps, policy_id, expected_content_sha256=expected_content_sha256)
        with self._lock:
            config = QConfig.from_dict(value['q_config'])
            require(open_loop_steps == config.horizon, 'provider execution horizon differs from Q')
            entry = self.catalog.select(policy_id)
            require(entry.family == 'pi05' and entry.content_sha256 == value['snapshot']['content_sha256']
                    and (expected_content_sha256 is None or expected_content_sha256 == entry.content_sha256), 'catalog identity differs')
            actor = str(entry.actor) if entry.actor else None
            require(actor == value['snapshot']['actor'] and str(entry.base_checkpoint) == value['snapshot']['base_checkpoint'],
                    'selected platform artifacts differ')
            if (self.loaded and self.current_policy_id == policy_id and self._q_contract is not None
                    and self._q_contract['composite_sha256'] == value['composite_sha256']):
                return
            self.unload()
            self.responses = queue.Queue()
            self.process = subprocess.Popen(['conda', 'run', '--no-capture-output', '-n', entry.settings['environment'],
                                             'python', '-B', '-u', '-m', 'qplanning.runtime.worker'],
                                            env={**os.environ, 'PYTHONPATH':os.pathsep.join([str(HERE.parents[1]),os.environ.get('PYTHONPATH','')])},
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            text=True, bufsize=1, start_new_session=True)
            process, responses = self.process, self.responses

            def receive():
                try:
                    for line in process.stdout:
                        if line.startswith('PI05_RPC '):
                            responses.put(json.loads(line[len('PI05_RPC '):]))
                finally:
                    responses.put(None)
            threading.Thread(target=receive, daemon=True).start()
            try:
                ready = self._send(dict(model=entry.settings, owner_pid=os.getpid(), parent=entry.parent,
                                        actor=actor, actor_sha256=entry.component_sha256.get("actor"),
                                        base_revision=entry.base_revision, content_sha256=entry.content_sha256, policy_id=policy_id,
                                        runtime_contract=str(self.contract_path), runtime_composite_sha256=value['composite_sha256']), timeout=900)
                require(ready.get('identity') == entry.content_sha256
                        and ready.get('runtime_composite_sha256') == value['composite_sha256']
                        and ready.get('inference_algorithm') == algorithm(value) and ready.get('candidate_count') == value['candidate_count']
                        and ready.get('pick') == value['pick'] and ready.get('device') == 'cuda:0'
                        and ready.get('flow_steps') == value.get('flow_steps', entry.settings['num_inference_steps'])
                        and ready.get('elite_count') == value.get('elite_count') and ready.get('temperature') == value.get('temperature')
                        and ready.get('critic') == value['critic'] and ready.get('noise_namespace') == value['noise_namespace']
                        and type(ready.get('native_horizon')) is int and ready['native_horizon'] >= config.horizon,
                        'worker initialization differs')
                self._native_horizon = ready['native_horizon']
                self._q_config = config
                require(self.catalog.select(policy_id).content_sha256 == entry.content_sha256, 'catalog changed during load')
                self.current_policy_entry, self.current_policy_id = entry, policy_id
                self._q_contract = value
            except BaseException:
                self.unload()
                raise

    def _trace(self, value):
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open('a') as handle:
            handle.write(json.dumps(value, allow_nan=False)+'\n')
            handle.flush()

    def seed(self, seed):
        if self._q_contract is None:
            return super().seed(seed)
        super().seed(seed)
        self._selection_rng = np.random.default_rng(np.random.SeedSequence([self._q_contract['selection_seed'],int(seed)]))
        self._seeded, self._queries, self._trial = True, 0, self._trial+1
        self._trace(dict(event='seed', seed=int(seed), trial=self._trial, composite_sha256=self._q_contract['composite_sha256']))

    def predict(self, observation, prompt, disabled_policy_cameras=()):
        if self._q_contract is None:
            return super().predict(observation, prompt, disabled_policy_cameras)
        with self._lock:
            require(self.loaded and self._seeded, 'prediction requires a loaded worker and a trial seed')
            require(not disabled_policy_cameras, 'protocol has no disabled cameras')
            config = self._q_config
            remaining = config.max_steps - config.horizon * self._queries
            require(remaining > 0, 'unexpected query beyond configured episode limit')
            from trajectory_utils import quaternion_to_axis_angle
            contract = model_module('base_models').model_contract(self.current_policy_entry.settings)
            transforms = model_module('image_transforms')
            io_contract = transforms.observation_config(contract) if hasattr(transforms, 'observation_config') else contract['io']
            orientation = transforms.image_orientation(io_contract)
            require(orientation == 'flip_vertical', 'policy image orientation differs')
            images = {name:transforms.orient_image(observation[key],orientation)
                      for key,name in zip(config.platform_camera_keys,config.policy_camera_keys,strict=True)}
            images['observation/state'] = np.concatenate([observation['robot0_eef_pos'],
                                                          quaternion_to_axis_angle(observation['robot0_eef_quat']),
                                                          observation['robot0_gripper_qpos']]).astype(np.float32)
            buffer = io.BytesIO()
            np.savez(buffer, **images)
            result = self._send(dict(op='infer', arrays=base64.b64encode(buffer.getvalue()).decode(),
                                     prompt=prompt, remaining_steps=remaining))
            require(result.get('runtime_composite_sha256') == self._q_contract['composite_sha256']
                    and result.get('query_index') == self._queries+1, 'worker query or composite changed')
            actions = np.asarray(result['actions'], dtype=np.float32)
            require(actions.shape == (self._native_horizon, config.action_dim) and np.isfinite(actions).all() and np.abs(actions).max() <= 1, 'invalid clipped chunk')
            d = result['diagnostics']
            count, pick = self._q_contract['candidate_count'], self._q_contract['pick']
            require(d['candidate_count'] == count and d['candidate_shape'] == [count, self._native_horizon, config.action_dim]
                    and d['selection_mode'] == pick and d['selection_seed'] == self._q_contract['selection_seed']
                    and d['noise_namespace'] == self._q_contract['noise_namespace'] and d['noise_candidates_distinct']
                    and d['flowsteps'] == self._q_contract.get('flow_steps', self.current_policy_entry.settings['num_inference_steps'])
                    and d['remaining_horizon_mask'] == min(config.horizon,remaining), 'invalid candidate pool or horizon')
            if pick in Q_MODES:
                critic = self._q_contract['critic']
                require(d['scorer_backend'] == critic['backend'] and d['scorer_device'] == 'cuda:0'
                        and d['scorer_sha256'] == critic['sha256'] and d['critic_seed'] == critic['seed']
                        and d['critic_training_step'] == critic['training_step'], 'unexpected Q scorer')
            else:
                require(d['scores'] is None and d['scorer_backend'] is None, 'random selector used a scorer')
            if pick in MEAN_MODES:
                elites = self._q_contract['elite_count']
                temperature = self._q_contract['temperature']
                idx, weights = elite_weights(count, d['scores'], elites, temperature, self._selection_rng)
                require(d['sampling_recipe']=='shared_prefix_independent_noise' and d['elite_count']==elites
                        and d['temperature']==temperature and d['selected_indices']==idx.tolist() and d['weights']==weights.tolist()
                        and d['selected_action_sha256']==digest(actions)
                        and d['scored_horizon']==(config.horizon if pick in Q_MODES else None), 'invalid aggregate or scorer horizon')
            else:
                require(d['sampling_recipe'] == f'bc_anchor_single1_plus_independent{count-1}_v1'
                        and d['slot0_original_policy_infer'] and d['selected_action_matches_candidate'], 'invalid selected candidate')
                expected = np.argmax(d['scores']) if pick=='greedy' else self._selection_rng.integers(count)
                require(d['selected_index'] == expected, 'selected candidate differs')
            self._queries += 1
            self._trace(dict(event='infer', trial=self._trial, query=self._queries, remaining_steps=remaining,
                             composite_sha256=self._q_contract['composite_sha256'], **d))
            return actions

    def unload(self):
        super().unload()
        self._q_contract = None
        self._seeded, self._queries = False, 0

    def metadata(self):
        value = super().metadata()
        if self._q_contract is not None:
            value.update(inference_algorithm=algorithm(self._q_contract), candidate_count=self._q_contract['candidate_count'],
                         critic_sha256=(self._q_contract['critic'] or {}).get('sha256'),
                         runtime_composite_sha256=self._q_contract['composite_sha256'])
        return value
