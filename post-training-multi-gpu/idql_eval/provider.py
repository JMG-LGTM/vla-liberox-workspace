"""Platform policy provider for IDQL, swapped in only inside run_idql_evaluation.py's own process.

The configured BC policy ID is served by idql_worker.py (BC candidates + Q selection); every other policy ID
keeps the platform's original pi05 worker. No platform file is changed.
"""
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
from contract import HERE, POLICY_ID, STEP, validate_contract, require


class IDQLProvider(OriginalProvider):
    def __init__(self, runtime, eval_config, catalog, *, contract_path, trace_path):
        super().__init__(runtime, eval_config, catalog)
        self.contract_path, self.trace_path = Path(contract_path), Path(trace_path)
        self._idql_contract = None
        self._seeded, self._queries, self._trial = False, 0, -1

    def load(self, open_loop_steps, policy_id='pi05-libero-base', *, expected_content_sha256=None):
        if policy_id != POLICY_ID:
            if self._idql_contract is not None: self.unload()
            return super().load(open_loop_steps, policy_id, expected_content_sha256=expected_content_sha256)
        with self._lock:
            value = validate_contract(self.contract_path)
            require(open_loop_steps == 8, 'IDQL requires unchanged open_loop8')
            entry = self.catalog.select(policy_id)
            require(entry.family == 'pi05' and entry.content_sha256 == value['policy_content_sha256']
                and (expected_content_sha256 is None or expected_content_sha256 == entry.content_sha256), 'IDQL catalog identity differs')
            require(str(entry.actor) == value['actor']['path'] and entry.component_sha256['actor'] == value['actor']['sha256'], 'IDQL BC actor differs')
            # Same actor/catalog content alone must never reuse an ordinary BC worker.
            if (self.loaded and self.current_policy_id == policy_id and self._idql_contract is not None
                    and self._idql_contract['composite_sha256'] == value['composite_sha256']):
                return
            self.unload(); self.responses = queue.Queue()
            self.process = subprocess.Popen(['conda', 'run', '--no-capture-output', '-n', entry.settings['environment'],
                'python', '-B', '-u', str(HERE/'idql_worker.py')], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True, bufsize=1, start_new_session=True)
            process, responses = self.process, self.responses
            def receive():
                try:
                    for line in process.stdout:
                        if line.startswith('PI05_RPC '): responses.put(json.loads(line[len('PI05_RPC '):]))
                finally: responses.put(None)
            threading.Thread(target=receive, daemon=True).start()
            try:
                ready = self._send(dict(model=entry.settings, owner_pid=os.getpid(), parent=entry.parent,
                    base_revision=entry.base_revision, content_sha256=entry.content_sha256,
                    actor=str(entry.actor), actor_sha256=entry.component_sha256['actor'], policy_id=policy_id,
                    runtime_contract=str(self.contract_path), runtime_composite_sha256=value['composite_sha256']), timeout=600)
                require(ready.get('identity') == entry.content_sha256
                    and ready.get('runtime_composite_sha256') == value['composite_sha256']
                    and ready.get('inference_algorithm') == 'official_idql_jax_max' and ready.get('candidate_count') == 4,
                    'IDQL worker did not prove composite initialization')
                require(ready.get('scorer_backend') == 'official_idql_jax'
                    and ready.get('scorer_device') == 'cpu' and ready.get('critic_training_step') == STEP,
                    'Worker did not load the selected official scorer')
                require(self.catalog.select(policy_id).content_sha256 == entry.content_sha256, 'IDQL catalog changed during load')
                self.current_policy_entry, self.current_policy_id = entry, policy_id
                self._idql_contract = value
            except BaseException:
                self.unload(); raise

    def _trace(self, value):
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open('a') as handle:
            handle.write(json.dumps(value, allow_nan=False)+'\n'); handle.flush()

    def seed(self, seed):
        if self._idql_contract is None: return super().seed(seed)
        super().seed(seed)
        self._seeded, self._queries, self._trial = True, 0, self._trial+1
        self._trace(dict(event='seed', seed=int(seed), trial=self._trial,
            composite_sha256=self._idql_contract['composite_sha256']))

    def predict(self, observation, prompt, disabled_policy_cameras=()):
        if self._idql_contract is None: return super().predict(observation, prompt, disabled_policy_cameras)
        with self._lock:
            require(self.loaded and self._seeded, 'IDQL prediction requires an owned loaded worker and trial seed')
            require(not disabled_policy_cameras, 'IDQL protocol has no disabled cameras')
            remaining = 950-8*self._queries
            require(remaining > 0, 'Unexpected extra IDQL query beyond950 actions')
            from trajectory_utils import quaternion_to_axis_angle
            contract = model_module('base_models').model_contract(self.current_policy_entry.settings)['io']
            transforms = model_module('image_transforms')
            orientation = transforms.image_orientation(contract)
            require(orientation == 'flip_vertical', 'BC image orientation differs')
            images = {name: transforms.orient_image(observation[key], orientation) for key, name in (
                ('agentview_image', 'observation/image'), ('robot0_eye_in_hand_image', 'observation/wrist_image'))}
            images['observation/state'] = np.concatenate([observation['robot0_eef_pos'],
                quaternion_to_axis_angle(observation['robot0_eef_quat']), observation['robot0_gripper_qpos']]).astype(np.float32)
            buffer = io.BytesIO(); np.savez(buffer, **images)
            result = self._send(dict(op='infer', arrays=base64.b64encode(buffer.getvalue()).decode(),
                prompt=prompt, remaining_steps=remaining))
            require(result.get('runtime_composite_sha256') == self._idql_contract['composite_sha256']
                and result.get('query_index') == self._queries+1, 'IDQL worker query/composite changed')
            actions = np.asarray(result['actions'], dtype=np.float32)
            require(actions.shape == (10, 7) and np.isfinite(actions).all() and np.abs(actions).max() <= 1,
                'IDQL returned invalid clipped execution chunk')
            diagnostic = result['diagnostics']
            require(diagnostic.get('scorer_backend') == 'official_idql_jax'
                and diagnostic.get('scorer_device') == 'cpu' and diagnostic.get('critic_training_step') == STEP
                and diagnostic.get('sampling_recipe') == 'bc_anchor_single1_plus_independent3_v1'
                and diagnostic.get('slot0_original_policy_infer') is True,
                'Official scorer or unchanged BC anchor evidence differs')
            require(diagnostic['candidate_count'] == 4 and diagnostic['candidate_shape'] == [4,10,7]
                and diagnostic['flowsteps'] == 10 and diagnostic['target_reduction'] == 'min'
                and diagnostic['remaining_horizon_mask'] == min(8,remaining)
                and all(diagnostic.get(key) is True for key in ('target_min_matches', 'selected_index_matches_argmax',
                    'selected_action_matches_candidate', 'scored_action_is_executed_clipped')), 'Incomplete IDQL selection evidence')
            self._queries += 1
            self._trace(dict(event='infer', trial=self._trial, query=self._queries, remaining_steps=remaining,
                composite_sha256=self._idql_contract['composite_sha256'], **diagnostic))
            return actions

    def unload(self):
        super().unload()
        self._idql_contract = None
        self._seeded, self._queries = False, 0

    def metadata(self):
        value = super().metadata()
        if self._idql_contract is not None:
            value.update(inference_algorithm='official_idql_jax_max', candidate_count=4,
                critic_sha256=self._idql_contract['critic']['sha256'],
                runtime_composite_sha256=self._idql_contract['composite_sha256'])
        return value
