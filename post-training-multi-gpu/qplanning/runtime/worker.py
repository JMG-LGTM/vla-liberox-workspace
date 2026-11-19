#!/usr/bin/env python3
"""Load the platform actor and the configured candidate selector."""
from __future__ import annotations
import base64
import contextlib
import io
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

from ..config import QConfig
from .contract import algorithm, require, validate_contract


def load_runtime(request, *, device='cuda:0'):
    value = validate_contract(request['runtime_contract'], verify_payloads=True)
    snap = value['snapshot']
    require(request['policy_id'] == snap['policy_id'] == value['candidate_policy_id']
            and request['content_sha256'] == snap['content_sha256'], 'worker policy differs')
    require(request['runtime_composite_sha256'] == value['composite_sha256'], 'worker runtime composite differs')
    require(request['base_revision'] == snap['base_revision'] and request.get('actor') == snap.get('actor'),
            'worker base or actor differs from sealed snapshot')
    settings = request['model']
    require(settings['num_inference_steps'] == snap['model_config']['num_inference_steps'] and settings['base_checkpoint'] == snap['base_checkpoint'],
            'worker model settings differ')
    workspace = Path(value['workspace'])
    sys.path.insert(0,str(workspace/'vla-adapter-rynn-iql/src'))
    from types import SimpleNamespace
    from vla_rynn_iql import pi05
    from vla_rynn_iql.pi05_assets import identity_digest
    from .runtime_core import CandidateRuntime, QScorer
    config = SimpleNamespace(raw={'model': settings}, section=lambda key: {
        'training': {'device': device, 'dtype': 'bfloat16'}, 'model': settings}[key])
    components = pi05.load_components(config, training=False, parent=request.get('parent'))
    require(identity_digest(components.identity) == snap['base_revision'], 'loaded base differs')
    if request.get('actor'):
        actor = Path(request['actor'])
        require(actor.name == 'actor.pt' and request['actor_sha256'] == value['assets_sha256'][str(actor)],
                'actor checksum differs from validated platform catalog')
        pi05.restore_actor(components, settings, actor.parent)
    require(all(not p.requires_grad for p in components.model.parameters()), 'base policy must stay frozen')
    config = QConfig.from_dict(value['q_config'])
    scorer = QScorer(value['critic'], device, config) if value['pick']=='greedy' else None
    return CandidateRuntime(components, scorer, device=device, candidate_count=value['candidate_count'],
                            pick=value['pick'], selection_seed=value['selection_seed'], config=config,
                            flow_steps=settings['num_inference_steps'], noise_namespace=value['noise_namespace']), value


def main():
    logging.disable(logging.INFO)
    output = sys.stdout

    def reply(value):
        output.write('PI05_RPC '+json.dumps(value, allow_nan=False)+'\n')
        output.flush()
    request = json.loads(sys.stdin.readline())
    owner = request['owner_pid']

    def watch_owner():
        while True:
            time.sleep(2)
            try:
                os.kill(owner, 0)
            except ProcessLookupError:
                os._exit(1)
    threading.Thread(target=watch_owner, daemon=True).start()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            runtime, value = load_runtime(request)
        import numpy as np
        reply(dict(ready=True, identity=value['snapshot']['content_sha256'], runtime_composite_sha256=value['composite_sha256'],
                   inference_algorithm=algorithm(value), candidate_count=runtime.candidate_count,
                   native_horizon=runtime.native_horizon, noise_namespace=runtime.noise_namespace,
                   pick=runtime.pick, device=str(runtime.device), critic=value['critic']))
        seeded, queries = False, 0
        for line in sys.stdin:
            request = json.loads(line)
            if request['op'] == 'seed':
                runtime.reset_seed(int(request['seed']))
                seeded, queries = True, 0
                reply({'ok': True})
                continue
            require(request['op'] == 'infer' and seeded, 'inference requires a trial seed')
            remaining = runtime.config.max_steps - runtime.config.horizon * queries
            require(remaining > 0 and request['remaining_steps'] == remaining, 'worker and provider query counts differ')
            with np.load(io.BytesIO(base64.b64decode(request['arrays'])), allow_pickle=False) as arrays:
                observation = {key: arrays[key] for key in arrays.files}
            observation['prompt'] = request['prompt']
            with contextlib.redirect_stdout(sys.stderr):
                actions, diagnostic = runtime.infer(observation, remaining_steps=remaining)
            queries += 1
            reply(dict(actions=actions.tolist(), diagnostics=diagnostic,
                       runtime_composite_sha256=value['composite_sha256'], query_index=queries))
    except Exception as exc:
        reply({'error': f'{type(exc).__name__}: {exc}'})
        raise


if __name__ == '__main__':
    main()
