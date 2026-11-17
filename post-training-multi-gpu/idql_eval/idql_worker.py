#!/usr/bin/env python3
"""Frozen Torch-CUDA BC candidate actor with the actual official JAX-CPU scorer."""
from __future__ import annotations
import base64
import contextlib
import copy
import io
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# This restricts JAX only. Do not hide CUDA from the unchanged Torch BC actor.
os.environ['JAX_PLATFORMS']='cpu'
os.environ['JAX_PLATFORM_NAME']='cpu'

from contract import (HERE, REPOSITORY, PROJECT, DEPENDENCIES, ACTOR_SHA, REVISION, POLICY_ID,
    SOURCE_COMMIT, UPSTREAM, STEP, NETWORK, CRITIC_UPDATES, validate_contract, require, canonical, sha)

# A private, hash-bound Gym dependency scope; do not mutate the pi05 environment.
sys.path.insert(0,str(DEPENDENCIES))


def load_official_scorer(critic):
    """Load actual target-only bytes with the ORIGINAL immutable training contract.

    A single-CPU architecture template changes execution placement only. It is
    not a new training contract, a resumed optimizer, or a Torch translation.
    """
    sys.path[:0]=[str(REPOSITORY/'src'),str(DEPENDENCIES)]
    require(critic['backend']=='official_idql_jax' and critic['upstream_commit']==UPSTREAM
        and critic['training_step']==STEP and critic['network']==NETWORK and critic['source_commit']==SOURCE_COMMIT,'Official scorer identity differs')
    directory=Path(critic['scorer_path'])
    require(sha(directory/'manifest.json')==critic['manifest_sha256']
        and sha(directory/'target_q.msgpack')==critic['target_q_sha256']==critic['sha256']
        and sha(directory/'scorer.json')==critic['scorer_json_sha256'],'Official scorer bytes differ')
    metadata=json.loads((directory/'scorer.json').read_text());training=metadata['contract']
    require(metadata['step']==STEP and metadata['backend']=='official_idql_jax' and metadata['upstream_commit']==UPSTREAM
        and metadata['network']==NETWORK and metadata['actor_loaded'] is False
        and metadata['actor_updates']==0 and metadata['resume_capable'] is False,'Actual scorer metadata differs')
    require(canonical(training)==critic['training_contract_sha256']
        and training['config']['training']['optimizer_updates']==CRITIC_UPDATES
        and training['config']['frozen_behavior']['actor_sha256']==ACTOR_SHA,'Original training contract/BC binding differs')
    for name,digest in training['source_sha256'].items():
        require(sha(REPOSITORY/'src/post_training'/name)==digest,'Scoring interpreter differs: '+name)
    normalization=json.loads(Path(critic['normalization_path']).read_text())
    require(sha(critic['normalization_path'])==critic['normalization_file_sha256']
        and canonical(normalization)==critic['normalization_sha256']
        and normalization==metadata['normalization']==training['prepared']['normalization'],'Scorer normalization differs')
    import jax
    import numpy as np
    from types import SimpleNamespace
    from post_training.official_idql.runner import create_state
    from post_training.official_idql.scoring import load_scorer
    devices=jax.local_devices(backend='cpu')
    require(devices and all(device.platform=='cpu' for device in jax.devices()),'Official Q must use JAX CPU exclusively')
    adapter_contract=metadata['adapter_contract'];pixels=training['config']['adapter']['pixels']
    require(adapter_contract==training['prepared']['adapter_contract'] and adapter_contract['pixel_keys']==['agent_pixels','wrist_pixels']
        and adapter_contract['observation_kind']=='pixels' and pixels['camera_layout']=='separate_rgb_views'
        and training['config']['data']['image_size']==128,'Official pixel geometry differs')
    adapter=SimpleNamespace(contract=adapter_contract,batch=lambda indices:dict(
        observations={'agent_pixels':np.zeros((1,128,128,3),np.uint8),
                      'wrist_pixels':np.zeros((1,128,128,3),np.uint8),'state':np.zeros((1,8),np.float32)},
        actions=np.zeros((1,64),np.float32)))
    template_config=copy.deepcopy(training['config'])
    template_config['runtime']={'platform':'cpu','device_count':1}
    template=create_state(template_config,adapter,devices=[devices[0]])
    scorer=load_scorer(directory,template,training,expected_adapter_contract=adapter_contract)
    require(scorer.metadata==metadata,'Loaded official scorer metadata differs')
    require(all(device.platform=='cpu' for leaf in jax.tree_util.tree_leaves(scorer._state.target_critic.params)
        for device in getattr(leaf,'devices',lambda: {devices[0]})()),'Target Q arrays left CPU')
    return scorer,metadata


def load_runtime(request, *, device='cuda:0'):
    """Only root's actual GPU smoke or the opted-in provider loads the actor."""
    value=validate_contract(request['runtime_contract'],verify_payloads=True)
    require(request['policy_id']==value['policy_id']==POLICY_ID
        and request['content_sha256']==value['policy_content_sha256'],'Worker policy/catalog differs')
    require(request['runtime_composite_sha256']==value['composite_sha256'],'Worker runtime composite differs')
    require(request['actor']==value['actor']['path'] and request['actor_sha256']==ACTOR_SHA
        and request['base_revision']==REVISION,'Worker frozen actor/base differs')
    settings=request['model']
    require(settings['num_inference_steps']==10 and settings['backbone']=='full'
        and settings['base_checkpoint']==value['actor']['base_checkpoint'],'Worker BC settings differ')
    sys.path[:0]=[str(PROJECT/'src'),str(REPOSITORY/'src'),str(HERE)]
    from types import SimpleNamespace
    from vla_rynn_iql import pi05
    from vla_rynn_iql.pi05_assets import identity_digest
    from runtime_core import CandidateRuntime,SAMPLING
    require(value['selector']['sampling']==SAMPLING and value['selector']['mode']=='greedy','Candidate sampling contract differs')
    config=SimpleNamespace(raw={'model':settings},section=lambda key:{
        'training':{'device':device,'dtype':'bfloat16'},'model':settings}[key])
    components=pi05.load_components(config,training=False,parent=request.get('parent'))
    require(identity_digest(components.identity)==REVISION,'Loaded BC base differs')
    pi05.restore_actor(components,settings,Path(request['actor']).parent)
    scorer,metadata=load_official_scorer(value['critic'])
    require(all(not p.requires_grad for p in components.model.parameters()),'BC actor must remain fully frozen')
    return CandidateRuntime(components,scorer,metadata['normalization'],expected_step=STEP,device=device,scorer_sha256=value['critic']['target_q_sha256']),value


def main():
    logging.disable(logging.INFO);output=sys.stdout
    def reply(value):output.write('PI05_RPC '+json.dumps(value,allow_nan=False)+'\n');output.flush()
    request=json.loads(sys.stdin.readline());owner=request['owner_pid']
    def watch_owner():
        while True:
            time.sleep(2)
            try:os.kill(owner,0)
            except ProcessLookupError:os._exit(1)
    threading.Thread(target=watch_owner,daemon=True).start()
    try:
        with contextlib.redirect_stdout(sys.stderr):runtime,value=load_runtime(request)
        import numpy as np
        reply(dict(ready=True,identity=value['policy_content_sha256'],runtime_composite_sha256=value['composite_sha256'],
            inference_algorithm='official_idql_jax_max',candidate_count=4,scorer_backend='official_idql_jax',
            scorer_device='cpu',critic_training_step=STEP))
        seeded,queries=False,0
        for line in sys.stdin:
            request=json.loads(line)
            if request['op']=='seed':
                runtime.reset_seed(int(request['seed']));seeded,queries=True,0;reply({'ok':True});continue
            require(request['op']=='infer' and seeded,'Official IDQL inference requires a trial seed')
            # 950-step episode, 8 executed steps per query: the protocol in contract.py.
            remaining=950-8*queries
            require(remaining>0 and request['remaining_steps']==remaining,'Worker/provider query count differs')
            with np.load(io.BytesIO(base64.b64decode(request['arrays'])),allow_pickle=False) as arrays:
                observation={key:arrays[key] for key in arrays.files}
            observation['prompt']=request['prompt']
            with contextlib.redirect_stdout(sys.stderr):actions,diagnostic=runtime.infer(observation,remaining_steps=remaining)
            queries+=1
            reply(dict(actions=actions.tolist(),diagnostics=diagnostic,runtime_composite_sha256=value['composite_sha256'],query_index=queries))
    except Exception as exc:
        reply({'error':f'{type(exc).__name__}: {exc}'});raise


if __name__=='__main__':main()
