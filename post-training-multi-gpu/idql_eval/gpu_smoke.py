"""GPU check before an evaluation: loads the BC (GPU) and Q (CPU), runs 8 decisions on a saved observation.

Checks that candidate 0 equals the plain BC output, that the executed chunk is the argmax of min(Q1, Q2),
and the 8/6-step masks. No simulator rollout."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
from contract import POLICY_ID,STEP,validate_contract,require,atomic_json

PROMPT='place the green bowl onto the plate that is in the table center and place the black bowl onto the plate that is next to the blue bowl'

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--runtime-contract',type=Path,required=True);p.add_argument('--observation-npz',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
 require(not a.output.exists(),'preserve real probe attempts')
 c=validate_contract(a.runtime_contract)
 import yaml
 raw=yaml.safe_load((Path(c['actor']['path']).parent/'policy.yaml').read_text())
 request=dict(runtime_contract=str(a.runtime_contract),policy_id=POLICY_ID,content_sha256=c['policy_content_sha256'],
  actor=c['actor']['path'],actor_sha256=c['actor']['sha256'],base_revision=c['actor']['base_revision'],
  runtime_composite_sha256=c['composite_sha256'],model=raw['model_config'],parent=raw.get('parent'))
 from idql_worker import load_runtime
 runtime,_=load_runtime(request,device='cuda:0')
 import torch
 from openpi.policies.policy import Policy
 require(runtime.device.type=='cuda' and torch.cuda.is_initialized(),'true GPU actor inference required')
 with np.load(a.observation_npz,allow_pickle=False) as data:observation={k:data[k] for k in data.files}
 observation['prompt']=PROMPT
 baseline=Policy(runtime.components.model,transforms=[runtime.components.input_transform],
  output_transforms=[runtime.components.output_transform],is_pytorch=True,pytorch_device='cuda:0',sample_kwargs={'num_steps':10})
 remaining=(950,942,934,6);diagnostics=[];checks=[];original_sample=runtime.components.model.sample_noise
 for seed in (0,3):
  baseline_noise=[]
  def capture_noise(shape,device):
   noise=original_sample(shape,device);baseline_noise.append(noise.detach().float().cpu().numpy().copy());return noise
  runtime.components.model.sample_noise=capture_noise
  try:
   with torch.random.fork_rng(devices=[runtime.device.index or 0]):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    expected=[np.clip(np.asarray(baseline.infer(observation)['actions'],dtype=np.float32),-1.,1.) for _ in remaining]
  finally:runtime.components.model.sample_noise=original_sample
  require(len(baseline_noise)==len(remaining),'BC default stream not observed')
  runtime.reset_seed(seed);captured=[];original_pool=runtime.generate_pool
  def capture_pool(obs):
   actions,noise=original_pool(obs);captured.append((actions.copy(),noise.copy()));return actions,noise
  runtime.generate_pool=capture_pool
  try:
   for query,length in enumerate(remaining):
    selected,d=runtime.infer(observation,remaining_steps=length)
    pool,noise=captured[-1];index=d['selected_index'];q1,q2,scores=(np.asarray(d[k]) for k in ('q1','q2','scores'))
    require(np.array_equal(noise[0],baseline_noise[query][0]),'anchor noise differs from BC default stream')
    require(np.array_equal(pool[0],expected[query]),'anchor actions differ from original BC')
    require(np.array_equal(selected,pool[index]),'executed actions differ from scored candidate')
    require(pool.shape==(4,10,7) and noise.shape==(4,10,32) and len({n.tobytes() for n in noise})==4,'candidate pool/noise differs')
    require(np.isfinite(pool).all() and np.abs(pool).max()<=1.,'clipped candidates invalid')
    require(np.array_equal(scores,np.minimum(q1,q2)) and index==int(np.argmax(scores)),'target minimum or argmax differs')
    require(d['remaining_horizon_mask']==min(8,length),'H8/H6 mask differs')
    require(d['scorer_backend']=='official_idql_jax' and d['scorer_device']=='cpu' and d['critic_training_step']==STEP,'wrong scorer backend')
    require(d['selected_action_matches_candidate'] and d['scored_action_is_executed_clipped'],'selection execution check failed')
    diagnostics.append(d);checks.append(dict(seed=seed,query=query,noise_exact=True,actions_exact=True))
  finally:runtime.generate_pool=original_pool
 require(all(not p.requires_grad for p in runtime.components.model.parameters()),'actor is trainable')
 proof=dict(status='PASS_OFFICIAL_IDQL_N4_TARGET_MIN_Q_GPU_BC_SMOKE',observed_at=datetime.now(timezone.utc).isoformat(),
  policy_id=POLICY_ID,policy_content_sha256=c['policy_content_sha256'],composite_sha256=c['composite_sha256'],
  candidate_count=4,flowsteps=10,target_reduction='min',selection_mode='greedy',selected_index_matches_argmax=True,
  selected_action_matches_candidate=True,scored_action_is_executed_clipped=True,remaining_horizon_masks=[8,6],
  device='cuda',actor_device='cuda',scorer_backend='official_idql_jax',scorer_device='cpu',critic_training_step=STEP,
  scorer_sha256=c['critic']['target_q_sha256'],n1_bc_transform_equivalence=True,bc0_default_policy_multiseed_multiquery_equivalence=True,
  code_sha256=c['code_sha256'],input_source=str(a.observation_npz),simulator_rollout_executed=False,
  runtime_diagnostics=diagnostics,bc_checks=checks,latency_ms=float(np.median([d['total_ms'] for d in diagnostics])),
  peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
 atomic_json(a.output,proof);print(json.dumps({k:v for k,v in proof.items() if k not in ('runtime_diagnostics','bc_checks','code_sha256')}))

if __name__=='__main__':main()
