"""Train a Q critic on the original roots and split of a rollout manifest."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .checkpoints import checkpoint, line, load_cp, read, reconcile, restore
from .data import action_stats, members, paths, save_json, sha
from .config import load_config
from .features import Samples, audit_cache, build_cache, verify_dino
from .train import GRAD_CLIP, GRAD_HIGH, GRAD_SPIKE, HIGH_STREAK, Trainer

STEPS = 40_000


@contextmanager
def platform_lock(device, path):
    """Hold the existing platform flock for the entire CUDA run."""
    if device.type == 'cpu':
        yield
    else:
        if path is None:
            raise ValueError('CUDA requires --gpu-lock pointing to the existing platform GPU lock')
        with path.open('rb') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield


def write_once(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f'run input or contract changed: {path}')
    else:
        save_json(path, value)


def prepare(args, device, config):
    rows = members(args.manifest)
    for row in rows:
        _, obs = paths(row)
        if sha(obs) != row['observations_sha256']:
            raise ValueError(f'observations hash mismatch: {obs}')
    dino_identity = verify_dino(args.dino_dir,args.dino_receipt)
    stats = action_stats(rows,config)
    if not args.cache.exists():
        build_cache(args.cache, rows, args.dino_dir, device, config, dino_identity)
    receipt = args.cache_receipt or args.cache/'sha256.json'
    audit_cache(args.cache, rows, receipt,config,dino_identity)
    return rows, stats, receipt,dino_identity


def telemetry(history, validation):
    losses = np.asarray([r['loss'] for r in history])
    windows = [dict(step=r['step'], train=float(losses[r['step']-500:r['step']].mean()), val=r['loss'])
               for r in validation]
    return dict(history=windows, q_range=[min(r['q_bounds'][0] for r in history),max(r['q_bounds'][1] for r in history)],
                clipped_steps=sum(bool(r.get('clipped')) for r in history),
                skipped_steps=[r['step'] for r in history if r['outcome']=='SKIPPED'])


def contract_for(manifest, receipt, stats_path, seed, validation_seed, config, dino_identity):
    package = Path(__file__).resolve().parent
    code = {f:sha(package/f) for f in ('train.py','critic.py','data.py','features.py','checkpoints.py','fit.py','config.py','defaults.json')}
    return dict(method='offline_q_planning',seed=seed,validation_seed=validation_seed,manifest_sha256=sha(manifest),cache_sha256=sha(receipt),
                normalization_sha256=sha(stats_path),dino=dino_identity,
                q_config=config.to_dict(),block_gamma=.99,reward_success=1,ema_tau=.005,bins=101,support=[-.01,1.01],
                sigma=.0075,layers=18,dim=1024,heads=16,ff=4096,batch=48,steps=STEPS,
                learning_rate=3e-4,warmup=2000,lr_schedule='cosine to 1e-6',grad_clip=GRAD_CLIP,
                raw_gradient_guards=[GRAD_SPIKE,GRAD_HIGH,HIGH_STREAK],source_sha256=code)


def train_one(args, device, rows, stats, receipt, config, dino_identity):
    folder = args.output/f'seed_{args.seed}'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'stopped.json').exists():
        raise RuntimeError('a recorded training stop requires review')
    stats_path = folder/'train_action_stats.json'
    write_once(stats_path,stats)
    contract_file = folder/'protocol.json'
    write_once(contract_file,contract_for(args.manifest,receipt,stats_path,args.seed,args.validation_seed,config,dino_identity))
    contract,cache_hash = sha(contract_file),sha(receipt)
    if (folder/'complete.json').exists():
        if read(folder/'complete.json')['contract'] != contract:
            raise ValueError('completed run contract mismatch')
        return
    for path in folder.glob('event_*.json'):
        if read(path).get('stop_reason'):
            raise RuntimeError(f'training already reached a stop condition: {path}')
    speed_file = folder/'speed.json'
    if speed_file.exists() and not read(speed_file)['passed']:
        raise RuntimeError('recorded full-cache throughput gate failed')
    train = Samples(args.cache,'train',rows,stats,config)
    val = Samples(args.cache,'val',rows,stats,config)
    trainer = Trainer(device,18,args.seed,config,args.validation_seed)
    receipts = sorted(folder.glob('step_*.pt.json'))
    history,validation = [],[]
    if receipts:
        step = read(receipts[-1])['step']
        state = load_cp(folder,step,contract,cache_hash)
        history,validation = restore(trainer,state)
        del state
        reconcile(folder,history)
    else:
        if (folder/'steps.jsonl').exists():
            raise RuntimeError('training journal exists without its checkpoint')
        checkpoint(folder,trainer,history,validation,args.seed,contract,cache_hash)

    def record_event(event):
        save_json(folder/f"event_{event['step']:05d}.json",event)

    if device.type=='cuda':
        torch.cuda.synchronize(device)
    start = time.monotonic()
    with (folder/'steps.jsonl').open('ab', buffering=0) as journal:
        for t in range(trainer.updates+1,STEPS+1):
            row = trainer.step(train,record_event)
            journal.write(line(row))
            if row.get('stop_reason'):
                os.fsync(journal.fileno())
                save_json(folder/'stopped.json',row)
                raise FloatingPointError(f"seed {args.seed}, step {t}: {row['stop_reason']}")
            history.append(row)
            if t==200 and not speed_file.exists():
                if device.type=='cuda':
                    torch.cuda.synchronize(device)
                elapsed = time.monotonic()-start
                speed = 200/elapsed
                passed = device.type!='cuda' or speed>=2
                save_json(speed_file,dict(steps=200,device=str(device),elapsed_seconds=elapsed,steps_per_second=speed,passed=passed))
                if not passed:
                    raise RuntimeError(f'full-cache throughput {speed:.3f} < 2 steps/s')
            if t%500==0:
                result = dict(step=t,**trainer.validation(val))
                validation.append(result)
                save_json(folder/'validation.json',validation)
                print(f"seed {args.seed} step {t} train {row['loss']:.5f} val {result['loss']:.5f}",flush=True)
            if t%2000==0:
                os.fsync(journal.fileno())
                checkpoint(folder,trainer,history,validation,args.seed,contract,cache_hash)
    audit_cache(args.cache,rows,receipt,config,dino_identity)
    from .c_score import validation_diagnostic
    scores = validation_diagnostic(trainer.net,val,val.members,device)
    destination = folder/f'q_seed{args.seed}_step{STEPS}.pt'
    temporary = destination.with_suffix('.partial')
    torch.save(dict(model=trainer.net.state_dict(),q_config=config.to_dict()),temporary)
    temporary.replace(destination)
    save_json(folder/'complete.json',dict(contract=contract,cache_hash=cache_hash,step=STEPS,
              seed=args.seed,curves=telemetry(history,validation),validation=scores,
              model_file=destination.name,model_sha256=sha(destination)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('manifest','dino-dir','cache','output'):
        parser.add_argument('--'+arg,type=Path,required=True)
    parser.add_argument('--config',type=Path,help='Shared rollout, camera and action layout JSON')
    parser.add_argument('--dino-receipt',type=Path,help='DINO provenance receipt for verification and legacy caches')
    parser.add_argument('--validation-seed',type=int,default=0)
    parser.add_argument('--cache-receipt',type=Path,help='Existing SHA256 file for a separately archived cache')
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--gpu-lock',type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    args.output.mkdir(parents=True,exist_ok=True)
    # Prevent two writers from concurrently repairing or appending to the same seed.
    with platform_lock(device,args.gpu_lock), (args.output/f'.seed_{args.seed}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.output/f'seed_{args.seed}'/'stopped.json').exists():
            raise RuntimeError('a recorded training stop requires review')
        config = load_config(args.config)
        rows,stats,receipt,dino_identity = prepare(args,device,config)
        train_one(args,device,rows,stats,receipt,config,dino_identity)


if __name__=='__main__':
    main()
