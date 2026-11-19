"""Durable scheduled checkpoints and their corresponding per-step journal."""
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .data import save_json, sha
from .train import set_states, torch_states

PERMANENT=(10000,20000,30000,40000)


def sync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read(path):
    return json.loads(path.read_text())


def line(row):
    return (json.dumps(row,allow_nan=False,separators=(',',':'))+'\n').encode()


def reconcile(folder,history):
    """The receipt commits the prefix. Archive any interrupted suffix before replay."""
    path=folder/'steps.jsonl'
    expected=b''.join(line(r) for r in history)
    old=path.read_bytes() if path.exists() else b''
    pending=[p for p in folder.glob('event_*.json') if int(p.stem.split('_')[1])>len(history)]
    if old==expected and not pending:
        return
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive=folder/'interruptions'/stamp
    archive.mkdir(parents=True)
    (archive/'previous_steps.jsonl').write_bytes(old)
    save_json(archive/'resume.json',dict(restored_step=len(history),previous_log_bytes=len(old)))
    for p in pending:
        p.rename(archive/p.name)
    temp=path.with_suffix('.repair')
    with temp.open('wb') as f:
        f.write(expected)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)
    sync_dir(folder)


def retain(folder):
    receipts=sorted(folder.glob('step_*.pt.json'))
    keep=set(PERMANENT)|{read(r)['step'] for r in receipts[-2:]}
    for r in receipts:
        entry=read(r)
        if entry['step'] not in keep:
            r.unlink()
            (folder/entry['file']).unlink(missing_ok=True)
    sync_dir(folder)


def checkpoint(folder,trainer,history,validation,seed,contract,cache_hash):
    step=trainer.updates
    assert len(history)==step
    path=folder/f'step_{step:05d}.pt'
    cpu,cuda=torch_states(trainer.device)
    payload=dict(format_version=1,q_config=trainer.config.to_dict(),seed=seed,step=step,contract=contract,cache_hash=cache_hash,
                 model=trainer.net.state_dict(),target=trainer.target.state_dict(),optimizer=trainer.optimizer.state_dict(),
                 rng=trainer.rng.bit_generator.state,torch_cpu=cpu,torch_cuda=cuda,python_rng=random.getstate(),
                 numpy_global=np.random.get_state(),history=history,validation=validation,
                 skipped=trainer.skipped,consecutive=trainer.consecutive,high_streak=trainer.high_streak)
    tmp=path.with_suffix('.partial')
    with tmp.open('wb') as f:
        torch.save(payload,f)
        f.flush()
        os.fsync(f.fileno())
    digest=sha(tmp)
    tmp.replace(path)
    sync_dir(folder)
    save_json(path.with_suffix('.pt.json'),dict(format_version=1,step=step,seed=seed,file=path.name,sha256=digest,contract=payload['contract']))
    retain(folder)


def load_cp(folder,step,contract,cache_hash):
    path=folder/f'step_{step:05d}.pt'
    rec=read(path.with_suffix('.pt.json'))
    if not (rec['file']==path.name and rec['format_version']==1 and rec['contract']==contract and sha(path)==rec['sha256']):
        raise ValueError(f'checkpoint identity mismatch: {path}')
    state=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    assert (state['format_version'],state['step'],state['seed'],state['contract'],state['cache_hash'])==(1,step,rec['seed'],rec['contract'],cache_hash)
    assert folder.name==f"seed_{state['seed']}"
    return state


def restore(trainer,state):
    if state['q_config'] != trainer.config.to_dict():
        raise ValueError('restored Q configuration differs from requested layout')
    trainer.net.load_state_dict(state['model'])
    trainer.target.load_state_dict(state['target'])
    trainer.optimizer.load_state_dict(state['optimizer'])
    trainer.rng.bit_generator.state=state['rng']
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_global'])
    set_states(trainer.device,(state['torch_cpu'],state['torch_cuda']))
    trainer.updates=state['step']
    trainer.skipped=state['skipped']
    trainer.consecutive=state['consecutive']
    trainer.high_streak=state['high_streak']
    records=state['history']
    assert [r['step'] for r in records]==list(range(1,trainer.updates+1))
    assert sum(r['outcome']=='SKIPPED' for r in records)==trainer.skipped
    assert not records or records[-1]['consecutive_skips']==trainer.consecutive
    return records,state['validation']
