"""Frozen, pinned DINOv2-large features indexed by root and original time."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Dinov2Model

from .config import DEFAULT_CONFIG, QConfig
from .data import actions_for, paths, save_json, sha, success_in_episode, transitions

DINO_FILES = ('config.json','preprocessor_config.json','model.safetensors')


def feature_shape(config):
    return (len(config.camera_keys), 256, 1024)


def verify_dino(folder, receipt=None):
    """Identify the local DINO files and optionally verify an external provenance receipt."""
    files = {name:sha(Path(folder)/name) for name in DINO_FILES}
    revision = None
    if receipt is not None:
        pin = json.loads(Path(receipt).read_text())
        if pin['sha256'] != files:
            raise ValueError('DINO files differ from the supplied receipt')
        revision = pin.get('revision')
    return dict(sha256=files,revision=revision)


@torch.inference_mode()
def encode(backbone, images, device, config=DEFAULT_CONFIG):
    # Batch x configured raw-orientation RGB views x height x width x channels.
    if images.ndim!=5 or images.shape[1]!=len(config.camera_keys) or images.shape[-1]!=3:
        raise ValueError('image views do not match the Q configuration')
    x = torch.from_numpy(np.ascontiguousarray(images)).to(device).flatten(0,1).permute(0,3,1,2).float().div_(255)
    x = F.interpolate(x,size=(224,224),mode='bilinear',align_corners=False,antialias=True)
    mean = x.new_tensor([.485,.456,.406])[None,:,None,None]
    std = x.new_tensor([.229,.224,.225])[None,:,None,None]
    with sdpa_kernel(SDPBackend.MATH):
        tokens = backbone(pixel_values=(x-mean)/std).last_hidden_state[:,1:,:]
    y = tokens.reshape(len(images), *feature_shape(config)).half()
    if not torch.isfinite(y).all():
        raise ValueError('nonfinite DINO cache')
    return y.cpu().numpy()


def build_cache(folder, rows, dino_dir, device, config=DEFAULT_CONFIG, dino_identity=None):
    """Build the full split in a sibling staging directory, then publish its verified index."""
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError(folder)
    dino_identity = dino_identity or verify_dino(dino_dir)
    model = Dinov2Model.from_pretrained(dino_dir, local_files_only=True).requires_grad_(False).eval().to(device)
    folder.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=folder.parent, prefix=folder.name+'.partial-') as staging:
        payload = Path(staging)/'cache'
        payload.mkdir()
        records = []
        files = {}
        for row in rows:
            _, obs = paths(row)
            if sha(obs) != row['observations_sha256']:
                raise ValueError(f'observations hash mismatch: {obs}')
            n = len(actions_for(row,config))
            with np.load(obs, allow_pickle=False) as source:
                cameras = [source[k][:n] for k in config.camera_keys]
            if any(len(v)!=n for v in cameras):
                raise ValueError(f'missing observations: {obs}')
            name = row['run_id']+'.npy'
            path = payload/name
            arr = np.lib.format.open_memmap(path, mode='w+', dtype='float16', shape=(n,*feature_shape(config)))
            for i in range(0,n,12):
                arr[i:i+12] = encode(model,np.stack([v[i:i+12] for v in cameras],axis=1),device,config)
            arr.flush()
            del arr
            files[name] = sha(path)
            times = list(range(n))
            records.append(dict(id=row['run_id'], starts=times, frames=times, split=row['split'],
                                observations_sha256=row['observations_sha256']))
        size = sum((payload/name).stat().st_size for name in files)
        index = dict(dino=dino_identity, q_config=config.to_dict(), smoke=False, roots=records, frames=sum(len(r['frames']) for r in records),bytes=size)
        save_json(payload/'index.json',index)
        files['index.json'] = sha(payload/'index.json')
        save_json(payload/'sha256.json',dict(files=files))
        payload.rename(folder)


def audit_cache(folder, rows, receipt, config=DEFAULT_CONFIG, dino_identity=None):
    """Verify cache identity, source membership, on-disk bytes and full-frame order."""
    folder = Path(folder)
    index = json.loads((folder/'index.json').read_text())
    recorded = json.loads(Path(receipt).read_text())['files']
    cached_config = QConfig.from_dict(index['q_config']) if 'q_config' in index else DEFAULT_CONFIG
    if config!=cached_config:
        raise ValueError('feature cache Q configuration differs from requested layout')
    if dino_identity is not None:
        if 'dino' in index:
            if index['dino']['sha256'] != dino_identity['sha256']:
                raise ValueError('cache encoder identity differs from requested DINO files')
        elif not dino_identity.get('revision') or index.get('revision')!=dino_identity['revision']:
            raise ValueError('legacy feature cache requires its matching --dino-receipt')
    if index['smoke'] or [r['id'] for r in index['roots']] != [r['run_id'] for r in rows]:
        raise ValueError('cache/model/manifest membership mismatch')
    names = {'index.json', *(r['run_id']+'.npy' for r in rows)}
    if set(recorded)!=names:
        raise ValueError('cache hash receipt membership mismatch')
    total = 0
    for name in sorted(names):
        if sha(folder/name)!=recorded[name]:
            raise ValueError(f'cache SHA mismatch: {name}')
    for row, record in zip(rows,index['roots']):
        n = len(actions_for(row,config))
        full = list(range(n))
        if record['starts']!=full or record['frames']!=full or record['split']!=row['split']:
            raise ValueError(f'incorrect full cache index: {record["id"]}')
        if record.get('observations_sha256',row['observations_sha256']) != row['observations_sha256']:
            raise ValueError(f'cache observations differ for {record["id"]}')
        a = np.load(folder/(record['id']+'.npy'),mmap_mode='r',allow_pickle=False)
        if a.shape!=(n,*feature_shape(config)) or a.dtype!=np.float16:
            raise ValueError(f'invalid cached feature shape or dtype: {record["id"]}')
        total += n
    if total!=index['frames']:
        raise ValueError('cache frame count mismatch')
    return total


class Samples:
    def __init__(self, folder, split, rows, stats, config=DEFAULT_CONFIG):
        self.config=config
        self.roots=[]
        self.rows=[]
        self.last_draws=None
        self.members=[dict(row,label_success=success_in_episode(row,config)) for row in rows if row['split']==split]
        folder=Path(folder)
        index=json.loads((folder/'index.json').read_text())
        records=[r for r in index['roots'] if r['split']==split]
        if [r['id'] for r in records] != [r['run_id'] for r in self.members]:
            raise ValueError(f'cache/manifest order mismatch in {split}')
        for rec,row in zip(records,self.members):
            actions=actions_for(row,config)
            starts=np.asarray(rec['starts'])
            w=transitions(actions,row['label_success'],starts,np.array(stats['mean']),np.array(stats['std']),config)
            lookup={step:i for i,step in enumerate(rec['frames'])}
            w['frame']=np.array([lookup[t] for t in starts])
            w['follow']=np.array([lookup[t] if t>=0 else -1 for t in w['next_index']])
            feature=np.load(folder/(rec['id']+'.npy'),mmap_mode='r',allow_pickle=False)
            self.rows.extend((len(self.roots),i) for i in range(len(starts)))
            self.roots.append((feature,w))

    def batch(self, rng, device, size=48):
        draws=rng.integers(0,len(self.rows),size)
        self.last_draws=draws
        batch={k:[] for k in ('features','action','mask','reward','bootstrap','next_features','next_action','next_mask')}
        for draw in draws:
            root,i=self.rows[draw]
            feature,w=self.roots[root]
            batch['features'].append(feature[w['frame'][i]])
            if w['bootstrap'][i]:
                batch['next_features'].append(feature[w['follow'][i]])
            for k in ('action','mask','reward','bootstrap','next_action','next_mask'):
                batch[k].append(w[k][i])
        return {k:torch.from_numpy(np.stack(v) if v else np.empty((0,*feature_shape(self.config)),np.float16)).to(device)
                for k,v in batch.items()}
