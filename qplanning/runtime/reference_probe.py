"""Compare an explicitly identified Q checkpoint against a frozen set of candidate scores."""
import json
import math
from pathlib import Path

import numpy as np


def load_spec(contract):
    from .contract import require, sha
    binding = contract['smoke_config']
    path = Path(binding['path'])
    require(sha(path)==binding['sha256'], 'reference probe configuration changed')
    spec = json.loads(path.read_text())
    for asset in spec['assets'].values():
        asset['path'] = str((path.parent/asset['path']).resolve())
    return spec


def score_errors(predicted, expected):
    if predicted.shape != expected.shape or expected.ndim != 2 or not expected.size:
        raise ValueError('Reference and predicted matrices must have identical nonempty shapes')
    if not np.isfinite(expected).all() or not np.isfinite(predicted).all():
        raise ValueError('Invalid score value')
    error = np.abs(predicted-expected)
    return dict(shape=list(expected.shape),max_abs_error=float(error.max()),mean_abs_error=float(error.mean()),
                top1_agreement=float(np.mean(predicted.argmax(1)==expected.argmax(1))))


def check_limits(report, spec):
    from .contract import require
    limits = spec['limits']
    require(all(math.isfinite(v) for v in (limits['max_abs_error'],limits['mean_abs_error'],limits['min_top1_agreement'])),
            'invalid reference score tolerance')
    require(report['shape']==spec['score_shape'] and 0<=report['max_abs_error']<=limits['max_abs_error']
            and 0<=report['mean_abs_error']<=limits['mean_abs_error']
            and limits['min_top1_agreement']<=report['top1_agreement']<=1,
            f'reference score agreement failed: {report}')


def validate_reference(report,spec):
    from .contract import require
    require(report.get('device_type')=='cuda' and report.get('asset_sha256')==
            {name:a['sha256'] for name,a in spec['assets'].items()},'reference scorer assets or device differ')
    check_limits(report,spec)


def offline_scores(net, backbone, fixture, spec, device, mean, std, config):
    from ..features import encode
    from ..c_score import candidate_scores
    views = [fixture[key] for key in spec['view_fields']]
    if spec['flip_vertical']:
        views = [v[:, ::-1] for v in views]
    images = np.stack(views,axis=1)
    features = np.concatenate([encode(backbone,images[i:i+8],device,config) for i in range(0,len(images),8)])
    return candidate_scores(net,features,fixture[spec['candidate_field']],device,mean,std)


def run_reference(spec,device):
    import torch
    from transformers import Dinov2Model
    from ..config import QConfig, resolve_model_config
    from ..critic import Critic
    from .contract import require, sha
    assets=spec['assets']
    for asset in assets.values():
        require(sha(Path(asset['path']))==asset['sha256'],'reference asset checksum differs: '+asset['path'])
    config=QConfig.from_dict(spec['q_config'])
    weights=Path(assets['checkpoint']['path'])
    stats=json.loads(Path(assets['normalization']['path']).read_text())
    scores=json.loads(Path(assets['scores']['path']).read_text())
    for key in spec['score_field_path']:
        scores=scores[key]
    scores=np.asarray(scores,dtype=np.float64)
    require(list(scores.shape)==spec['score_shape'],'archived score matrix shape differs')
    dino_dir=Path(assets['backbone_weights']['path']).parent
    require(Path(assets['backbone_config']['path'])==dino_dir/'config.json'
            and Path(assets['backbone_weights']['path'])==dino_dir/'model.safetensors',
            'expected local backbone config.json and model.safetensors')
    with torch.inference_mode(),np.load(assets['fixture']['path'],allow_pickle=False) as fixture:
        if 'fixture_score_field' in spec:
            require(np.array_equal(scores,fixture[spec['fixture_score_field']]),'archived score rows or candidate order differ')
        backbone=Dinov2Model.from_pretrained(dino_dir,local_files_only=True,use_safetensors=True).float().eval().to(device)
        state=torch.load(weights,map_location='cpu',mmap=True,weights_only=True)
        resolve_model_config(state,config)
        with torch.device('meta'):
            net=Critic(config=config,**spec.get('architecture',{}))
        net.load_state_dict(state.get('model',state),strict=True,assign=True)
        net.requires_grad_(False).eval().to(device)
        observed=offline_scores(net,backbone,fixture,spec,device,stats['mean'],stats['std'],config)
    report=dict(device_type=device.type,asset_sha256={k:a['sha256'] for k,a in assets.items()},
                **score_errors(observed,scores))
    return report
