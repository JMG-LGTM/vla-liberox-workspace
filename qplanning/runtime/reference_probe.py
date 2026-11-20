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
    if contract.get('include_archive_in_score_tolerance'):
        spec['score_max_abs_tolerance'] = contract['score_max_abs_tolerance']
        spec['elite_count'] = contract.get('elite_count')
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


def score_matrix(online, offline, shape, elite_count, tolerance, *, rank_singleton=False):
    """Return every state's independent comparison, including rows after a numerical failure."""
    from .planning import compare_scores
    online, offline = np.asarray(online), np.asarray(offline)
    if online.ndim != 2 or not online.size or online.shape != offline.shape or online.shape != tuple(shape):
        raise ValueError('Expected aligned nonempty score matrices of the declared shape')
    rows = [compare_scores(x, y, elite_count, tolerance, rank_singleton=rank_singleton)
            for x, y in zip(online, offline, strict=True)]
    failed = [i for i, row in enumerate(rows) if not row['within_tolerance']]
    maximum = mean = top1 = None
    if all(row['score_max_abs'] is not None for row in rows):
        maximum = max(row['score_max_abs'] for row in rows)
        mean = float(maximum * np.mean([row['score_mean_abs'] / maximum for row in rows])) if maximum else 0.
        if all(row['top1_same'] is not None for row in rows):
            top1 = float(np.mean([row['top1_same'] for row in rows]))
    return dict(shape=list(shape), tolerance=tolerance, within_tolerance=not failed,
                max_abs_error=maximum, mean_abs_error=mean, top1_agreement=top1,
                failed_rows=failed, first_failed_row=failed[0] if failed else None, score_comparisons=rows)


def require_matrix_pass(report, stage):
    from .planning import PlanningParityError
    if not report['within_tolerance']:
        index = report['first_failed_row']
        raise PlanningParityError(dict(stage=stage, fixture_row=index, shape=report['shape'],
                                       score_matrix=report, **report['score_comparisons'][index]))


def verify_score_matrix(report, shape, elite_count, tolerance, stage, *, expected=None):
    """Reconstruct the decision and all diagnostics from the recorded same-state scores."""
    from .contract import require
    rows = report.get('score_comparisons', [])
    require(len(rows) == shape[0] and shape[0] > 0, 'missing state-by-state score evidence')
    online = np.array([r['online_scores'] for r in rows], dtype=np.float64)
    offline = np.array([r['offline_scores'] for r in rows], dtype=np.float64)
    if expected is not None:
        require(np.array_equal(offline, expected), 'recorded reference scores differ from the sealed archive')
    rebuilt = score_matrix(online, offline, shape, elite_count, tolerance)
    require_matrix_pass(rebuilt, stage)
    require(all(key in report and report[key] == value for key,value in rebuilt.items()),
            'score evidence differs from recorded values or the sealed tolerance')


def archive_scores(spec):
    from .contract import require, sha
    asset = spec['assets']['scores']
    path = Path(asset['path'])
    require(sha(path) == asset['sha256'], 'archived scores checksum differs')
    scores = json.loads(path.read_text())
    for key in spec['score_field_path']:
        scores = scores[key]
    result = np.asarray(scores, dtype=np.float64)
    require(list(result.shape) == spec['score_shape'], 'archived score matrix shape differs')
    return result


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
    if 'score_max_abs_tolerance' in spec:
        verify_score_matrix(report, spec['score_shape'], spec.get('elite_count'),
                            spec['score_max_abs_tolerance'], 'archive_reference_scores', expected=archive_scores(spec))
    else:
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
    scores=archive_scores(spec)
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
    if 'score_max_abs_tolerance' in spec:
        measured=score_matrix(observed,scores,spec['score_shape'],spec.get('elite_count'),spec['score_max_abs_tolerance'])
    else:
        measured=score_errors(observed,scores)
    report=dict(device_type=device.type,asset_sha256={k:a['sha256'] for k,a in assets.items()},**measured)
    return report
