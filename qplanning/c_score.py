"""Score actual action chunks with the frozen Q critic and report within-state variation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from scipy.stats import spearmanr

from .config import load_config, resolve_model_config
from .critic import Critic, expectation
from .metrics import average_metrics


@torch.inference_mode()
def predict(net,f,a,mask,device):
    net.eval()
    vals=[]
    for i in range(0,len(a),24):
        x=[torch.from_numpy(np.ascontiguousarray(v[i:i+24])).to(device) for v in (f,a,mask)]
        with sdpa_kernel(SDPBackend.MATH), torch.autocast(torch.device(device).type,dtype=torch.bfloat16):
            q=expectation(net(x[0].float(),x[1],x[2]))
        vals.append(q.cpu().numpy())
    scores=np.concatenate(vals)
    if not np.isfinite(scores).all() or scores.min()<-.01 or scores.max()>1.01:
        raise ValueError('invalid held-out Q')
    return scores


def candidate_scores(net,features,actions,device,mean,std):
    """Score N raw action chunks per state using training-set normalization."""
    config = net.config
    mean, std = np.asarray(mean,dtype=np.float64), np.asarray(std,dtype=np.float64)
    if mean.shape != (config.action_dim,) or std.shape != (config.action_dim,) or not np.isfinite(mean).all() or not np.all(np.isfinite(std) & (std>0)):
        raise ValueError('invalid training-set action normalization')
    if actions.ndim != 4 or actions.shape[2:] != (config.horizon,config.action_dim) or min(actions.shape[:2]) < 1 or len(features) != len(actions):
        raise ValueError('expected state-aligned actions [states, candidates, configured horizon, action dimension]')
    out=[]
    for begin in range(0,len(features),4):
        raw=actions[begin:begin+4]
        a=((raw-mean)/std).reshape(-1,config.horizon,config.action_dim).astype('float32')
        f=np.repeat(features[begin:begin+4],raw.shape[1],axis=0)
        out.append(predict(net,f,a,np.ones(a.shape[:2],bool),device).reshape(len(raw),raw.shape[1]))
    return np.concatenate(out)


def score_dispersion(scores):
    scores = np.asarray(scores, dtype=np.float64)
    within = float(scores.std(axis=1).mean())
    across = float(scores.mean(axis=1).std())
    return dict(within_state_std=within, across_state_std=across,
                ratio=within / across if across > 0 else None)


def validation_diagnostic(net, dataset, records, device):
    """Aggregate Q by held-out root, comparing success and failure with equal root weights."""
    roots = []
    for record, (feature, windows) in zip(records, dataset.roots, strict=True):
        values = []
        for start in range(0, len(windows['action']), 24):
            s = slice(start,start+24)
            values.extend(predict(net,feature[windows['frame'][s]],windows['action'][s],windows['mask'][s],device))
        scores = np.array(values)
        quartile = max(len(scores)//4, 1)
        roots.append(dict(id=record['run_id'], success=record['label_success'], mean_q=float(scores.mean()),
                          spearman_time_q=float(spearmanr(np.arange(len(scores)),scores).statistic) if np.ptp(scores)>0 else 0.,
                          early=float(scores[:quartile].mean()),late=float(scores[-quartile:].mean())))
    pos = np.array([r['mean_q'] for r in roots if r['success']])
    neg = np.array([r['mean_q'] for r in roots if not r['success']])
    return dict(roots=roots, success_mean=float(pos.mean()) if len(pos) else None,
                failure_mean=float(neg.mean()) if len(neg) else None,
                trajectory_auc=float((np.sign(pos[:, None]-neg)+1).mean()/2) if len(pos) and len(neg) else None,
                success_mean_spearman=float(np.mean([r['spearman_time_q'] for r in roots if r['success']])) if len(pos) else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,help='Optional Q layout override or legacy model configuration')
    parser.add_argument('--bootstrap-seed',type=int,default=0)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Q state_dict export or training checkpoint')
    parser.add_argument('--candidates', type=Path, required=True, help='NPZ with frozen DINO features, raw actions, and optional binary labels [state,candidate,repeat]')
    parser.add_argument('--action-stats', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    args = parser.parse_args()
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    config=resolve_model_config(state,load_config(args.config) if args.config else None)
    model = Critic(config=config)
    model.load_state_dict(state.get('model',state), strict=True)
    model.to(device).eval()
    stats = json.loads(args.action_stats.read_text())
    with np.load(args.candidates, allow_pickle=False) as inputs:
        scores = candidate_scores(model,inputs['features'],inputs['actions'],device,stats['mean'],stats['std'])
        report = dict(scores=scores.tolist(),dispersion=score_dispersion(scores))
        if 'labels' in inputs:
            labels = inputs['labels']
            if labels.ndim != 3 or labels.shape[:2] != scores.shape or labels.shape[2] < 2 or not np.isin(labels,[0,1]).all():
                raise ValueError('expected binary labels [states,candidates,at least two repeats]')
            report['ranking'] = average_metrics([scores],labels,seed=args.bootstrap_seed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
