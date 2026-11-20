"""Identity contract for π₀.₅ base + offline Q-planning Q, one contract per Q seed. No model imports."""
from __future__ import annotations
import argparse
import hashlib
import contextlib
import fcntl
import json
import os
from pathlib import Path

from ..config import QConfig, load_config
from .planning import MEAN_MODES, Q_MODES, compare_scores, validate_options, validate_score_tolerance

HERE = Path(__file__).resolve().parent
DEFAULT_NOISE_NAMESPACE = 'qplanning-n4-v1'
PROTOCOL_FIELDS = ('max_steps', 'open_loop_steps', 'control_hz', 'realtime', 'disabled_policy_cameras', 'manipulability_enabled')


def algorithm(value):
    return f"pi05_n{value['candidate_count']}_qplanning_{value['pick']}"


def require(condition, message):
    if not condition:
        raise ValueError('Q-planning runtime: '+message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def fresh_sha(path):
    """Hash after dropping clean cached pages: this machine has shown page-cache bit flips."""
    with Path(path).open('rb') as f:
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return sha(path)


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = path.with_name('.'+path.name+'.'+str(os.getpid())+'.tmp')
    with t.open('w') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False)+'\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(t, path)


@contextlib.contextmanager
def gpu_lock(contract):
    path = Path(contract['workspace'])/'dataset-root/projects/libero_x_vla/.gpu-task.lock'
    with path.open('rb') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Platform GPU lock busy: {path}') from None
        yield


def verify_score_comparison(record, contract, candidate_count):
    require(isinstance(record,dict) and len(record.get('online_scores',[]))==candidate_count,
            'missing or incomplete score comparison')
    expected=compare_scores(record['online_scores'],record['offline_scores'],contract.get('elite_count'),
                            contract['score_max_abs_tolerance'],rank_singleton=not contract.get('include_archive_in_score_tolerance'))
    require(expected==record and expected['within_tolerance'], 'recorded scores exceed tolerance or diagnostics differ')


def validate_smoke(proof, contract):
    require(proof['status']=='PASS_QPLANNING_GPU_SMOKE' and proof['composite_sha256']==contract['composite_sha256']
            and proof['candidate_policy_id']==contract['candidate_policy_id'] and proof['pick']==contract['pick'],
            'missing or stale GPU smoke proof')
    require(proof['bc_checks'] and all(c['noise_exact'] and c['actions_exact']
            and c['per_slot_max_abs']==[0.0]*contract['candidate_count'] for c in proof['bc_checks']),
            'GPU smoke did not verify every candidate slot')
    if contract['pick'] in MEAN_MODES:
        require(all(c.get('normalized_exact') and c.get('raw_exact') and c.get('aggregation_exact') and
                    c.get('candidates_compared')==c.get('matched_count')==contract['candidate_count'] and c.get('mismatched_indices')==[] and
                    c.get('per_slot_normalized_max_abs')==[0.0]*contract['candidate_count'] and
                    len(c['selected_indices'])==len(c['weights'])==contract['elite_count'] for c in proof['bc_checks']),
                'missing normalized candidate, original BC or aggregation parity')
        if contract['pick'] in Q_MODES:
            if 'score_max_abs_tolerance' in contract:
                diagnostics=proof.get('candidate_diagnostics') or []
                require(len(diagnostics)==len(proof['bc_checks']), 'expected one diagnostic per candidate check')
                for sample,query in zip(proof['bc_checks'],diagnostics,strict=True):
                    record=sample.get('score_comparison')
                    verify_score_comparison(record,contract,contract['candidate_count'])
                    require(record['online_scores']==query.get('scores') and sample['selected_indices']==query.get('selected_indices')
                            and sample['weights']==query.get('weights'), 'score comparison differs from the recorded decision')
                    require(sample.get('offline_scores_exact') is (record['score_max_abs']==0.),
                            'recorded exact equality differs from measured scores')
            else:
                require(all(c.get('offline_scores_exact') for c in proof['bc_checks']), 'missing current offline candidate score parity')
    if contract['pick'] in Q_MODES:
        from .reference_probe import load_spec, validate_reference, verify_score_matrix
        validate_reference(proof.get('reference_gpu',{}),load_spec(contract))
        if contract.get('include_archive_in_score_tolerance'):
            parity=proof['scorer_parity']
            verify_score_matrix(parity,parity['shape'],contract.get('elite_count'),
                                contract['score_max_abs_tolerance'],'fixture_offline_scores')
        elif 'score_max_abs_tolerance' in contract:
            parity=proof['scorer_parity']
            rows=parity.get('score_comparisons',[])
            require(rows and len(rows)==parity['shape'][0], 'missing same-device fixture score evidence')
            for record in rows:
                verify_score_comparison(record,contract,parity['shape'][1])
            require(parity.get('within_tolerance') is True and parity.get('tolerance')==contract['score_max_abs_tolerance']
                    and parity['max_abs_error']==max(record['score_max_abs'] for record in rows),
                    'fixture error does not match the source scores')
        require(proof.get('scorer_parity',{}).get('checkpoint_sha256')==contract['critic']['sha256'],
                'current policy scorer was not checked with its own checkpoint')


def code_files(workspace):
    terminal = workspace/'liberox-vla-adapter-terminal'
    project = workspace/'vla-adapter-rynn-iql'
    openpi = workspace/'third_party/OpenPI'
    sources = [HERE/n for n in ('contract.py', 'q_model.py', 'runtime_core.py', 'planning.py', 'planning_smoke.py', 'worker.py', 'provider.py', 'run_eval.py', 'smoke.py', 'reference_probe.py')]
    sources += [HERE.parent/n for n in ('config.py','defaults.json','c_score.py','critic.py','features.py','metrics.py','data.py')]
    sources += [terminal/'backend/app/policies/pi05.py', terminal/'backend/app/evaluation/batch.py',
                project/'scripts/pi05_worker.py', project/'src/vla_rynn_iql/pi05.py',
                project/'src/vla_rynn_iql/image_transforms.py', openpi/'src/openpi/policies/policy.py',
                openpi/'src/openpi/models_pytorch/pi0_pytorch.py']
    return sources


def validate_effective(value, contract):
    policy, config = value['policy_snapshot'], value['config']
    require(policy == contract['snapshot'], 'platform policy snapshot changed')
    require(policy['policy_id'] == contract['candidate_policy_id'] == contract['continuation_policy_id'],
            'the same actor replans at every decision in this protocol')
    require(policy['family'] == 'pi05' and policy['model_config']['num_inference_steps'] > 0, 'invalid pi05 actor or flow steps')
    q_config = QConfig.from_dict(contract['q_config'])
    require(config['max_steps']==q_config.max_steps and config['open_loop_steps']==q_config.horizon,
            'evaluation episode or chunk length differs from Q configuration')
    require(not config['disabled_policy_cameras'], 'the frozen actor requires its configured cameras')
    for k,v in contract['protocol'].items():
        require(config[k] == v, 'protocol differs: '+k)
    require(value['schedule_sha256'] in contract['schedule_sha256'], 'evaluation schedule changed')


def validate_contract(path, *, verify_payloads=False):
    c = read(path)
    require(c['schema_version']==3 and c['composite_sha256']==canonical({k:v for k,v in c.items() if k!='composite_sha256'}),
            'runtime identity changed')
    require(c['candidate_policy_id']==c['continuation_policy_id']==c['snapshot']['policy_id'], 'actor identities differ')
    require(type(c['candidate_count']) is int and c['candidate_count']>=1 and c['pick'] in ('greedy','random',*MEAN_MODES),
            'invalid selector')
    if 'score_max_abs_tolerance' in c:
        validate_score_tolerance(c['score_max_abs_tolerance'])
    require(type(c.get('include_archive_in_score_tolerance',False)) is bool and
            (not c.get('include_archive_in_score_tolerance') or 'score_max_abs_tolerance' in c),
            'archive comparison requires a sealed maximum absolute score difference')
    steps=c.get('flow_steps', c['snapshot']['model_config']['num_inference_steps'])
    validate_options(c['pick'], c['candidate_count'], c.get('elite_count'), c.get('temperature'), steps)
    require((c['critic'] is not None)==(c['pick'] in Q_MODES), 'Q binding differs from selection mode')
    require(type(c['selection_seed']) is int and 0<=c['selection_seed']<1<<63, 'invalid selection seed')
    require(isinstance(c['noise_namespace'],str) and c['noise_namespace'].strip(), 'empty noise namespace')
    q_config=QConfig.from_dict(c['q_config'])
    require(q_config.max_steps==c['protocol']['max_steps'] and q_config.horizon==c['protocol']['open_loop_steps'],
            'sealed lengths differ')
    if c['pick'] in Q_MODES:
        smoke=c['smoke_config']
        require(sha(smoke['path'])==smoke['sha256'],'smoke configuration changed')
    sources=code_files(Path(c['workspace']))
    require({str(p):sha(p) for p in sources}==c['code_sha256'], 'runtime source or platform integration changed')
    if verify_payloads:
        for name,digest in c['assets_sha256'].items():
            require(fresh_sha(name)==digest, 'runtime asset changed: '+name)
    return c


def build_contract(args):
    import yaml
    require(not args.output.exists(), 'preserve sealed contract')
    configs=[yaml.safe_load(p.read_text()) for p in args.config]
    snapshot=configs[0]['policy_snapshot']
    require(type(args.candidate_count) is int and args.candidate_count>=1, 'positive candidate count required')
    require(type(args.selection_seed) is int and 0<=args.selection_seed<1<<63,'invalid selection seed')
    workspace=args.workspace.resolve()
    require(args.noise_namespace.strip(), 'empty noise namespace')
    q_config=load_config(args.q_config)
    if args.pick in Q_MODES:
        require(all(v is not None for v in (args.q_weights,args.q_seed,args.q_step,args.action_stats,args.dino,args.smoke_config)),
                'greedy requires Q assets, training seed/step and smoke configuration')
        require(args.q_step > 0, 'Q checkpoint step must be positive')
        from ..config import resolve_model_config
        import torch
        checkpoint=torch.load(args.q_weights,map_location='cpu',weights_only=True,mmap=True)
        q_config=resolve_model_config(checkpoint,q_config if args.q_config is not None else None)
        del checkpoint
    c=dict(schema_version=3,workspace=str(workspace),snapshot=snapshot,candidate_policy_id=args.candidate_policy_id,
           continuation_policy_id=args.continuation_policy_id,pick=args.pick,candidate_count=args.candidate_count,
           selection_seed=args.selection_seed,noise_namespace=args.noise_namespace,q_config=q_config.to_dict(),
           protocol={k:configs[0]['config'][k] for k in PROTOCOL_FIELDS},schedule_sha256=sorted({v['schedule_sha256'] for v in configs}),critic=None)
    tolerance=getattr(args,'score_max_abs_tolerance',None)
    if tolerance is not None:
        validate_score_tolerance(tolerance)
        c['score_max_abs_tolerance']=tolerance
    include_archive=getattr(args,'include_archive_in_score_tolerance',False)
    require(type(include_archive) is bool, 'invalid archive score tolerance option')
    if include_archive:
        require(tolerance is not None,'archive comparison requires an explicit score tolerance')
        c['include_archive_in_score_tolerance']=True
    c['flow_steps']=args.flow_steps if args.flow_steps is not None else snapshot['model_config']['num_inference_steps']
    c['elite_count'],c['temperature']=args.elite_count,args.temperature
    validate_options(args.pick, args.candidate_count, c['elite_count'], c['temperature'], c['flow_steps'])
    for effective in configs:
        validate_effective(effective,c)
    assets=[Path(snapshot['base_checkpoint'])/'model.safetensors']
    if snapshot['actor']:
        assets.append(Path(snapshot['actor']))
    if args.pick in Q_MODES:
        smoke=args.smoke_config.resolve()
        c['smoke_config']={'path':str(smoke),'sha256':sha(smoke)}
        weights=args.q_weights.resolve(); stats=args.action_stats.resolve(); dino=args.dino.resolve()
        from ..features import verify_dino
        dino_identity=verify_dino(dino,args.dino_receipt)
        assets.extend([weights,stats])
        c['critic']=dict(path=str(weights),seed=args.q_seed,training_step=args.q_step,action_stats=str(stats),
                         dino_path=str(dino),dino_revision=dino_identity['revision'],backend='torch_qplanning_offline_q',
                         layers=18,dim=1024,heads=16)
    c['assets_sha256']={str(p):fresh_sha(p) for p in assets}
    if c['critic']:
        c['assets_sha256'].update({str(dino/name):h for name,h in dino_identity['sha256'].items()})
        c['critic']['sha256']=c['assets_sha256'][str(weights)]
    c['code_sha256']={str(p):sha(p) for p in code_files(workspace)}
    c['composite_sha256']=canonical(c)
    atomic_json(args.output,c)
    return validate_contract(args.output)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description='Seal runtime options for one platform effective config and schedule.')
    p.add_argument('--workspace',type=Path,required=True,help='Platform workspace with runtime dependencies')
    p.add_argument('--config',type=Path,required=True,action='append',help='Platform effective_config.yaml; repeat for each scheduled half')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--candidate-policy-id',required=True)
    p.add_argument('--continuation-policy-id',required=True,help='Same actor as candidate policy, replanned once per configured chunk')
    p.add_argument('--candidate-count',type=int,default=4)
    p.add_argument('--flow-steps',type=int,help='Number of flow steps per candidate; defaults to the actor setting')
    p.add_argument('--elite-count',type=int,help='Number of complete BC-normalized chunks to average')
    p.add_argument('--score-max-abs-tolerance',type=float,
                   help='Maximum absolute runtime/offline score difference for the current critic')
    p.add_argument('--include-archive-in-score-tolerance',action='store_true',
                   help='Apply the same explicitly sealed maximum absolute difference to archive replay')
    p.add_argument('--temperature',type=float,help='Positive softmax temperature for Q-weighted averaging')
    p.add_argument('--pick',choices=('greedy','random',*MEAN_MODES),default='greedy')
    p.add_argument('--selection-seed',type=int,default=0)
    p.add_argument('--noise-namespace',default=DEFAULT_NOISE_NAMESPACE,help='Domain prefix for the independent candidate noise stream')
    p.add_argument('--q-config',type=Path,help='Q and camera configuration; defaults to exported metadata or model defaults')
    p.add_argument('--smoke-config',type=Path,help='Path to the parity and smoke fixture specification (required for greedy)')
    p.add_argument('--q-step',type=int,help='Training step of the selected Q checkpoint (required for greedy)')
    p.add_argument('--q-seed',type=int); p.add_argument('--q-weights',type=Path)
    p.add_argument('--dino-receipt',type=Path,help='Optional external DINO hashes and revision')
    p.add_argument('--dino',type=Path); p.add_argument('--action-stats',type=Path)
    print(build_contract(p.parse_args())['composite_sha256'])
