"""Verified, held-out same-state human/policy action ranking, evaluation only.

The JSON names raw recordings and an exact state/action index. It contains no
features or invented labels. Verification reads manifest-hashed original files,
checks saved simulator state plus both raw cameras/proprioception, and encodes
both continuations with the identical execution mask and native normalization.
Human intervention is an assumed preference, not an oracle success label.
"""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import numpy as np

from ..archive_io import load_npz_arrays
from ..data import safe_path, _manifest_identity
from ..io import stable_hash
from .adapters import encode_action_chunks, file_binding

FORMAT = 'official_idql_native_same_state_pairs_v1'
LABEL = 'human_intervention_assumed_preferred'


def _require(condition, message):
    if not condition:
        raise ValueError('Official IDQL same-state pairs: ' + message)


def _array_hash(value):
    value = np.ascontiguousarray(value)
    import hashlib
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


class VerifiedSameStatePairs:
    """Prepared, immutable observation/actions; scoring does not consume RNG."""
    def __init__(self, records, contract):
        self.records, self.contract = records, contract

    def evaluate(self, agent):
        from .core import score_actions
        rows=[]
        for row in self.records:
            score=score_actions(agent,row['observation'],row['actions'])
            q=np.asarray(score['q_min'],dtype=np.float64)
            _require(q.shape==(2,) and np.isfinite(q).all(), 'target-Q pair scores invalid')
            margin=float(q[0]-q[1])
            rows.append(dict(pair_id=row['pair_id'],root_run_id=row['root_run_id'],state_index=row['state_index'],
                horizon=row['horizon'],chosen_target_min_q=float(q[0]),rejected_target_min_q=float(q[1]),
                margin=margin,human_higher=margin>0,tie=margin==0))
        return dict(available=True,status='AVAILABLE_VERIFIED_HELDOUT_RAW_SAME_STATE',
            evidence_verified=True,held_out=True,finite=True,pair_count=len(rows),
            root_count=len({x['root_run_id'] for x in rows}),root_run_ids=sorted({x['root_run_id'] for x in rows}),
            human_higher_fraction=float(np.mean([x['human_higher'] for x in rows])),
            tie_count=sum(x['tie'] for x in rows),mean_margin=float(np.mean([x['margin'] for x in rows])),
            per_pair=rows,source_contract=self.contract,source_contract_sha256=stable_hash(self.contract),
            score_definition='target twin-Q minimum at exactly identical raw saved state; strict human>policy, ties are not wins',
            label=LABEL,interpretation='human intervention assumed preferred; small held-out pair ranking is not proof of policy success')


def load_same_state_pairs(path, adapter, *, data_root, heldout_root_ids):
    """Prepare actual evidence or return None for an explicitly absent package.

    External features require an EXACT cached observation at this root/start;
    otherwise no inference feature extractor is available and preparation fails.
    No floor/nearest cache matching or raw t+1 substitution is permitted.
    """
    if path is None:
        return None
    root=Path(data_root).expanduser().resolve(strict=True)
    source=file_binding(path); payload=json.loads(Path(source['path']).read_text())
    required={'schema_version','format','dataset_sha256','dataset_manifest_sha256','label','pairs'}
    _require(isinstance(payload,dict) and set(payload)==required and type(payload['schema_version']) is int
        and payload['schema_version']==1 and payload['format']==FORMAT and payload['label']==LABEL, 'unsupported package schema/label')
    native_binding=file_binding(safe_path(root,'manifest.json'))
    manifest=json.loads(Path(native_binding['path']).read_text())
    _require(manifest.get('dataset_sha256')==_manifest_identity(manifest)
        and payload['dataset_sha256']==manifest['dataset_sha256']
        and payload['dataset_manifest_sha256']==native_binding['sha256'], 'native dataset/manifest identity differs')
    _require(adapter.dataset.metadata['identity']['dataset_sha256']==manifest['dataset_sha256'], 'adapter/native dataset differs')
    heldout=set(heldout_root_ids)
    _require(heldout and all(isinstance(x,str) and x for x in heldout), 'held-out roots required')
    local_roots={r['root_run_id'] for r in adapter.dataset.metadata['records']
                 if r.get('split') in {'val','validation'}}
    manifest_roots={ep['root_run_id'] for ep in manifest['episodes']}
    train_roots={r['root_run_id'] for r in adapter.dataset.metadata['records'] if r.get('split')=='train'}
    _require(heldout<=manifest_roots and not heldout & train_roots,
             'declared held-out root is absent from native roster or belongs to training')
    _require(isinstance(payload['pairs'],list) and payload['pairs'], 'empty pair package')
    episodes={x['run_id']:x for x in manifest['episodes']}
    records=[]; bindings={};loaded=OrderedDict();identities=set();pair_ids=set()
    stats=adapter.dataset.metadata['normalization']
    if adapter.dataset.arrays is None:
        adapter.dataset[0]
    def arrays(ep,field):
        relative=ep[field]
        _require(relative in manifest['files'], 'unhashed raw episode file')
        if relative not in loaded:
            binding=file_binding(safe_path(root,relative)); expected=manifest['files'][relative]
            _require(binding['sha256']==expected['sha256'] and binding['bytes']==expected['size'], 'raw source SHA/size differs')
            bindings[relative]=binding
            keys = (('sim_state','env_action','eef_position','eef_axis_angle','gripper_qpos','action_source')
                    if field == 'trajectory_path' else ('agentview_image','wrist_image'))
            loaded[relative]=load_npz_arrays(Path(binding['path']),keys=keys,expected_sha256=binding['sha256'])
            # A package can span many long recordings. Retain at most the two
            # current trajectories and two camera archives, not the full pool.
            while len(loaded)>4:
                loaded.popitem(last=False)
        loaded.move_to_end(relative)
        return loaded[relative]
    for row in payload['pairs']:
        _require(isinstance(row,dict) and set(row)=={'pair_id','root_run_id','chosen_run_id','rejected_run_id','state_index','horizon'}, 'invalid pair fields')
        _require(isinstance(row['pair_id'],str) and row['pair_id'] and row['pair_id'] not in pair_ids, 'duplicate/invalid pair ID')
        pair_ids.add(row['pair_id'])
        root_id=row['root_run_id'];t=row['state_index'];h=row['horizon']
        _require(root_id in heldout and root_id in local_roots and type(t) is int and t>=0
                 and type(h) is int and 1<=h<=8, 'non-held-out pair or invalid t/horizon')
        a,b=episodes.get(row['chosen_run_id']),episodes.get(row['rejected_run_id'])
        _require(a is not None and b is not None and a['root_run_id']==b['root_run_id']==root_id
            and a['run_id']!=b['run_id'] and a.get('source')=='takeover_chain' and a.get('success') is True
            and b['run_id']==root_id and b.get('source')=='autonomous_failure' and b.get('success') is False,
            'pair must be a successful human takeover versus its original failed-policy root')
        key=(a['run_id'],b['run_id'],t,h)
        _require(key not in identities, 'duplicate same-state comparison')
        identities.add(key)
        ta,tb=arrays(a,'trajectory_path'),arrays(b,'trajectory_path')
        oa,ob=arrays(a,'observations_path'),arrays(b,'observations_path')
        _require(t+h<=len(ta['env_action']) and t+h<=len(tb['env_action']), 'action horizon outside recording')
        _require(a.get('terminal_step') is None or t+h<=int(a['terminal_step'])+1,
                 'pair horizon extends past confirmed successful training endpoint')
        _require('sim_state' in ta and 'sim_state' in tb, 'saved simulator state evidence unavailable')
        evidence={}
        for name in ('sim_state','eef_position','eef_axis_angle','gripper_qpos'):
            x,y=ta[name][t],tb[name][t]
            _require(x.shape==y.shape and x.dtype==y.dtype and np.isfinite(x).all()
                     and np.isfinite(y).all() and np.array_equal(x,y), 'raw same-state '+name+' mismatch')
            evidence[name+'_sha256']=_array_hash(x)
        for name in ('agentview_image','wrist_image'):
            x,y=oa[name][t],ob[name][t]
            _require(x.dtype==y.dtype==np.uint8 and x.ndim==3 and x.shape[-1]==3
                and x.shape==y.shape and np.array_equal(x,y), 'raw same-state '+name+' mismatch')
            evidence[name+'_sha256']=_array_hash(x)
        _require(np.all(ta['action_source'][t:t+h]=='human') and np.all(tb['action_source'][t:t+h]=='policy'),
                 'chosen/rejected action producer must remain human/policy for the entire shared horizon')
        raw=np.stack([np.pad(x['env_action'][t:t+h],((0,8-h),(0,0))) for x in (ta,tb)]).astype(np.float32)
        _require(raw.shape==(2,8,7), 'raw action dimensions differ')
        mask=np.broadcast_to(np.arange(8)<h,(2,8)).copy()
        actions=encode_action_chunks(raw,mask,normalization=stats['actions'])
        from ..replay import _pixels, normalize
        proprio=np.concatenate([ta[name][t] for name in ('eef_position','eef_axis_angle','gripper_qpos')]).astype(np.float32)
        _require(proprio.shape==(8,), 'proprio dimensions differ')
        state=normalize(proprio,stats['state'])
        size=int(adapter.dataset.metadata['identity']['critic_image_size'])
        pixels=_pixels(oa['agentview_image'][t,::-1].copy(),oa['wrist_image'][t,::-1].copy(),size)
        if adapter.mode=='pixels':
            image=pixels.transpose(1,2,0)
            observation=dict(agent_pixels=image[...,:3].copy(),wrist_pixels=image[...,3:].copy(),state=state)
        elif adapter.mode=='vector':
            observation=state
        else:
            candidates=[]
            for index,record in enumerate(adapter.dataset.metadata['records']):
                if record['root_run_id']==root_id and record['start']==t and record.get('split') in {'val','validation'} \
                    and np.array_equal(adapter.dataset.arrays['raw_proprio'][index],proprio) \
                    and np.array_equal(adapter.dataset.arrays['pixels'][index],pixels):
                    candidates.append(index)
            _require(candidates, 'no exact cached fork observation for bound external feature encoder; no automatic raw feature extraction')
            feature=adapter.features['observations'][candidates[0]]
            _require(all(np.array_equal(adapter.features['observations'][i],feature) for i in candidates), 'conflicting same-state external features')
            observation=feature.copy()
            evidence['external_feature_cache_indices']=candidates
        records.append({**row,'observation':observation,'actions':actions,'evidence':evidence})
    contract=dict(schema_version=1,format=FORMAT,package=source,native_manifest=native_binding,
        dataset_sha256=manifest['dataset_sha256'],adapter_sha256=stable_hash(adapter.contract),
        adapter_contract=adapter.contract,normalization_sha256=stable_hash(stats),
        heldout_root_ids=sorted(heldout),selected_raw_files=bindings,label=LABEL,
        geometry=dict(horizon_max=8,action_dim=7,encoded_dim=64,mask='same_nonempty_prefix_for_both_candidates'),
        pairing='exact_raw_state_t_and_actions_t_to_t_plus_h; simulator+both_raw_cameras+proprio_equal',
        evidence=[{k:v for k,v in row.items() if k not in {'observation','actions'}} for row in records])
    return VerifiedSameStatePairs(records,contract)


def evaluate_same_state_pairs(pairs, agent):
    if pairs is None:
        from .diagnostics import unavailable_same_state
        return unavailable_same_state()
    return pairs.evaluate(agent)
