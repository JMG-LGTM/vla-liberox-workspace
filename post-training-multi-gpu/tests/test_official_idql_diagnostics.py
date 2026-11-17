"""CPU synthetic evidence tests; these are not GPU or policy-success proofs."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from post_training.io import sha256_file, stable_hash
from post_training.official_idql.adapters import ReplayAdapter, build_diagnostic_memberships
from post_training.official_idql.diagnostics import summarize_diagnostics, classify_gates, unavailable_same_state
from post_training.official_idql.pairs import load_same_state_pairs, evaluate_same_state_pairs, FORMAT, LABEL


def diagnostic_case():
    path={'root_run_id':'r','split':'val','cache_indices':[0,1,2]}
    rows=[]
    for i in range(5):
        member=[dict(run_id='h',root_run_id='r',remaining_blocks=2-i,success_terminal=i==2)] if i<3 else []
        rows.append(dict(cache_index=i,run_id='h' if i in (1,2) else 'r',root_run_id='r',split='val',
            start=[0,8,16,8,16][i],end=[8,16,24,16,24][i], action_source='human' if i in (1,2) else 'policy',
            recording_success=i in (1,2),remaining_blocks=2-i if i in (1,2) else -1,success_terminal=i==2,
            progress_scope='complete_successful_paths',membership_sha256='m'*64,
            success_memberships=member,expected_success_paths={'h':path}))
    v=np.array([1.,2.,3.,0.,0.]);q=v+np.array([0.,.1,.1,-.1,-.1])
    outputs=dict(v=v,q1=q,q2=q+.01,target_q1=q+.2,target_q2=q+.3)
    pair=dict(available=True,evidence_verified=True,held_out=True,pair_count=2,root_count=1,
              root_run_ids=['r'],human_higher_fraction=1.)
    return rows,outputs,pair


def report_case():
    rows,outputs,pair=diagnostic_case()
    return summarize_diagnostics(rows,outputs,same_state_result=pair,expected_cache_indices=list(range(5)))


def test_full_paths_shared_failed_prefix_unique_global_and_hypothetical_weights():
    result=report_case()
    assert result['spearman_v_remaining_blocks']==pytest.approx(-1)
    assert result['median_per_trajectory_spearman']==pytest.approx(-1)
    assert result['success_progress_count']==3 and result['count']==5
    assert result['human_comparison']['human_higher_fraction']==1
    assert 'different-state' in result['human_comparison']['definition']
    assert result['gates']['eligible_for_idql_evaluation']
    assert result['success_terminal_weight_share']<.5
    assert 'diagnostic only' in result['weight_definition']
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('defect',['no_expected','duplicate','partial','train','no_membership','missing_path','constant'])
def test_incomplete_undefined_or_training_evidence_cannot_qualify(defect):
    rows,outputs,pair=diagnostic_case();expected=list(range(5))
    if defect=='no_expected':expected=None
    elif defect=='duplicate':rows[-1]['cache_index']=3
    elif defect=='partial':expected.append(5)
    elif defect=='train':rows[0]['split']='train'
    elif defect=='no_membership':
        for row in rows:row['progress_scope']='retained_unique_trajectory_intervals'
    elif defect=='missing_path':rows[0]['success_memberships']=[]
    else:outputs['v'][:]=1
    result=summarize_diagnostics(rows,outputs,same_state_result=pair,expected_cache_indices=expected)
    assert not result['gates']['eligible_for_idql_evaluation']
    assert result['gates']['unavailable']


def test_human_heuristic_never_substitutes_for_true_same_state_and_ties_fail():
    rows,outputs,_=diagnostic_case()
    report=summarize_diagnostics(rows,outputs,expected_cache_indices=list(range(5)))
    assert report['gates']['human_passed'] and not report['gates']['same_state_passed']
    assert report['same_state_action_ranking']['status']=='UNAVAILABLE'
    for name,threshold,expected in [('human_higher_fraction',.5,False),('human_higher_fraction',.500001,True)]:
        patched=report_case();patched['human_comparison'][name]=threshold
        assert classify_gates(patched)['human_passed'] is expected
    patched=report_case();patched['same_state_action_ranking']['human_higher_fraction']=.5
    assert not classify_gates(patched)['same_state_passed']
    assert evaluate_same_state_pairs(None,None)==unavailable_same_state()


def test_undefined_success_path_not_silently_removed_and_gate_boundaries():
    rows,outputs,pair=diagnostic_case()
    for row in rows:row['expected_success_paths']['one']={'root_run_id':'r','split':'val','cache_indices':[4]}
    rows[4]['success_memberships']=[dict(run_id='one',root_run_id='r',remaining_blocks=0,success_terminal=True)]
    report=summarize_diagnostics(rows,outputs,same_state_result=pair,expected_cache_indices=range(5))
    assert report['undefined_trajectory_count']==1 and report['median_per_trajectory_spearman'] is None
    patched=report_case();patched['ess_fraction']=.1;patched['success_terminal_weight_share']=.5
    assert classify_gates(patched)['eligible_for_idql_evaluation']
    patched['ess_fraction']=.0999
    assert not classify_gates(patched)['ess_passed']
    with pytest.raises(ValueError):classify_gates(report_case(),{'unknown':0})


class DiagnosticCache:
    def __init__(self,records):
        self.indices=list(range(len(records)))
        self.metadata=dict(count=len(records),fingerprint='f'*64,records=records,normalization={})
        self.arrays={}
    def __len__(self):return len(self.indices)


def test_adapter_restores_exact_full_success_memberships_not_canonical_failure():
    chunks=[dict(start=i*8,end=(i+1)*8,length=8,action_source='policy' if i==0 else 'human',copied_prefix=i==0) for i in range(3)]
    failure=dict(run_id='r',root_run_id='r',split='val',success=False,recorded_action_count=24,terminal_step=None,
        chunks=[dict(start=i*8,end=(i+1)*8,length=8,action_source='policy') for i in range(3)])
    success=dict(run_id='h',root_run_id='r',split='val',success=True,recorded_action_count=24,terminal_step=23,chunks=chunks)
    records=[dict(run_id='r',root_run_id='r',split='val',origin_run_id='r',start=0,end=8,action_source='policy',remaining_blocks=-1,recording_success=False),
        *[dict(run_id='h',root_run_id='r',split='val',origin_run_id='h',start=i*8,end=(i+1)*8,action_source='human',remaining_blocks=2-i,recording_success=True) for i in (1,2)],
        *[dict(run_id='r',root_run_id='r',split='val',origin_run_id='r',start=i*8,end=(i+1)*8,action_source='policy',remaining_blocks=-1,recording_success=False) for i in (1,2)]]
    cache=DiagnosticCache(records)
    adapter=ReplayAdapter(cache,dict(mode='vector',vector_source='normalized_proprio'))
    mapping=build_diagnostic_memberships(cache.metadata,[failure,success])
    adapter.bind_diagnostic_memberships(mapping)
    rows=adapter.diagnostic_rows()
    assert rows[0]['remaining_blocks']==-1 and rows[0]['success_memberships'][0]['remaining_blocks']==2
    assert rows[0]['expected_success_paths']['h']['cache_indices']==[0,1,2]
    corrupt=deepcopy(mapping);corrupt['successful_paths']['h']['cache_indices'].pop()
    corrupt['sha256']=stable_hash({k:v for k,v in corrupt.items() if k!='sha256'})
    with pytest.raises(ValueError):adapter.bind_diagnostic_memberships(corrupt)
    broken=deepcopy(cache.metadata);broken['records'][0]['start']=1
    with pytest.raises(ValueError,match='absent'):build_diagnostic_memberships(broken,[failure,success])


def pair_fixture(tmp_path, *, mode='vector', t=3, horizon=6, mutate=None):
    from post_training.data import _manifest_identity
    from post_training.replay import _pixels, normalize
    root=tmp_path/'native';root.mkdir()
    n=14
    trajectories={};observations={}
    for name in ('r','h'):
        trajectories[name]=dict(sim_state=np.arange((n+1)*4,dtype=np.float64).reshape(n+1,4),
            eef_position=np.zeros((n+1,3),np.float32),eef_axis_angle=np.zeros((n+1,3),np.float32),
            gripper_qpos=np.zeros((n+1,2),np.float32),env_action=np.zeros((n,7),np.float32),
            action_source=np.full(n,'policy',dtype='<U8'))
        observations[name]=dict(agentview_image=np.arange((n+1)*12*12*3,dtype=np.uint8).reshape(n+1,12,12,3),
            wrist_image=np.full((n+1,12,12,3),81,np.uint8))
    trajectories['h']['action_source'][t:t+horizon]='human';trajectories['h']['env_action'][t:t+horizon]=.2
    if mutate:mutate(trajectories,observations)
    episodes=[];files={}
    for name in ('r','h'):
        for suffix,values in [('trajectory',trajectories[name]),('observations',observations[name])]:
            path=root/f'{name}_{suffix}.npz';np.savez(path,**values)
            files[path.name]=dict(size=path.stat().st_size,sha256=sha256_file(path))
        episodes.append(dict(run_id=name,root_run_id='r',source='takeover_chain' if name=='h' else 'autonomous_failure',
            success=name=='h',trajectory_path=f'{name}_trajectory.npz',observations_path=f'{name}_observations.npz'))
    manifest=dict(schema_version=1,episodes=episodes,files=files)
    manifest['dataset_sha256']=_manifest_identity(manifest)
    mp=root/'manifest.json';mp.write_text(json.dumps(manifest))
    package=dict(schema_version=1,format=FORMAT,dataset_sha256=manifest['dataset_sha256'],
        dataset_manifest_sha256=sha256_file(mp),label=LABEL,pairs=[dict(pair_id='p',root_run_id='r',chosen_run_id='h',
            rejected_run_id='r',state_index=t,horizon=horizon)])
    path=tmp_path/'pairs.json';path.write_text(json.dumps(package))
    stats=dict(actions=dict(q01=[-1.]*7,q99=[1.]*7),state=dict(q01=[-1.]*8,q99=[1.]*8))
    record=dict(run_id='h',root_run_id='r',split='val',start=t,end=t+horizon,action_source='human',origin_run_id='h')
    cache=DiagnosticCache([record]);cache.metadata.update(normalization=stats,
        identity=dict(dataset_sha256=manifest['dataset_sha256'],critic_image_size=8))
    cache.arrays=dict(raw_proprio=np.zeros((1,8),np.float32),pixels=np.stack([_pixels(observations['h']['agentview_image'][t,::-1].copy(),
        observations['h']['wrist_image'][t,::-1].copy(),8)]))
    spec=dict(mode='pixels',pixels=dict(architecture='upstream_pixel_multiplexer_d4pg',camera_layout='separate_rgb_views')) if mode=='pixels' else dict(mode='vector',vector_source='normalized_proprio')
    return path,ReplayAdapter(cache,spec),root,package


@pytest.mark.parametrize('mode',['vector','pixels'])
def test_true_non_aligned_same_state_t_with_h6_shared_mask_and_target_score(tmp_path,monkeypatch,mode):
    path,adapter,root,_=pair_fixture(tmp_path,mode=mode)
    package=load_same_state_pairs(path,adapter,data_root=root,heldout_root_ids=['r'])
    row=package.records[0]
    assert row['state_index']==3 and row['actions'].shape==(2,64)
    np.testing.assert_array_equal(row['actions'][:,-8:],[[1]*6+[0]*2]*2)
    assert np.all(row['actions'][:,42:56]==0)
    if mode=='pixels':assert row['observation']['agent_pixels'].shape==(8,8,3)
    pytest.importorskip('jax');pytest.importorskip('flax');pytest.importorskip('optax');pytest.importorskip('gym')
    from post_training.official_idql import core
    def fake_score(agent,observation,actions):
        assert agent is None
        return {'q_min':np.array([2.,1.])}
    monkeypatch.setattr(core,'score_actions',fake_score)
    state=np.random.get_state()
    report=package.evaluate(None)
    assert report['human_higher_fraction']==1 and report['evidence_verified'] and report['held_out']
    assert report['source_contract']['geometry']['encoded_dim']==64
    assert np.array_equal(np.random.get_state()[1],state[1])


@pytest.mark.parametrize('field',['sim_state','eef_position','agentview_image','wrist_image','action_source'])
def test_true_pair_mismatched_raw_evidence_or_producer_rejected(tmp_path,field):
    def mutate(trajectories,observations):
        if field in observations['h']:observations['h'][field][3,0,0,0]^=1
        elif field=='action_source':trajectories['h'][field][5]='policy'
        else:trajectories['h'][field][3,0]+=.01
    path,adapter,root,_=pair_fixture(tmp_path,mutate=mutate)
    with pytest.raises(ValueError):load_same_state_pairs(path,adapter,data_root=root,heldout_root_ids=['r'])


def test_pair_source_binding_heldout_exact_index_and_duplicate_rejected(tmp_path):
    path,adapter,root,payload=pair_fixture(tmp_path)
    with pytest.raises(ValueError):load_same_state_pairs(path,adapter,data_root=root,heldout_root_ids=['train_root'])
    assert load_same_state_pairs(None,adapter,data_root=root,heldout_root_ids=['r']) is None
    payload['pairs'].append(deepcopy(payload['pairs'][0]));payload['pairs'][-1]['pair_id']='another'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='duplicate same-state'):load_same_state_pairs(path,adapter,data_root=root,heldout_root_ids=['r'])
    payload['pairs'].pop();payload['pairs'][0]['state_index']=4;path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):load_same_state_pairs(path,adapter,data_root=root,heldout_root_ids=['r'])
    payload['pairs'][0]['state_index']=3;path.write_text(json.dumps(payload))
    with (root/'r_trajectory.npz').open('ab') as stream:stream.write(b'tamper')
    with pytest.raises(ValueError,match='SHA/size'):load_same_state_pairs(path,adapter,data_root=root,heldout_root_ids=['r'])


def test_partial_human_roots_and_out_of_range_metrics_fail_closed():
    result=report_case()
    result['human_comparison']['unmatched_human_roots']=['another_heldout_takeover']
    gate=classify_gates(result)
    assert not gate['human_passed'] and 'human_higher_fraction' in gate['unavailable']
    for container,key,bad in [(None,'ess_fraction',1.1),(None,'success_terminal_weight_share',-.01),
                              ('human_comparison','human_higher_fraction',2.),
                              ('same_state_action_ranking','human_higher_fraction',2.)]:
        result=report_case();destination=result if container is None else result[container]
        destination[key]=bad
        assert not classify_gates(result)['eligible_for_idql_evaluation']
    result=report_case();result['finite']=False
    assert not classify_gates(result)['eligible_for_idql_evaluation']
    rows,outputs,pair=diagnostic_case();outputs['q1'][:]=1e308;outputs['q2'][:]=1e308;outputs['v'][:]=-1e308
    with pytest.raises(ValueError,match='derived advantage'):summarize_diagnostics(rows,outputs,same_state_result=pair,expected_cache_indices=range(5))


def test_multiple_successful_memberships_use_unique_global_median_and_full_per_path():
    rows,outputs,pair=diagnostic_case()
    for row in rows:
        row['expected_success_paths']['second']={'root_run_id':'r','split':'val','cache_indices':[0,1]}
    for i in (0,1):
        rows[i]['success_memberships'].append(dict(run_id='second',root_run_id='r',remaining_blocks=1-i,success_terminal=i==1))
    result=summarize_diagnostics(rows,outputs,same_state_result=pair,expected_cache_indices=range(5))
    assert result['count']==5 and result['success_progress_count']==3
    assert result['defined_trajectory_count']==2 and result['median_per_trajectory_spearman']==pytest.approx(-1)
    assert result['success_terminal_count']==2


def test_external_features_require_exact_verified_fork_observation_not_floor_match(tmp_path):
    path,adapter,root,_=pair_fixture(tmp_path)
    features=tmp_path/'features.npz';encoder=tmp_path/'encoder.py';pre=tmp_path/'pre.py'
    np.savez(features,observations=np.array([[2.,3.]],np.float32),next_observations=np.array([[4.,5.]],np.float32))
    encoder.write_text('# test frozen encoder\n');pre.write_text('# test preprocessing\n')
    spec=dict(path=str(features),sha256=sha256_file(features),cache_fingerprint=adapter.dataset.metadata['fingerprint'],
        encoder_path=str(encoder),encoder_sha256=sha256_file(encoder),preprocess_path=str(pre),
        preprocess_sha256=sha256_file(pre),feature_layer='test_output')
    external=ReplayAdapter(adapter.dataset,dict(mode='external_features',external_features=spec))
    package=load_same_state_pairs(path,external,data_root=root,heldout_root_ids=['r'])
    np.testing.assert_array_equal(package.records[0]['observation'],[2.,3.])
    assert package.contract['adapter_contract']['encoder']['sha256']==spec['encoder_sha256']
    adapter.dataset.metadata['records'][0]['start']=0
    with pytest.raises(ValueError,match='no exact cached'):load_same_state_pairs(path,external,data_root=root,heldout_root_ids=['r'])


def test_complete_frozen_roster_may_include_native_roots_without_cache_chunks(tmp_path):
    from post_training.data import _manifest_identity
    path,adapter,root,payload=pair_fixture(tmp_path)
    mp=root/'manifest.json';manifest=json.loads(mp.read_text())
    manifest['episodes'].append(dict(run_id='empty',root_run_id='empty'))
    manifest['dataset_sha256']=_manifest_identity(manifest);mp.write_text(json.dumps(manifest))
    payload['dataset_sha256']=manifest['dataset_sha256'];payload['dataset_manifest_sha256']=sha256_file(mp)
    path.write_text(json.dumps(payload));adapter.dataset.metadata['identity']['dataset_sha256']=manifest['dataset_sha256']
    package=load_same_state_pairs(path,adapter,data_root=root,heldout_root_ids=['r','empty'])
    assert package.contract['heldout_root_ids']==['empty','r']
