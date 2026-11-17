"""Actual CPU native packaging/cache -> official Q/V -> full restore integration."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

import numpy as np
import pytest

for dependency in ('jax','flax','optax','gym'):
    pytest.importorskip(dependency)

from post_training.data import package_dataset,inspect_dataset
from post_training.official_idql import runner
from test_data_replay import make_run


@pytest.fixture
def native(tmp_path):
    source=tmp_path/'source';target=tmp_path/'native'
    for index,name in enumerate(['a','b','c','d']):make_run(source,name,offset=.1*index)
    package_dataset(source,target,run_ids=['a','b','c','d'],horizon=8)
    return target


def configuration(tmp_path):
    config=runner.load_config(Path(__file__).parents[1]/'configs/official_idql/idql_official_vector_cpu.yaml')
    config.update(output_dir=str(tmp_path/'run'),cache_dir=str(tmp_path/'cache'))
    config['training'].update(optimizer_updates=4,checkpoint_interval=2,validation_interval=3)
    return config


def test_actual_native_cpu_train_stop_resume_export_and_frozen_root_members(native,tmp_path):
    from flax import serialization
    import jax
    config=configuration(tmp_path)
    info=inspect_dataset(native,runner.replay_config(config))
    frozen=tmp_path/'frozen.json';frozen.write_text(json.dumps({'dataset_sha256':info['dataset_sha256'],
        'members':[{'root_run_id':r['root_run_id'],'split':r['split']} for r in info['episodes']]}))
    config['data']['frozen_members']=str(frozen)
    golden=deepcopy(config);golden['output_dir']=str(tmp_path/'golden')
    expected=runner.train(golden,native)
    stopped=runner.train(config,native,stop_after=2)
    assert stopped['status']=='STOPPED' and stopped['optimizer_updates']==2
    (tmp_path/'run/stop.request').write_text('previous safe-stop request')
    shutil.rmtree(tmp_path/'run/scorers/update_00000002')
    final=runner.train(config,native,resume=stopped['checkpoint'])
    assert final['status']=='COMPLETED' and final['optimizer_updates']==4 and final['sample_draws']==16
    assert final['actor_optimizer_updates']==0
    # These real native fixtures have no verified same-state action pairs.
    # Good loss or a later checkpoint must never manufacture qualification.
    assert final['selected_scorer'] is None
    assert final['selection']['status']=='NO_QUALIFIED_SCORER'
    assert final['selection']['policy']['fallback_to_last'] is False
    assert [item['optimizer_update'] for item in final['selection']['retained_candidates']]==[2,4]
    assert all(item['status']!='QUALIFIED' for item in final['selection']['retained_candidates'])
    control=json.loads((Path(final['checkpoint'])/'control.json').read_text())
    assert control['metadata']['selection']==final['selection']
    assert json.loads((tmp_path/'run/selection.json').read_text())==final['selection']
    assert sorted(p.name for p in (tmp_path/'run/validation').glob('update_*.json'))==[
        f'update_{i:08d}.json' for i in [2,3,4]]
    for update in [2,3,4]:
        evidence=json.loads((tmp_path/'run/validation'/f'update_{update:08d}.json').read_text())
        assert evidence['optimizer_update']==update and evidence['held_out'] is True and evidence['updated'] is False
        assert evidence['diagnostics']['gates']['eligible_for_idql_evaluation'] is False
    assert not (tmp_path/'run/stop.request').exists()
    left=serialization.msgpack_restore((Path(expected['checkpoint'])/'state.msgpack').read_bytes())
    right=serialization.msgpack_restore((Path(final['checkpoint'])/'state.msgpack').read_bytes())
    for a,b in zip(jax.tree_util.tree_leaves(left),jax.tree_util.tree_leaves(right)):
        np.testing.assert_array_equal(a,b)
    rows=[json.loads(line) for line in (tmp_path/'run/metrics.jsonl').read_text().splitlines()]
    assert [row['optimizer_update'] for row in rows]==[1,2,3,4]
    assert sorted((tmp_path/'run/scorers').iterdir())==[tmp_path/'run/scorers/update_00000002',tmp_path/'run/scorers/update_00000004']
    # Rollback is rejected before truncating any already-published history.
    original=(tmp_path/'run/metrics.jsonl').read_bytes()
    with pytest.raises(ValueError,match='latest boundary'):runner.train(config,native,resume=stopped['checkpoint'])
    assert (tmp_path/'run/metrics.jsonl').read_bytes()==original
    complete=runner.train(config,native,resume=final['checkpoint'])
    assert complete['already_complete_no_parameter_update'] is True
    assert complete['selected_scorer'] is None and complete['selection']==final['selection']
    assert (tmp_path/'run/metrics.jsonl').read_bytes()==original


def test_dryrun_has_explicit_adapter_exact_roster_and_no_actor(native,tmp_path):
    config=configuration(tmp_path);result=runner.dry_run(config,native)
    assert result['gpu_used'] is False and result['actor_loaded'] is False
    assert result['prepared']['adapter_contract']['action_codec']['encoded_dim']==64
    assert result['contract']['upstream_commit'].startswith('7582571')
    wrong=deepcopy(config);wrong['adapter']['mode']=None
    with pytest.raises(ValueError,match='explicit'):runner.validate_config(wrong)


def test_accumulation_torchrun_preference_and_old_pt_do_not_silently_enter(native,tmp_path,monkeypatch):
    config=configuration(tmp_path)
    wrong=deepcopy(config);wrong['training']['gradient_accumulation_steps']=2
    with pytest.raises(ValueError,match='accumulation1'):runner.validate_config(wrong)
    monkeypatch.setenv('WORLD_SIZE','2')
    with pytest.raises(ValueError,match='torchrun'):runner.train(config,native)
    monkeypatch.setenv('WORLD_SIZE','1')
    path=tmp_path/'preference.yaml';path.write_text('backend: official_idql_jax\npreference: {weight: 0.1}\n')
    with pytest.raises(ValueError,match='Unknown config'):runner.load_config(path)
    old=tmp_path/'run/checkpoints/old.pt';old.parent.mkdir(parents=True);old.write_bytes(b'not a JAX checkpoint')
    with pytest.raises(ValueError,match='.pt'):runner.train(config,native,resume=old)


def test_fresh_process_dryrun_never_imports_policy_openpi_or_preference(native,tmp_path):
    config=configuration(tmp_path);path=tmp_path/'config.json';path.write_text(json.dumps(config))
    script='''
import importlib.abc,json,sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='post_training.policy' or fullname.startswith(('openpi','post_training.preference')):
   raise AssertionError(fullname)
sys.meta_path.insert(0,Guard())
from post_training.official_idql.runner import load_config,dry_run
value=dry_run(load_config(sys.argv[1]),sys.argv[2])
assert value['actor_loaded'] is False and value['gpu_used'] is False
assert 'jax' not in sys.modules
'''
    result=subprocess.run([sys.executable,'-B','-c',script,str(path),str(native)],
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'},capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize('visible',[None,'','0','0,0','0,-1'])
def test_gpu_selection_is_rejected_before_any_jax_device_call(tmp_path,monkeypatch,visible):
    config=configuration(tmp_path);config['runtime'].update(platform='gpu',device_count=2)
    config['training'].update(global_batch_size=64,per_device_batch_size=32)
    if visible is None:monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    else:monkeypatch.setenv('CUDA_VISIBLE_DEVICES',visible)
    import jax
    monkeypatch.setattr(jax,'local_devices',lambda **kwargs:pytest.fail('GPU preflight must reject before device initialization'))
    with pytest.raises(ValueError,match='CUDA_VISIBLE_DEVICES'):runner._devices(config)


def test_two_virtual_cpu_devices_run_full_pmap_train_validation_checkpoint_restore_and_export(native,tmp_path):
    """Run the actual pmap runner in a fresh CPU-only interpreter, not a mock."""
    import re
    config=configuration(tmp_path);config['runtime']['device_count']=2
    config['training']['per_device_batch_size']=2
    path=tmp_path/'pmap.json';path.write_text(json.dumps(config))
    script='''
from copy import deepcopy
from pathlib import Path
import json,sys,numpy as np,jax
from flax import serialization
from post_training.official_idql.runner import load_config,train
c=load_config(sys.argv[1]);gold=deepcopy(c);gold['output_dir']+='-uninterrupted'
a=train(gold,sys.argv[2]);paused=train(c,sys.argv[2],stop_after=2)
b=train(c,sys.argv[2],resume=paused['checkpoint'])
assert b['status']=='COMPLETED' and b['optimizer_updates']==4 and b['actor_optimizer_updates']==0
left=serialization.msgpack_restore((Path(a['checkpoint'])/'state.msgpack').read_bytes())
right=serialization.msgpack_restore((Path(b['checkpoint'])/'state.msgpack').read_bytes())
for x,y in zip(jax.tree_util.tree_leaves(left),jax.tree_util.tree_leaves(right)):np.testing.assert_array_equal(x,y)
assert np.asarray(right['step']).tolist()==[4,4]
assert (Path(c['output_dir'])/'scorers/update_00000004/manifest.json').is_file()
'''
    flags=re.sub(r'--xla_force_host_platform_device_count=\d+','',os.environ.get('XLA_FLAGS',''))
    result=subprocess.run([sys.executable,'-B','-c',script,str(path),str(native)],
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu',
             'XLA_FLAGS':flags+' --xla_force_host_platform_device_count=2'},
        capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr


def _synthetic_selection_evidence(agent,adapter,*args,**kwargs):
    """Control-flow fixture only: no claim these toy trajectories pass a gate."""
    update=int(np.asarray(agent.step).flat[0]);passed=update!=4
    rho=-.8 if update==2 else -.95
    return dict(optimizer_update=update,count=len(adapter),held_out=True,updated=False,
        critic_loss=.1 if passed else .00001,value_loss=.2,q1_mse=.1,q2_mse=.1,
        diagnostics=dict(finite=True,spearman_v_remaining_blocks=rho,median_per_trajectory_spearman=-.8,
            ess_fraction=.8,success_terminal_weight_share=.1,
            human_comparison=dict(human_higher_fraction=.9 if passed else .45),
            same_state_action_ranking=dict(human_higher_fraction=.8,available=True,evidence_verified=True,
                held_out=True,finite=True,pair_count=2,root_count=2),
            gates=dict(progress_passed=True,human_passed=passed,same_state_passed=True,
                ess_passed=True,terminal_mass_passed=True,eligible_for_idql_evaluation=passed,
                failures=[] if passed else ['human_higher_fraction'],unavailable=[])))


def test_best_selection_survives_resume_and_failed_later_lower_td_checkpoint(native,tmp_path,monkeypatch):
    """Exercise real model/CP I/O with synthetic gate results, not model quality."""
    config=configuration(tmp_path);config['training']['validation_interval']=1
    monkeypatch.setattr(runner,'evaluate',_synthetic_selection_evidence)
    stopped=runner.train(config,native,stop_after=2)
    best=stopped['selected_scorer']
    assert best['optimizer_update']==2
    assert [v['optimizer_update'] for v in stopped['selection']['retained_candidates']]==[2]
    # A validation-only update1 can have better rho but is not a retained full CP.
    assert stopped['selection']['best_qualified_scorer']['optimizer_update']!=1
    stored=json.loads((Path(stopped['checkpoint'])/'control.json').read_text())['metadata']['selection']
    assert stored==stopped['selection']
    # CP metadata is the recovery authority even if the derived sidecar is gone.
    (tmp_path/'run/selection.json').unlink()
    final=runner.train(config,native,resume=stopped['checkpoint'])
    assert final['selected_scorer']==best
    assert final['selected_scorer']['optimizer_update']==2
    assert final['selection']['best_progress_candidate']['optimizer_update']==4
    assert [v['optimizer_update'] for v in final['selection']['retained_candidates']]==[2,4]
    assert json.loads((tmp_path/'run/selection.json').read_text())==final['selection']
    assert json.loads((Path(final['checkpoint'])/'control.json').read_text())['metadata']['selection']==final['selection']
    assert sorted(p.name for p in (tmp_path/'run/validation').glob('update_*.json'))==[f'update_{i:08d}.json' for i in range(1,5)]


def test_resume_cannot_change_quality_thresholds_or_rewrite_history(native,tmp_path):
    config=configuration(tmp_path);stopped=runner.train(config,native,stop_after=2)
    history=(tmp_path/'run/metrics.jsonl').read_bytes();selection=(tmp_path/'run/selection.json').read_bytes()
    changed=deepcopy(config);changed['diagnostics']['thresholds']['rho_max']=-.6
    with pytest.raises(ValueError,match='contract'):
        runner.train(changed,native,resume=stopped['checkpoint'])
    assert (tmp_path/'run/metrics.jsonl').read_bytes()==history
    assert (tmp_path/'run/selection.json').read_bytes()==selection
