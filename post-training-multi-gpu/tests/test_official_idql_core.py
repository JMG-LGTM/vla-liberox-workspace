"""Independent upstream-create and numerical CPU references for official IDQL.

The production side is critic-only. The reference side constructs the original
7582571 DDPMIQLLearner (including its unused DDPM actor) and invokes its actual
update methods. It never calls the adapter's creator/update as a reference.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import pytest

# Before importing JAX: tests are forbidden from opening a CUDA backend.
os.environ['JAX_PLATFORMS']='cpu'
os.environ['JAX_PLATFORM_NAME']='cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''
if 'jax' not in sys.modules:
 flags=[x for x in os.environ.get('XLA_FLAGS','').split() if not x.startswith('--xla_force_host_platform_device_count=')]
 os.environ['XLA_FLAGS']=' '.join(flags+['--xla_force_host_platform_device_count=2'])
gym=pytest.importorskip('gym',reason='Install the official-idql optional dependencies to run upstream constructor checks')
jax=pytest.importorskip('jax',reason='Install the official-idql optional dependencies to run JAX numerical checks')
pytest.importorskip('flax',reason='Official IDQL numerical tests require Flax')
pytest.importorskip('optax',reason='Official IDQL numerical tests require Optax')
import jax.numpy as jnp
import numpy as np
from post_training.official_idql import core,upstream

COMMIT='7582571ce686c77b4ee0f592dd639ab8151295fa'
PINNED={
 'jaxrl5/agents/ddpm_iql_simple/ddpm_iql_learner.py':'8dc87a85e5f5720c493556df8ec3be489786a34db9e989bebc9b96eeae27566f',
 'jaxrl5/networks/ensemble.py':'3d7e92028b0193fb806554edc226694dfcf177904f5f78a7f42559d381415d5e',
 'jaxrl5/networks/mlp.py':'e90a14f82ddc819397a7d7dc7dc4b93d5366d04f13f64eceea5aeef7677b69d8',
 'jaxrl5/networks/state_action_value.py':'eec608a5d535afc57116a46fedb24897edcaacd1581eba51592a5d42248c1779',
 'jaxrl5/networks/state_value.py':'6da0dc238a15b48d3f7f6484f191f9001bf4d6316b4a141406f26a5b053fc4ad',
 'jaxrl5/networks/pixel_multiplexer.py':'324c8c13cbe2c85e19587f7d76e3bedf268de7c0907cadd454872365df39cbfb',
 'jaxrl5/agents/agent.py':'38cdfa470c4f78161e3011a77d5ba80c86c88ff2c8ab7fdcf04ceeaee05cbb63',
 'jaxrl5/networks/diffusion.py':'ff0e3b01ddb5739e11f17f3a199d3d2faed308680270c17e0f0de062f34dda9a',
}
MEASUREMENTS={}
OBS=11
ACT=64  # Eight7D execution actions plus eight validity-mask coordinates.
SEED=23
ATOL=2e-6
RTOL=2e-5


@pytest.fixture(scope='session',autouse=True)
def cpu_receipt():
 devices=jax.devices();assert devices and all(d.platform=='cpu' for d in devices)
 MEASUREMENTS['devices']=[{'platform':d.platform,'id':d.id,'kind':d.device_kind} for d in devices]
 MEASUREMENTS['cpu_affinity']=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None
 MEASUREMENTS['versions']={name:importlib.metadata.version(name) for name in ['jax','jaxlib','flax','optax','gym','numpy','pytest']}
 MEASUREMENTS['upstream_commit']=COMMIT
 MEASUREMENTS['reference_execution']={'constructor':'unmodified DDPMIQLLearner.create',
  'update':'unmodified DDPMIQLLearner.critic_update with its original @jax.jit',
  'separate_eager_update_v_then_update_q':'diagnostic only; different XLA fusion boundaries are not the official compiled reference'}
 MEASUREMENTS['tolerance']={'absolute':ATOL,'relative':RTOL,'initialization_and_checkpoint_absolute':0.0}
 yield
 destination=os.environ.get('OFFICIAL_IDQL_TEST_METRICS_PATH')
 if destination:Path(destination).write_text(json.dumps(MEASUREMENTS,indent=2,allow_nan=False)+'\n')


def parts(agent):
 return {'critic_params':agent.critic.params,'critic_adam':agent.critic.opt_state,'critic_step':agent.critic.step,
  'target_params':agent.target_critic.params,'target_step':agent.target_critic.step,
  'value_params':agent.value.params,'value_adam':agent.value.opt_state,'value_step':agent.value.step,
  'rng':agent.rng}


def compare_tree(actual,expected,*,atol=ATOL,rtol=RTOL):
 left,lt=jax.tree_util.tree_flatten(actual);right,rt=jax.tree_util.tree_flatten(expected)
 assert lt==rt,'PyTree structure differs'
 largest=0.;count=0
 for a,b in zip(left,right):
  a,b=np.asarray(a),np.asarray(b)
  assert a.shape==b.shape and a.dtype==b.dtype
  assert np.isfinite(a).all() and np.isfinite(b).all()
  np.testing.assert_allclose(a,b,atol=atol,rtol=rtol)
  if a.size:largest=max(largest,float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))))
  count+=1
 return {'leaves':count,'max_absolute_error':largest}


def max_tree_difference(actual,expected):
 return max(float(np.max(np.abs(np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64))))
  for a,b in zip(jax.tree_util.tree_leaves(actual),jax.tree_util.tree_leaves(expected)) if np.asarray(a).size)


def reference(seed=SEED):
 # The upstream constructor consumes real gym samples solely to determine shape.
 obs=gym.spaces.Box(low=-np.inf,high=np.inf,shape=(OBS,),dtype=np.float32)
 act=gym.spaces.Box(low=-1.,high=1.,shape=(ACT,),dtype=np.float32)
 obs.seed(100);act.seed(101)
 return upstream.load_upstream().DDPMIQLLearner.create(seed,obs,act,
  critic_hidden_dims=(256,256),critic_lr=3e-4,value_lr=3e-4,
  critic_hyperparam=.7,discount=.99,tau=.005)


def fresh(*,axis_name=None):
 return core.create_agent(SEED,np.zeros(OBS,dtype=np.float32),np.zeros(ACT,dtype=np.float32),
  hidden_dims=(256,256),critic_lr=3e-4,value_lr=3e-4,expectile=.7,discount=.99,tau=.005,axis_name=axis_name)


def batch(generator,size=64):
 actions=generator.uniform(-1,1,(size,ACT)).astype(np.float32)
 actions[:,-8:]=1.
 return {'observations':jnp.asarray(generator.normal(size=(size,OBS)).astype(np.float32)),
  'actions':jnp.asarray(actions),'rewards':jnp.asarray(generator.normal(-.5,.8,size=size).astype(np.float32)),
  'masks':jnp.asarray((generator.uniform(size=size)>.2).astype(np.float32)),
  'next_observations':jnp.asarray(generator.normal(size=(size,OBS)).astype(np.float32))}


def official_step(agent,data):
 # Execute the original class's compiled critic_update, not a local loss
 # rewrite or an eager-only reference with different XLA fusion boundaries.
 return agent.critic_update(data)


def test_fixed_source_bytes_and_real_bound_function_objects():
 module=upstream.load_upstream();source=Path(inspect.getsourcefile(module.DDPMIQLLearner)).resolve()
 verified=upstream.verify_upstream();assert verified['commit']==COMMIT
 vendor=source.parents[3]
 got={name:hashlib.sha256((vendor/name).read_bytes()).hexdigest() for name in PINNED}
 assert got==PINNED
 agent=fresh()
 assert getattr(type(agent).update_v,'__func__',type(agent).update_v) is module.DDPMIQLLearner.update_v
 assert getattr(type(agent).update_q,'__func__',type(agent).update_q) is module.DDPMIQLLearner.update_q
 MEASUREMENTS['source_identity']={'files_sha256':got,'functions_are_upstream_objects':True,'module_file':str(source),
  'all_vendored_files_verified':len(verified['files'])}


def test_default_vector_network_initialization_and_forward_match_official_create():
 a,b=fresh(),reference();rng=np.random.default_rng(90);data=batch(rng)
 result=compare_tree(parts(a),parts(b),atol=0,rtol=0)
 for name,arguments in [('critic',(data['observations'],data['actions'])),('value',(data['observations'],))]:
  aa,bb=getattr(a,name),getattr(b,name)
  compare_tree(aa.apply_fn({'params':aa.params},*arguments),bb.apply_fn({'params':bb.params},*arguments),atol=0,rtol=0)
 assert a.critic.apply_fn({'params':a.critic.params},data['observations'],data['actions']).shape==(2,64)
 for invalid in [{'critic_lr':float('nan')},{'value_lr':True},{'expectile':1.},{'discount':0.},
  {'hidden_dims':(True,256)},{'hidden_dims':(256.5,256)},{'encoder_features':(4,4)},{'latent_dim':False}]:
  with pytest.raises(ValueError):core.create_agent(SEED,np.zeros(OBS,dtype=np.float32),np.zeros(ACT,dtype=np.float32),**invalid)
 MEASUREMENTS['official_create_initialization']=result


def test_five_updates_match_independent_full_official_learner_params_adam_and_ema():
 a,b=fresh(),reference();eager_reference=reference();rng=np.random.default_rng(901);records=[];eager_diagnostic=[]
 for step in range(1,6):
  data=batch(rng);a,ai=core.update(a,data);b,bi=official_step(b,data)
  record={'update':step,'state':compare_tree(parts(a),parts(b),atol=0,rtol=0)}
  for name in ['value_loss','critic_loss']:compare_tree(ai[name],bi[name],atol=0,rtol=0)
  assert int(a.critic.step)==int(a.value.step)==step
  if hasattr(a,'step'):assert int(a.step)==step
  records.append(record)
  eager_reference,_=eager_reference.update_v(data);eager_reference,_=eager_reference.update_q(data)
  eager_diagnostic.append({'update':step,'max_absolute_difference_vs_compiled_official':max_tree_difference(parts(eager_reference),parts(b))})
 MEASUREMENTS['five_updates_vs_official_full_creator']=records
 MEASUREMENTS['separate_eager_subcalls_diagnostic_not_equivalence_reference']=eager_diagnostic


def test_ema_updates_once_and_value_target_uses_minimum_of_both_target_qs():
 a=fresh();data=batch(np.random.default_rng(42));before=a
 qs=np.asarray(a.target_critic.apply_fn({'params':a.target_critic.params},data['observations'],data['actions']))
 value=np.asarray(a.value.apply_fn({'params':a.value.params},data['observations']))
 difference=qs.min(axis=0)-value
 expected_loss=np.where(difference>0,.7,.3)*difference**2
 diagnostics=core.evaluate_losses(a,data)
 compare_tree(parts(a),parts(before),atol=0,rtol=0)
 np.testing.assert_allclose(np.asarray(diagnostics['value_loss']),expected_loss.mean(),atol=ATOL,rtol=RTOL)
 a,info=core.update(a,data)
 np.testing.assert_allclose(np.asarray(info['value_loss']),expected_loss.mean(),atol=ATOL,rtol=RTOL)
 expected=jax.tree_util.tree_map(lambda new,old:.005*new+.995*old,a.critic.params,before.target_critic.params)
 measurement=compare_tree(a.target_critic.params,expected)
 poisoned=jax.tree_util.tree_map(lambda x:jnp.full_like(x,jnp.nan) if jnp.issubdtype(jnp.asarray(x).dtype,jnp.floating) else x,a.critic.opt_state)
 _,bad_info=core.update(a.replace(critic=a.critic.replace(opt_state=poisoned)),data)
 assert not bool(bad_info['finite'])
 MEASUREMENTS['target_q_min_and_single_ema']=measurement
 MEASUREMENTS['read_only_validation_state_unchanged']=True
 MEASUREMENTS['nonfinite_adam_rejected_by_update_status']=True


def test_two_virtual_cpu_pmap_reduces_gradients_before_adam_vs_global64():
 if jax.local_device_count()!=2:
  pytest.skip('Full two-device check requires a fresh process: tools/verify_official_idql.py; not verified in this pytest process')
 distributed=fresh(axis_name='devices');reference_agent=reference();rng=np.random.default_rng(134)
 replicated=jax.device_put_replicated(distributed,jax.local_devices())
 update=jax.pmap(core.update,axis_name='devices');records=[]
 for step in range(1,4):
  data=batch(rng)
  # Opposed reward halves strongly distinguish mean gradients from averaged Adam updates.
  data['rewards']=data['rewards']+jnp.concatenate((jnp.full((32,),7.),jnp.full((32,),-6.)))
  local=jax.tree_util.tree_map(lambda x:x.reshape((2,32)+x.shape[1:]),data)
  replicated,info=update(replicated,local);reference_agent,reference_info=official_step(reference_agent,data)
  hosts=[jax.tree_util.tree_map(lambda x:x[rank],replicated) for rank in range(2)]
  records.append({'update':step,'rank0_vs_global64':compare_tree(parts(hosts[0]),parts(reference_agent)),
   'rank1_vs_global64':compare_tree(parts(hosts[1]),parts(reference_agent)),
   'replicas':compare_tree(parts(hosts[0]),parts(hosts[1]),atol=0,rtol=0)})
  for key in ['value_loss','critic_loss']:
   np.testing.assert_allclose(np.asarray(info[key]),np.broadcast_to(np.asarray(reference_info[key]),(2,)),atol=ATOL,rtol=RTOL)
  if step==1:
   local_agents=[official_step(reference(),jax.tree_util.tree_map(lambda x:x[32*rank:32*(rank+1)],data))[0] for rank in range(2)]
   incorrect_average=jax.tree_util.tree_map(lambda x,y:(x+y)/2,local_agents[0].critic.params,local_agents[1].critic.params)
   gap=max_tree_difference(incorrect_average,reference_agent.critic.params)
   assert gap>1e-5,'Fixture did not distinguish averaging Adam updates from averaging gradients'
   MEASUREMENTS['incorrect_average_after_adam_max_error']=gap
 MEASUREMENTS['virtual_cpu2_global64_three_updates']=records


def test_target_q_min_and_greedy_match_upstream_compute_q_for_candidates():
 agent=fresh();rng=np.random.default_rng(321)
 observations=jnp.asarray(rng.normal(size=(3,OBS)).astype(np.float32));actions=jnp.asarray(rng.uniform(-1,1,(3,4,ACT)).astype(np.float32))
 scores=core.score_actions(agent,observations,actions)
 flat_obs=jnp.repeat(observations,4,axis=0);flat_actions=actions.reshape(12,ACT)
 module=upstream.load_upstream()
 expected=module.compute_q(agent.target_critic.apply_fn,agent.target_critic.params,flat_obs,flat_actions).reshape(3,4)
 compare_tree(scores['q_min'],expected)
 np.testing.assert_allclose(np.asarray(scores['q_min']),np.minimum(np.asarray(scores['q1']),np.asarray(scores['q2'])),atol=0,rtol=0)
 chosen,indices,returned=core.greedy_actions(agent,observations,actions)
 indices=np.asarray(indices);np.testing.assert_array_equal(indices,np.asarray(expected).argmax(axis=-1))
 np.testing.assert_array_equal(np.asarray(chosen),np.asarray(actions)[np.arange(3),indices])
 # Exact ties must choose the first slot as original jnp.argmax does.
 tied=jnp.repeat(actions[:,:1,:],4,axis=1);_,tied_indices,_=core.greedy_actions(agent,observations,tied)
 np.testing.assert_array_equal(np.asarray(tied_indices),np.zeros(3,dtype=np.int32))
 MEASUREMENTS['target_q_candidate_selection']={'candidate_count':4,'observations':3,'upstream_compute_q_reference':True,'tie_first_index':True}


def test_full_checkpoint_restore_preserves_all_state_and_numpy_sampler(tmp_path):
 from post_training.official_idql import checkpoint
 agent=fresh();rng=np.random.default_rng(117)
 for _ in range(2):agent,_=core.update(agent,batch(rng))
 sampling_state=rng.bit_generator.state
 contract={'schema_version':1,'backend':'official_idql_jax','test_contract':'official_758_vector_global64_cpu',
  'upstream_commit':COMMIT,'action_dim':ACT,'observation_dim':OBS}
 directory=tmp_path/'checkpoint_00000002'
 checkpoint.save_checkpoint(directory,agent,contract,sampling_state,{'update':2,'test_only':True})
 resumed,restored_rng,metadata=checkpoint.load_checkpoint(directory,fresh(),contract)
 initial=compare_tree(parts(resumed),parts(agent),atol=0,rtol=0)
 rng2=np.random.default_rng();rng2.bit_generator.state=restored_rng
 assert metadata['update']==2
 for _ in range(3):
  b1,b2=batch(rng),batch(rng2);compare_tree(b1,b2,atol=0,rtol=0)
  agent,_=core.update(agent,b1);resumed,_=core.update(resumed,b2)
 final=compare_tree(parts(resumed),parts(agent),atol=0,rtol=0)
 assert rng.bit_generator.state==rng2.bit_generator.state
 assert int(agent.critic.step)==int(resumed.critic.step)==5
 MEASUREMENTS['checkpoint_resume2_to5']={'at_restore':initial,'after_three_more_updates':final,'numpy_sampler_exact':True}
 if jax.local_device_count()==2:
  template=jax.device_put_replicated(fresh(axis_name='devices'),jax.local_devices())
  # Distinct device RNGs must survive the full-state checkpoint, even though
  # deterministic Q/V updates do not consume those keys.
  template=template.replace(rng=jax.random.split(jax.random.PRNGKey(447),2))
  distributed=template;sampler=np.random.default_rng(118);update=jax.pmap(core.update,axis_name='devices')
  def next_sharded(generator):return jax.tree_util.tree_map(lambda x:x.reshape((2,32)+x.shape[1:]),batch(generator))
  for _ in range(2):distributed,_=update(distributed,next_sharded(sampler))
  checkpoint.save_checkpoint(tmp_path/'pmap_checkpoint_00000002',distributed,contract,sampler.bit_generator.state,{'update':2,'world_size':2})
  reloaded,sampler_state,loaded_metadata=checkpoint.load_checkpoint(tmp_path/'pmap_checkpoint_00000002',template,contract)
  restored=compare_tree(parts(reloaded),parts(distributed),atol=0,rtol=0)
  sampler2=np.random.default_rng();sampler2.bit_generator.state=sampler_state
  for _ in range(3):
   b1,b2=next_sharded(sampler),next_sharded(sampler2);compare_tree(b1,b2,atol=0,rtol=0)
   distributed,_=update(distributed,b1);reloaded,_=update(reloaded,b2)
  resumed_diff=compare_tree(parts(reloaded),parts(distributed),atol=0,rtol=0)
  assert loaded_metadata['world_size']==2 and sampler.bit_generator.state==sampler2.bit_generator.state
  assert not np.array_equal(np.asarray(distributed.rng[0]),np.asarray(distributed.rng[1]))
  MEASUREMENTS['pmap_full_checkpoint_resume2_to5']={'at_restore':restored,'after_three_more_updates':resumed_diff,
   'numpy_sampler_exact':True,'distinct_device_rngs_restored':True}


def test_explicit_two_camera_upstream_pixel_adapter_cpu_smoke():
 """Domain-adaptation smoke only, not a claim of state-benchmark identity."""
 rng=np.random.default_rng(932)
 sample={'agent_image':np.zeros((32,32,3),dtype=np.uint8),'wrist_image':np.zeros((32,32,3),dtype=np.uint8),'state':np.zeros(8,dtype=np.float32)}
 agent=core.create_agent(SEED,sample,np.zeros(ACT,dtype=np.float32),observation_kind='pixels',
  pixel_keys=('agent_image','wrist_image'),hidden_dims=(16,16),encoder_features=(4,4,4,4),latent_dim=8)
 def pixels():return {'agent_image':jnp.asarray(rng.integers(0,256,(2,32,32,3),dtype=np.uint8)),
  'wrist_image':jnp.asarray(rng.integers(0,256,(2,32,32,3),dtype=np.uint8)),
  'state':jnp.asarray(rng.normal(size=(2,8)).astype(np.float32))}
 observations=pixels()
 data={'observations':observations,'next_observations':pixels(),
  'actions':jnp.asarray(rng.uniform(-1,1,(2,ACT)).astype(np.float32)),
  'rewards':jnp.asarray([-1.,0.],dtype=jnp.float32),'masks':jnp.asarray([1.,0.],dtype=jnp.float32)}
 before=agent;agent,info=core.update(agent,data)
 assert bool(info['finite']) and int(agent.step)==int(agent.critic.step)==int(agent.value.step)==1
 assert max_tree_difference(agent.critic.params,before.critic.params)>0
 assert max_tree_difference(agent.value.params,before.value.params)>0
 candidates=jnp.asarray(rng.uniform(-1,1,(2,4,ACT)).astype(np.float32))
 scored=core.score_actions(agent,observations,candidates)
 assert scored['q_min'].shape==(2,4) and np.isfinite(np.asarray(scored['q_min'])).all()
 chosen,indices,_=core.greedy_actions(agent,observations,candidates)
 np.testing.assert_array_equal(np.asarray(indices),np.asarray(scored['q_min']).argmax(axis=1))
 np.testing.assert_array_equal(np.asarray(chosen),np.asarray(candidates)[np.arange(2),np.asarray(indices)])
 names=set(agent.critic.params)
 assert {'encoder_0','encoder_1'}<=names,'Both named cameras must have original PixelMultiplexer encoders'
 MEASUREMENTS['explicit_two_camera_pixel_cpu_smoke']={'image_shape':[32,32,3],'camera_count':2,'batch_size':2,
  'candidates':4,'critic_updates':1,'value_updates':1,'encoders':['encoder_0','encoder_1'],
  'test_hidden_dims':[16,16],'test_encoder_features':[4,4,4,4],'test_latent_dim':8,
  'domain_adaptation_not_official_state_network':True,'gpu_validated':False}
