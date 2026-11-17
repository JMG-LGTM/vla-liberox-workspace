"""CPU-only exact transport tests, including deliberately corrupted copies.

The old replication API is used ONLY as a known-good CPU reference. Production
transport must never use it. These fixtures do not constitute real GPU evidence.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest import mock
import numpy as np
import pytest

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['JAX_PLATFORMS']='cpu'
os.environ['JAX_PLATFORM_NAME']='cpu'
if 'jax' not in sys.modules:
    flags=[x for x in os.environ.get('XLA_FLAGS','').split() if not x.startswith('--xla_force_host_platform_device_count=')]
    os.environ['XLA_FLAGS']=' '.join(flags+['--xla_force_host_platform_device_count=2'])
jax=pytest.importorskip('jax')
pytest.importorskip('flax');pytest.importorskip('optax');pytest.importorskip('gym')
from post_training.official_idql import core,runner
from post_training.official_idql.replication import canonical_host_snapshot,replicate_state_via_host,verify_replicated_state


def devices():
    selected=jax.local_devices(backend='cpu')[:2]
    assert len(selected)==2, 'CPU transport verification requires two virtual CPU devices'
    assert all(d.platform=='cpu' for d in selected)
    return selected


def agent():
    return core.create_agent(7,np.zeros(8,np.float32),np.zeros(64,np.float32),
                             hidden_dims=(8,8),axis_name='devices',expectile=.8)


def exact_tree(left,right):
    a,ta=jax.tree_util.tree_flatten(left);b,tb=jax.tree_util.tree_flatten(right)
    assert ta==tb
    for x,y in zip(a,b):
        x,y=np.asarray(x),np.asarray(y)
        assert x.shape==y.shape and x.dtype==y.dtype
        assert x.tobytes()==y.tobytes()


def test_full_agent_initialization_matches_old_cpu_reference_exactly_without_fanout():
    initial=agent();selected=devices()
    reference=jax.device_put_replicated(initial,selected)  # CPU-only old baseline.
    with mock.patch.object(jax,'device_put_replicated',side_effect=AssertionError('GPU fanout API forbidden')):
        result=replicate_state_via_host(initial,selected)
    exact_tree(result,reference)
    assert np.asarray(result.step).tolist()==[0,0]
    assert np.asarray(result.discount).tolist()==[np.float32(.99)]*2
    assert np.asarray(result.tau).tolist()==[np.float32(.005)]*2
    assert np.asarray(result.critic_hyperparam).tolist()==[np.float32(.8)]*2
    proof=verify_replicated_state(result,canonical_host_snapshot(initial),selected)
    assert proof['independent_device_shards_checked'] and proof['assembled_host_view_checked']
    assert proof['verified_leaf_count']>20


def test_each_host_upload_is_an_independent_numpy_buffer_on_the_target_device():
    state=dict(vector=np.arange(5,dtype=np.float32)+1,scalar=.99)
    selected=devices();original_put=jax.device_put;original_sharded=jax.device_put_sharded;captured=[]
    def put(value,**kwargs):
        assert isinstance(value,np.ndarray) and kwargs['may_alias'] is False
        captured.append((value,kwargs['device']))
        return original_put(value,**kwargs)
    def assemble(shards,targets):
        assert list(targets)==selected
        for replica,device in zip(shards,targets):
            assert all(leaf.devices()=={device} for leaf in jax.tree_util.tree_leaves(replica))
        return original_sharded(shards,targets)
    with mock.patch.object(jax,'device_put',side_effect=put),mock.patch.object(jax,'device_put_sharded',side_effect=assemble):
        result=replicate_state_via_host(state,selected)
    assert len(captured)==4
    for i,(value,_) in enumerate(captured):
        for other,_ in captured[i+1:]:assert not np.shares_memory(value,other)
    np.testing.assert_array_equal(np.asarray(result['vector']),np.tile(state['vector'],(2,1)))


@pytest.mark.parametrize('x64',[False,True])
def test_python_scalar_dtypes_follow_jax_configuration_without_changing_existing_float32_bits(x64):
    with jax.experimental.enable_x64(x64):
        snapshot=canonical_host_snapshot(dict(float=.99,integer=7,boolean=True,
            explicit_float32=np.array([.005,-0.,.8],np.float32)))
        assert snapshot['float'].dtype==np.dtype('float64' if x64 else 'float32')
        assert snapshot['integer'].dtype==np.dtype('int64' if x64 else 'int32')
        assert snapshot['boolean'].dtype==np.dtype('bool')
        assert snapshot['explicit_float32'].dtype==np.dtype('float32')
        assert snapshot['explicit_float32'].tobytes()==np.array([.005,-0.,.8],np.float32).tobytes()
        replicated=replicate_state_via_host(snapshot,devices())
        for key,value in snapshot.items():assert np.asarray(replicated[key]).dtype==value.dtype


def test_zeroed_second_device_upload_fails_at_initialization_before_any_update():
    initial=agent();selected=devices();original=jax.device_put;update=mock.Mock()
    def broken(value,**kwargs):
        if kwargs['device']==selected[1]:value=np.zeros_like(value)
        return original(value,**kwargs)
    with mock.patch.object(jax,'device_put',side_effect=broken),pytest.raises(ValueError,match='rank 1 direct host upload'):
        replicated=replicate_state_via_host(initial,selected)
        update(replicated)  # Must be unreachable.
    update.assert_not_called();assert int(initial.step)==0


class HostAssemblyView:
    """Synthetic readback wrapper; all device shards remain real CPU arrays."""
    def __init__(self,array,*,bad_global=False,bad_shard=False):
        self.original=array;self.bad_global=bad_global
        self.addressable_shards=list(array.addressable_shards)
        if bad_shard:
            first=self.addressable_shards[1]
            self.addressable_shards[1]=SimpleNamespace(device=first.device,index=first.index,
                                                     data=np.zeros_like(np.asarray(first.data)))
    def __array__(self,dtype=None,copy=None):
        value=np.asarray(self.original)
        return np.zeros_like(value) if self.bad_global else value


def test_assembled_host_corruption_is_detected_even_when_individual_device_shards_match():
    selected=devices();canonical=np.array([.99,.005,.8],np.float32)
    good=replicate_state_via_host(canonical,selected)
    with pytest.raises(ValueError,match='assembled host view'):
        verify_replicated_state(HostAssemblyView(good,bad_global=True),canonical,selected)


def test_independent_shard_corruption_is_detected_even_when_assembled_host_view_looks_correct():
    selected=devices();canonical=np.array([.99,.005,.8],np.float32)
    good=replicate_state_via_host(canonical,selected)
    with pytest.raises(ValueError,match='rank 1 independent shard'):
        verify_replicated_state(HostAssemblyView(good,bad_shard=True),canonical,selected)


def test_nonfinite_empty_state_and_duplicate_devices_fail_before_upload():
    selected=devices()
    with mock.patch.object(jax,'device_put') as put:
        for state,target in [(dict(x=np.array([np.nan],np.float32)),selected),({},selected),
                             (dict(x=np.ones(2,np.float32)),[selected[0],selected[0]])]:
            with pytest.raises(ValueError):replicate_state_via_host(state,target)
        put.assert_not_called()


def test_actual_pmap_updates_keep_params_optimizer_targets_rng_and_scalars_identical_to_cpu_reference():
    initial=agent();selected=devices()
    old=jax.device_put_replicated(initial,selected)  # Test-only known-good CPU baseline.
    candidate=replicate_state_via_host(initial,selected)
    update=jax.pmap(core.update,axis_name='devices',devices=selected)
    rng=np.random.default_rng(17)
    for _ in range(3):
        batch=dict(observations=rng.normal(size=(2,2,8)).astype(np.float32),
            next_observations=rng.normal(size=(2,2,8)).astype(np.float32),
            actions=rng.normal(size=(2,2,64)).astype(np.float32),rewards=-np.ones((2,2),np.float32),
            masks=np.ones((2,2),np.float32))
        old,old_metrics=update(old,batch);candidate,new_metrics=update(candidate,batch)
        exact_tree(old,candidate);exact_tree(old_metrics,new_metrics)
    assert np.asarray(candidate.step).tolist()==[3,3]


def test_runner_single_device_does_not_call_new_replication_transport(monkeypatch):
    config=dict(runtime=dict(device_count=1),critic=dict(hidden_dims=[8,8],critic_lr=3e-4,value_lr=3e-4,expectile=.8,tau=.005),
        reward=dict(discount=.99),training=dict(seed=7),adapter={})
    adapter=SimpleNamespace(contract={'observation_kind':'vector'},batch=lambda indices:dict(
        observations=np.zeros((1,8),np.float32),actions=np.zeros((1,64),np.float32)))
    from post_training.official_idql import replication
    monkeypatch.setattr(replication,'replicate_state_via_host',lambda *args:pytest.fail('single-device path changed'))
    actual=runner.create_state(config,adapter,devices=[devices()[0]])
    reference=core.create_agent(7,np.zeros(8,np.float32),np.zeros(64,np.float32),hidden_dims=(8,8),expectile=.8)
    from flax.serialization import to_state_dict
    exact_tree(to_state_dict(actual),to_state_dict(reference))


def test_existing_float64_arrays_are_rejected_rather_than_silently_downcast_when_x64_disabled():
    selected=devices();host=np.array([.99,.005,.8],np.float64)
    with jax.experimental.enable_x64(True):
        existing=jax.device_put(host,selected[0])
    for value in (host,existing):
        before=np.asarray(value).tobytes()
        with jax.experimental.enable_x64(False),mock.patch.object(jax,'device_put') as upload:
            with pytest.raises(ValueError,match='refuse implicit precision change'):
                replicate_state_via_host({'existing_array':value},selected)
            upload.assert_not_called()
        assert np.asarray(value).dtype==np.dtype('float64') and np.asarray(value).tobytes()==before
