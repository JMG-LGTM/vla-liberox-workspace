"""Explicit CPU adapters: no policy loading or hidden image encoder."""
from copy import deepcopy
import json

import numpy as np
import pytest

from post_training.io import sha256_file
from post_training.official_idql.adapters import ReplayAdapter, encode_action_chunks, frozen_membership


class FakeCache:
    def __init__(self):
        self.indices=[0,1]
        self.metadata={'count':2,'fingerprint':'f'*64,'normalization':{
            'actions':{'q01':[-1.]*7,'q99':[1.]*7},'state':{'q01':[-1.]*8,'q99':[1.]*8}}}
        self.arrays={'actions':np.zeros((2,8,7),np.float32),
            'action_mask':np.array([[1]*8,[1]*6+[0]*2],bool),
            'proprio':np.zeros((2,8),np.float32),'next_proprio':np.ones((2,8),np.float32),
            'reward':np.array([-1.,0.],np.float32),'bootstrap_mask':np.array([1.,0.],np.float32),
            'pixels':np.concatenate((np.zeros((2,3,16,16),np.uint8),np.full((2,3,16,16),200,np.uint8)),axis=1),
            'next_pixels':np.ones((2,6,16,16),np.uint8)}
    def __len__(self):return 2


def test_h6_tail_mask_padding_nan_and_raw_scoring_use_same_codec():
    from post_training.replay import normalize
    raw=np.arange(56,dtype=np.float32).reshape(1,8,7)/100
    mask=np.array([[1]*6+[0]*2],bool);raw[:,6:]=np.nan
    stats={'q01':[-.5]*7,'q99':[.8]*7}
    normalized=np.zeros_like(raw);normalized[mask]=normalize(raw[mask],stats)
    a=encode_action_chunks(raw,mask,normalization=stats)
    b=encode_action_chunks(normalized,mask)
    np.testing.assert_array_equal(a,b)
    assert a.shape==(1,64) and np.isfinite(a).all()
    assert np.all(a[:,42:56]==0) and a[0,-8:].tolist()==[1]*6+[0]*2
    assert np.isnan(raw[0,7]).all()


@pytest.mark.parametrize('kind',['valid_nan','empty','nonprefix'])
def test_invalid_valid_action_or_mask_rejected(kind):
    actions=np.zeros((1,8,7),np.float32);mask=np.ones((1,8),bool)
    if kind=='valid_nan':actions[0,0,0]=np.nan
    elif kind=='empty':mask[:]=False
    else:mask[0,2]=False
    with pytest.raises(ValueError):encode_action_chunks(actions,mask)


def test_vector_and_separate_pixel_views_preserve_native_reward_and_bootstrap():
    cache=FakeCache()
    vector=ReplayAdapter(cache,{'mode':'vector','vector_source':'normalized_proprio'})
    batch=vector.batch(np.array([1,0],np.int64))
    np.testing.assert_array_equal(batch['rewards'],[0,-1]);np.testing.assert_array_equal(batch['masks'],[0,1])
    pixels=ReplayAdapter(cache,{'mode':'pixels','pixels':{'architecture':'upstream_pixel_multiplexer_d4pg',
        'camera_layout':'separate_rgb_views'}})
    data=pixels.batch(np.array([0],np.int64))['observations']
    assert set(data)=={'agent_pixels','wrist_pixels','state'}
    assert data['agent_pixels'].shape==(1,16,16,3) and np.all(data['agent_pixels']==0)
    assert np.all(data['wrist_pixels']==200) and pixels.contract['pixel_keys']==['agent_pixels','wrist_pixels']
    with pytest.raises(ValueError):ReplayAdapter(cache,{'mode':'pixels','pixels':{'architecture':'upstream_pixel_multiplexer_d4pg'}})
    with pytest.raises(ValueError):ReplayAdapter(cache,{'mode':None})


def test_external_features_bind_encoder_preprocess_layer_and_global_cache_rows(tmp_path):
    cache=FakeCache();path=tmp_path/'features.npz';encoder=tmp_path/'encoder.py';pre=tmp_path/'pre.py'
    encoder.write_text('# supplied frozen feature encoder\n');pre.write_text('# supplied preprocessing\n')
    np.savez(path,observations=np.arange(6,dtype=np.float32).reshape(2,3),
             next_observations=np.arange(6,dtype=np.float32).reshape(2,3)+10)
    spec={'path':str(path),'sha256':sha256_file(path),'cache_fingerprint':cache.metadata['fingerprint'],
          'encoder_path':str(encoder),'encoder_sha256':sha256_file(encoder),
          'preprocess_path':str(pre),'preprocess_sha256':sha256_file(pre),'feature_layer':'frozen_encoder.output'}
    adapter=ReplayAdapter(cache,{'mode':'external_features','external_features':spec})
    np.testing.assert_array_equal(adapter.batch(np.array([1],np.int64))['observations'],[[3,4,5]])
    assert adapter.contract['feature_layer']=='frozen_encoder.output'
    bad=deepcopy(spec);bad['encoder_sha256']='0'*64
    with pytest.raises(ValueError):ReplayAdapter(cache,{'mode':'external_features','external_features':bad})


def test_frozen_roster_includes_roots_without_eligible_chunks(tmp_path):
    info={'dataset_sha256':'d'*64,'episodes':[{'root_run_id':'a','split':'train'},
         {'root_run_id':'b','split':'val'}],'rows':[{'root_run_id':'a','split':'train'}]}
    path=tmp_path/'members.json';value={'dataset_sha256':info['dataset_sha256'],
        'members':[{'root_run_id':'a','split':'train'},{'root_run_id':'b','split':'val'}]}
    path.write_text(json.dumps(value));proof=frozen_membership(path,info)
    assert proof['members']=={'a':'train','b':'val'}
    value['members'].pop();path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='complete frozen'):frozen_membership(path,info)
