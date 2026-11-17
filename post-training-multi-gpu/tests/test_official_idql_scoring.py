"""Real target-network export/restore and input-identity failure checks."""
from __future__ import annotations

import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import pytest
pytest.importorskip('gym')
jax = pytest.importorskip('jax')
pytest.importorskip('flax')
pytest.importorskip('optax')

from post_training.io import stable_hash
from post_training.official_idql import core, checkpoint, scoring
from post_training.official_idql.adapters import ACTION_CODEC, encode_action_chunks


def fixture_state():
    state = core.create_agent(7, np.zeros(8, np.float32), np.zeros(64, np.float32), hidden_dims=(8, 8))
    rng = np.random.default_rng(9)
    batch = dict(observations=rng.normal(size=(4, 8)).astype(np.float32),
                 next_observations=rng.normal(size=(4, 8)).astype(np.float32),
                 actions=rng.normal(size=(4, 64)).astype(np.float32),
                 rewards=np.full(4, -1, np.float32), masks=np.ones(4, np.float32))
    state, _ = core.update(state, batch)
    norm = {key: dict(q01=[-1.] * dim, q99=[1.] * dim, codec='openpi_quantile')
            for key, dim in [('actions', 7), ('state', 8)]}
    adapter = dict(mode='vector', vector_source='normalized_proprio', action_codec=ACTION_CODEC,
                   normalization_sha256=stable_hash(norm))
    contract = dict(backend=checkpoint.BACKEND, upstream_commit=checkpoint.UPSTREAM_COMMIT,
                    test='official_target_export', prepared=dict(adapter_contract=adapter,
                    normalization=norm, normalization_sha256=stable_hash(norm)))
    return state, norm, adapter, contract


def test_target_export_roundtrip_raw_h6_and_h8_scoring_and_metadata_isolation(tmp_path):
    state, norm, adapter, contract = fixture_state()
    directory = tmp_path / 'scorer'
    scoring.export_scorer(directory, state, contract, normalization=norm, adapter_contract=adapter)
    template = core.create_agent(99, np.zeros(8, np.float32), np.zeros(64, np.float32), hidden_dims=(8, 8))
    loaded = scoring.load_scorer(directory, template, contract, expected_adapter_contract=adapter)
    rng = np.random.default_rng(90)
    obs = rng.normal(size=(2, 8)).astype(np.float32)
    actions = rng.uniform(-.5, .5, (2, 4, 8, 7)).astype(np.float32)
    mask = np.ones((2, 4, 8), bool); mask[1, :, 6:] = False; actions[1, :, 6:] = np.nan
    encoded = encode_action_chunks(actions, mask, normalization=norm['actions'])
    expected = core.score_actions(state, obs, encoded)
    got = loaded.score_chunks(obs, actions, mask)
    for key in expected:
        np.testing.assert_array_equal(np.asarray(got[key]), np.asarray(expected[key]))
    mutated = loaded.metadata; mutated['normalization']['actions']['q01'][0] = -900.
    again = loaded.score_chunks(obs, actions, mask)
    np.testing.assert_array_equal(np.asarray(again['q_min']), np.asarray(expected['q_min']))
    assert loaded.metadata['resume_capable'] is False and loaded.metadata['actor_updates'] == 0
    assert not hasattr(loaded, 'update')
    with pytest.raises(ValueError):
        checkpoint.load_checkpoint(directory, template, contract)
    with pytest.raises(ValueError, match='immutable'):
        scoring.export_scorer(directory, state, contract, normalization=norm, adapter_contract=adapter)


def test_export_rejects_replicated_divergence_and_undeclared_device_axis(tmp_path):
    state, norm, adapter, contract = fixture_state()
    replicated = jax.tree_util.tree_map(lambda x: np.stack([np.asarray(x), np.asarray(x)]), state)
    with pytest.raises(ValueError, match='replicated=True'):
        scoring.export_scorer(tmp_path/'undeclared', replicated, contract,
                              normalization=norm, adapter_contract=adapter)
    directory = scoring.export_scorer(tmp_path/'valid', replicated, contract, normalization=norm,
                                     adapter_contract=adapter, replicated=True)
    loaded = scoring.load_scorer(directory, state, contract)
    assert loaded.metadata['step'] == 1
    leaves, structure = jax.tree_util.tree_flatten(replicated.target_critic.params)
    leaves[0] = leaves[0].copy(); leaves[0][1].flat[0] += 1
    bad = replicated.replace(target_critic=replicated.target_critic.replace(
        params=jax.tree_util.tree_unflatten(structure, leaves)))
    with pytest.raises(ValueError, match='replicas differ'):
        scoring.export_scorer(tmp_path/'divergent', bad, contract, normalization=norm,
                              adapter_contract=adapter, replicated=True)
    assert not (tmp_path/'divergent').exists()


def test_scorer_rejects_corruption_wrong_contract_and_network_shape(tmp_path):
    state, norm, adapter, contract = fixture_state()
    directory = scoring.export_scorer(tmp_path/'valid', state, contract, normalization=norm,
                                     adapter_contract=adapter)
    with pytest.raises(ValueError, match='contract differs'):
        scoring.load_scorer(directory, state, {**contract, 'test': 'other'})
    with pytest.raises(ValueError, match='input adapter differs'):
        scoring.load_scorer(directory, state, contract, expected_adapter_contract={**adapter, 'mode': 'pixels'})
    wrong = core.create_agent(7, np.zeros(8, np.float32), np.zeros(64, np.float32), hidden_dims=(9, 8))
    with pytest.raises(ValueError, match='parameter shape'):
        scoring.load_scorer(directory, wrong, contract)
    payload = directory/'target_q.msgpack'
    data = bytearray(payload.read_bytes()); data[-1] ^= 1; payload.write_bytes(data)
    with pytest.raises(ValueError, match='SHA256'):
        scoring.load_scorer(directory, state, contract)


def test_export_rejects_mutually_consistent_inputs_from_a_different_run(tmp_path):
    state, norm, adapter, contract = fixture_state()
    other_norm = {**norm, 'provenance': {'other_dataset': True}}
    other_adapter = {**adapter, 'normalization_sha256': stable_hash(other_norm)}
    with pytest.raises(ValueError, match='full training contract'):
        scoring.export_scorer(tmp_path/'wrong', state, contract, normalization=other_norm,
                              adapter_contract=other_adapter)
    assert not (tmp_path/'wrong').exists()


def test_different_candidate_durations_cannot_be_ranked_as_one_pool(tmp_path):
    state, norm, adapter, contract = fixture_state()
    directory = scoring.export_scorer(tmp_path/'valid', state, contract, normalization=norm,
                                     adapter_contract=adapter)
    loaded = scoring.load_scorer(directory, state, contract)
    actions = np.zeros((4, 8, 7), np.float32)
    mask = np.ones((4, 8), bool); mask[0, 6:] = False
    with pytest.raises(ValueError, match='same pool'):
        loaded.score_chunks(np.zeros(8, np.float32), actions, mask)
    encoded = encode_action_chunks(actions, mask)
    with pytest.raises(ValueError, match='same pool'):
        loaded.score_encoded(np.zeros(8, np.float32), encoded)
