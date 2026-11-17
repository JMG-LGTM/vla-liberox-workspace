"""Explicit domain adapters for the separate official JAX Q/V runner.

The replay package, normalization and root split remain owned by the existing
native cache. No actor or new default image encoder is loaded here. Pixel mode
is an explicit upstream DrQ/D4PG domain adaptation, not the paper's state MLP.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from ..io import sha256_file, stable_hash

ACTION_CODEC = dict(name='normalized_h8_a7_masked_flat56_plus_mask8_v1', horizon=8,
                    action_dim=7, encoded_dim=64, padding='zero_before_flatten',
                    mask='shared_nonempty_contiguous_prefix')


def _require(condition, message):
    if not condition:
        raise ValueError('Official IDQL adapter: ' + message)


def file_binding(path):
    path = Path(path).expanduser().resolve(strict=True)
    _require(path.is_file(), 'input must be a regular file')
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}


def encode_action_chunks(actions, action_mask, *, normalization=None):
    """Encode training or scoring chunks identically, masking padding NaNs first.

    Cache actions are already normalized and omit ``normalization``. For raw
    inference actions, supply the same checkpoint's action quantile dictionary.
    Leading dimensions are preserved; the final axis is exactly56+8=64.
    """
    actions, mask = np.asarray(actions, np.float32), np.asarray(action_mask)
    _require(actions.ndim >= 2 and actions.shape[-2:] == (8, 7)
             and mask.shape == actions.shape[:-1], 'expected [...,8,7] actions and [...,8] mask')
    _require(mask.dtype.kind in 'buif' and np.isfinite(mask).all() and np.isin(mask, [0, 1]).all(),
             'action mask must be binary')
    mask = mask.astype(bool)
    lengths = mask.sum(-1)
    _require(np.all(lengths > 0) and np.array_equal(mask, np.arange(8) < lengths[..., None]),
             'every chunk needs a nonempty contiguous prefix')
    _require(np.isfinite(actions[mask]).all(), 'valid actions must be finite')
    encoded = np.zeros_like(actions)
    if normalization is None:
        encoded[mask] = actions[mask]
    else:
        from ..replay import normalize
        encoded[mask] = normalize(actions[mask], normalization)
    return np.concatenate((encoded.reshape(*actions.shape[:-2], 56), mask.astype(np.float32)), axis=-1)


def frozen_membership(path, info):
    """Bind the complete native root roster, including roots with no used chunks."""
    actual = {}
    for row in info['episodes']:
        root, split = row['root_run_id'], row['split']
        _require(root not in actual or actual[root] == split, 'native root crosses splits')
        actual[root] = split
    if path is None:
        return {'source': 'complete_native_dataset', 'members': actual,
                'dataset_sha256': info['dataset_sha256'], 'sha256': stable_hash(actual)}
    binding = file_binding(path)
    value = json.loads(Path(binding['path']).read_text())
    _require(value.get('dataset_sha256') == info['dataset_sha256'], 'frozen dataset identity differs')
    frozen = {}
    for row in value.get('members', []):
        root, split = row['root_run_id'], row['split']
        split = 'val' if split == 'validation' else split
        _require(split in {'train', 'val'} and (root not in frozen or frozen[root] == split),
                 'invalid or conflicting frozen membership')
        frozen[root] = split
    _require(frozen and frozen == actual, 'complete frozen root membership differs from native split')
    return {**binding, 'members': frozen}


def build_diagnostic_memberships(cache_metadata, episodes):
    """Recover complete successful paths without changing deduplicated replay.

    A retained interval can canonically belong to a failed original recording
    while also being a prefix of one or more successful takeover paths. Matching
    uses the existing exact origin/start/end/producer key, never a nearby block.
    """
    from ..data import replay_chunk_key, replay_chunks, replay_end_step
    records = cache_metadata['records']
    lookup = {}
    for index, row in enumerate(records):
        key = (str(row['origin_run_id']), int(row['start']), int(row['end']), str(row['action_source']))
        _require(key not in lookup, 'duplicate unique replay interval')
        lookup[key] = index
    members, paths = {}, {}
    for ep in episodes:
        if not ep.get('success'):
            continue
        end = replay_end_step(ep, False)
        chunks = [c for c in replay_chunks(ep) if int(c['end']) <= end]
        _require(chunks and int(chunks[-1]['end']) == end, 'successful path misses its terminal boundary')
        indices = []
        for position, chunk in enumerate(chunks):
            key = replay_chunk_key(ep, chunk)
            _require(key in lookup, 'complete successful path interval absent from cache')
            index = lookup[key]
            _require(records[index]['root_run_id'] == ep['root_run_id']
                     and records[index]['split'] == ep['split'], 'successful path crosses root/split')
            member = dict(run_id=ep['run_id'], root_run_id=ep['root_run_id'],
                remaining_blocks=len(chunks)-position-1, success_terminal=int(chunk['end']) == end)
            members.setdefault(str(index), []).append(member)
            indices.append(index)
        _require(ep['run_id'] not in paths and len(indices) == len(set(indices)), 'duplicate successful path')
        paths[ep['run_id']] = dict(root_run_id=ep['root_run_id'], split=ep['split'], cache_indices=indices)
    value = dict(schema_version=1, cache_fingerprint=cache_metadata['fingerprint'],
        cache_count=cache_metadata['count'], complete_success_paths=True,
        members=members, successful_paths=paths)
    return {**value, 'sha256': stable_hash(value)}


class ReplayAdapter:
    """Read existing mmap rows into explicit vector/features/pixel observations."""
    def __init__(self, dataset, specification):
        self.dataset, self.specification = dataset, dict(specification)
        mode = specification.get('mode')
        _require(mode in {'vector', 'external_features', 'pixels'}, 'adapter.mode is required')
        self.mode, self.features = mode, None
        self.diagnostic_memberships = None
        self.contract = {'mode': mode, 'action_codec': ACTION_CODEC,
            'normalization_sha256': stable_hash(dataset.metadata['normalization']),
            'cache_fingerprint': dataset.metadata['fingerprint']}
        if mode == 'vector':
            _require(specification.get('vector_source') == 'normalized_proprio',
                     'vector mode explicitly requires normalized_proprio')
            self.contract.update(observation_kind='vector', observation_dimension=8,
                                 vector_source='normalized_proprio')
        elif mode == 'pixels':
            from .upstream import UPSTREAM_COMMIT
            pixels = specification.get('pixels')
            _require(isinstance(pixels, dict) and pixels.get('architecture') == 'upstream_pixel_multiplexer_d4pg',
                     'pixels mode must explicitly select upstream PixelMultiplexer/D4PG')
            _require(pixels.get('camera_layout') == 'separate_rgb_views',
                     'pixels mode requires an explicit separate_rgb_views camera layout')
            self.contract.update(observation_kind='pixels', pixels=dict(pixels),
                image_layout='two_BHWC3_uint8', pixel_keys=['agent_pixels', 'wrist_pixels'], state='normalized_proprio8',
                image_orientation='existing_cache_vertical_flip_once',
                encoder_sharing='one_encoder_per_camera; Q_ensemble_shared_visual_encoder; separate_V_encoders',
                augmentation=dict(implementation='jaxrl5.agents.drq.augmentations.batched_random_crop',
                    upstream_commit=UPSTREAM_COMMIT, pad=pixels.get('random_crop_pad',0),
                    train=pixels.get('random_crop_pad',0)>0, current_and_next_observations=True,
                    independent_cameras=True, per_rank_keys='jax.random.fold_in(axis_index)',
                    validation=False, scoring=False, rng='agent.rng preserved in full checkpoint'),
                domain_adaptation='upstream_DrQ_pixels_not_paper_state_MLP')
        else:
            feature = specification.get('external_features')
            required = {'path', 'sha256', 'cache_fingerprint', 'encoder_path', 'encoder_sha256',
                        'preprocess_path', 'preprocess_sha256', 'feature_layer'}
            _require(isinstance(feature, dict) and set(feature) == required,
                     'external features need aligned data and an inference encoder/preprocess contract')
            data, encoder, preprocess = (file_binding(feature[key]) for key in ('path', 'encoder_path', 'preprocess_path'))
            _require(data['sha256'] == feature['sha256'] and encoder['sha256'] == feature['encoder_sha256']
                     and preprocess['sha256'] == feature['preprocess_sha256'], 'external feature/encoder/preprocess bytes differ')
            _require(feature['cache_fingerprint'] == dataset.metadata['fingerprint']
                     and isinstance(feature['feature_layer'], str) and bool(feature['feature_layer'].strip()),
                     'feature alignment or named encoder layer is not bound')
            with np.load(data['path'], allow_pickle=False) as archive:
                _require(set(archive.files) == {'observations', 'next_observations'},
                         'feature NPZ requires current and next observations')
                self.features = {key: archive[key] for key in archive.files}
            a, b = self.features['observations'], self.features['next_observations']
            _require(a.dtype == b.dtype == np.float32 and a.ndim == b.ndim == 2 and a.shape == b.shape
                     and a.shape[0] == dataset.metadata['count'] and a.shape[1] > 0
                     and np.isfinite(a).all() and np.isfinite(b).all(), 'aligned features must be finite float32[N,D]')
            self.contract.update(observation_kind='vector', observation_dimension=a.shape[1],
                external_features=data, encoder=encoder, preprocess=preprocess, feature_layer=feature['feature_layer'],
                inference_requirement='caller must run this bound encoder/preprocess and layer; no automatic feature extraction')

    def __len__(self):
        return len(self.dataset)

    def bind_diagnostic_memberships(self, value):
        """Bind the supplementary full-path metadata validated during prepare."""
        value = deepcopy(value)
        _require(value.get('sha256') == stable_hash({k:v for k,v in value.items() if k != 'sha256'})
                 and value.get('schema_version') == 1 and value.get('complete_success_paths') is True
                 and value.get('cache_fingerprint') == self.dataset.metadata['fingerprint']
                 and value.get('cache_count') == self.dataset.metadata['count'], 'diagnostic membership identity differs')
        for key, members in value['members'].items():
            _require(str(int(key)) == key and 0 <= int(key) < value['cache_count'], 'invalid membership cache index')
            for member in members:
                path = value['successful_paths'].get(member['run_id'])
                _require(path is not None and int(key) in path['cache_indices']
                         and member['root_run_id'] == path['root_run_id'], 'inconsistent success membership')
                record = self.dataset.metadata['records'][int(key)]
                _require(record['root_run_id'] == path['root_run_id'] and record['split'] == path['split'],
                         'bound success membership crosses native root/split')
                position = path['cache_indices'].index(int(key))
                _require(member['remaining_blocks'] == len(path['cache_indices'])-position-1
                         and member['success_terminal'] == (position == len(path['cache_indices'])-1),
                         'invalid successful path progress')
        for run, path in value['successful_paths'].items():
            _require(len(path['cache_indices']) == len(set(path['cache_indices'])) > 0,
                     'invalid successful path roster')
            for index in path['cache_indices']:
                _require(sum(m['run_id'] == run for m in value['members'].get(str(index), [])) == 1,
                         'successful path coverage incomplete')
        self.diagnostic_memberships = value
        return self

    def diagnostic_rows(self, indices=None):
        """Actual native labels, with each unique held-out interval returned once."""
        indices = np.arange(len(self), dtype=np.int64) if indices is None else np.asarray(indices)
        _require(indices.ndim == 1 and indices.dtype.kind in 'iu'
                 and (not len(indices) or indices.min() >= 0 and indices.max() < len(self)),
                 'invalid diagnostic local indices')
        if self.dataset.arrays is None:
            self.dataset[0]
        result = []
        for index in np.asarray(self.dataset.indices)[indices]:
            row = deepcopy(self.dataset.metadata['records'][int(index)])
            for key in ('remaining_blocks', 'success_terminal'):
                if key in self.dataset.arrays:
                    row[key] = self.dataset.arrays[key][index].item()
            row['cache_index'] = int(index)
            row['progress_scope'] = ('complete_successful_paths' if self.diagnostic_memberships is not None
                                     else 'retained_unique_trajectory_intervals')
            if self.diagnostic_memberships is not None:
                row['success_memberships'] = deepcopy(self.diagnostic_memberships['members'].get(str(index), []))
                row['membership_sha256'] = self.diagnostic_memberships['sha256']
                row['expected_success_paths'] = {run:path for run,path in self.diagnostic_memberships['successful_paths'].items()
                    if path['split'] == row['split']}
            result.append(row)
        return result

    def batch(self, indices):
        indices = np.asarray(indices)
        _require(indices.ndim == 1 and indices.size > 0 and indices.dtype.kind in 'iu'
                 and indices.min() >= 0 and indices.max() < len(self), 'invalid local replay indices')
        if self.dataset.arrays is None:
            self.dataset[0]  # Existing lazy mmap initialization; no policy/model.
        selected = np.asarray(self.dataset.indices)[indices]
        names = ('actions', 'action_mask', 'proprio', 'next_proprio', 'reward', 'bootstrap_mask')
        values = {key: np.array(self.dataset.arrays[key][selected], copy=True) for key in names}
        if self.mode == 'external_features':
            observations = self.features['observations'][selected].copy()
            next_observations = self.features['next_observations'][selected].copy()
        elif self.mode == 'pixels':
            pixels = np.array(self.dataset.arrays['pixels'][selected]).transpose(0, 2, 3, 1)
            next_pixels = np.array(self.dataset.arrays['next_pixels'][selected]).transpose(0, 2, 3, 1)
            observations = {'agent_pixels': pixels[..., :3], 'wrist_pixels': pixels[..., 3:],
                            'state': values['proprio']}
            next_observations = {'agent_pixels': next_pixels[..., :3], 'wrist_pixels': next_pixels[..., 3:],
                                 'state': values['next_proprio']}
        else:
            observations, next_observations = values['proprio'], values['next_proprio']
        actions = encode_action_chunks(values['actions'], values['action_mask'])
        rewards, masks = values['reward'].astype(np.float32), values['bootstrap_mask'].astype(np.float32)
        _require(np.isfinite(rewards).all() and np.isfinite(masks).all() and np.isin(masks, [0, 1]).all(),
                 'native macro reward/bootstrap masks invalid')
        # Discount remains the configured native per-chunk discount in core;
        # masks preserve success terminals and truncation bootstrap unchanged.
        return dict(observations=observations, next_observations=next_observations,
                    actions=actions, rewards=rewards, masks=masks)
