"""Root-bounded, overlapping action windows. All source data stay read-only."""
import os
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import DEFAULT_CONFIG

GAMMA = .99


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        f.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def members(manifest):
    """Read the manifest's ordered original rollouts and its explicit root split."""
    manifest = Path(manifest)
    rows = [r.copy() for r in json.loads(manifest.read_text())['members'] if r['kind'] == 'original']
    ids = [r['run_id'] for r in rows]
    if len(ids) != len(set(ids)) or any(r.get('root_run_id', r['run_id']) != r['run_id'] for r in rows):
        raise ValueError('duplicate or non-root original rollout')
    if {r['split'] for r in rows} != {'train', 'val'}:
        raise ValueError('the manifest must define train and val root splits')
    for row in rows:
        if Path(row['run_id']).name != row['run_id'] or row['run_id'] in ('', '.', '..'):
            raise ValueError('invalid run ID')
        path = Path(row['platform_dir'])
        if not path.is_absolute():
            row['platform_dir'] = str(manifest.resolve().parent / path)
    return rows


def paths(row):
    folder = Path(row['platform_dir']) / 'episodes/episode_000'
    return folder / 'trajectory.npz', folder / 'trajectory_observations.npz'


def actions_for(row, config=DEFAULT_CONFIG):
    """Use the recorded terminal action (inclusive) and the actual length of short failures."""
    path, _ = paths(row)
    if sha(path) != row['trajectory_sha256']:
        raise ValueError(f'trajectory hash mismatch: {path}')
    with np.load(path, allow_pickle=False) as archive:
        actions = archive['env_action']
    if actions.ndim != 2 or actions.shape[1] != config.action_dim or not np.isfinite(actions).all():
        raise ValueError(f'invalid env_action: {path}')
    if row['label_success']:
        terminal = row['terminal_step']
        if not isinstance(terminal, int) or isinstance(terminal, bool) or not 0 <= terminal < len(actions):
            raise ValueError(f'invalid terminal action index: {path}')
        return actions[:min(terminal + 1,config.max_steps)]
    if not len(actions):
        raise ValueError(f'empty failed rollout: {path}')
    return actions[:config.max_steps]


def success_in_episode(row, config=DEFAULT_CONFIG):
    return row['label_success'] and row['terminal_step'] < config.max_steps


def action_stats(rows, config=DEFAULT_CONFIG):
    train_roots = [r for r in rows if r['split'] == 'train']
    train = np.concatenate([actions_for(r,config) for r in train_roots]).astype('float64')
    norm = dict(mean=train.mean(0), std=train.std(0), minimum=train.min(0), maximum=train.max(0))
    if min(norm['std']) <= 0:
        raise ValueError('a training action dimension has zero variance')
    return dict(**{k:v.tolist() for k,v in norm.items()}, roots=sorted(r['run_id'] for r in train_roots), n=len(train))


def windows(actions, starts, mean, std, config=DEFAULT_CONFIG):
    """Input actions already truncated to the last usable transition. Zero after normalization."""
    index = np.asarray(starts)[:, None] + np.arange(config.horizon)
    valid = index < len(actions)
    x = (actions[np.minimum(index, len(actions)-1)] - mean) / std
    return np.where(valid[:, :, None], x, 0).astype('float32'), valid


def transitions(actions, success, starts, mean, std, config=DEFAULT_CONFIG):
    starts = np.asarray(starts)
    if (starts < 0).any() or (starts >= len(actions)).any():
        raise IndexError('sample outside the root')
    action, mask = windows(actions, starts, mean, std, config)
    follow = starts + config.horizon
    bootstrap = follow < len(actions)
    reward = (~bootstrap & success).astype('float32')
    next_action, next_mask = windows(actions, follow, mean, std, config)
    return dict(action=action, mask=mask, reward=reward, bootstrap=bootstrap,
                next_index=np.where(bootstrap, follow, -1), next_action=next_action, next_mask=next_mask)
