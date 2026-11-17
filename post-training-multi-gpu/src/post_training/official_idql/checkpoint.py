"""Atomic, pickle-free full-state checkpoints for the pinned official JAX runner."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from ..io import sha256_file, stable_hash

BACKEND = 'official_idql_jax'
UPSTREAM_COMMIT = '7582571ce686c77b4ee0f592dd639ab8151295fa'
FORMAT = 'official_idql_full_state_v1'


def _require(condition, message):
    if not condition:
        raise ValueError('Official IDQL checkpoint: ' + message)


def _contract(value):
    _require(isinstance(value, dict) and value.get('backend') == BACKEND
             and value.get('upstream_commit') == UPSTREAM_COMMIT,
             'backend or official source identity differs; old PyTorch checkpoints are unsupported')
    stable_hash(value)


def _path(value):
    path = Path(value).expanduser().absolute()
    _require(path.suffix != '.pt' and not path.is_symlink()
             and not any(parent.is_symlink() for parent in path.parents),
             'old .pt files and symlink checkpoint paths are forbidden')
    return path


def _step(state):
    required = ('critic', 'value', 'target_critic', 'rng', 'step')
    _require(all(hasattr(state, key) for key in required) and not hasattr(state, 'actor'),
             'expected the official critic/value/target/optimizer/RNG state, with no actor')
    values = np.asarray(state.step)
    _require(values.size > 0 and values.dtype.kind in 'iu' and np.all(values == values.flat[0])
             and int(values.flat[0]) >= 0, 'replicated optimizer update counters differ')
    step = int(values.flat[0])
    for name in ('critic', 'value'):
        optimizer = getattr(state, name)
        _require(np.all(np.asarray(optimizer.step) == step), name + ' optimizer step differs')
        def check_count(value):
            if hasattr(value, '_fields') and {'count', 'mu', 'nu'} <= set(value._fields):
                _require(np.all(np.asarray(value.count) == step), name + ' Adam count differs')
            if isinstance(value, dict):
                for child in value.values(): check_count(child)
            elif isinstance(value, (tuple, list)):
                for child in value: check_count(child)
        check_count(optimizer.opt_state)
    return step


def _layout(template, restored):
    """Flax deserialization alone does not enforce array geometry/dtypes."""
    from flax import serialization
    def compare(a, b):
        if isinstance(a, dict):
            _require(isinstance(b, dict) and a.keys() == b.keys(), 'restored state tree differs')
            for key in a: compare(a[key], b[key])
        else:
            x, y = np.asarray(a), np.asarray(b)
            _require(x.shape == y.shape, 'restored array shape differs')
            # JIT canonicalizes Python scalar hyperparameters to JAX dtypes.
            if x.ndim > 0 or hasattr(a, 'dtype'):
                _require(x.dtype == y.dtype, 'restored array dtype differs')
    compare(serialization.to_state_dict(template), serialization.to_state_dict(restored))


def _write(path, data):
    with path.open('xb') as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())


def _sync(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_checkpoint(directory, state, contract, sampling_state, metadata=None):
    """Save every state leaf, including replicated device RNG and all Adam state.

    ``sampling_state`` is the actual NumPy bit-generator state and counters; it
    is restored by the caller. A checkpoint is a new immutable three-file
    directory, published only after complete payload hashes and fsync.
    """
    from flax import serialization
    _contract(contract); step = _step(state); target = _path(directory)
    _require(not target.exists(), 'refuse to replace an immutable checkpoint')
    target.parent.mkdir(parents=True, exist_ok=True)
    control = dict(schema_version=1, format=FORMAT, backend=BACKEND, upstream_commit=UPSTREAM_COMMIT,
        step=step, contract=contract, contract_sha256=stable_hash(contract),
        sampling_state=sampling_state, metadata={} if metadata is None else metadata,
        actor_loaded=False, actor_updates=0, replicated_state_preserved=True)
    control_bytes = (json.dumps(control, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    staging = Path(tempfile.mkdtemp(prefix='.' + target.name + '.incoming-', dir=target.parent))
    try:
        _write(staging / 'state.msgpack', serialization.to_bytes(state))
        _write(staging / 'control.json', control_bytes)
        files = {name: dict(bytes=(staging / name).stat().st_size, sha256=sha256_file(staging / name))
                 for name in ('state.msgpack', 'control.json')}
        manifest = dict(schema_version=1, format=FORMAT, backend=BACKEND, upstream_commit=UPSTREAM_COMMIT,
                        step=step, contract_sha256=stable_hash(contract), files=files)
        _write(staging / 'manifest.json', (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode())
        _sync(staging)
        _require(not target.exists(), 'checkpoint destination appeared while writing')
        os.rename(staging, target); _sync(target.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target


def load_checkpoint(directory, template_state, contract):
    """Restore only an exact official backend/data/adapter/batch contract."""
    from flax import serialization
    _contract(contract); target = _path(directory)
    _require(target.is_dir() and {p.name for p in target.iterdir()} ==
             {'state.msgpack', 'control.json', 'manifest.json'}, 'incomplete or unknown checkpoint file set')
    for path in target.iterdir():
        _require(path.is_file() and not path.is_symlink(), 'checkpoint payload must be a regular file')
    manifest = json.loads((target / 'manifest.json').read_text())
    _require(set(manifest) == {'schema_version', 'format', 'backend', 'upstream_commit', 'step', 'contract_sha256', 'files'}
             and manifest['schema_version'] == 1 and manifest['format'] == FORMAT and manifest['backend'] == BACKEND
             and manifest['upstream_commit'] == UPSTREAM_COMMIT and manifest['contract_sha256'] == stable_hash(contract),
             'official source/schema/resume contract differs')
    _require(set(manifest['files']) == {'state.msgpack', 'control.json'}, 'unknown checkpoint payload identity')
    payload = {}
    for name, expected in manifest['files'].items():
        path = target / name
        _require(path.stat().st_size == expected['bytes'] and sha256_file(path) == expected['sha256'],
                 'checkpoint payload SHA256 differs: ' + name)
        payload[name] = path.read_bytes()
        import hashlib
        _require(hashlib.sha256(payload[name]).hexdigest() == expected['sha256'], 'checkpoint changed while reading')
    control = json.loads(payload['control.json'])
    _require(control['format'] == FORMAT and control['backend'] == BACKEND and control['upstream_commit'] == UPSTREAM_COMMIT
             and control['contract'] == contract and control['contract_sha256'] == stable_hash(contract)
             and control['actor_loaded'] is False and control['actor_updates'] == 0
             and control['step'] == manifest['step'], 'full-state control contract differs')
    state = serialization.from_bytes(template_state, payload['state.msgpack'])
    _layout(template_state, state)
    _require(_step(state) == manifest['step'], 'restored optimizer update counter differs')
    return state, control['sampling_state'], control['metadata']
