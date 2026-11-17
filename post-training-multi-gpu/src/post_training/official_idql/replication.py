"""Exact host-mediated initialization transport for the single-process pmap runner.

This is a transport adapter, not an optimizer or synchronization algorithm.
Official losses, gradient pmean, Optax and target EMA are unchanged. In JAX
0.5.3 device_put_replicated first creates a buffer on devices[0] and fans that
GPU buffer out. Here every device receives an independent host buffer instead.
Every local shard and the assembled host view must reproduce the canonical
snapshot exactly before the caller can perform its first optimizer update.
"""
from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError('Official IDQL replication: ' + message)


def canonical_host_snapshot(state):
    """Preserve existing array bits; canonicalize Python scalar dtypes as JAX does.

    With x64 disabled, Python float/int become float32/int32, rather than
    accidentally changing scalar precision through NumPy float64/int64 defaults.
    Existing array dtypes must remain representable under the current x64
    configuration; otherwise fail rather than silently downcast their bits.
    No random values, parameters, optimizer state or RNG are recomputed.
    """
    import jax
    def host(value):
        array = np.asarray(jax.device_get(value))
        dtype = jax.dtypes.canonicalize_dtype(array.dtype)
        _require(type(value) in (bool, int, float, complex) or dtype == array.dtype,
                 'existing array dtype is disabled by JAX configuration; refuse implicit precision change')
        _require(jax.dtypes.issubdtype(dtype, np.number) or jax.dtypes.issubdtype(dtype, np.bool_),
                 'state leaves must be numeric arrays or scalars')
        result = np.array(array, dtype=dtype, copy=True)
        _require(np.isfinite(result).all(), 'canonical state contains nonfinite values')
        return result
    snapshot = jax.tree_util.tree_map(host, state)
    _require(bool(jax.tree_util.tree_leaves(snapshot)), 'state has no array/scalar leaves')
    return snapshot


def _exact(actual, expected, context):
    actual, expected = np.asarray(actual), np.asarray(expected)
    _require(actual.shape == expected.shape and actual.dtype == expected.dtype,
             context + ': shape/dtype mismatch')
    _require(np.isfinite(actual).all() and actual.tobytes(order='C') == expected.tobytes(order='C'),
             context + ': exact canonical value mismatch')


def verify_replicated_state(state, canonical, devices):
    """Check independent device-to-host reads AND the assembled leading axis.

    ``canonical`` is the unreplicated snapshot. This validator deliberately does
    not select rank zero, average replicas, tolerate differences, or repair data.
    """
    import jax
    devices = tuple(devices)
    _require(devices and len(set(devices)) == len(devices), 'devices must be distinct and nonempty')
    expected, expected_tree = jax.tree_util.tree_flatten(canonical)
    actual, actual_tree = jax.tree_util.tree_flatten(state)
    _require(expected_tree == actual_tree and bool(expected), 'state tree structure differs')
    for leaf_index, (array, wanted) in enumerate(zip(actual, expected)):
        wanted = np.asarray(wanted)
        _require(hasattr(array, 'addressable_shards'), 'replicated leaf lacks addressable device shards')
        shards = list(array.addressable_shards)
        _require(len(shards) == len(devices) and {s.device for s in shards} == set(devices),
                 f'leaf {leaf_index}: incomplete or unexpected device shard coverage')
        for rank, device in enumerate(devices):
            selected = [shard for shard in shards if shard.device == device]
            _require(len(selected) == 1, f'leaf {leaf_index}: duplicate device shard')
            shard = selected[0]
            _require(isinstance(shard.index, tuple) and bool(shard.index), 'missing leading replica index')
            index = shard.index[0]
            correct_index = (index == rank if isinstance(index, (int, np.integer)) else
                isinstance(index, slice) and (index.start or 0) == rank and index.stop == rank+1
                and index.step in (None, 1))
            _require(correct_index, f'leaf {leaf_index}, rank {rank}: wrong replica shard index')
            value = np.asarray(jax.device_get(shard.data))
            # JAX 0.5.3 supports both pmap rank-reduction modes. Only the
            # singleton replica axis may be removed, never an observation axis.
            if value.shape == (1,) + wanted.shape:
                value = value[0]
            _exact(value, wanted, f'leaf {leaf_index}, rank {rank} independent shard')
        assembled = np.asarray(array)
        _exact(assembled, np.broadcast_to(wanted, (len(devices),) + wanted.shape),
               f'leaf {leaf_index} assembled host view')
    return dict(status='EXACT_CANONICAL_HOST_REPLICATION_VERIFIED', replica_count=len(devices),
        verified_leaf_count=len(expected), independent_device_shards_checked=True,
        assembled_host_view_checked=True, parameters_or_optimizer_recomputed=False)


def replicate_state_via_host(state, devices):
    """Upload the same original state from independent host buffers to each GPU.

    Each device_put receives its own NumPy copy with may_alias=False. The final
    device_put_sharded receives buffers already resident on their target devices,
    so it must not fan a first-device source buffer out to the remaining GPUs.
    Any copied-zero, stale-buffer or host-assembly mismatch raises before return.
    """
    import jax
    devices = tuple(devices)
    _require(devices and len(set(devices)) == len(devices), 'devices must be distinct and nonempty')
    _require(all(device.process_index == jax.process_index() for device in devices),
             'all devices must be addressable by this single process')
    canonical = canonical_host_snapshot(state)
    per_device = []
    for rank, device in enumerate(devices):
        uploaded = jax.tree_util.tree_map(
            lambda host: jax.device_put(np.array(host, copy=True), device=device, may_alias=False), canonical)
        # Inspect the single-device buffers before constructing the leading
        # pmap axis, isolating bad host uploads from bad assembled-array reads.
        got, tree = jax.tree_util.tree_flatten(uploaded)
        wanted, expected_tree = jax.tree_util.tree_flatten(canonical)
        _require(tree == expected_tree, 'uploaded state tree structure differs')
        for index, (leaf, reference) in enumerate(zip(got, wanted)):
            _require(leaf.devices() == {device}, f'leaf {index}, rank {rank}: upload landed on wrong device')
            _exact(jax.device_get(leaf), reference, f'leaf {index}, rank {rank} direct host upload')
        per_device.append(uploaded)
    replicated = jax.device_put_sharded(per_device, devices)
    verify_replicated_state(replicated, canonical, devices)
    return replicated
