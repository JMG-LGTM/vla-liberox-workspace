"""Immutable target-Q exports; these are never optimizer resume checkpoints.

Observations must already follow the exact exported observation adapter. This
module neither loads the behavior policy nor extracts its visual features.
"""
from __future__ import annotations

import hashlib
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from ..io import sha256_file, stable_hash
from .adapters import ACTION_CODEC, encode_action_chunks
from .checkpoint import BACKEND, UPSTREAM_COMMIT, _contract, _path, _step, _sync, _write

FORMAT = "official_idql_target_q_scorer_v1"


def _require(condition, message):
    if not condition:
        raise ValueError("Official IDQL scorer: " + message)


def _template_match(actual, template):
    import jax
    left, structure = jax.tree_util.tree_flatten(actual)
    right, expected_structure = jax.tree_util.tree_flatten(template)
    _require(structure == expected_structure, "target network structure differs")
    for a, b in zip(left, right):
        a, b = np.asarray(a), np.asarray(b)
        _require(a.shape == b.shape and a.dtype == b.dtype and np.isfinite(a).all(),
                 "target parameter shape, dtype or finiteness differs")


def _bound_inputs(contract, normalization, adapter):
    prepared = contract.get("prepared", {})
    _require(prepared.get("normalization") == normalization
             and prepared.get("normalization_sha256") == stable_hash(normalization)
             and prepared.get("adapter_contract") == adapter,
             "normalization/adapter differ from the full training contract")
    _require(adapter.get("action_codec") == ACTION_CODEC, "unsupported action codec")


def _same_candidate_duration(mask):
    _require(mask.ndim in (2, 3), "expected one or batched candidate pools")
    first = mask[:1] if mask.ndim == 2 else mask[:, :1]
    _require(np.array_equal(mask, np.broadcast_to(first, mask.shape)),
             "all candidates in the same pool must have identical execution masks")


def export_scorer(directory, state, contract, *, normalization, adapter_contract,
                  replicated=False):
    """Export only target Q, with all data/input identities and complete hashes.

    A replicated pmap state must be declared. Every target parameter must agree
    exactly across replicas before the leading device dimension is removed.
    """
    from flax import serialization
    import jax
    from ..replay import validate_normalization
    _contract(contract)
    _require(type(replicated) is bool, "replicated must be explicit boolean")
    normalization = validate_normalization(normalization)
    _require(isinstance(adapter_contract, dict) and bool(adapter_contract),
             "an observation/action adapter contract is required")
    _require(adapter_contract.get("normalization_sha256") == stable_hash(normalization),
             "adapter and exported normalization differ")
    _bound_inputs(contract, normalization, adapter_contract)
    step = _step(state)
    params = state.target_critic.params
    if replicated:
        counters = np.asarray(state.step)
        _require(counters.ndim == 1 and counters.size > 0,
                 "expected pmap replica update counters")
        replicas = counters.size
        def first(value):
            value = np.asarray(value)
            _require(value.ndim > 0 and value.shape[0] == replicas
                     and np.isfinite(value).all()
                     and np.array_equal(value, np.broadcast_to(value[0], value.shape)),
                     "target Q replicas differ; no export was published")
            return value[0]
        params = jax.tree_util.tree_map(first, params)
    else:
        _require(np.asarray(state.step).ndim == 0, "pmap state requires replicated=True")
        _require(all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(params)),
                 "target Q parameters must be finite")
    control = dict(schema_version=1, format=FORMAT, backend=BACKEND,
                   upstream_commit=UPSTREAM_COMMIT, step=step, contract=contract,
                   contract_sha256=stable_hash(contract), normalization=normalization,
                   adapter_contract=adapter_contract, adapter_sha256=stable_hash(adapter_contract),
                   network="target_twin_q_min", actor_loaded=False, actor_updates=0,
                   resume_capable=False)
    data = serialization.to_bytes(params)
    metadata = (json.dumps(control, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    target = _path(directory)
    _require(not target.exists(), "refuse to replace an immutable scorer export")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="." + target.name + ".incoming-", dir=target.parent))
    try:
        _write(stage / "target_q.msgpack", data)
        _write(stage / "scorer.json", metadata)
        manifest = dict(schema_version=1, format=FORMAT, backend=BACKEND,
                        upstream_commit=UPSTREAM_COMMIT, step=step,
                        contract_sha256=stable_hash(contract), files={
                            name: {"bytes": (stage / name).stat().st_size,
                                   "sha256": sha256_file(stage / name)}
                            for name in ("target_q.msgpack", "scorer.json")})
        _write(stage / "manifest.json", (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode())
        _sync(stage)
        _require(not target.exists(), "export destination appeared while writing")
        os.rename(stage, target)
        _sync(target.parent)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return target


class FrozenScorer:
    """Target-only scoring wrapper: no actor, optimizer update or resume API."""

    def __init__(self, state, metadata):
        self._state = state
        self._metadata = copy.deepcopy(metadata)

    @property
    def metadata(self):
        return copy.deepcopy(self._metadata)

    def score_encoded(self, observations, encoded_candidates):
        """Accept already-adapted observations and normalized encoded candidates."""
        import jax
        from .core import score_actions
        encoded = np.asarray(encoded_candidates)
        _require(encoded.ndim in (2, 3) and encoded.shape[-1] == 64
                 and np.isfinite(encoded).all(), "candidates must be finite encoded64 pools")
        masks = encoded[..., 56:]
        _same_candidate_duration(masks)
        canonical = encode_action_chunks(encoded[..., :56].reshape(*encoded.shape[:-1], 8, 7), masks)
        _require(np.array_equal(encoded, canonical), "encoded padding or validity mask differs from codec")
        _require(all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(observations)),
                 "adapted observations must be finite")
        return score_actions(self._state, observations, encoded_candidates)

    def score_chunks(self, observations, candidates, action_mask, *, actions_normalized=False):
        """Encode [N,8,7] or [B,N,8,7] candidates with this export's normalization."""
        _require(type(actions_normalized) is bool, "actions_normalized must be explicit boolean")
        _same_candidate_duration(np.asarray(action_mask))
        encoded = encode_action_chunks(candidates, action_mask,
            normalization=None if actions_normalized else self._metadata["normalization"]["actions"])
        return self.score_encoded(observations, encoded)


def load_scorer(directory, template_state, contract, *, expected_adapter_contract=None):
    """Load target Q into a matching single-device architecture supplied by caller.

    The template's online Q/V and optimizer states are not restored and are not
    exposed for training. Use checkpoint.load_checkpoint for actual continuation.
    """
    from flax import serialization
    from ..replay import validate_normalization
    _contract(contract)
    _require(np.asarray(template_state.step).ndim == 0,
             "scoring requires an unreplicated architecture template")
    target = _path(directory)
    _require(target.is_dir() and {p.name for p in target.iterdir()} ==
             {"target_q.msgpack", "scorer.json", "manifest.json"}, "invalid scorer file set")
    _require(all(p.is_file() and not p.is_symlink() for p in target.iterdir()),
             "scorer payload must be regular files")
    manifest = json.loads((target / "manifest.json").read_text())
    _require(set(manifest) == {"schema_version", "format", "backend", "upstream_commit", "step", "contract_sha256", "files"}
             and manifest["schema_version"] == 1 and manifest["format"] == FORMAT
             and manifest["backend"] == BACKEND and manifest["upstream_commit"] == UPSTREAM_COMMIT
             and manifest["contract_sha256"] == stable_hash(contract)
             and set(manifest["files"]) == {"target_q.msgpack", "scorer.json"},
             "official scorer source/schema/contract differs")
    payload = {}
    for name, record in manifest["files"].items():
        payload[name] = (target / name).read_bytes()
        _require(len(payload[name]) == record["bytes"]
                 and hashlib.sha256(payload[name]).hexdigest() == record["sha256"],
                 "scorer payload SHA256 differs: " + name)
    control = json.loads(payload["scorer.json"])
    _require(control["schema_version"] == 1 and control["format"] == FORMAT
             and control["backend"] == BACKEND and control["upstream_commit"] == UPSTREAM_COMMIT
             and control["contract"] == contract and control["contract_sha256"] == stable_hash(contract)
             and control["step"] == manifest["step"] and control["network"] == "target_twin_q_min"
             and control["actor_loaded"] is False and control["actor_updates"] == 0
             and control["resume_capable"] is False
             and stable_hash(control["adapter_contract"]) == control["adapter_sha256"],
             "target-only scorer metadata differs")
    normalization = validate_normalization(control["normalization"])
    _require(control["adapter_contract"].get("normalization_sha256") == stable_hash(normalization),
             "adapter normalization differs")
    _bound_inputs(contract, normalization, control["adapter_contract"])
    if expected_adapter_contract is not None:
        _require(control["adapter_contract"] == expected_adapter_contract, "expected input adapter differs")
    params = serialization.from_bytes(template_state.target_critic.params, payload["target_q.msgpack"])
    _template_match(params, template_state.target_critic.params)
    scorer_state = template_state.replace(target_critic=template_state.target_critic.replace(params=params))
    return FrozenScorer(scorer_state, control)
