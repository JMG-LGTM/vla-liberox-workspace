"""Evaluation-time IDQL settings and the sealed runtime contract. No model imports.

Every run-specific value (paths, frozen BC identity, Q step, normalization, protocol) comes from the JSON
config named by the IDQL_EVAL_CONFIG environment variable; see config.example.json. The checks are the
ones used for the Scene24 L1 evaluations (73/100 and 76/100): a contract built here binds the exact BC
actor, the exact Q scorer, the training source files and this folder's code by SHA-256.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent                      # post-training-multi-gpu/: the Q training code
CONFIG_PATH = Path(os.environ.get('IDQL_EVAL_CONFIG', '')).expanduser()


def require(condition, message):
    if not condition:
        raise ValueError('Official IDQL runtime: ' + message)


def read(path):
    return json.loads(Path(path).read_text())


require(CONFIG_PATH.is_file(), 'set IDQL_EVAL_CONFIG to a config file (see config.example.json)')
CONFIG = read(CONFIG_PATH)

WORKSPACE = Path(CONFIG['workspace'])          # vla-liberox-workspace checkout running the platform
TERMINAL = WORKSPACE / 'liberox-vla-adapter-terminal'
PROJECT = WORKSPACE / 'vla-adapter-rynn-iql'
OPENPI = WORKSPACE / 'third_party/OpenPI'
REGISTRY = WORKSPACE / 'policy-registry'

# Frozen BC that proposes the candidates (an existing platform catalog entry; nothing is registered).
BC = CONFIG['bc']
POLICY_ID = BC_ID = BC['policy_id']
ACTOR_SHA = BC['actor_sha256']
ACTOR_BYTES = BC['actor_bytes']
ACTOR_STEP = BC['training_step']
BASE = Path(BC['base_checkpoint'])
REVISION = BC['base_revision']

# Q scorer exported by official_idql/runner.py, and the files of its checkpoint selection.
Q = CONFIG['critic']
SCORER_DIR = Path(Q['scorer_dir'])
NORMALIZATION_PATH = Path(Q['normalization'])
NORMALIZATION_SHA = Q['normalization_sha256']
SELECTION_PATH = Path(Q['selection_json'])
VALIDATION_PATH = Path(Q['validation_json'])
VALIDATION_CHUNKS = Q['validation_chunks']
STEP = Q['step']
CRITIC_UPDATES = Q['optimizer_updates']
NETWORK = 'target_twin_q_min'                 # min(Q1, Q2) of the target critic; the code has no num_q option
SOURCE_COMMIT = Q['source_commit']
UPSTREAM = '7582571ce686c77b4ee0f592dd639ab8151295fa'   # author IDQL commit vendored in src/post_training/_vendor

# gym is needed to import the author code; it lives in a private folder, not in the pi05 environment.
DEPENDENCIES = Path(CONFIG['dependencies_dir'])
DEPENDENCY_RECEIPT = Path(CONFIG['dependency_receipt']) if CONFIG.get('dependency_receipt') else None

# Candidate recipe. Changing how candidates are drawn or chosen means editing runtime_core.py
# (generate_pool / infer) and these labels together; the contract then no longer matches old runs.
SAMPLING = {'recipe': 'bc_anchor_single1_plus_independent3_v1',
            'anchor_forward': 'original_openpi_policy_infer_batch1', 'extra_forward': 'batch3',
            'anchor_noise': 'torch.normal_float32_shape1x10x32_default_bc_seed_stream',
            'extra_noise': 'independent_cuda_generator', 'selection_noise': 'independent_cpu_generator',
            'seed_derivation': 'sha256_bc78-n4-v1_domain_seed_first8_big_mask63',
            'extra_domain': 'extra3', 'selection_domain': 'selector'}
SELECTOR = dict(candidate_count=4, mode='greedy', target_reduction='min', num_inference_steps=10,
                native_horizon=10, score_horizon=8, action_dim=7, proprio_dim=8, image_size=128, sampling=SAMPLING)
PROTOCOL = dict(max_steps=950, open_loop_steps=8, control_hz=20, realtime=True,
                schedule_sha256s=sorted(CONFIG['schedule_sha256s']),
                disabled_policy_cameras=[], manipulability_enabled=False)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), 'symlink destination forbidden')
    t = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    with t.open('w') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n'); f.flush(); os.fsync(f.fileno())
    os.replace(t, path)


def code_files():
    source = training_contract()['source_sha256']
    files = [HERE / name for name in ('contract.py', 'runtime_core.py', 'provider.py', 'idql_worker.py',
                                      'run_idql_evaluation.py', 'gpu_smoke.py')] + [CONFIG_PATH.resolve()]
    files += [REPOSITORY / 'src/post_training' / name for name in source]
    vendor = REPOSITORY / 'src/post_training/_vendor/idql_official'
    files += list(vendor.rglob('*.py')) + [vendor / 'UPSTREAM.json']
    files += [TERMINAL / 'backend/app/policies/pi05.py', TERMINAL / 'backend/app/evaluation/batch.py',
              PROJECT / 'src/vla_rynn_iql/pi05.py', PROJECT / 'src/vla_rynn_iql/image_transforms.py',
              OPENPI / 'src/openpi/policies/policy.py', OPENPI / 'src/openpi/models_pytorch/pi0_pytorch.py']
    if DEPENDENCY_RECEIPT is not None:
        files += [DEPENDENCY_RECEIPT]
    files += [p for p in DEPENDENCIES.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    require(not any(p.suffix in ('.pyc', '.pyo') for p in files), 'bytecode cannot be sealed as dependency source')
    return sorted(set(files))


def training_contract():
    return read(SCORER_DIR / 'scorer.json')['contract']


def qualification():
    selection = read(SELECTION_PATH); best = selection['best_qualified_scorer']
    require(selection['status'] == 'QUALIFIED_SCORER_SELECTED' and best is not None and best['optimizer_update'] == STEP,
            'pre-registered qualified selection does not identify this scorer')
    value = read(VALIDATION_PATH)
    require(best['status'] == 'QUALIFIED' and best['gates']['eligible_for_idql_evaluation'] is True
            and not best['gates']['failures'] and not best['gates']['unavailable'], 'scorer did not pass all gates')
    require(value['optimizer_update'] == STEP and value['count'] == VALIDATION_CHUNKS and value['updated'] is False
            and value['diagnostics']['gates'] == best['gates'], 'same-update held-out evidence differs')
    return dict(status='QUALIFIED', optimizer_update=STEP, selection_path=str(SELECTION_PATH), selection_sha256=sha(SELECTION_PATH),
                validation_path=str(VALIDATION_PATH), validation_sha256=sha(VALIDATION_PATH),
                thresholds=selection['policy']['thresholds'])


def expected_critic():
    metadata = read(SCORER_DIR / 'scorer.json'); manifest = read(SCORER_DIR / 'manifest.json')
    contract = training_contract()
    require(metadata['contract'] == contract and metadata['step'] == manifest['step'] == STEP and metadata['network'] == NETWORK,
            'scorer update/contract differs')
    require(contract['backend'] == 'official_idql_jax' and contract['upstream_commit'] == UPSTREAM, 'not the official backend')
    require(metadata['contract_sha256'] == canonical(contract) and contract['config']['critic'].get('num_q', 2) == 2,
            'training contract differs')
    require(canonical(read(NORMALIZATION_PATH)) == NORMALIZATION_SHA, 'normalization identity differs')
    for name in ('target_q.msgpack', 'scorer.json'):
        p = SCORER_DIR / name; identity = manifest['files'][name]
        require(p.is_file() and not p.is_symlink() and p.stat().st_size == identity['bytes'] and sha(p) == identity['sha256'],
                'scorer payload differs: ' + name)
    target = sha(SCORER_DIR / 'target_q.msgpack')
    return dict(backend='official_idql_jax', upstream_commit=UPSTREAM, scorer_path=str(SCORER_DIR),
                path=str(SCORER_DIR / 'target_q.msgpack'), sha256=target, target_q_sha256=target,
                manifest_sha256=sha(SCORER_DIR / 'manifest.json'), scorer_json_sha256=sha(SCORER_DIR / 'scorer.json'),
                training_step=STEP, source_commit=SOURCE_COMMIT, network=NETWORK, training_contract_sha256=canonical(contract),
                normalization_path=str(NORMALIZATION_PATH), normalization_file_sha256=sha(NORMALIZATION_PATH),
                normalization_sha256=NORMALIZATION_SHA)


def expected_actor():
    return dict(policy_id=BC_ID, training_step=ACTOR_STEP, path=str(REGISTRY / POLICY_ID / 'actor.pt'),
                sha256=ACTOR_SHA, base_checkpoint=str(BASE), base_revision=REVISION)


def check_training_sources():
    """The repository's Q code must be byte-identical to the code that trained the scorer."""
    for name, digest in training_contract()['source_sha256'].items():
        require(sha(REPOSITORY / 'src/post_training' / name) == digest, 'training/scoring source differs: ' + name)


def validate_contract(path, *, verify_payloads=False):
    value = read(path)
    require(set(value) == {'schema_version', 'policy_id', 'policy_content_sha256', 'actor', 'critic', 'selector', 'protocol',
                           'qualification', 'code_sha256', 'composite_sha256'}, 'runtime fields differ')
    require(value['schema_version'] == 4 and value['policy_id'] == POLICY_ID, 'unexpected runtime scope')
    require(value['composite_sha256'] == canonical({k: v for k, v in value.items() if k != 'composite_sha256'}), 'composite changed')
    require(value['selector'] == SELECTOR and value['protocol'] == PROTOCOL, 'candidate/protocol changed')
    require(value['qualification'] == qualification(), 'qualified scorer evidence changed')
    actor = value['actor']; p = Path(actor['path'])
    require(actor == expected_actor(), 'BC actor identity differs')
    require(p.is_file() and not p.is_symlink() and p.stat().st_size == ACTOR_BYTES, 'actor file/size differs')
    if verify_payloads:
        require(sha(p) == ACTOR_SHA, 'actor payload SHA differs')
    require(value['critic'] == expected_critic(), 'official scorer identity differs')
    files = code_files(); require(set(value['code_sha256']) == {str(p) for p in files}, 'runtime source set differs')
    for name, digest in value['code_sha256'].items():
        require(sha(name) == digest, 'runtime source changed: ' + name)
    check_training_sources()
    return value


def build_contract(content_sha, output):
    check_training_sources()
    value = dict(schema_version=4, policy_id=POLICY_ID, policy_content_sha256=content_sha, actor=expected_actor(),
                 critic=expected_critic(), selector=SELECTOR, protocol=PROTOCOL, qualification=qualification(),
                 code_sha256={str(p): sha(p) for p in code_files()})
    value['composite_sha256'] = canonical(value)
    require(not Path(output).exists(), 'do not overwrite a sealed runtime contract')
    atomic_json(output, value); validate_contract(output, verify_payloads=True)
    return value


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Seal the runtime contract for one evaluation.')
    p.add_argument('--content-sha', required=True, help='content_sha256 of the BC catalog entry (GET /api/models/<id>)')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    value = build_contract(a.content_sha, a.output)
    print(json.dumps(dict(policy_id=POLICY_ID, composite_sha256=value['composite_sha256'])))
