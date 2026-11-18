#!/usr/bin/env python3
"""Run one 50-trial IDQL evaluation through the platform's unchanged batch loop (imported read-only).

The effective config is a copy of a platform BC evaluation job (prepare_eval.py); results go to its result_path."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import signal
import sys
import threading

from contract import TERMINAL, REPOSITORY, ACTOR_STEP, validate_contract, require


def validate_effective(value, contract):
    import yaml
    policy, config = value['policy_snapshot'], value['config']
    require(policy['policy_id'] == contract['policy_id'] and policy['content_sha256'] == contract['policy_content_sha256'], 'Runner policy identity differs')
    manifest = yaml.safe_load(Path(policy['manifest']).read_text())
    require(policy['family'] == 'pi05' and manifest['algorithm'] == 'bc' and policy['training_step'] == ACTOR_STEP,
        'IDQL must retain the frozen BC actor training provenance')
    require(manifest['component_sha256']['actor'] == contract['actor']['sha256']
        and policy['actor'] == contract['actor']['path'], 'Runner BC component differs')
    require(policy['model_config']['num_inference_steps'] == 10, 'Flow step count differs')
    for key in ('max_steps','open_loop_steps','control_hz','realtime','disabled_policy_cameras'):
        require(config[key] == contract['protocol'][key], 'Runner protocol changed: '+key)
    require(value['schedule_sha256'] in contract['protocol']['schedule_sha256s'], 'Runner schedule is not a configured one')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--runtime-contract', type=Path, required=True)
    args = parser.parse_args(argv)
    contract = validate_contract(args.runtime_contract)
    sys.path[:0] = [str(TERMINAL), str(TERMINAL/'scripts'), str(REPOSITORY/'src')]
    from backend.app.evaluation import batch
    from backend.app.policies import router
    from provider import IDQLProvider
    effective = batch.load_effective_config(args.config)
    validate_effective(effective, contract)
    trace = Path(effective['result_path']).parent/'idql_runtime_trace.jsonl'
    class BoundIDQLProvider(IDQLProvider):
        def __init__(self, runtime, eval_config, catalog):
            super().__init__(runtime, eval_config, catalog, contract_path=args.runtime_contract, trace_path=trace)
    router.Pi05PolicyProvider = BoundIDQLProvider
    marker = Path(effective['result_path']).parent/'idql_runner_activation.json'
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(dict(status='EXACT_IDQL_RUNTIME_ACTIVATED_BEFORE_PLATFORM_LOOP',
        policy_id=contract['policy_id'], policy_content_sha256=contract['policy_content_sha256'],
        composite_sha256=contract['composite_sha256'], trace_path=str(trace)), indent=2)+'\n')
    stopped = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT): signal.signal(signum, lambda *_: stopped.set())
    result = batch.run_evaluation(effective, stop_requested=stopped.is_set)
    return 0 if result['status'] == 'COMPLETED' else 130 if result['status'] == 'CANCELED' else 1


if __name__ == '__main__': raise SystemExit(main())
