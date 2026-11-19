#!/usr/bin/env python3
"""Run one half of one arm through the unchanged platform batch loop, in this process only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import signal
import sys
import threading

from .planning import MEAN_MODES
from .contract import gpu_lock, read, validate_contract, validate_effective, validate_smoke, require


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--runtime-contract', type=Path, required=True)
    parser.add_argument('--pick', choices=('greedy','random',*MEAN_MODES), help='Assert the selection mode sealed in the runtime contract')
    parser.add_argument('--output-dir', type=Path, help='Assert effective_config result directory (platform output owner)')
    parser.add_argument('--smoke-proof', type=Path, required=True)
    args = parser.parse_args(argv)
    contract = validate_contract(args.runtime_contract)
    require(args.pick is None or args.pick == contract['pick'], 'selection mode differs from sealed contract')
    validate_smoke(read(args.smoke_proof),contract)
    with gpu_lock(contract):
        return run_locked(args,contract)


def run_locked(args,contract):
    terminal = Path(contract['workspace'])/'liberox-vla-adapter-terminal'
    sys.path[:0] = [str(terminal), str(terminal/'scripts')]
    from backend.app.evaluation import batch
    from backend.app.policies import router
    from .provider import QPlanningProvider
    effective = batch.load_effective_config(args.config)
    validate_effective(effective, contract)
    output = Path(effective['result_path']).parent
    require(args.output_dir is None or args.output_dir.resolve() == output.resolve(), 'use the result directory configured in platform effective config')
    result_path = Path(effective['result_path'])
    marker = output/'runner_activation.json'
    if result_path.exists():
        if (read(result_path)['status']=='COMPLETED' and marker.is_file()
                and read(marker)['composite_sha256']==contract['composite_sha256']):
            return 0
        raise RuntimeError(f'Existing evaluation belongs to another selector or is incomplete: {result_path}')
    trace = output/'q_runtime_trace.jsonl'

    class BoundProvider(QPlanningProvider):
        def __init__(self, runtime, eval_config, catalog):
            super().__init__(runtime, eval_config, catalog, contract_path=args.runtime_contract, trace_path=trace)
    router.Pi05PolicyProvider = BoundProvider
    marker.write_text(json.dumps(dict(status='Q_RUNTIME_ACTIVATED_BEFORE_PLATFORM_LOOP', policy_id=contract['candidate_policy_id'], pick=contract['pick'],
                                      critic_seed=(contract['critic'] or {}).get('seed'), composite_sha256=contract['composite_sha256'],
                                      trace_path=str(trace)), indent=2)+'\n')
    stopped = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stopped.set())
    result = batch.run_evaluation(effective, stop_requested=stopped.is_set)
    return 0 if result['status'] == 'COMPLETED' else 130 if result['status'] == 'CANCELED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
