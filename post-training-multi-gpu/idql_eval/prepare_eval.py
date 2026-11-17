#!/usr/bin/env python3
"""Prepare one IDQL evaluation from its config: per-half effective configs and the sealed runtime contract.

Each half (50 trials) copies the effective_config.yaml of an existing platform BC evaluation job, so the task,
initial states, seed schedule and success rule are exactly those of that BC evaluation. Only evaluation_id
and result_path change. Run with IDQL_EVAL_CONFIG set, in an environment with PyYAML (e.g. vla-liberox).
"""
from __future__ import annotations
import json
from pathlib import Path
import sys

import yaml

from contract import CONFIG, WORKSPACE, POLICY_ID, build_contract, validate_contract, require

JOBS = WORKSPACE / 'dataset-root/projects/libero_x_vla/jobs'


def main():
    results = Path(CONFIG['results_dir'])
    content = set()
    for half in CONFIG['halves']:
        out = results / half['name']
        target = out / 'effective_config.yaml'
        source = JOBS / half['source_evaluation_id'] / 'effective_config.yaml'
        value = yaml.safe_load(source.read_text())
        require(value['policy_snapshot']['policy_id'] == POLICY_ID, f'{source}: source job is not the configured BC')
        require(value['schedule_sha256'] in CONFIG['schedule_sha256s'], f'{source}: schedule not in config')
        content.add(value['policy_snapshot']['content_sha256'])
        value['evaluation_id'] = f"{CONFIG['name']}_{half['name']}"
        value['result_path'] = str(out / 'evaluation.json')
        if target.exists():
            require(yaml.safe_load(target.read_text()) == value, f'{target} exists and differs')
        else:
            out.mkdir(parents=True, exist_ok=True)
            target.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False))
        print('half', half['name'], '<-', half['source_evaluation_id'])
    require(len(content) == 1, 'source jobs use different BC catalog contents')
    contract = results / 'runtime_contract.json'
    value = validate_contract(contract, verify_payloads=True) if contract.exists() else build_contract(content.pop(), contract)
    print(json.dumps(dict(runtime_contract=str(contract), composite_sha256=value['composite_sha256'])))


if __name__ == '__main__':
    sys.exit(main())
