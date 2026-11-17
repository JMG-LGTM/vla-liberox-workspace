#!/usr/bin/env python3
"""Separate official-JAX critic entrypoint; use one process, never torchrun."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True,help='Verified native transition package')
    parser.add_argument('--dry-run',action='store_true',help='Verify/cache CPU inputs; no JAX model/device initialization')
    parser.add_argument('--resume',type=Path,help='This same run\'s immutable official full-state directory; .pt is refused')
    args=parser.parse_args()
    from post_training.official_idql.runner import load_config,dry_run,train,reject_torchrun
    reject_torchrun();config=load_config(args.config)
    result=dry_run(config,args.data) if args.dry_run else train(config,args.data,resume=args.resume)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
