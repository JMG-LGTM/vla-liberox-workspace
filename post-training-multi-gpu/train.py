#!/usr/bin/env python3
"""Noninteractive standalone torchrun entry point."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--method", choices=["bc", "iql", "critic"])
    parser.add_argument("--dry-run", action="store_true", help="Verify data/config/identity without loading a policy or CUDA")
    parser.add_argument("--resume", type=Path, help="Trusted optimizer-boundary checkpoint in the same output directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    from post_training.config import load_config
    from post_training.training import dry_run, train
    config = load_config(args.config, method=args.method)
    if args.output:
        config["output_dir"] = str(args.output)
    if args.cache:
        config["cache_dir"] = str(args.cache)
    if args.dry_run:
        if int(os.environ.get("RANK", "0")) == 0:
            report = dry_run(config, args.data, int(os.environ.get("WORLD_SIZE", "1")))
            print(json.dumps(report, indent=2))
            if not report["environment"]["ready"]:
                raise SystemExit(2)
        return
    train(config, args.data, resume=args.resume)


if __name__ == "__main__":
    main()
