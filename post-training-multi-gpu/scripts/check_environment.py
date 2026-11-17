#!/usr/bin/env python3
"""Check the server environment without data, weights or GPU allocation."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from post_training.config import load_config
from post_training.preflight import environment_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.world_size <= 8:
        parser.error("--world-size must be between 1 and 8")
    report = environment_report(load_config(args.config), args.world_size)
    print(json.dumps(report, indent=2))
    if not report["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
