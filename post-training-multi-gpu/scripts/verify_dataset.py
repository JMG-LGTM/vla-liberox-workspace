#!/usr/bin/env python3
"""Verify every packaged SHA-256 and print exact replay counts."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from post_training.data import inspect_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    info = inspect_dataset(args.data)
    print(json.dumps({k: v for k, v in info.items() if k not in ("rows", "episodes")}, indent=2))


if __name__ == "__main__":
    main()
