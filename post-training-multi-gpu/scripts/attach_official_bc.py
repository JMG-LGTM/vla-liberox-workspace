#!/usr/bin/env python3
"""Create a new, checksum-bound native + official-observation/action BC package."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from post_training.official_bc import attach_official_demonstrations
from post_training.data import inspect_dataset

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--native", type=Path, required=True)
parser.add_argument("--official", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
manifest = attach_official_demonstrations(args.native, args.official, args.output)
info = inspect_dataset(args.output, {"method": "bc"})
print(json.dumps({"dataset_sha256": manifest["dataset_sha256"], **{k: info[k] for k in
    ("native_members", "official_demonstrations", "total_training_episodes", "native_chunks", "official_bc_chunks", "count")}}, indent=2))
