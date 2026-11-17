#!/usr/bin/env python3
"""Copy selected platform recordings into a verified, portable dataset."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from post_training.data import package_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--dataset-id", help="Platform dataset ID or dataset.json path")
    selection.add_argument("--run-ids", nargs="+", help="Selected native run IDs")
    parser.add_argument("--project", default="libero_x_vla")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = package_dataset(args.dataset_root, args.output, dataset_id=args.dataset_id, run_ids=args.run_ids, project=args.project)
    print(json.dumps({k: manifest[k] for k in ("dataset_id", "dataset_sha256", "member_count", "record_count", "composition")}, indent=2))


if __name__ == "__main__":
    main()
