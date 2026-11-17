#!/usr/bin/env python3
"""Resume a dataset download, respect HF_ENDPOINT, then verify all SHA-256 values."""
import argparse
import json
import subprocess
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from post_training.data import verify_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Default is the SCENE23 L2 dataset repository of the first experiments; pass --repo-id for any other dataset.
    parser.add_argument("--repo-id", default="JMG-NTU123/liberox-scene23-l2-posttrain")
    parser.add_argument("--revision", default="main", help="Pin the dataset commit for reproducible downloads")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subset", choices=("bc50", "scene23_61"), help="Download only this package from a combined dataset repository")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--weights-mode", choices=("upstream", "converted"), help="Optionally invoke the separate pinned weight download workflow")
    parser.add_argument("--weights-output", type=Path)
    parser.add_argument("--weights-repo", help="Owner's optional converted model repository")
    parser.add_argument("--weights-revision", help="Full 40-character converted model revision")
    parser.add_argument("--convert-weights", action="store_true", help="Convert upstream JAX weights on CPU")
    parser.add_argument("--converted-output", type=Path)
    args = parser.parse_args()
    if args.weights_mode and args.weights_output is None:
        parser.error("--weights-mode requires --weights-output")
    if args.weights_mode == "converted" and (not args.weights_repo or not args.weights_revision):
        parser.error("converted weights require --weights-repo and --weights-revision")
    if args.convert_weights and (args.weights_mode != "upstream" or not args.converted_output):
        parser.error("--convert-weights requires upstream mode and --converted-output")
    from huggingface_hub import snapshot_download
    # local_dir persists .cache/huggingface metadata; rerunning resumes downloads.
    snapshot_download(repo_id=args.repo_id, repo_type="dataset", revision=args.revision,
                      local_dir=args.output, local_files_only=args.local_files_only,
                      allow_patterns=[f"{args.subset}/**"] if args.subset else None)
    root = args.output / args.subset if args.subset else args.output
    packages = [root] if (root / "manifest.json").is_file() else sorted(path.parent for path in root.glob("*/manifest.json"))
    if not packages:
        raise ValueError(f"No dataset manifest was downloaded to {root}")
    for package in packages:
        manifest = verify_dataset(package)
        print(json.dumps(dict(path=str(package.resolve()), dataset_sha256=manifest["dataset_sha256"], verified=True)))
    if args.weights_mode:
        command = [sys.executable, str(Path(__file__).with_name("download_weights.py")), "--output", str(args.weights_output)]
        if args.local_files_only:
            command.append("--local-files-only")
        if args.weights_mode == "converted":
            command.extend(["--converted-repo", args.weights_repo, "--revision", args.weights_revision])
        elif args.convert_weights:
            command.extend(["--convert", "--converted-output", str(args.converted_output)])
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
