#!/usr/bin/env python3
"""Plan or explicitly execute a resumable upload to a private Hugging Face dataset."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from post_training.data import verify_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="A package, or parent containing bc50/ and scene23_61/")
    parser.add_argument("--repo-id", default="JMG-NTU123/liberox-scene23-l2-posttrain")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--execute", action="store_true", help="Perform the upload after owner approval; otherwise only verify and print a plan")
    args = parser.parse_args()
    folder = args.data.resolve()
    packages = [folder] if (folder / "manifest.json").is_file() else sorted(path.parent for path in folder.glob("*/manifest.json"))
    if not packages:
        parser.error("No portable dataset packages found")
    manifests = [verify_dataset(path) for path in packages]
    plan = dict(repo_id=args.repo_id, private=True, folder=str(folder), execute=args.execute,
                datasets=[dict(path=str(path.relative_to(folder)), sha256=manifest["dataset_sha256"]) for path, manifest in zip(packages, manifests)])
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return
    from huggingface_hub import HfApi
    api = HfApi()
    api.whoami()  # Resolve credentials locally; tokens are never printed or accepted as CLI arguments.
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo_id=args.repo_id, repo_type="dataset").private:
        raise ValueError("Destination dataset must be private; refusing public upload")
    allow = []
    for package, manifest in zip(packages, manifests):
        prefix = "" if package == folder else package.relative_to(folder).as_posix() + "/"
        allow.extend(prefix + relative for relative in ["manifest.json", *manifest["files"]])
    # The Hub's Xet/LFS transport chunks large content and persists resumable task metadata.
    api.upload_large_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=folder,
                            allow_patterns=allow, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
