#!/usr/bin/env python3
"""Download pinned original weights or a user-supplied converted model repository."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from post_training.pi05_assets import (IDENTITY_FILE, NORM_FILE, SOURCE, SOURCE_REVISION,
                                       checkpoint_identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--converted-repo", help="Optional user-owned HF model repo containing converted assets")
    parser.add_argument("--revision", help="Full immutable commit; required with --converted-repo")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--convert", action="store_true", help="Convert original weights using prepare_pi05.py (CPU RAM required)")
    parser.add_argument("--converted-output", type=Path, default=PROJECT / "models/pi05_liberox_torch")
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--prefetch-tokenizer", action="store_true",
                        help="Download the PaliGemma tokenizer into OPENPI_DATA_HOME for offline training")
    args = parser.parse_args()
    if args.converted_repo and (not args.revision or not re.fullmatch(r"[0-9a-f]{40}", args.revision)):
        parser.error("--converted-repo requires --revision with a full immutable HF commit")
    if args.converted_repo and args.convert:
        parser.error("--converted-repo is already converted; do not combine with --convert")
    if args.revision and not args.converted_repo and args.revision != SOURCE_REVISION:
        parser.error("Original LIBERO-X weights must use the pinned source revision")
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint
    from huggingface_hub import snapshot_download
    output = args.output.expanduser().resolve()
    source = args.converted_repo or SOURCE
    revision = args.revision if args.converted_repo else SOURCE_REVISION
    patterns = ["model.safetensors", IDENTITY_FILE, NORM_FILE, "config.json"] if args.converted_repo else ["params/**", NORM_FILE]
    snapshot_download(repo_id=source, revision=revision, local_dir=str(output),
                      allow_patterns=patterns, local_files_only=args.local_files_only)
    if args.converted_repo:
        print(json.dumps(checkpoint_identity(output), indent=2))
    else:
        # prepare_pi05 verifies this marker and its complete per-file SHA manifest.
        from post_training.io import atomic_json
        from post_training.pi05_assets import sha256_file
        files = sorted(path for path in output.rglob("*") if path.is_file()
                       and (path.relative_to(output).parts[0] == "params" or path.relative_to(output).as_posix() == NORM_FILE))
        if not (output / NORM_FILE).is_file() or not any(path.relative_to(output).parts[0] == "params" for path in files):
            raise RuntimeError("Original snapshot is missing Orbax parameters or LIBERO-X normalization")
        atomic_json(output / "source_snapshot.json", {"source": SOURCE, "source_revision": SOURCE_REVISION,
                    "files": {path.relative_to(output).as_posix(): sha256_file(path) for path in files}})
    if args.prefetch_tokenizer:
        if args.local_files_only:
            parser.error("Tokenizer prefetch needs network; transfer OPENPI_DATA_HOME for offline operation")
        os.environ["JAX_PLATFORMS"] = "cpu"
        from openpi.shared.download import maybe_download
        print(f"Tokenizer cached at {maybe_download('gs://big_vision/paligemma_tokenizer.model', gs={'token': 'anon'})}")
    if args.convert:
        subprocess.run([sys.executable, str(PROJECT / "scripts/prepare_pi05.py"), "--source", str(output),
                        "--output", str(args.converted_output), "--precision", args.precision], check=True)
    print(f"Downloaded {source}@{revision} to {output}; partial HTTP downloads resume on rerun.")


if __name__ == "__main__":
    main()
