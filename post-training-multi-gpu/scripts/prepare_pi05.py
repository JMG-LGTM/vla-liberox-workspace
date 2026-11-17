#!/usr/bin/env python3
"""Convert the pinned JAX LIBERO-X snapshot on CPU and bind every output asset."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from post_training.pi05_assets import (IDENTITY_FILE, NORM_FILE, OPENPI_COMMIT, SOURCE,
                                       SOURCE_REVISION, checkpoint_identity, sha256_file)
from post_training.io import atomic_json


def verify_source(source: Path) -> None:
    manifest = json.loads((source / "source_snapshot.json").read_text())
    if manifest.get("source") != SOURCE or manifest.get("source_revision") != SOURCE_REVISION:
        raise ValueError("Source snapshot must come from download_weights.py at the pinned revision")
    actual = {path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file()
              and (path.relative_to(source).parts[0] == "params" or path.relative_to(source).as_posix() == NORM_FILE)}
    files = manifest.get("files", {})
    if NORM_FILE not in files or len(files) < 2 or set(files) != actual:
        raise ValueError("Source snapshot checksum inventory is incomplete or changed")
    for name, digest in files.items():
        path = (source / name).resolve()
        if not path.is_relative_to(source.resolve()) or sha256_file(path) != digest:
            raise ValueError(f"Source snapshot SHA-256 mismatch: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Output of scripts/download_weights.py")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    source, output = args.source.expanduser().resolve(), args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite an existing checkpoint: {output}")
    verify_source(source)
    from post_training.policy import verify_runtime
    openpi_root = verify_runtime()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pi05-convert-", dir=output.parent))
    print(f"CPU conversion: {source} -> {output}. Requires substantial host RAM and disk space.", flush=True)
    try:
        subprocess.run([sys.executable, str(openpi_root / "examples/convert_jax_model_to_pytorch.py"),
                        "--checkpoint-dir", str(source), "--config-name", "pi05_libero",
                        "--output-path", str(temporary), "--precision", args.precision], check=True)
        norm = temporary / NORM_FILE
        norm.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / NORM_FILE, norm)
        identity = {"schema_version": 1, "family": "pi05", "openpi_commit": OPENPI_COMMIT,
                    "source": SOURCE, "source_revision": SOURCE_REVISION, "config_name": "pi05_libero",
                    "conversion_precision": args.precision, "native_action_horizon": 10,
                    "native_action_dim": 32, "discrete_state_input": False,
                    "files": {name: sha256_file(temporary / name) for name in ("model.safetensors", NORM_FILE)}}
        atomic_json(temporary / IDENTITY_FILE, identity)
        checkpoint_identity(temporary)
        temporary.rename(output)
    except BaseException:
        print(f"Partial conversion retained for diagnosis: {temporary}", file=sys.stderr)
        raise
    print(json.dumps(identity, indent=2))


if __name__ == "__main__":
    main()
