#!/usr/bin/env python3
"""Install pinned OpenPI and its Transformers patches in a new dedicated environment."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from post_training.pi05_assets import OPENPI_COMMIT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", type=Path, default=PROJECT / "third_party/openpi")
    parser.add_argument("--wheel-dir", type=Path, default=PROJECT / "wheelhouse")
    parser.add_argument("--offline", action="store_true", help="Use a transferred source tree, uv cache and wheelhouse")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11) or Path(sys.prefix).name != "post-training-pi05":
        raise SystemExit("Use a NEW Python 3.11 venv/Conda environment named post-training-pi05; never the platform environment")
    root = args.openpi_root.expanduser().resolve()
    if not root.exists():
        if args.offline:
            raise SystemExit("Offline setup requires a transferred pinned OpenPI checkout, including .git and submodules")
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/Physical-Intelligence/openpi.git", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "checkout", "--detach", OPENPI_COMMIT], check=True)
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()
    if commit != OPENPI_COMMIT or dirty:
        raise SystemExit("OpenPI must be clean at the exact pinned commit; existing source is never overwritten")
    if not args.offline:
        subprocess.run(["git", "-C", str(root), "submodule", "update", "--init", "--recursive"], check=True)
    environment = {**os.environ, "UV_PROJECT_ENVIRONMENT": sys.prefix,
                   "UV_LINK_MODE": "copy", "GIT_LFS_SKIP_SMUDGE": "1", "JAX_PLATFORMS": "cpu"}
    lock = tomllib.loads((root / "uv.lock").read_text())
    torch_package = next(package for package in lock["package"] if package["name"] == "torch")
    cuda = [item["name"] for item in torch_package["dependencies"]
            if item["name"].startswith("nvidia-") or item["name"] == "triton"]
    skip = [arg for name in ("torch", "torchvision", *cuda) for arg in ("--no-install-package", name)]
    command = [sys.executable, "-m", "uv", "sync", "--frozen", "--inexact", "--no-default-groups",
               *skip, "--project", str(root)]
    if args.offline:
        command.append("--offline")
    subprocess.run(command, env=environment, check=True)
    wheels = args.wheel_dir.expanduser().resolve()
    wheels.mkdir(parents=True, exist_ok=True)
    protobuf = next(package["version"] for package in lock["package"] if package["name"] == "protobuf")
    packages = ["torch==2.7.1+cu128", "torchvision==0.22.1+cu128", "PyYAML>=6,<7",
                "tensorboard==2.20.0", f"protobuf=={protobuf}", "pytest>=8,<9"]
    if not args.offline:
        # Keep the complete pip-installed dependency closure for subsequent offline installs.
        subprocess.run([sys.executable, "-m", "pip", "download", "--dest", str(wheels),
                        "--extra-index-url", "https://download.pytorch.org/whl/cu128", *packages], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", str(wheels),
                    *packages], check=True)
    target = Path(importlib.util.find_spec("transformers").origin).parent
    if not target.resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise SystemExit("Transformers resolved outside the dedicated environment; refusing to patch it")
    for source in (root / "src/openpi/models_pytorch/transformers_replace").iterdir():
        if source.is_dir():
            shutil.copytree(source, target / source.name, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target / source.name)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", str(PROJECT)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    print("Environment installed. No model was loaded. Run scripts/download_weights.py next.")


if __name__ == "__main__":
    main()
