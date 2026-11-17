#!/usr/bin/env python3
"""Export a trusted local training checkpoint to the platform's unchanged package format."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Trusted local update_*.pt checkpoint")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--platform-base-checkpoint", required=True,
                        help="Exact converted base path on the evaluation machine, before compatibility hashes are computed")
    parser.add_argument("--policy-id", help="Optional platform-safe identifier")
    args = parser.parse_args()
    # Sharded checkpoints publish a symlink to one immutable model generation.
    args.checkpoint = args.checkpoint.resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from post_training.export import export_policy, validate_policy_package
    from post_training.pi05_assets import sha256_file
    sidecar = json.loads(args.checkpoint.with_suffix(".json").read_text())
    if sidecar.get("sha256") != sha256_file(args.checkpoint):
        raise ValueError("Checkpoint SHA-256 mismatch")
    # Full optimizer checkpoints contain Python/NumPy RNG states. Load only your own files.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    if (checkpoint.get("schema_version") not in {1, 2} or not checkpoint.get("actor")
            or not checkpoint.get("accumulation_boundary")):
        raise ValueError("Expected a schema-1/2 BC/IQL checkpoint containing native actor parameters")
    warmup = checkpoint.get("contract", {}).get("critic_warmup_updates", 0)
    actor_updates = checkpoint["update"] - (warmup if metadata["method"] == "iql" else 0)
    path = export_policy(args.output, checkpoint["actor"], metadata["model_identity"],
                         metadata["model_settings"], metadata["dataset_sha256"], actor_updates,
                         metadata["method"], metadata.get("reward_sha256"),
                         args.platform_base_checkpoint, provenance=metadata, policy_id=args.policy_id)
    validate_policy_package(path.parent)
    print(json.dumps({"policy": str(path), "validated": True, "platform_modified": False}, indent=2))


if __name__ == "__main__":
    main()
