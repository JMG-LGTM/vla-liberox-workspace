"""Read-only environment checks without CUDA initialization or model creation."""
from __future__ import annotations

import csv
import os
import platform
import re
import subprocess
import sys


CUDA_12_MINOR_DRIVER = (525, 60, 13)
CUDA_128_NATIVE_DRIVER = (570, 26)
NVIDIA_REFERENCE = "https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html"


def _version(value):
    if not isinstance(value, str) or re.fullmatch(r"\d+(?:\.\d+)*", value) is None:
        raise ValueError(f"Unrecognized NVIDIA driver version: {value!r}")
    parts = tuple(map(int, value.split(".")))
    return parts + (0,) * max(0, 3 - len(parts))


def environment_report(config, world_size, *, run=subprocess.run, torch_module=None,
                       environment=None):
    """Inspect Torch build and NVIDIA management output; never call torch.cuda.

    nvidia-smi reads driver/device metadata without allocating GPU memory. This
    cannot certify NCCL, PTX/JIT compatibility, free memory, or model execution.
    A driver in the CUDA-12 minor-compatibility range requires a real GPU smoke.
    """
    if torch_module is None:
        import torch as torch_module
    env = os.environ if environment is None else environment
    cuda = torch_module.version.cuda
    version = str(torch_module.__version__)
    device = config["distributed"]["device"]
    errors, warnings = [], []
    result = {"python": platform.python_version(), "torch": version, "torch_cuda_build": cuda,
              "requested_device": device, "world_size": world_size,
              "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
              "driver_check": "not_requested", "gpus": [], "errors": errors,
              "warnings": warnings, "cuda_context_created": False,
              "runtime_verified": False, "nvidia_reference": NVIDIA_REFERENCE}
    if version.split("+")[0] != "2.7.1":
        errors.append("This repository pins Torch 2.7.1; install the documented dedicated environment.")
    if config["method"] != "critic" and sys.version_info[:2] != (3, 11):
        errors.append("The pinned pi0.5 environment requires Python 3.11.")
    if device == "cpu":
        warnings.append("CPU framework checks do not verify CUDA/NCCL or pi0.5 GPU execution.")
        result["ready"] = not errors
        return result
    if platform.system() != "Linux":
        errors.append("GPU training is supported on Linux only.")
    if cuda != "12.8":
        errors.append("GPU configuration requires Torch 2.7.1 built for CUDA 12.8 (cu128); detected " + str(cuda))
    try:
        completed = run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total",
                         "--format=csv,noheader,nounits"], capture_output=True, text=True,
                        check=True, timeout=10)
        gpus = []
        for row in csv.reader(completed.stdout.splitlines()):
            if len(row) != 5:
                raise ValueError("Unrecognized nvidia-smi CSV output")
            index, uuid, name, driver, memory = [item.strip() for item in row]
            _version(driver)
            gpus.append({"index": int(index), "uuid": uuid, "name": name,
                         "driver_version": driver, "memory_total_mib": int(memory)})
        visible = env.get("CUDA_VISIBLE_DEVICES")
        if visible is None:
            selected = gpus
        elif visible.strip() in {"", "-1"}:
            selected = []
        else:
            selected, seen = [], set()
            for token in visible.split(","):
                token = token.strip()
                matches = [gpu for gpu in gpus if token == str(gpu["index"]) or
                           token.startswith("GPU-") and gpu["uuid"].startswith(token)]
                if len(matches) != 1 or matches[0]["uuid"] in seen:
                    errors.append(f"Cannot resolve CUDA_VISIBLE_DEVICES token {token!r} uniquely; MIG is not supported by this preflight.")
                    continue
                selected.append(matches[0])
                seen.add(matches[0]["uuid"])
        result["gpus"], result["visible_gpu_count"] = selected, len(selected)
        result["driver_check"] = "read_only_nvidia_smi"
        if len(selected) < world_size:
            errors.append(f"Requested {world_size} ranks but CUDA_VISIBLE_DEVICES exposes {len(selected)} GPUs.")
        for driver in sorted({gpu["driver_version"] for gpu in selected}):
            parsed = _version(driver)
            if parsed < CUDA_12_MINOR_DRIVER:
                errors.append(f"Driver {driver} is below NVIDIA's CUDA-12 Linux minor-compatibility minimum 525.60.13; upgrade the driver.")
            elif parsed < CUDA_128_NATIVE_DRIVER + (0,):
                warnings.append(f"Driver {driver} is in the CUDA-12 minor-compatibility range but below CUDA-12.8 GA's 570.26 driver level. PTX/JIT and newer features may fail; run the frozen GPU smoke before training.")
        warnings.append("Driver metadata does not prove free VRAM, NCCL, CUDA kernels or full-model memory fit; run both GPU smoke configurations.")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        result["driver_check"] = "unavailable"
        errors.append(f"Read-only nvidia-smi check failed: {exc}. Expose the NVIDIA driver/devices and rerun on the server.")
    result["ready"] = not errors
    return result
