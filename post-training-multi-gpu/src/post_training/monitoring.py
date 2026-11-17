"""Small collective memory snapshots for training, checkpoint and export stages."""
from __future__ import annotations

from pathlib import Path
import resource

import torch
import torch.distributed as dist


_GIB = 1024 ** 3


def _local_memory(device: torch.device) -> tuple[float, float, float, float]:
    """Read Linux process RSS in KiB and CUDA allocator peaks in bytes."""
    status = Path("/proc/self/status").read_text()
    rss_line = next((line for line in status.splitlines() if line.startswith("VmRSS:")), None)
    if rss_line is None:
        raise RuntimeError("Linux /proc/self/status does not contain VmRSS")
    rss = float(rss_line.split()[1]) * 1024 / _GIB
    # Linux resource counters and /proc counters may be sampled asynchronously.
    peak_rss = max(rss, float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024 / _GIB)
    allocated, reserved = 0.0, 0.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        allocated = float(torch.cuda.max_memory_allocated(device)) / _GIB
        reserved = float(torch.cuda.max_memory_reserved(device)) / _GIB
    return rss, peak_rss, allocated, reserved


def memory_metrics(device: torch.device | str) -> dict[str, float]:
    """Return identical process/allocator memory summaries on every rank.

    All ranks must call this function at the same stage. Current RSS is summed
    across trainer processes; the rank maximum and rank-zero peaks help diagnose
    checkpoint/export pressure. These process counters exclude data-loader worker
    RSS and are not a measurement of total host RAM use. CUDA allocator peaks are
    cumulative since the caller's last reset; CPU calls never access torch.cuda.
    """
    packet = {}
    try:
        packet["values"] = _local_memory(torch.device(device))
    except Exception as exc:
        packet["error"] = f"{type(exc).__name__}: {exc}"
    if dist.is_available() and dist.is_initialized():
        packets = [None] * dist.get_world_size()
        dist.all_gather_object(packets, packet)
    else:
        packets = [packet]
    failures = [f"rank {index}: {item['error']}" for index, item in enumerate(packets) if "error" in item]
    if failures:
        raise RuntimeError("Memory metric collection failed: " + "; ".join(failures))
    values = [item["values"] for item in packets]
    return {"host_rss_gib_sum": float(sum(item[0] for item in values)),
            "host_peak_rss_gib_max": float(max(item[1] for item in values)),
            "host_peak_rss_gib_rank0": float(values[0][1]),
            "peak_cuda_memory_gib_max": float(max(item[2] for item in values)),
            "peak_cuda_reserved_gib_max": float(max(item[3] for item in values))}
