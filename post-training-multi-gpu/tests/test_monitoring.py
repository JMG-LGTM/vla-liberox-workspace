"""CPU checks for collective process-memory summaries without touching CUDA."""
import math

import pytest
import torch

from post_training import monitoring


class ForbiddenCuda:
    def __getattr__(self, name):
        raise AssertionError(f"CPU monitoring accessed torch.cuda.{name}")


def test_cpu_memory_metrics_are_finite_and_never_access_cuda(monkeypatch):
    monkeypatch.setattr(torch, "cuda", ForbiddenCuda())
    result = monitoring.memory_metrics(torch.device("cpu"))
    assert set(result) == {"host_rss_gib_sum", "host_peak_rss_gib_max", "host_peak_rss_gib_rank0",
                           "peak_cuda_memory_gib_max", "peak_cuda_reserved_gib_max"}
    assert all(type(value) is float and math.isfinite(value) and value >= 0 for value in result.values())
    assert result["host_peak_rss_gib_max"] == result["host_peak_rss_gib_rank0"]
    assert result["host_peak_rss_gib_max"] >= result["host_rss_gib_sum"] > 0
    assert result["peak_cuda_memory_gib_max"] == result["peak_cuda_reserved_gib_max"] == 0.0


def _collect_packets(monkeypatch, packets):
    monkeypatch.setattr(monitoring.dist, "is_available", lambda: True)
    monkeypatch.setattr(monitoring.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(monitoring.dist, "get_world_size", lambda: len(packets))
    def gather(result, local):
        result[:] = packets
    monkeypatch.setattr(monitoring.dist, "all_gather_object", gather)


def test_rank_summary_sums_current_rss_and_keeps_rank_zero_peak(monkeypatch):
    _collect_packets(monkeypatch, [{"values": (2.0, 5.0, 1.0, 1.5)},
                                  {"values": (3.0, 7.0, 2.0, 4.0)}])
    result = monitoring.memory_metrics("cpu")
    assert result == {"host_rss_gib_sum": 5.0, "host_peak_rss_gib_max": 7.0,
                      "host_peak_rss_gib_rank0": 5.0, "peak_cuda_memory_gib_max": 2.0,
                      "peak_cuda_reserved_gib_max": 4.0}


def test_a_remote_sampling_error_fails_on_every_rank(monkeypatch):
    _collect_packets(monkeypatch, [{"values": (2.0, 5.0, 0.0, 0.0)},
                                  {"error": "OSError: cannot read memory counters"}])
    with pytest.raises(RuntimeError, match="rank 1.*cannot read memory counters"):
        monitoring.memory_metrics("cpu")
