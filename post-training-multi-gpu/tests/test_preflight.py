"""No CUDA runtime calls: management queries and version branches are mocked."""
import copy
import subprocess
from types import SimpleNamespace

import pytest

from post_training.config import DEFAULTS
from post_training.preflight import environment_report


def torch_build(cuda="12.8", version="2.7.1+cu128"):
    class ForbiddenCuda:
        def __getattr__(self, name):
            raise AssertionError("Preflight called torch.cuda." + name)
    return SimpleNamespace(__version__=version, version=SimpleNamespace(cuda=cuda), cuda=ForbiddenCuda())


def query(driver="570.124.06", count=8):
    def run(command, **kwargs):
        assert command[0] == "nvidia-smi"
        assert kwargs["timeout"] == 10
        return SimpleNamespace(stdout="\n".join(f"{i}, GPU-id{i}, NVIDIA H20, {driver}, 97871" for i in range(count)))
    return run


def config(device="cuda"):
    result = copy.deepcopy(DEFAULTS)
    result["method"] = "critic"
    result["distributed"]["device"] = device
    return result


def test_cpu_does_not_probe_driver_or_cuda():
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU preflight queried NVIDIA")
    report = environment_report(config("cpu"), 2, run=forbidden, torch_module=torch_build(), environment={})
    assert report["ready"]
    assert report["driver_check"] == "not_requested"
    assert report["cuda_context_created"] is False


def test_native_cuda_and_visible_device_selection():
    report = environment_report(config(), 2, run=query(), torch_module=torch_build(),
                                environment={"CUDA_VISIBLE_DEVICES": "3,1"})
    assert report["ready"]
    assert [gpu["index"] for gpu in report["gpus"]] == [3, 1]
    assert not any("PTX" in warning for warning in report["warnings"])
    assert not report["runtime_verified"]


@pytest.mark.parametrize("driver,ready,limited", [("520.10.0", False, False),
    ("525.60.13", True, True), ("550.120", True, True), ("570.25.9", True, True),
    ("570.26", True, False), ("580.10", True, False)])
def test_driver_minimum_vs_minor_compatibility(driver, ready, limited):
    report = environment_report(config(), 1, run=query(driver), torch_module=torch_build(), environment={})
    assert report["ready"] is ready
    assert any("PTX" in warning for warning in report["warnings"]) is limited


@pytest.mark.parametrize("visible", ["", "-1", "0", "GPU-id0", "0,0", "9"])
def test_insufficient_or_invalid_visibility(visible):
    report = environment_report(config(), 2, run=query(), torch_module=torch_build(),
                                environment={"CUDA_VISIBLE_DEVICES": visible})
    assert not report["ready"]


def test_missing_driver_and_wrong_cuda_wheel():
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")
    report = environment_report(config(), 8, run=missing,
                                torch_module=torch_build(cuda=None), environment={})
    assert not report["ready"]
    assert report["driver_check"] == "unavailable"
    assert len(report["errors"]) == 2
