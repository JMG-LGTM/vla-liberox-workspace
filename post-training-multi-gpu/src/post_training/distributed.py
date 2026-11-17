"""DDP, deterministic global sampling and dtype-partitioned ZeRO-1."""
from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Sampler


def active():
    return dist.is_available() and dist.is_initialized()


def rank():
    return dist.get_rank() if active() else 0


def world_size():
    return dist.get_world_size() if active() else 1


def initialize(config):
    cfg = config["distributed"]
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if int(os.environ.get("LOCAL_WORLD_SIZE", str(world))) != world:
        raise ValueError("Only single-node training is supported")
    if cfg["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use distributed.device=cpu for CPU tests")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    backend = cfg["backend"]
    if backend == "auto":
        backend = "nccl" if device.type == "cuda" else "gloo"
    if "RANK" in os.environ:
        dist.init_process_group(backend, timeout=timedelta(seconds=cfg["timeout_seconds"]))
    elif world != 1:
        raise ValueError("Use torchrun to launch multiple processes")
    return device


def primary_call(function):
    """Broadcast rank-zero I/O errors so peers do not hang at a barrier."""
    packet = [None]
    if rank() == 0:
        try:
            packet[0] = {"value": function()}
        except Exception as exc:
            packet[0] = {"error": f"{type(exc).__name__}: {exc}"}
    if active():
        dist.broadcast_object_list(packet, src=0)
    if "error" in packet[0]:
        raise RuntimeError(packet[0]["error"])
    return packet[0]["value"]


def all_rank_call(function):
    """Run local work and propagate any rank's exception before continuing.

    Only error strings are gathered; the result, including large checkpoint
    tensors, stays local. This cannot recover a process that has been killed.
    """
    value, error = None, None
    try:
        value = function()
    except Exception as exc:
        error = f"rank {rank()}: {type(exc).__name__}: {exc}"
    errors = [None] * world_size()
    if active():
        dist.all_gather_object(errors, error)
    else:
        errors[0] = error
    failures = [item for item in errors if item is not None]
    if failures:
        raise RuntimeError("Checkpoint operation failed: " + "; ".join(failures))
    return value


def unwrap(module):
    return module.module if isinstance(module, DDP) else module


def wrap(module, device, *, unused=False, move_inputs=True, gradient_as_bucket_view=False):
    if world_size() == 1:
        return module
    return DDP(module, device_ids=[device.index] if device.type == "cuda" and move_inputs else None,
               broadcast_buffers=False, find_unused_parameters=unused,
               gradient_as_bucket_view=gradient_as_bucket_view)


def reduce_metrics(metrics, device):
    names = sorted(metrics)
    values = torch.tensor([metrics[name] for name in names], dtype=torch.float64, device=device)
    if active():
        dist.all_reduce(values)
        values /= world_size()
    return dict(zip(names, values.cpu().tolist()))


class GlobalBatchSampler(Sampler):
    """Each rank receives a distinct slice of the same deterministic global draw.

    Shuffle sampling drops the small tail at each epoch, never pads with duplicate
    samples. Replacement sampling reproduces the platform's sampling distribution;
    repeated indices are then intentional. Resume is independent of worker prefetch.
    """
    def __init__(self, size, global_batch, rank, world, seed, start, end, sampling="shuffle"):
        if size < 1 or global_batch < world or global_batch % world or not 0 <= rank < world:
            raise ValueError("Invalid global batch sampler shape")
        if start < 0 or end < start or sampling not in {"shuffle", "replacement"}:
            raise ValueError("Invalid global batch sampler range/mode")
        if sampling == "shuffle" and size < global_batch:
            raise ValueError("Training split is smaller than a global micro batch; reduce batch or use replacement sampling")
        self.size, self.batch, self.rank, self.world = size, global_batch, rank, world
        self.seed, self.start, self.end, self.sampling = seed, start, end, sampling

    def __len__(self):
        return self.end - self.start

    def __iter__(self):
        local = self.batch // self.world
        lo, hi = self.rank * local, (self.rank + 1) * local
        generator = torch.Generator()
        if self.sampling == "replacement":
            generator.manual_seed(self.seed)
            for step in range(self.end):
                indices = torch.randint(self.size, (self.batch,), generator=generator)
                if step >= self.start:
                    yield indices[lo:hi].tolist()
            return
        per_epoch = self.size // self.batch
        previous_epoch, order = None, None
        for step in range(self.start, self.end):
            epoch, offset = divmod(step, per_epoch)
            if epoch != previous_epoch:
                generator.manual_seed(self.seed + epoch)
                order = torch.randperm(self.size, generator=generator)
                previous_epoch = epoch
            yield order[offset * self.batch + lo:offset * self.batch + hi].tolist()


class OptimizerGroup:
    """Separate ZeRO-1 instances by dtype; OpenPI has mixed parameter dtypes."""
    def __init__(self, parameters, *, zero, optimizer_class=torch.optim.AdamW, **kwargs):
        grouped = {}
        for parameter in parameters:
            if parameter.requires_grad:
                grouped.setdefault(str(parameter.dtype), []).append(parameter)
        if not grouped:
            raise ValueError("Optimizer has no trainable parameters")
        self.dtypes = sorted(grouped)
        self.zero = bool(zero and world_size() > 1)
        if self.zero:
            from torch.distributed.optim import ZeroRedundancyOptimizer
            self.optimizers = [ZeroRedundancyOptimizer(grouped[dtype], optimizer_class=optimizer_class,
                               overlap_with_ddp=False, **kwargs) for dtype in self.dtypes]
        else:
            self.optimizers = [optimizer_class(grouped[dtype], **kwargs) for dtype in self.dtypes]

    @property
    def param_groups(self):
        return [group for optimizer in self.optimizers for group in optimizer.param_groups]

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def consolidate(self):
        if self.zero:
            for optimizer in self.optimizers:
                optimizer.consolidate_state_dict(to=0)

    def state_dict(self):
        return {"dtypes": self.dtypes, "states": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state):
        if state["dtypes"] != self.dtypes or len(state["states"]) != len(self.optimizers):
            raise ValueError("Optimizer dtype layout differs from checkpoint")
        for optimizer, saved in zip(self.optimizers, state["states"]):
            optimizer.load_state_dict(saved)

    def _local_layout(self, optimizer):
        parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        indices = {id(parameter): index for index, parameter in enumerate(parameters)}
        if len(indices) != len(parameters):
            raise ValueError("Optimizer has duplicate parameters")
        local = optimizer.optim if self.zero else optimizer
        return {"optimizer_class": type(local).__module__ + "." + type(local).__qualname__,
                "parameters": [{"shape": list(parameter.shape), "dtype": str(parameter.dtype)}
                               for parameter in parameters],
                "global_groups": [[indices[id(parameter)] for parameter in group["params"]]
                                  for group in optimizer.param_groups],
                "local_groups": [[indices[id(parameter)] for parameter in group["params"]]
                                 for group in local.param_groups]}

    def local_state_dict(self):
        """Save each ZeRO optimizer's actual local shard, without consolidation.

        The pinned PyTorch ZeRO API exposes the underlying optimizer as ``optim``.
        Explicit global parameter indices detect a different partition at resume;
        an incompatible PyTorch version is rejected rather than reinterpreted.
        """
        return {"schema_version": 1, "torch_version": str(torch.__version__),
                "rank": rank(), "world_size": world_size(), "zero": self.zero,
                "dtypes": self.dtypes,
                "layouts": [self._local_layout(optimizer) for optimizer in self.optimizers],
                "global_groups": [[{key: value for key, value in group.items() if key != "params"}
                                   for group in optimizer.param_groups] for optimizer in self.optimizers],
                "states": [(optimizer.optim if self.zero else optimizer).state_dict()
                           for optimizer in self.optimizers]}

    def load_local_state_dict(self, state):
        expected = {"schema_version": 1, "torch_version": str(torch.__version__),
                    "rank": rank(), "world_size": world_size(), "zero": self.zero,
                    "dtypes": self.dtypes,
                    "layouts": [self._local_layout(optimizer) for optimizer in self.optimizers]}
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("Local optimizer layout/runtime differs from checkpoint")
        if len(state["states"]) != len(self.optimizers) or len(state["global_groups"]) != len(self.optimizers):
            raise ValueError("Local optimizer state count differs from checkpoint")
        for optimizer, saved, groups in zip(self.optimizers, state["states"], state["global_groups"]):
            if len(groups) != len(optimizer.param_groups):
                raise ValueError("Optimizer parameter group count differs from checkpoint")
            local = optimizer.optim if self.zero else optimizer
            local.load_state_dict(saved)
            # ZeRO copies outer hyperparameters into its local optimizer on step().
            for target in (optimizer.param_groups, local.param_groups):
                for group, settings in zip(target, groups):
                    parameters = group["params"]
                    group.clear()
                    group.update(settings, params=parameters)
