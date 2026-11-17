"""Two-process CPU equivalence for the real staged critic and accumulation."""
from __future__ import annotations

import copy
from contextlib import nullcontext
from datetime import timedelta
import io

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from post_training.iql.critic import PixelIQL


def _batch():
    generator = torch.Generator().manual_seed(4)
    return {"pixels": torch.randint(0, 256, (8, 6, 16, 16), dtype=torch.uint8, generator=generator),
            "next_pixels": torch.randint(0, 256, (8, 6, 16, 16), dtype=torch.uint8, generator=generator),
            "proprio": torch.randn(8, 8, generator=generator),
            "next_proprio": torch.randn(8, 8, generator=generator),
            "actions": torch.randn(8, 2, 7, generator=generator),
            "action_mask": torch.ones(8, 2, dtype=torch.bool),
            "reward": torch.full((8,), -1.), "bootstrap_mask": torch.ones(8)}


def _optimizers(model, optimizer_class=torch.optim.Adam):
    return (optimizer_class(model.value.parameters(), lr=1e-3),
            optimizer_class(list(model.q1.parameters()) + list(model.q2.parameters()), lr=1e-3))


def _update(module, optimizers, batches):
    gradients = {}
    model = module.module if hasattr(module, "module") else module
    for stage, optimizer, loss_key in zip(("value", "q"), optimizers, ("v_loss", "q_loss")):
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(batches):
            context = module.no_sync() if hasattr(module, "no_sync") and index + 1 < len(batches) else nullcontext()
            with context:
                (module(batch, stage=stage)[loss_key].mean() / len(batches)).backward()
        selected = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        gradients[stage] = {}
        for name, parameter in model.named_parameters():
            if id(parameter) in selected:
                assert parameter.grad is not None, name
                gradients[stage][name] = parameter.grad.clone()
        optimizer.step()
    model.update_targets()
    return gradients


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=rendezvous,
                            timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(12)
        model = PixelIQL(horizon=2)
        reference = copy.deepcopy(model)
        distributed = DistributedDataParallel(model, find_unused_parameters=True, broadcast_buffers=False)
        # SGD preserves small gradient errors linearly. Adam's first step divides
        # by |g| + eps, amplifying roundoff for gradients close to its 1e-8 eps.
        # Compare the actual reduced gradients before either optimizer steps.
        optimizers = _optimizers(model, torch.optim.SGD)
        reference_optimizers = _optimizers(reference, torch.optim.SGD)
        full = _batch()
        indices = [torch.tensor([rank * 2, rank * 2 + 1]),
                   torch.tensor([4 + rank * 2, 5 + rank * 2])]
        local = [{key: value[index] for key, value in full.items()} for index in indices]
        actual_gradients = _update(distributed, optimizers, local)
        expected_gradients = _update(reference, reference_optimizers, [full])
        for stage, gradients in actual_gradients.items():
            assert gradients.keys() == expected_gradients[stage].keys()
            for name, gradient in gradients.items():
                torch.testing.assert_close(gradient, expected_gradients[stage][name], atol=1e-7, rtol=1e-5,
                                           msg=lambda detail: f"{stage}/{name}: {detail}")
        # The distributed average must equal a single optimizer update on all 8 rows.
        for name, parameter in model.state_dict().items():
            torch.testing.assert_close(parameter, reference.state_dict()[name], atol=1e-7, rtol=1e-5)
            gathered = [torch.empty_like(parameter) for _ in range(2)]
            dist.all_gather(gathered, parameter)
            assert torch.equal(gathered[0], gathered[1]), name

        # Exercise both Adam optimizers with populated moments before recording a
        # restart boundary, then compare the exact next Adam update after resume.
        optimizers = _optimizers(model)
        _update(distributed, optimizers, local)
        buffer = io.BytesIO()
        torch.save({"model": model.state_dict(), "optimizers": [item.state_dict() for item in optimizers]}, buffer)
        _update(distributed, optimizers, local)
        uninterrupted = {name: value.clone() for name, value in model.state_dict().items()}
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=True)
        model.load_state_dict(restored["model"])
        for optimizer, saved in zip(optimizers, restored["optimizers"]):
            optimizer.load_state_dict(saved)
        _update(distributed, optimizers, local)
        for name, parameter in model.state_dict().items():
            torch.testing.assert_close(parameter, uninterrupted[name], atol=0, rtol=0)
            gathered = [torch.empty_like(parameter) for _ in range(2)]
            dist.all_gather(gathered, parameter)
            assert torch.equal(gathered[0], gathered[1]), name
    finally:
        dist.destroy_process_group()


def test_staged_ddp_gradients_accumulation_and_adam_resume_match_reference(tmp_path):
    mp.spawn(_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)
