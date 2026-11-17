"""CPU-only production sampler / ZeRO checkpoint regression tests."""
import os
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from post_training.checkpoint import load_checkpoint, save_checkpoint
from post_training.distributed import GlobalBatchSampler, OptimizerGroup, primary_call, unwrap, wrap
from post_training.io import atomic_json, sha256_file


def test_sampler_no_overlap_and_resume():
    for world in (2, 4):
        ranks = [list(GlobalBatchSampler(37, 8, rank, world, 7, 0, 13)) for rank in range(world)]
        for batches in zip(*ranks):
            ids = [index for batch in batches for index in batch]
            assert len(ids) == len(set(ids)) == 8
        resumed = list(GlobalBatchSampler(37, 8, 1, world, 7, 9, 13))
        assert resumed == ranks[1][9:]
        combined = [sum((ranks[r][i] for r in range(world)), []) for i in range(13)]
        assert combined == list(GlobalBatchSampler(37, 8, 0, 1, 7, 0, 13))
    with pytest.raises(ValueError, match="smaller"):
        GlobalBatchSampler(3, 8, 0, 2, 7, 0, 1)


class TinyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.2), torch.nn.Linear(4, 1))
        self.offset = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))

    def forward(self, x):
        return self.net(x) + self.offset.to(x.dtype)

    def trainable_state_dict(self):
        return self.state_dict()

    def load_trainable_state_dict(self, state):
        self.load_state_dict(state)


def _ddp_worker(rank, world, rendezvous, directory, checkpoint_format):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=world)
    device = torch.device("cpu")
    try:
        torch.manual_seed(13)
        actor = wrap(TinyActor(), device)
        optimizer = OptimizerGroup(actor.parameters(), zero=True, lr=.01)
        if checkpoint_format == "sharded":
            def no_consolidation():
                raise AssertionError("Sharded checkpoints must not consolidate optimizer states")
            optimizer.consolidate = no_consolidation
        generator = torch.Generator().manual_seed(99 + rank)
        inputs = torch.randn(3, 3, generator=generator)
        torch.manual_seed(42 + rank)
        random.seed(42 + rank)
        np.random.seed(42 + rank)
        def step():
            optimizer.zero_grad()
            actor(inputs).square().mean().backward()
            optimizer.step()
        step()
        for group in optimizer.param_groups:
            group["lr"] = .007
        path = Path(directory) / "resume.pt"
        save_checkpoint(path, update=1, actor=actor, critic=None, optimizers={"actor": optimizer},
                        contract={"world": world}, metadata={"test": True}, device=device,
                        checkpoint_format=checkpoint_format)
        expected_random = (random.random(), float(np.random.random()), torch.rand(2))
        step()
        expected = {key: value.clone() for key, value in unwrap(actor).state_dict().items()}
        # Restore into fresh modules/optimizers to expose accidentally retained state.
        actor = wrap(TinyActor(), device)
        optimizer = OptimizerGroup(actor.parameters(), zero=True, lr=.9)
        update, metadata = load_checkpoint(path, actor=actor, critic=None, optimizers={"actor": optimizer},
                                            contract={"world": world}, device=device)
        assert update == 1 and metadata == {"test": True}
        actual_random = (random.random(), float(np.random.random()), torch.rand(2))
        assert expected_random[:2] == actual_random[:2]
        assert torch.equal(expected_random[2], actual_random[2])
        assert all(group["lr"] == .007 for group in optimizer.param_groups)
        if checkpoint_format == "sharded":
            assert all(group["lr"] == .007 for opt in optimizer.optimizers for group in opt.optim.param_groups)
        step()
        for key, value in unwrap(actor).state_dict().items():
            assert torch.equal(value, expected[key])
            replicas = [torch.empty_like(value) for _ in range(world)]
            dist.all_gather(replicas, value)
            assert all(torch.equal(value, other) for other in replicas)
        with pytest.raises(RuntimeError, match="Resume contract changed"):
            load_checkpoint(path, actor=actor, critic=None, optimizers={"actor": optimizer},
                            contract={"world": 123}, device=device)
        with pytest.raises(RuntimeError, match="intentional"):
            primary_call(lambda: (_ for _ in ()).throw(ValueError("intentional startup failure")))
        if checkpoint_format == "sharded":
            _sharded_failure_checks(path, actor, optimizer, rank, world, device)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("checkpoint_format", ["consolidated", "sharded"])
def test_gloo_zero_checkpoint_rng_and_sync(tmp_path, world, checkpoint_format):
    mp.spawn(_ddp_worker, args=(world, str(tmp_path / "rendezvous"), str(tmp_path), checkpoint_format),
             nprocs=world, join=True)


def _sharded_failure_checks(path, actor, optimizer, rank, world, device):
    def load():
        return load_checkpoint(path, actor=actor, critic=None, optimizers={"actor": optimizer},
                               contract={"world": world}, device=device)
    generation = path.resolve().parent
    manifest = json.loads((generation / "model.json").read_text())
    assert len(manifest["shards"]) == world
    local_path = generation / manifest["shards"][rank]["file"]
    local = torch.load(local_path, weights_only=False)
    assert local["optimizers"]["actor"]["zero"] is True
    layouts = local["optimizers"]["actor"]["layouts"]
    all_layouts = [None] * world
    dist.all_gather_object(all_layouts, layouts)
    for dtype_index, layout in enumerate(layouts):
        ownership = [index for other in all_layouts for group in other[dtype_index]["local_groups"] for index in group]
        assert sorted(ownership) == list(range(len(layout["parameters"])))

    # A failed rewrite on a non-primary rank leaves the previous generation usable.
    original_local_state = optimizer.local_state_dict
    if rank == 1:
        def fail():
            raise OSError("intentional shard write failure")
        optimizer.local_state_dict = fail
    with pytest.raises(RuntimeError, match="rank 1.*intentional shard write failure"):
        save_checkpoint(path, update=1, actor=actor, critic=None, optimizers={"actor": optimizer},
                        contract={"world": world}, metadata={"test": True}, device=device,
                        checkpoint_format="sharded")
    optimizer.local_state_dict = original_local_state
    assert path.resolve().parent == generation
    load()

    # Even failure after the convenience JSON alias switches keeps the old model
    # alias resumable: readers always resolve that alias before reading metadata.
    import post_training.checkpoint as checkpoint_module
    original_symlink = checkpoint_module._atomic_symlink
    if rank == 0:
        def fail_public_commit(target, destination):
            if destination == path.absolute():
                raise OSError("intentional public commit failure")
            return original_symlink(target, destination)
        checkpoint_module._atomic_symlink = fail_public_commit
    try:
        with pytest.raises(RuntimeError, match="intentional public commit failure"):
            save_checkpoint(path, update=1, actor=actor, critic=None, optimizers={"actor": optimizer},
                            contract={"world": world}, metadata={"test": True}, device=device,
                            checkpoint_format="sharded")
    finally:
        checkpoint_module._atomic_symlink = original_symlink
    assert path.resolve().parent == generation
    load()

    # A validly hashed shard with a different partition/runtime is rejected by all ranks.
    def mutate_layout():
        shard_path = generation / "rank_00001.pt"
        state = torch.load(shard_path, weights_only=False)
        state["optimizers"]["actor"]["layouts"][0]["local_groups"] = [[-1]]
        torch.save(state, shard_path)
        manifest["shards"][1].update(sha256=sha256_file(shard_path), bytes=shard_path.stat().st_size)
        atomic_json(generation / "model.json", manifest)
    primary_call(mutate_layout)
    with pytest.raises(RuntimeError, match="rank 1.*layout/runtime"):
        load()
    # Missing or corrupted shards on a non-primary rank also fail collectively.
    def corrupt_shard():
        shard_path = generation / "rank_00001.pt"
        payload = bytearray(shard_path.read_bytes())
        payload[-1] ^= 1
        shard_path.write_bytes(payload)
    primary_call(corrupt_shard)
    with pytest.raises(RuntimeError, match="rank 1.*SHA-256"):
        load()
    primary_call(lambda: (generation / "rank_00001.pt").unlink())
    with pytest.raises(RuntimeError, match="rank 1.*FileNotFoundError"):
        load()
    def incomplete_manifest():
        manifest["shards"].pop()
        atomic_json(generation / "model.json", manifest)
    primary_call(incomplete_manifest)
    with pytest.raises(RuntimeError, match="one complete shard"):
        load()
