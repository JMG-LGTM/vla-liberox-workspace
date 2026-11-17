"""Official BC is one part of the actor batch, never a fabricated transition."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import numpy as np
import torch
from torch import nn
import yaml

from post_training import training
from post_training.actor_bc import ACTOR_FIELDS, build_actor_bc_cache, inspect_actor_bc
from post_training.config import batch_plan, validate_config
from post_training.critic import PixelIQL
from post_training.data import _manifest_identity, grouped_split, inspect_dataset, package_dataset
from post_training.distributed import OptimizerGroup
from post_training.replay import CachedReplayDataset
from test_data_replay import make_run
from test_official_bc import mixed
from test_policy_export import base
from test_training import configuration, install_fake_policy, load_snapshot
from test_iql_all_data_ddp import _assert_identical_state


@pytest.fixture
def official_package(mixed, tmp_path):
    root = tmp_path / "official_actor_only"
    old = mixed[-1]
    root.mkdir()
    shutil.copytree(old / "official_bc", root / "official_bc")
    source = json.loads((old / "manifest.json").read_text())
    manifest = {"schema_version": 1, "format": "liberox-official-bc-demonstrations",
        "action_horizon": 8, "action_dim": 7, "proprio_dim": 8,
        "bc_demonstrations": source["bc_demonstrations"],
        "bc_demonstration_contract": source["bc_demonstration_contract"],
        "files": {k: v for k, v in source["files"].items() if k.startswith("official_bc/")},
        "split_seed": 7, "validation_fraction": .2, "split_strategy": "haizhou_hash_threshold"}
    manifest["root_splits"] = grouped_split(manifest["bc_demonstrations"], .2, 7, "haizhou_hash_threshold")
    manifest["dataset_sha256"] = _manifest_identity(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


@pytest.fixture
def native_dataset(tmp_path):
    root, package = tmp_path / "native_source", tmp_path / "native_dataset"
    for index, name in enumerate(["native", "success_a", "success_b", "success_c"]):
        path = make_run(root, name, offset=.1 * index)
        # Match real native256px observations so no source is silently resized
        # merely to concatenate it with the published256px official samples.
        images = np.full((11, 256, 256, 3), 30 + index, np.uint8)
        np.savez_compressed(path / "episodes/episode_000/trajectory_observations.npz",
                            agentview_image=images, wrist_image=images)
    package_dataset(root, package, run_ids=["native", "success_a", "success_b", "success_c"])
    return package


def config_for_aux(tmp_path, official, model_root=None):
    config = configuration(tmp_path, "iql", updates=4)
    config["model"].update(backbone="full", checkpoint=str(model_root or "/unused/base"))
    config["critic"]["warmup_updates"] = 0
    config["data"].update(validation_fraction=.2, seed=7, split_strategy="haizhou_hash_threshold")
    config["training"].update(global_batch_size=3, per_device_batch_size=3,
        gradient_accumulation_steps=1, sampling="replacement")
    config["actor_bc"].update(dataset_root=str(official), weight=.1, per_device_batch_size=1)
    return config


@pytest.mark.parametrize("mutation", ["critic", "all_official", "accumulation", "wrong_world", "negative_weight"])
def test_invalid_auxiliary_method_batch_or_weight_rejected(tmp_path, mutation):
    config = config_for_aux(tmp_path, "/not-read-during-config-validation")
    if mutation == "critic":
        config["method"] = "critic"
    elif mutation == "all_official":
        config["actor_bc"]["per_device_batch_size"] = 3
    elif mutation == "accumulation":
        config["training"]["gradient_accumulation_steps"] = 2
    elif mutation == "wrong_world":
        config["training"].update(global_batch_size=64, per_device_batch_size=32, gradient_accumulation_steps="auto")
    else:
        config["actor_bc"]["weight"] = -1
    with pytest.raises(ValueError):
        validate_config(config)
        batch_plan(config, 1)


def test_official_cache_is_train_only_raw_and_never_a_critic_pool(official_package, native_dataset, tmp_path):
    config = config_for_aux(tmp_path, official_package)
    native = inspect_dataset(native_dataset, config)
    auxiliary = build_actor_bc_cache(official_package, tmp_path / "aux_cache", config, native)
    dataset = CachedReplayDataset(auxiliary["cache"])
    assert len(dataset) == 4 and all(r["split"] == "train" for r in dataset.records)
    assert set(dataset.metadata["arrays"]) == set(ACTOR_FIELDS)
    sample = dataset[0]
    assert sample["agent_image"][0, 0, 0].item() == 19
    assert sample["agent_image"][-1, 0, 0].item() == 241
    assert sample["raw_actions"][0, 0].item() == pytest.approx(-1.6)
    assert all(not {"reward", "next_pixels", "bootstrap_mask", "done"}.intersection(row) for row in dataset.records)
    assert not {r["root_run_id"] for r in dataset.records}.intersection(r["root_run_id"] for r in native["rows"])
    assert build_actor_bc_cache(official_package, tmp_path / "aux_cache", config, native) == auxiliary


@pytest.mark.parametrize("kind", ["heldout_hash", "native_root_overlap", "fake_transition", "changed_payload"])
def test_auxiliary_rejects_leakage_fake_transitions_and_corruption(official_package, native_dataset, tmp_path, kind):
    config = config_for_aux(tmp_path, official_package)
    native = inspect_dataset(native_dataset, config)
    p = official_package / "manifest.json"
    manifest = json.loads(p.read_text())
    if kind == "heldout_hash":
        for seed in range(100):
            splits = grouped_split(manifest["bc_demonstrations"], .2, seed, "haizhou_hash_threshold")
            if "val" in splits.values():
                break
        config["data"]["seed"] = manifest["split_seed"] = seed
        manifest["root_splits"] = splits
    elif kind == "native_root_overlap":
        native["episodes"][0]["root_run_id"] = manifest["bc_demonstrations"][0]["root_run_id"]
    elif kind == "fake_transition":
        manifest["bc_demonstrations"][0]["reward"] = 0
    else:
        (official_package / manifest["bc_demonstrations"][0]["numeric_path"]).write_bytes(b"corrupt")
    manifest["dataset_sha256"] = _manifest_identity(manifest)
    p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        inspect_actor_bc(official_package, config, native)


def test_one_actor_forward_and_backward_uses_group_means_with_original_critic_batch(tmp_path):
    from test_critic_ddp import _batch
    config = config_for_aux(tmp_path, "/configuration-only")
    config["training"].update(global_batch_size=32, per_device_batch_size=32)
    critic = PixelIQL(horizon=2)
    raw = _batch()
    # Repeat a valid small critic batch to32; actor raw values remain independent
    # test targets and are never passed into critic's normalized-action fields.
    batch = {k: v.repeat(4, *([1] * (v.ndim - 1))) for k, v in raw.items()}
    batch.update(raw_actions=torch.arange(1, 33).float()[:, None, None].expand(32, 2, 7).clone(),
        raw_proprio=torch.zeros(32, 8), agent_image=torch.zeros(32, 4, 4, 3, dtype=torch.uint8),
        wrist_image=torch.zeros(32, 4, 4, 3, dtype=torch.uint8), prompt=["task"] * 32)
    official = {"raw_actions": torch.full((1, 2, 7), 10.), "action_mask": torch.ones(1, 2, dtype=torch.bool),
        "raw_proprio": torch.zeros(1, 8), "agent_image": torch.zeros(1, 4, 4, 3, dtype=torch.uint8),
        "wrist_image": torch.zeros(1, 4, 4, 3, dtype=torch.uint8), "prompt": ["task"]}
    sizes, gradients = [], []
    original_forward = critic.forward
    def forward(data, stage="both", **kwargs):
        sizes.append((stage, data["actions"].shape[0]))
        return original_forward(data, stage=stage, **kwargs)
    critic.forward = forward
    def scores(data, **kwargs):
        assert data["actions"].shape[0] == 31
        return {"weights": torch.full((31,), 2.), "advantage": torch.full((31,), .231)}
    critic.scores = scores
    class Actor(nn.Module):
        def __init__(self):
            super().__init__(); self.weight = nn.Parameter(torch.tensor(0.)); self.calls = 0
        def forward(self, data):
            self.calls += 1
            assert data["actor_source"] == ["native"] * 31 + ["official_bc"]
            return (self.weight - data["raw_actions"][:, 0, 0]).square()
    actor = Actor()
    actor.weight.register_hook(lambda value: gradients.append(float(value)))
    optimizers = {"actor": OptimizerGroup(actor.parameters(), zero=False, lr=1e-4),
        "q": OptimizerGroup([*critic.q1.parameters(), *critic.q2.parameters()], zero=False, lr=1e-4),
        "value": OptimizerGroup(critic.value.parameters(), zero=False, lr=1e-4)}
    metrics = training.optimization_step(config, [batch], actor, critic, optimizers, torch.device("cpu"),
        actor_update=0, official_bc_batches=[official])
    assert actor.calls == 1 and gradients == pytest.approx([-66.])
    assert sizes == [("value", 32), ("q", 32)]
    assert metrics["unweighted_actor_loss"] == pytest.approx(336.)
    assert metrics["native_iql_actor_loss"] == pytest.approx(672.)
    assert metrics["official_bc_loss"] == pytest.approx(100.)
    assert metrics["official_bc_aux_loss"] == pytest.approx(10.)
    assert metrics["actor_loss"] == pytest.approx(682.)


@pytest.mark.parametrize("mutation", ["lambda", "source_content"])
def test_auxiliary_resume_binds_weight_and_complete_source_content(
        official_package, native_dataset, base, tmp_path, monkeypatch, mutation):
    model_root = install_fake_policy(monkeypatch, base)
    config = config_for_aux(tmp_path, official_package, model_root)
    original = training.optimization_step
    def stop(*args, **kwargs):
        result = original(*args, **kwargs)
        (Path(config["output_dir"]) / "stop.request").write_text("pause\n")
        return result
    monkeypatch.setattr(training, "optimization_step", stop)
    summary = training.train(config, native_dataset)
    checkpoint = load_snapshot(summary["checkpoint"])
    assert checkpoint["contract"]["actor_bc_dataset_sha256"] == checkpoint["metadata"]["actor_bc_auxiliary"]["dataset_sha256"]
    if mutation == "lambda":
        config["actor_bc"]["weight"] = .2
    else:
        manifest_path = official_package / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["audit_note"] = "different fully bound source description"
        manifest["dataset_sha256"] = _manifest_identity(manifest)
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="Resume contract changed"):
        training.train(config, native_dataset, resume=summary["checkpoint"])


_WORKER = r'''
import json,sys
from pathlib import Path
import torch
from post_training import training,policy
from post_training.config import load_config
from post_training.critic import PixelIQL
from post_training.distributed import unwrap
from post_training.pi05_assets import checkpoint_identity,load_normalization,platform_model_settings
from post_training.policy import Pi05Policy
from test_training import StochasticFakeModel
torch.set_num_threads(1)
c=load_config(sys.argv[1]); root=Path(c['model']['checkpoint']); identity=checkpoint_identity(root)
def loader(model_config,device,*,verified_identity):
    return Pi05Policy(StochasticFakeModel(),lambda v:{'actions':v['actions']},identity,
        platform_model_settings(model_config,identity),load_normalization(root),
        observation_factory=lambda v:v).to(device)
policy.load_policy=loader
observed={'steps':[],'forward_calls':0,'backward_calls':0,
    'critic_training_forward_calls':0,'critic_validation_forward_calls':0}; inside={'step':False,'weights':None}
original_forward=Pi05Policy.forward
def actor_forward(self,batch):
    assert batch['actor_source']==['native']*31+['official_bc']
    observed['forward_calls']+=1
    result=original_forward(self,batch)
    def gradient(value):
        observed['backward_calls']+=1
        torch.testing.assert_close(value[:31],inside['weights']/31,rtol=1e-6,atol=1e-7)
        assert float(value[31])==__import__('pytest').approx(.1)
    result.register_hook(gradient)
    return result
Pi05Policy.forward=actor_forward
original_critic_forward=PixelIQL.forward
def critic_forward(self,batch,*args,**kwargs):
    if inside['step']:
        assert batch['pixels'].shape[0]==32
        observed['critic_training_forward_calls']+=1
    else:
        assert not torch.is_grad_enabled() and not self.training
        observed['critic_validation_forward_calls']+=1
    return original_critic_forward(self,batch,*args,**kwargs)
PixelIQL.forward=critic_forward
original_scores=PixelIQL.scores
def scores(self,batch,*args,**kwargs):
    result=original_scores(self,batch,*args,**kwargs)
    if inside['step']:
        assert batch['pixels'].shape[0]==31
        inside['weights']=result['weights'].detach().clone()
    return result
PixelIQL.scores=scores
original_step=training.optimization_step
def step(*args,**kwargs):
    assert len(args[1])==1 and args[1][0]['raw_actions'].shape[0]==32
    official=kwargs['official_bc_batches'][0]
    assert official['raw_actions'].shape[0]==1
    assert all(not r.startswith('official') for r in args[1][0]['root_run_id'])
    before=(observed['forward_calls'],observed['backward_calls'])
    inside['step']=True
    result=original_step(*args,**kwargs)
    inside['step']=False
    assert observed['forward_calls']==before[0]+1 and observed['backward_calls']==before[1]+1
    observed['steps'].append({'update':kwargs['actor_update']+1,'official_run':official['run_id'],
        'official_start':official['start'].tolist()})
    if int(sys.argv[4]) and kwargs['actor_update']+1==int(sys.argv[4]):
        (Path(c['output_dir'])/'stop.request').write_text('stop at2\n')
    return result
training.optimization_step=step
summary=training.train(c,sys.argv[2],resume=None if sys.argv[3]=='-' else sys.argv[3])
observed['summary']=summary
rank=int(__import__('os').environ['RANK'])
(Path(c['output_dir'])/f'rank_{rank}_aux_test.json').write_text(json.dumps(observed))
'''


def test_two_real_gloo_ranks_actor64_critic64_one_backward_and_exact_2_to4_resume(
        official_package, native_dataset, base, tmp_path):
    config = config_for_aux(tmp_path, official_package, base[0])
    config["training"].update(global_batch_size=64, per_device_batch_size=32,
        gradient_accumulation_steps=1, checkpoint_format="sharded", zero_optimizer=True)
    config["distributed"]["backend"] = "gloo"
    config_path, worker = tmp_path / "aux.yaml", tmp_path / "aux_worker.py"
    worker.write_text(_WORKER)
    project = Path(__file__).resolve().parents[1]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
        "PYTHONPATH": os.pathsep.join([str(project / "src"), str(project / "tests"), os.environ.get("PYTHONPATH", "")])}
    command = [sys.executable,"-m","torch.distributed.run","--standalone","--nnodes=1",
               "--nproc_per_node=2",str(worker),str(config_path),str(native_dataset)]
    def run(output, resume=None, stop=0):
        config["output_dir"] = str(output)
        config_path.write_text(yaml.safe_dump(config))
        result = subprocess.run(command+[str(resume) if resume else "-",str(stop)],env=env,
                                capture_output=True,text=True,timeout=120)
        assert result.returncode == 0, result.stdout+result.stderr
        return [json.loads((output/f"rank_{rank}_aux_test.json").read_text()) for rank in range(2)]
    full_dir, resume_dir = tmp_path / "full", tmp_path / "resumed"
    full = run(full_dir)
    paused = run(resume_dir,stop=2)
    resumed = run(resume_dir,resume=resume_dir/'checkpoints/update_00000002.pt')
    for rank in range(2):
        assert full[rank]["steps"] == paused[rank]["steps"]+resumed[rank]["steps"]
        assert full[rank]["forward_calls"] == full[rank]["backward_calls"] == 4
        assert full[rank]["critic_training_forward_calls"] == 8
    assert full[0]["critic_validation_forward_calls"] > 0
    assert full[1]["critic_validation_forward_calls"] == 0
    expected_path, actual_path = full_dir/'checkpoints/update_00000004.pt',resume_dir/'checkpoints/update_00000004.pt'
    expected, actual = load_snapshot(expected_path), load_snapshot(actual_path)
    _assert_identical_state(expected['actor'],actual['actor'])
    _assert_identical_state(expected['critic'],actual['critic'])
    for rank in range(2):
        first=load_snapshot(expected_path.resolve().parent/f'rank_{rank:05d}.pt')
        last=load_snapshot(actual_path.resolve().parent/f'rank_{rank:05d}.pt')
        _assert_identical_state(first['optimizers'],last['optimizers'])
        torch.testing.assert_close(first['rng']['torch'],last['rng']['torch'],atol=0,rtol=0)
    summary=resumed[0]['summary']
    assert (summary['policy_sample_draws'],summary['native_actor_sample_draws'],
            summary['official_bc_sample_draws'],summary['critic_sample_draws'])==(256,248,8,256)
    records=[json.loads(l) for l in (resume_dir/'metrics.jsonl').read_text().splitlines()]
    for row in records:
        step=row['optimizer_update']
        assert (row['global_actor_batch_size'],row['global_native_actor_batch_size'],
                row['global_official_bc_batch_size'],row['global_critic_batch_size'])==(64,62,2,64)
        assert row['native_actor_sample_draws']==62*step and row['official_bc_sample_draws']==2*step
        assert row['actor_weight_effective_sample_size']<=62
        assert row['actor_loss']==pytest.approx(row['native_iql_actor_loss']+row['official_bc_aux_loss'])
        assert row['official_bc_aux_loss']==pytest.approx(.1*row['official_bc_loss'])
        assert row['unweighted_actor_loss']==row['unweighted_native_actor_loss']
    provenance=json.loads((resume_dir/'policy/export_provenance.json').read_text())
    assert provenance['actor_objective']['kind']=='native_iql_plus_official_bc'
    assert provenance['actor_bc_auxiliary']['critic_samples']==0
