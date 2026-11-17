"""Verify critic optimizer selection against the actual CPU runner and artifacts."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from post_training import training
from post_training.config import DEFAULTS, load_config, validate_config
from post_training.iql.critic import read_critic_metadata
from post_training.distributed import OptimizerGroup
from test_training import configuration, runner_dataset


def test_default_critic_optimizer_matches_validated_experiment(tmp_path):
    path = tmp_path / "minimal.yaml"
    path.write_text("method: critic\n")
    config = load_config(path)
    assert config["critic"]["optimizer"] == "adamw"
    assert config["critic"]["weight_decay"] == .01


@pytest.mark.parametrize("invalid", ["AdamW", "sgd", "", None, True, 1, [], {}])
def test_critic_optimizer_config_is_strict(invalid):
    config = copy.deepcopy(DEFAULTS)
    config["critic"]["optimizer"] = invalid
    with pytest.raises(ValueError, match=r"critic.optimizer must be adamw or adam"):
        validate_config(config)


def test_local_shard_does_not_confuse_adam_with_adamw():
    parameter = torch.nn.Parameter(torch.ones(2))
    adam = OptimizerGroup([parameter], zero=False, optimizer_class=torch.optim.Adam, lr=1e-3)
    adamw = OptimizerGroup([parameter], zero=False, optimizer_class=torch.optim.AdamW, lr=1e-3)
    with pytest.raises(ValueError, match="Local optimizer layout/runtime differs"):
        adamw.load_local_state_dict(adam.local_state_dict())


@pytest.mark.parametrize("name,expected,decay", [
    ("adamw", torch.optim.AdamW, .01),
    ("adam", torch.optim.Adam, 0.),
])
def test_runner_uses_selected_optimizer_for_both_q_and_value(
        runner_dataset, tmp_path, monkeypatch, name, expected, decay):
    config = configuration(tmp_path, updates=1)
    config["critic"].update(optimizer=name, weight_decay=decay,
                            learning_rate=2e-4, value_learning_rate=4e-4)
    config["training"]["checkpoint_format"] = "sharded"
    # Critic optimizers are built in algorithms.py from post_training.distributed.
    from post_training import distributed as distributed_module
    real_group = distributed_module.OptimizerGroup
    created = []

    def record_group(*args, **kwargs):
        group = real_group(*args, **kwargs)
        created.append(group)
        return group

    monkeypatch.setattr(distributed_module, "OptimizerGroup", record_group)
    result = training.train(config, runner_dataset)
    assert result["status"] == "COMPLETED"
    assert len(created) == 2
    for group, rate in zip(created, (2e-4, 4e-4)):
        assert all(type(optimizer) is expected for optimizer in group.optimizers)
        assert all(settings["weight_decay"] == decay and settings["lr"] == rate
                   for settings in group.param_groups)
        assert all(optimizer.state for optimizer in group.optimizers)
        # Shard layout records the concrete class rather than identifying AdamW
        # through an ambiguous Adam substring match.
        state = group.local_state_dict()
        assert all(layout["optimizer_class"] == expected.__module__ + "." + expected.__qualname__
                   for layout in state["layouts"])
    effective = yaml.safe_load((Path(config["output_dir"]) / "config.yaml").read_text())
    assert effective["critic"]["optimizer"] == name
    assert effective["critic"]["weight_decay"] == decay
    provenance = json.loads((Path(config["output_dir"]) / "provenance.json").read_text())
    settings = provenance["critic_optimizer"]
    assert settings == {"optimizer": name,
                        "optimizer_class": expected.__module__ + "." + expected.__qualname__,
                        "q_learning_rate": 2e-4, "value_learning_rate": 4e-4,
                        "weight_decay": decay}
    exported = read_critic_metadata(Path(config["output_dir"]) / "critic_last.pt")
    assert exported["provenance"]["critic_optimizer"] == settings
