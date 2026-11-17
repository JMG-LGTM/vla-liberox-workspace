import copy
from pathlib import Path

import pytest

from post_training.config import DEFAULTS, actor_learning_rate, batch_plan, load_config, validate_config
from post_training.critic import PixelIQL


def test_bc50_update_units_and_world_sizes():
    config = load_config("configs/bc50_reproduce.yaml")
    one, eight = batch_plan(config, 1), batch_plan(config, 8)
    assert one["gradient_accumulation_steps"] == 8
    assert eight["gradient_accumulation_steps"] == 1
    assert one["micro_batches"] == 6400
    assert eight["micro_batches"] == 800
    assert one["expected_sample_draws"] == eight["expected_sample_draws"] == 12800
    assert one["policy_optimizer_updates"] == eight["policy_optimizer_updates"] == 800
    assert actor_learning_rate(79, 800, 3e-5, 3e-6, 80) == pytest.approx(3e-5)
    assert actor_learning_rate(799, 800, 3e-5, 3e-6, 80) == pytest.approx(3e-6)


def test_iql_extra_warmup_is_explicit():
    config = load_config("configs/iql.yaml")
    plan = batch_plan(config, 8)
    assert plan["critic_optimizer_updates"] == 17500
    assert plan["policy_optimizer_updates"] == 5000
    assert plan["expected_sample_draws"] == 17500 * 64
    assert plan["gradient_accumulation_steps"] == 4


def test_critic_defaults_match_init_checkpoint_configs():
    model = PixelIQL()
    assert model.expectile == model.model_config["expectile"] == DEFAULTS["critic"]["expectile"] == 0.8
    for name in ("critic", "iql", "iql_frozen", "iql_from_critic"):
        config = load_config(f"configs/{name}.yaml")
        assert config["critic"]["image_size"] == DEFAULTS["critic"]["image_size"] == 128
        assert config["critic"]["expectile"] == model.expectile
        assert config["critic"]["optimizer"] == DEFAULTS["critic"]["optimizer"] == "adamw"
        assert config["critic"]["weight_decay"] == DEFAULTS["critic"]["weight_decay"] == 0.01


def test_frozen_iql_actor_budget_matches_bc50_without_hiding_critic_warmup():
    bc = load_config("configs/bc50_reproduce.yaml")
    iql = load_config("configs/iql_frozen.yaml")
    assert iql["model"] == bc["model"]
    for field in ("global_batch_size", "per_device_batch_size", "optimizer_updates",
                  "warmup_updates", "learning_rate", "final_learning_rate", "seed", "sampling"):
        assert iql["training"][field] == bc["training"][field]
    for world in (1, 8):
        plan = batch_plan(iql, world)
        assert plan["policy_optimizer_updates"] == 800
        assert plan["critic_warmup_updates"] == 12500
        assert plan["critic_optimizer_updates"] == 13300
        assert plan["expected_sample_draws"] == 13300 * 16


def test_pretrained_iql_uses_all_actor_data_and_bc50_actor_budget():
    bc = load_config("configs/bc50_reproduce.yaml")
    critic = load_config("configs/critic.yaml")
    iql = load_config("configs/iql_from_critic.yaml")
    assert critic["data"]["validation_fraction"] == 0.2
    assert iql["data"]["validation_fraction"] == bc["data"]["validation_fraction"] == 0
    assert iql["model"] == bc["model"]
    assert iql["critic"]["init_checkpoint"] == "outputs/critic/critic_best.pt"
    for field in ("global_batch_size", "per_device_batch_size", "optimizer_updates",
                  "warmup_updates", "learning_rate", "final_learning_rate", "seed", "sampling"):
        assert iql["training"][field] == bc["training"][field]
    for world in (1, 8):
        plan = batch_plan(iql, world)
        assert plan["critic_warmup_updates"] == 0
        assert plan["critic_optimizer_updates"] == plan["policy_optimizer_updates"] == 800
        assert plan["expected_sample_draws"] == 12800


@pytest.mark.parametrize("name", ["iql", "iql_frozen", "iql_from_critic"])
def test_shipped_iql_configs_do_not_hold_back_actor_data(name):
    assert load_config(f"configs/{name}.yaml")["data"]["validation_fraction"] == 0


def test_full_smoke_includes_checkpoint_and_export_boundary():
    config = load_config("configs/smoke_full.yaml")
    plan = batch_plan(config, 8)
    assert config["model"]["backbone"] == "full"
    assert config["training"]["zero_optimizer"]
    assert config["training"]["checkpoint_format"] == "sharded"
    assert plan["optimizer_updates"] == 2
    assert plan["gradient_accumulation_steps"] == 1
    assert plan["optimizer_updates"] % config["training"]["checkpoint_interval"] == 0
    assert config["method"] == "bc"  # Policy methods export automatically after the final update.


@pytest.mark.parametrize("value", ["model_only", "", None, True])
def test_invalid_checkpoint_format_is_rejected(value):
    config = copy.deepcopy(DEFAULTS)
    config["training"]["checkpoint_format"] = value
    with pytest.raises(ValueError, match="checkpoint_format"):
        validate_config(config)


def test_strict_config_and_divisibility(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("training:\n  train_steps: 123\n")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)
    config = copy.deepcopy(DEFAULTS)
    config["training"]["global_batch_size"] = 15
    with pytest.raises(ValueError, match="divide"):
        batch_plan(config, 8)
    config["training"]["global_batch_size"] = 16
    config["training"]["gradient_accumulation_steps"] = 8
    with pytest.raises(ValueError, match="global_batch_size !="):
        batch_plan(config, 8)
    config = copy.deepcopy(DEFAULTS)
    config["data"]["image_flip_horizontal"] = True
    with pytest.raises(ValueError, match="mirror"):
        validate_config(config)
    config = copy.deepcopy(DEFAULTS)
    config["reward"]["normalize"] = True
    with pytest.raises(ValueError, match="reward"):
        validate_config(config)


@pytest.mark.parametrize("path", sorted(Path("configs").glob("*.yaml")), ids=lambda path: path.stem)
def test_all_shipped_configs(path):
    config = load_config(path)
    if config["model"]["backbone"] == "full":
        assert config["training"]["checkpoint_format"] == "sharded"
    for world in (1, 2, 4, 8):
        plan = batch_plan(config, world)
        assert plan["global_batch_size"] == plan["per_device_batch_size"] * world * plan["gradient_accumulation_steps"]
