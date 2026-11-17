"""What differs between the training methods. ``training.py`` runs everything they share.

Same layout as the platform's ``vla_rynn_iql/algorithms.py``: one class per method behind one
interface, so the runner calls the class instead of branching on the method.

- ``BehaviorCloning`` (``--method bc``): no critic. The actor loss is the masked flow-matching
  loss of ``policy.py``, averaged over the batch.
- ``ImplicitQLearning`` (``--method iql``): each update first trains V, then Q1/Q2, then moves the
  target networks (``iql/critic.py``). The actor loss is then weighted per sample by
  ``min(exp(beta * advantage), max_weight)``; optionally ``lambda * official-demonstration loss``
  is added (``iql/actor_bc.py``).
- ``CriticOnly`` (``--method critic``): the IQL critic update alone, without an actor.

The official IDQL critic is a separate runner (``official_idql/``) and does not use this module.
"""
from __future__ import annotations

import math
from contextlib import nullcontext


class BehaviorCloning:
    name = "bc"
    trains_actor = True
    trains_critic = False

    def __init__(self, config):
        self.config = config

    def build_critic(self, device, normalization):
        """Return (critic, optimizers, optimizer class); BC has none."""
        return None, {}, None

    def actor_inputs(self, batches, official_bc_batches, actor_update):
        """Per microbatch: (actor input, native batch, number of native samples)."""
        if official_bc_batches is not None:
            raise ValueError("Official BC batch supplied to pure IQL/BC")
        return [(batch, batch, batch["action_mask"].shape[0]) for batch in batches]

    def update_critic(self, batches, critic, optimizers, device):
        return {}

    def actor_scores(self, critic, native_batch, device):
        """Per-sample weights for the actor loss; None means an unweighted mean."""
        return None

    def actor_loss(self, losses, native_count, scores, record):
        return losses[:native_count].mean()

    def actor_metrics(self, record, accumulation):
        return {}


class ImplicitQLearning(BehaviorCloning):
    name = "iql"
    trains_critic = True

    @property
    def auxiliary(self):
        return self.config["actor_bc"]["dataset_root"] is not None

    def build_critic(self, device, normalization):
        import torch
        from .distributed import OptimizerGroup, unwrap, wrap
        from .iql.critic import PixelIQL, default_input_spec, load_critic
        config = self.config
        critic = PixelIQL(**critic_model_config(config))
        if config["critic"]["init_checkpoint"]:
            load_critic(config["critic"]["init_checkpoint"], critic,
                expected_input_spec=default_input_spec(critic, config["critic"]["image_size"]),
                expected_normalization=normalization, expected_reward_definition=config["reward"],
                expected_termination_rule=config["reward"]["termination_rule"])
        critic = wrap(critic.to(device), device, unused=True,
                      gradient_as_bucket_view=config["distributed"]["gradient_as_bucket_view"])
        optimizer_class = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam}[config["critic"]["optimizer"]]
        zero = config["training"]["zero_optimizer"]
        optimizers = {
            "q": OptimizerGroup([*unwrap(critic).q1.parameters(), *unwrap(critic).q2.parameters()],
                zero=zero, optimizer_class=optimizer_class,
                lr=config["critic"]["learning_rate"], weight_decay=config["critic"]["weight_decay"]),
            "value": OptimizerGroup(unwrap(critic).value.parameters(), zero=zero,
                optimizer_class=optimizer_class, lr=config["critic"]["value_learning_rate"],
                weight_decay=config["critic"]["weight_decay"])}
        return critic, optimizers, optimizer_class

    def actor_inputs(self, batches, official_bc_batches, actor_update):
        accumulation = len(batches)
        if self.auxiliary and actor_update is not None and (accumulation != 1 or official_bc_batches is None
                                                            or len(official_bc_batches) != accumulation):
            raise ValueError("Official BC auxiliary requires one combined actor batch and one backward")
        if not self.auxiliary and official_bc_batches is not None:
            raise ValueError("Official BC batch supplied to pure IQL/BC")
        if not (self.auxiliary and actor_update is not None):
            return [(batch, batch, batch["action_mask"].shape[0]) for batch in batches]
        from .iql.actor_bc import actor_batch
        combined = []
        for native, official in zip(batches, official_bc_batches):
            if official["raw_actions"].shape[0] != self.config["actor_bc"]["per_device_batch_size"]:
                raise ValueError("Official BC sample count differs from the actor batch contract")
            combined.append(actor_batch(native, official))
        return combined

    def update_critic(self, batches, critic, optimizers, device):
        """V, then Q1/Q2, then the target networks, over all microbatches of one update."""
        if critic is None:
            return {}
        from . import training as runner  # shared finite-loss/step guards
        from .distributed import unwrap
        from .iql.critic import augment_batch
        accumulation = len(batches)
        metrics = {}
        # Only small uint8 critic inputs remain resident across the two stages.
        augmented = [augment_batch(runner._tensor_batch(batch, device), self.config["critic"]["random_shift_pad"])
                     for batch in batches]
        for stage, key, module_names in [("value", "v_loss", ["value"]), ("q", "q_loss", ["q1", "q2"])]:
            optimizer = optimizers[stage]
            optimizer.zero_grad()
            losses = []
            for index, batch in enumerate(augmented):
                context = critic.no_sync() if hasattr(critic, "no_sync") and index + 1 < accumulation else nullcontext()
                with context:
                    result = critic(batch, stage=stage)
                    loss = result[key].mean()
                    runner._finite_loss(loss, device)
                    (loss / accumulation).backward()
                losses.append(float(loss.detach()))
            parameters = [p for name in module_names for p in getattr(unwrap(critic), name).parameters()]
            metrics[f"{stage}_grad_norm"] = runner._finite_step(optimizer, parameters, self.config["critic"]["grad_clip"], device)
            metrics[key] = sum(losses) / accumulation
        unwrap(critic).update_targets()
        del augmented
        return metrics

    def actor_scores(self, critic, native_batch, device):
        if critic is None:
            return None
        from . import training as runner
        from .distributed import unwrap
        return unwrap(critic).scores(runner._tensor_batch(native_batch, device), beta=self.config["critic"]["beta"],
                                     max_weight=self.config["critic"]["max_weight"])

    def actor_loss(self, losses, native_count, scores, record):
        native_losses = losses[:native_count]
        native_loss = native_losses.mean() if scores is None else (native_losses * scores["weights"]).mean()
        if scores is not None:
            weight, advantage = scores["weights"], scores["advantage"].double()
            record.setdefault("advantages_mean", []).append(float(advantage.mean()))
            record.setdefault("advantages_second_moment", []).append(float(advantage.square().mean()))
            record.setdefault("weights_second_moment", []).append(float(weight.double().square().mean()))
            record.setdefault("clipped", []).append(float((self.config["critic"]["beta"] * advantage >=
                                                           math.log(self.config["critic"]["max_weight"])).double().mean()))
        if not self.auxiliary:
            return native_loss
        official_loss = losses[native_count:].mean()
        auxiliary_loss = self.config["actor_bc"]["weight"] * official_loss
        record.setdefault("native_iql_losses", []).append(float(native_loss.detach()))
        record.setdefault("official_losses", []).append(float(official_loss.detach()))
        record.setdefault("auxiliary_losses", []).append(float(auxiliary_loss.detach()))
        return native_loss + auxiliary_loss

    def actor_metrics(self, record, accumulation):
        metrics = {}
        if record.get("advantages_mean"):
            metrics.update(advantage_mean=sum(record["advantages_mean"]) / accumulation,
                advantage_second_moment=sum(record["advantages_second_moment"]) / accumulation,
                actor_weight_second_moment=sum(record["weights_second_moment"]) / accumulation,
                actor_weight_clip_fraction=sum(record["clipped"]) / accumulation)
        if self.auxiliary:
            metrics.update(native_iql_actor_loss=sum(record["native_iql_losses"]) / accumulation,
                official_bc_loss=sum(record["official_losses"]) / accumulation,
                official_bc_aux_loss=sum(record["auxiliary_losses"]) / accumulation,
                unweighted_native_actor_loss=sum(record["unweighted_losses"]) / accumulation)
        return metrics


class CriticOnly(ImplicitQLearning):
    name = "critic"
    trains_actor = False


ALGORITHMS = {cls.name: cls for cls in (BehaviorCloning, ImplicitQLearning, CriticOnly)}


def build_algorithm(config):
    return ALGORITHMS[config["method"]](config)


def critic_model_config(config):
    return dict(horizon=config["data"]["action_horizon"], action_dim=config["data"]["action_dim"],
                proprio_dim=config["data"]["proprio_dim"], pixel_channels=6,
                discount=config["reward"]["discount"], expectile=config["critic"]["expectile"], tau=config["critic"]["tau"])
