# Haizhou root split and held-out BC loss

This preparation adds explicit opt-in settings; existing configurations retain
their fixed-count root split and their previous BC logging behavior.

```yaml
data:
  validation_fraction: 0.2
  seed: 7
  split_strategy: haizhou_hash_threshold
training:
  bc_validation: true
  validation_interval: 500
  validation_batch_size: 8
  validation_seed: 17007
```

`haizhou_hash_threshold` applies the pinned Haizhou `_split` rule to each complete
root: the first eight bytes of `SHA256("{seed}:{root_run_id}")`, interpreted as an
unsigned big-endian integer and divided by `float(2**64)`, are below the validation
fraction for a held-out root. It does not round the number of roots. Parent runs,
takeover branches and their shared prefixes stay in the same group. If every
selected root would be validation, the first selected root becomes training,
matching the upstream fallback. BC validation explicitly rejects an empty held-out
split rather than using training data under a validation label.

Official observation/action demonstrations participate in the same dynamic root
assignment. Their immutable source descriptors remain unchanged, including the
source schema's original `split` field; inspected/cache rows carry the effective
training split. Official masks continue to exclude padded action slots.

With `training.bc_validation: true`, rank zero evaluates the unwrapped actor over
the complete held-out cache every configured interval and at the final optimizer
boundary. Other ranks wait through the existing error-broadcast helper. No backward
or optimizer step is called. The reported mean weights each sample once, including
the smaller final batch; the actor already reduces each sample over its real action
mask. The separate batch size only controls evaluation memory.

The pinned OpenPI forward calls preprocessing with `train=True` even when its
module is in eval mode. Therefore this metric is a **fixed-seed Monte Carlo loss
for the original training flow objective**, including its original augmentation,
flow noise and flow time. It is neither an augmentation-free likelihood nor a
simulator rollout success rate. This patch does not alter pinned OpenPI code or
the training loss. A fixed validation seed makes checkpoint comparisons repeatable
at a fixed dataset, batch size and runtime; it is not an exact expectation over
all noise or augmentation draws.

Validation preserves and restores Python, NumPy, Torch CPU and the rank's assigned
CUDA generator, the separate loader generator and every submodule's training flag.
It seeds CPU and the assigned CUDA generator separately, without modifying the
other visible device's generator. Cached replay reads are deterministic. Training
parameters and their accumulated gradients are untouched.

Outputs:

- `validation_metrics.jsonl`: optimizer update, sample count, held-out label, fixed
  seed, aggregation contract and `val_actor_loss` for each validation.
- `metrics.jsonl`: `val_actor_loss` alongside the training record at validation
  boundaries; all original per-update metrics remain present.
- TensorBoard: `validation/actor_loss`.
- `summary.json`: last BC validation loss and update.
- `provenance.json`: actor evaluation settings and the existing immutable config,
  dataset/cache and base-model contract.

Resume retires future validation logs alongside training logs and rewrites the
retained boundary atomically, avoiding duplicate validation updates. Full actor,
optimizer and RNG checkpoints retain the existing save format.

Preparation checks use tiny fake actors on CPU, including real two-rank Gloo and
ZeRO/sharded resume. A real-model two-A100 batch64 smoke and held-out validation
remain required once the authorized cards are free; CPU checks do not certify GPU
memory or production throughput.
