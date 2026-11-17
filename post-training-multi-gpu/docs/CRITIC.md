# Standalone critic and IQL details

The `critic` method instantiates Q1, Q2, V, and two frozen target Q networks.
It does not instantiate or import the policy, OpenPI, or a model registry. A
critic export is an input to future IDQL experiments; candidate-action sampling
and inference-time Q selection are outside this release.

## Inputs and architecture

The cache supplies `pixels` and `next_pixels` as `[B, 6, S, S]` uint8 tensors:
agent-view RGB followed by wrist-view RGB. The encoder divides by 255 exactly
once. Floating tensors, if supplied, must still be in the 0–255 domain. The
cache applies the configured vertical image flip and never a horizontal flip.
Proprioception is `[B, 8]`; actions are `[B, H, 7]`; the boolean action mask is
`[B, H]`. The cache owns action/state normalization, and its exact statistics
and transform definitions must accompany the export. The network does not
silently normalize actions a second time.

Each Q and V has its own four-layer CNN with channels 32, 64, 128, 256,
stride-2 convolutions, GroupNorm, SiLU, and adaptive average pooling. A 64-wide
proprioception projection joins the visual features in a 256-wide embedding.
Q additionally consumes the flattened masked actions and expanded action mask.
Its head widths are 512 and 256; V uses a 256-wide head. This preserves the
small architecture of the standalone Q/V experiment rather than loading a
vision-language backbone. Padding actions are excluded by the mask, including
their values in the Q head input.

## Objectives and update order

For each block, the sparse reward is −1, or 0 for the confirmed-success block.
The discount is 0.99 once per block; it is not raised to the action length.
V uses expectile regression against the minimum target Q:

```text
delta = min(target_Q1(s, a), target_Q2(s, a)) - V(s)
L_V = abs(expectile - 1[delta < 0]) * delta**2
y = reward + gamma * bootstrap_mask * V(next_s)
L_Q = ((Q1(s, a) - y)**2 + (Q2(s, a) - y)**2) / 2
```

The runner first accumulates V gradients and updates V. It then accumulates Q
gradients over the same batches, using the newly updated V for `y`, and updates
Q. Target Q receives one Polyak update after the complete Q optimizer update:
`target = (1 - tau) * target + tau * online`. The runner computes actor weights
only after these stages, from online Q and V, with no gradient to the critic:

```text
A = min(Q1(s, a), Q2(s, a)) - V(s)
w = min(exp(beta * A), 20)
```

The implementation caps the exponent before exponentiation to avoid overflow.
Weights are not divided by their batch mean. `success_aware_recording_end_v1`
keeps `bootstrap_mask=1` at a recording-limit failure and sets it to 0 at
confirmed success. A failure's final observation therefore still contributes
to its Bellman target. Replay owns success confirmation and post-success crop
semantics; the loss does not reinterpret the recording boundary as terminal.

`PixelIQL.forward(batch, stage="value")` returns per-sample `v_loss`, `v`, and
`target_q`. `stage="q"` returns per-sample `q_loss`, `q1`, `q2`, `q`, and `target`.
Only the loss tensors require gradients. The default `stage="both"` is provided
for small standalone checks and computes joint losses with the pre-update V;
production training uses the separate stages above. DDP wrapping the complete
module requires `find_unused_parameters=True` for staged forwards. The runner
owns optimizers, `no_sync`, clipping, and accumulation scaling.

The runner defaults to AdamW for both the twin-Q optimizer and the V optimizer,
with weight decay 0.01. `critic.optimizer` accepts `adamw` or explicit `adam` for
controlled comparisons. Q/V learning rates remain independently configurable;
the shared weight decay applies to both. Effective optimizer settings are recorded
in run provenance and portable critic exports. These are independent of the
actor's optimizer settings.

## Augmentation, validation, and selection

Training-only augmentation replicate-pads each RGB camera by 4 pixels and
crops it back to its original size. Each camera/sample receives an independent
integer translation. Current and next images also receive separate draws.
There are no flips, rotations, color changes, or changes to actor inputs.
A dedicated CPU `torch.Generator` can be checkpointed independently of the
sampler and model RNG. Reusing the augmented batch in V and Q stages preserves
the reference two-stage update on one observation draw.

Validation uses original cached images with no random augmentation. Entire
root-trajectory families belong to one split, including their takeover paths.
The progress statistic is Spearman correlation between V and blocks remaining
on confirmed-success paths; failed paths use `remaining_blocks=-1` and are
excluded only from this progress correlation. They still contribute to weight
diagnostics. Lower, more negative correlation is the checkpoint-selection
direction. The final successful block has distance 0.

Diagnostics include the fraction at weight cap 20, fraction below 0.1, ESS/N,
and the fraction of total weight on successful final blocks. The terminal
indicator is explicit and is not inferred from end-of-file. Undefined
correlations and zero-total-weight shares are `null`, never misleading zeros
or nonstandard JSON NaNs. Constant V therefore gives an undefined correlation.
If trajectory IDs are available, per-trajectory correlations and their median
are reported too; an undefined path remains visible in the median.

`evaluate_critic(model, loader, device, beta=3, max_weight=20)` accepts a complete
validation loader on rank zero. It does not enter distributed collectives;
the runner coordinates other ranks and selects the best checkpoint. Each row
must appear once, without padding a distributed sampler with duplicate rows.
The model's original training modes are restored after evaluation. `critic` mode
requires a nonempty holdout and selects a best checkpoint from its `validation`
metrics. IQL may set `data.validation_fraction: 0`; then the same evaluator visits
the training pool and logs `train_diagnostic`, with `held_out: false` and no best
checkpoint selection. This ensures the actor can use every selected root without
mislabeling training scores as held-out validation.

The recommended sequence is held-out critic pretraining, followed by all-data
`iql_from_critic` using the selected export and unchanged normalization. During
IQL, Q/V are updated on all actor data, including former validation roots. The
source export's held-out score is therefore an initialization provenance record,
not a continuing validation guarantee. Source artifact/metadata hashes and
pretraining provenance accompany the new run and its critic export. Fresh-Q/V
IQL with no holdout is supported as a training-diagnostic ablation.

Progress correlations do not establish same-state action ranking or IDQL policy
success. A validation set used to select checkpoints is not an independent final
test set. GPU convergence and policy success remain unverified in this repository.

Production defaults are **128-pixel critic images and expectile 0.8**.
CPU fixtures may explicitly use smaller images for speed. Caches bind resolution
and normalization; changing 84 to 128 intentionally builds a distinct cache.
The default optimizer is AdamW with weight decay 0.01. Reproducing another
experiment also requires matching its root split, normalization, batch size,
seed, and training budget.

## Portable exports and initialization

`export_critic` writes a single atomic `.pt` containing CPU `state_dict` tensors
and JSON-compatible metadata:

- Architecture version and network dimensions/hyperparameters.
- Pixel layout, camera order, orientation, action horizon and masks.
- Actual normalization statistics and transform definitions.
- Sparse reward and block-discount definition.
- Termination rule version and provenance.

It includes Q1, Q2, V and target Q weights, so IQL initialization preserves the
target state. It contains no actor and no optimizer state. Full training
continuation uses the runner's checkpoint; an export initializes a new run.
`load_critic` uses `torch.load(weights_only=True)` and strict tensor loading.
The metadata checksum guards against accidental metadata edits; dataset and
artifact SHA-256 checks remain the transport-integrity mechanism.
`read_critic_metadata` memory-maps the export on CPU and validates its schema and
metadata checksum without constructing a network. Dry runs use this API and
compare explicit normalization files before any cache or model allocation.

For IQL initialization, supply all four `expected_*` arguments from the new
run's verified input specification, normalization, reward definition, and
termination version. Mismatches fail before loading weights into the model.
Model dimensions and discount/expectile/Polyak settings must also match an
explicitly supplied model. For standalone inspection, omitting the expected
arguments constructs a model from the export and returns its metadata:

```python
from post_training.iql.critic import load_critic

critic, metadata = load_critic("outputs/critic/critic.pt")
critic.eval()
# Normalize candidate actions according to metadata before calling critic.q1/q2.
```

## Source attribution and CPU checks

The CNN, Q/V heads, target networks, loss equations, and V→Q→target order are
adapted from `vla_rynn_iql/iql.py` at platform `origin/server` commit `e27f426`.
The DDP execution design references `distributed_algorithms.py` from that
commit, with optimizers moved to this repository's runner. The augmentation and
diagnostic contracts are documented above and implemented locally. Runtime
training does not import the platform repository.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  PYTHONPATH=src .venv/bin/python -m pytest tests/test_critic.py tests/test_critic_ddp.py -q
```

The tests check equations, success/failure bootstrap masks, padding invariance,
stage-specific gradients, the newly updated V target, Polyak updates,
deterministic non-mirroring camera shifts, tied/undefined Spearman correlations,
stable weight statistics, export round trips and compatibility rejection, and
validation mode/RNG preservation. No GPU or policy weights are needed.
The two-process gloo test additionally compares staged DDP accumulation with a
single global-batch reference, checks exact equality between ranks, and verifies
that restoring both optimizers and Q/V state reproduces the next update exactly.

The runner tests in `tests/test_training.py` exercise real synthetic record
packaging, replay, critic training, validation, and exports. BC/IQL use a small
fake policy with the real export adapter; they never allocate a real pi0.5.
Single-process tests cover all three methods and exact continuation after both
a safe stop and a rollback from a completed run. A two-process `torchrun` test
also exercises the production critic runner with gloo, ZeRO-1, staged gradient
accumulation, checkpoint save/load, and exact continued Q/V weights.

On rollback, the runner retains previous final artifacts and original logs in
`retired/<unique>/`. A retired policy keeps its `policy` directory basename so
its `policy_id` remains valid. Active metrics are truncated to the checkpoint
update; a partial trailing JSON line is discarded. Versioned best-critic files
are preserved, and `critic_best.pt` is restored from the checkpoint's recorded
best export before training continues. Explicit resume consumes a previous
`stop.request`. Resume requires the same training length and mathematical
configuration; a checkpoint already at the final update is rejected.
