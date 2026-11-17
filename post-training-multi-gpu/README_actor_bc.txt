Optional official BC auxiliary for IQL

actor_bc:
  dataset_root: /absolute/path/to/official2_bc_package
  weight: 0.1
  per_device_batch_size: 1

dataset_root:null disables this feature and preserves pure IQL behavior.
The auxiliary requires one effective actor batch without gradient accumulation.
With two ranks and per-device32, Q/V use all32 native transitions per rank.
The actor reuses the first31 native observations/actions and appends one
official observation/action sample: global actor64 =62 native+2 official.
One actor forward returns32 losses and a single backward optimizes:

  mean(native_flow_loss * exp(beta*advantage).clip(max_weight))
    +0.1 * mean(official_flow_loss)

The two group means are explicit. This is not a32-sample mean with the official
sample merely assigned weight0.1. Official rows never receive Q/V updates,
critic scores, fabricated rewards, done flags, bootstrap masks, or next states.
Published images/actions/state remain unchanged; no extra image flip or numeric
conversion is added. Standard inherited policy transforms still apply.

Only the two pinned Scene24 official episodes1719/1720 are accepted. Their
separate package binds every payload by size/SHA. Root grouping is recomputed
with the native Haizhou seed/fraction, and both official roots must be train.
Their dedicated171-chunk actor cache contains no transition arrays. Critic
normalization and its native replay remain independent of the official cache.

The official sampler uses a private GlobalBatchSampler, replacement seed
training.seed+104729, and completed actor microbatch index plus rank. Resume
reconstructs the exact stream independently of dataloader prefetch and does not
need another mutable RNG state. Auxiliary dataset/cache identity, weight, batch
share and source manifest SHA are bound into the checkpoint resume contract.

Logs distinguish native_iql_actor_loss, official_bc_loss,
official_bc_aux_loss (lambda-scaled), and total actor_loss.
unweighted_actor_loss and unweighted_native_actor_loss both refer to the native
group. Advantage/weight statistics and ESS use only62 native actor samples.

Counters separate policy_sample_draws, native_actor_sample_draws,
official_bc_sample_draws and critic_sample_draws. For30000 rounds these are
1920000/1860000/60000/1920000. sample_draws retains its previous meaning: draws
from the main native replay, including any critic warmup. Actor and critic draws
overlap intentionally; summing those counters would double-count shared samples.
Export provenance names this composite objective and its official source; the
platform policy manifest's dataset_sha256 remains the native transition dataset.

This is training loss only. Existing held-out evaluation diagnoses the critic;
it does not become an actor held-out loss or a rollout success measurement.
