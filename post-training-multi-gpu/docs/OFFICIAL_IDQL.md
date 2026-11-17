# Official IDQL twin-Q: implementation and interfaces

## Source and scope

The Q/V core uses [philippe-eecs/IDQL](https://github.com/philippe-eecs/IDQL),
pinned to `7582571ce686c77b4ee0f592dd639ab8151295fa`.
The active learner is `jaxrl5/agents/ddpm_iql_simple/ddpm_iql_learner.py`.
Author source files are preserved under `src/post_training/_vendor/idql_official/`;
`UPSTREAM.json` binds their original bytes and source identity.

The entry point `train_idql_official.py` calls the original JAX update functions.
The existing `train.py --method critic` entry point uses a separate custom
PyTorch PixelIQL implementation. Its checkpoints are incompatible with this path.

Vision, chunked actions and an external frozen pi0.5 policy are explicit domain
adaptations. The implementation does not reproduce the complete original DDPM
actor experiment and does not update a behavior actor or add preference losses.

## Modules

| Module under `src/post_training/official_idql/` | Responsibility |
| --- | --- |
| `upstream.py` | Verify and import the fixed author source. |
| `core.py` | Create official networks and invoke original Q/V updates. |
| `replication.py` | Copy and verify initial state on each device. |
| `adapters.py` | Bind observations, normalization and chunk encoding. |
| `runner.py` | Configure data, sample batches, train, validate and save. |
| `checkpoint.py` | Save and restore complete training state. |
| `scoring.py` | Export/load target-Q scorers and score action chunks. |
| `diagnostics.py` | Compute held-out value and action diagnostics. |
| `pairs.py` | Verify genuine same-state action comparisons. |
| `selection.py` | Select checkpoints using configured qualification gates. |

## Learning and scoring

- V directly calls the author's `DDPMIQLLearner.update_v`: expectile regression
  against the minimum of the two target Q heads.
- Q directly calls `DDPMIQLLearner.update_q`: the target is
  `reward + discount * mask * V(next_observation)`; target Q follows by EMA.
- Optimization uses Optax Adam. The official path does not silently inherit
  the custom PyTorch critic's AdamW weight decay or gradient clipping.
- One JAX process manages the configured devices. Gradients are averaged across
  devices before Adam updates; V is synchronized before the Q update.
- Candidate scoring uses target `min(Q1, Q2)`. An external candidate policy may
  select the candidate with the largest score. Candidate generation and platform
  rollout integration are separate from this training package.

## Data and adapters

Provide a verified native transition package. Root-group splitting keeps
original trajectories, takeover branches and shared prefixes in one split.
Normalization must be fitted to training data and remain fixed on resume.
Observation/action-only demonstrations cannot be supplied as fabricated
reward/termination/next-state transitions.

Supported observation modes are explicit:

- `vector`: normalized proprioceptive observations.
- `external_features`: aligned features with bound encoder, preprocessing and
  cache identities; this package does not automatically extract pi0.5 features.
- `pixels`: upstream DrQ PixelMultiplexer/D4PG encoders, separate RGB camera
  views, shared Q visual features and separate V encoders.

The action codec supports H8 chunks with seven action dimensions and a contiguous
valid-length mask. It zeros masked action slots, flattens the action chunk and
appends the mask. Training and scoring use the same codec and normalization;
short tail chunks retain their actual valid length.

Reward, discount unit, termination and recording truncation are bound to the
dataset/configuration contract. A bootstrapped transition requires a real next
observation. Recording end alone does not justify inventing a terminal state.

For pixels, augmentation uses the author's `batched_random_crop`. Camera,
current/next-state and device random streams are separated. Q and V share the
augmented batch for an update. Validation and scoring do not augment inputs or
advance the training RNG.

## Validation and selection interfaces

Validation covers the complete held-out transition roster. Diagnostics include
V/progress correlations, hypothetical advantage-weight statistics and action
comparisons. These statistics do not update the behavior policy.

The same-state loader requires matching simulator state, both camera images and
proprioception at the exact recorded timestep, with equal valid action horizons.
The label that a human action is preferred is an explicit assumption.

Thresholds are configured before training and bound to checkpoints. Missing or
undefined required evidence does not qualify a scorer. `best_qualified_scorer`
remains null when none qualify; `best_progress_candidate` is a separate
diagnostic pointer. Selection considers retained checkpoints with diagnostics
at the same update and is restored from checkpoint state.

## Running

Use a dedicated Python environment:

```bash
python -m pip install -e '.[idql-official,test,plot]'
python train_idql_official.py \
  --config configs/official_idql/idql_official_pixels.yaml \
  --data /path/to/native-package --dry-run
```

The dependency extra pins the CPU JAX stack. Install compatible CUDA plugins
and check the GPU environment separately when using GPUs. Do not launch this
entry point through torchrun.

The pixel and vector YAML files are generic examples. Bind required data paths,
frozen membership, normalization and behavior-policy identity for the intended
run; configure device count and global/per-device batches consistently.
Gradient accumulation is limited to one in the official runner.

`--dry-run` verifies inputs and creates or checks the CPU replay cache; it does
not create Q/V models or initialize JAX devices. Run without `--dry-run` to train.
Use `--resume /path/to/checkpoint-directory` for the same run's complete
checkpoint, with the matching immutable contract.

A complete checkpoint includes Q, V, target Q, Adam state, update counters,
sampling/RNG state, data identities and selection state. Scorer exports contain
target-Q scoring state and cannot serve as full training resumes.

## Tests and plotting tools

```bash
python tools/verify_official_idql.py --output /tmp/official_idql_verification.json
python scripts/plot_official_idql_metrics.py RUN/metrics.csv \
  --validation RUN/validation.csv --output-dir RUN/loss_plots
```

The tests cover upstream-source identity, numerical reference updates,
replication, augmentation, checkpoint restore, data adapters, scoring and
selection. The multi-device CPU versus global-batch numerical comparison has a
known tolerance issue requiring review. The original assertions remain intact.
CPU checks do not establish GPU throughput or rollout performance.
