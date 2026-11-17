# pi0.5 post-training on 1–8 GPUs

**分支定位：** `review/bc-iql-updates-20261114` 整理初版 `main` 之后的 BC、IQL 和共用训练改进；入口仍是 `train.py`。`review/official-idql-twinq-20261114` 以此为共享基座，单独审阅官方 IDQL 差异；`archive/custom-preference-critic-20261114` 保留旧 preference critic 历史。详见[中文变更与验证说明](docs/BC_IQL_REVIEW_20261114.md)。这里的 IQL 使用项目原有 PyTorch twin-Q/V。

Standalone BC, IQL and Q/V-only training for 1–8 GPUs, with configurations
targeting one 8 × H20 96 GB server. All commands are noninteractive. Portable
dataset packages let training run independently of the data collection platform.

**Validation status:** the framework has been tested on CPU, including
2/4-process gloo, real recording packaging/cache construction and fake-policy
export compatibility. Real pi0.5 weight loading, CUDA/NCCL, GPU memory/throughput,
resume/export/inference with real weights and trained-policy evaluation have
not yet been validated. Run the policy smoke configurations before long training.

| Method | Trained components | Policy dependency |
| --- | --- | --- |
| `bc` | Equal-weight masked flow matching | Pinned pi0.5 base |
| `iql` | V, twin Q, target Q, advantage-weighted policy | Pinned pi0.5 base |
| `critic` | V, twin Q and target Q only | None; no OpenPI or model weights |

BC and critic provide the two parts of a future IDQL system. Sampling candidate
actions and selecting with Q at inference time is outside this release.

## 1. Environment

Use Linux, Python **3.11**, and a dedicated environment. The policy stack is
PyTorch **2.7.1+cu128**, torchvision **0.22.1+cu128**, patched Transformers
**4.53.2**, and OpenPI commit
`215abfb217dbac7d5f1273282331b9b1866c0479`. The server driver must support the
CUDA 12.8 wheels.

```bash
git clone https://github.com/JMG-LGTM/post-training-multi-gpu.git
cd post-training-multi-gpu
python3.11 -m venv "$HOME/envs/post-training-pi05"
source "$HOME/envs/post-training-pi05/bin/activate"
python -m pip install --upgrade pip setuptools wheel uv
python scripts/setup_pi05.py
export OPENPI_DATA_HOME="$PWD/cache/openpi"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
```

The setup script installs the pinned upstream source, uses its lock file and
applies its Transformers replacements in this new environment. For a critic-only
or CPU development environment, use the smaller core stack:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
# Choose the CUDA index for H20 critic training, or the CPU index for CPU tests.
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

For CUDA critic-only training, replace the CPU index above with
`https://download.pytorch.org/whl/cu128`. Do not run the core requirements install
after policy setup; use the policy setup lock file as the policy environment's
dependency authority. [PI05.md](docs/PI05.md) describes full offline installation:
transfer the pinned checkout including `.git`/submodules, the populated uv cache,
wheelhouse, tokenizer cache and weights, then use `setup_pi05.py --offline`.
A wheelhouse alone does not contain OpenPI's pinned Git dependencies.

For core-only offline installation, prepare a wheelhouse on a matching connected
machine (`python -m pip download -r requirements.txt -d wheelhouse` with the
appropriate Torch index), transfer it, then install with
`python -m pip install --no-index --find-links wheelhouse -r requirements.txt`.
Transfer setuptools/wheel bootstrap wheels as well and install this repository
with `--no-deps --no-build-isolation -e .`.

Before downloading weights, run the read-only server check:

```bash
python scripts/check_environment.py --config configs/smoke.yaml --world-size 8
```

It prints Python, the installed Torch/CUDA build, visible GPU count, GPU memory
capacity and driver versions using `nvidia-smi`; it allocates no GPU memory.
`requirements.txt` pins the Torch release, while the chosen installation index
selects CPU versus cu128. A GPU configuration requires the CUDA 12.8 build.
Errors exit with code 2. `ready: true` means the metadata checks passed, not that
CUDA kernels or NCCL have run. Full `train.py --dry-run` includes the same report.

On Linux, NVIDIA lists CUDA-12 minor compatibility from driver 525.60.13 and a
CUDA-12.8 GA driver level of 570.26. The intermediate range gets an explicit
PTX/JIT compatibility warning rather than being rejected as always incompatible.
A driver below 525.60.13 fails preflight. Even a newer driver must pass GPU smoke.
The reported CUDA build is the wheel's runtime, not an installed `nvcc` toolkit.
See [NVIDIA's release-note tables](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html).

## 2. Download and verify data

The dataset examples use the private repository
`JMG-NTU123/liberox-scene23-l2-posttrain`. The two packages have been prepared;
the examples require them to be uploaded and your account to have access.
You can also use a locally supplied verified package. Credentials are read
from the Hugging Face credential store or `HF_TOKEN`; never put tokens in YAML,
commands saved in this repository, or Git.

```bash
# Optional mirror, when it supports authenticated access to this private repo:
# export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_dataset.py \
  --repo-id JMG-NTU123/liberox-scene23-l2-posttrain \
  --revision DATASET_COMMIT --subset bc50 --output data
python scripts/download_dataset.py \
  --repo-id JMG-NTU123/liberox-scene23-l2-posttrain \
  --revision DATASET_COMMIT --subset scene23_61 --output data
python scripts/verify_dataset.py data/bc50
```

Replace `DATASET_COMMIT` with the uploaded dataset commit for reproducibility.
Rerun the same download command after interruption; the Hub keeps local resume
metadata. `HF_ENDPOINT` must be set before starting Python. Download verifies
every manifest-bound file; training verifies again before constructing or reusing
a cache. SHA mismatches fail immediately. Never edit a checksum to hide damage.

Each portable package has `manifest.json`, an English dataset card, selected
members, complete ancestor records and per-file SHA-256 digests. Runtime resolves
only package-relative manifest paths. Historical source paths inside copied
`run.json` are provenance and are never needed to train. Ancestors are retained
for lineage; only selected members contribute training chunks, with inherited
prefixes deduplicated. The first packages are:

| Package | Selected members | Physical records with ancestors | Training chunks before holdout |
| --- | --- | --- | --- |
| `bc50` | 39 autonomous successes + 11 takeover chains | 73 | 4,296 |
| `scene23_61` | BC50 + 11 autonomous failures | 73 | 4,935 |

The package manifest identifies the exact selected records, ancestor records
and file hashes.

## 3. Obtain pi0.5 weights for BC/IQL

Critic mode skips this section. The original base is
`Adam-YAN/pi05-liberox-base`, pinned to
`4664e4725d2e72c3a8af8b09adfda41eaf74514c`.

```bash
python scripts/download_weights.py --output models/pi05_liberox_jax \
  --convert --converted-output models/pi05_liberox_torch --prefetch-tokenizer
```

Conversion runs on CPU and requires substantial host RAM/disk. The result must
contain `model.safetensors`, `assets/meituan/LIBERO-X/norm_stats.json`, and
`pi05_identity.json`. Identity verification binds the checkpoint bytes, statistics,
source revision and OpenPI commit. To download an existing converted checkpoint,
use `--converted-repo OWNER/REPO --revision FULL_COMMIT
--output models/pi05_liberox_torch`. Both workflows only download or convert weights.
See [PI05.md](docs/PI05.md) for conversion and offline details.

## 4. Dry-run, smoke, then train

Run from the repository root. YAML paths are relative to the working directory.
Use a new `output_dir` for each experiment; `--output` and `--cache` override these
two paths without editing a configuration.

```bash
# Hash data and model identity, validate configuration, print sample/update plan.
# This does not instantiate the policy, build a cache or initialize CUDA/NCCL.
torchrun --standalone --nproc_per_node=8 train.py \
  --config configs/bc50_reproduce.yaml --data data/bc50 --method bc --dry-run

# Frozen policy smoke: 4 optimizer updates, save, export.
torchrun --standalone --nproc_per_node=8 train.py \
  --config configs/smoke.yaml --data data/bc50 --method bc

# Full policy smoke: 2 updates, sharded checkpoint and export.
torchrun --standalone --nproc_per_node=8 train.py \
  --config configs/smoke_full.yaml --data data/bc50 --method bc

# Baseline reproduction: frozen, global batch 16, 800 policy updates, seed 7.
torchrun --standalone --nproc_per_node=8 train.py \
  --config configs/bc50_reproduce.yaml --data data/bc50 --method bc

# Independent critic: batch 256, 15,000 Q/V updates, no policy files required.
torchrun --standalone --nproc_per_node=8 train.py \
  --config configs/critic.yaml --data data/scene23_61 --method critic

# Recommended next stage: start from the held-out-selected critic above.
# The frozen actor sees the same complete BC50 membership and budget as BC50 BC.
torchrun --standalone --nproc_per_node=8 train.py \
  --config configs/iql_from_critic.yaml --data data/bc50 --method iql

# Optional fresh-critic ablation: full BC50 pool, with critic-only warmup.
# Its critic metrics are training diagnostics, with no held-out model selection.
torchrun --standalone --nproc_per_node=8 train.py \
  --config configs/iql_frozen.yaml --data data/bc50 --method iql

# Full IQL: run after the full policy memory smoke passes.
torchrun --standalone --nproc_per_node=8 train.py \
  --config configs/iql.yaml --data data/scene23_61 --method iql

```

The recommended IQL path has two stages. `critic.yaml` holds out 20% of root
trajectories and selects `outputs/critic/critic_best.pt`. Then
`iql_from_critic.yaml` initializes Q/V from it, inherits its exact normalization,
and uses **all** selected actor data (`validation_fraction: 0`). It does not add
critic-only warmup after pretraining. Q/V continue updating during IQL. The
former held-out roots can now enter training, so the pretraining validation score
is a historical selection result, not a held-out guarantee for the updated IQL
critic. The later metrics are explicitly labeled training diagnostics.

Use `--data data/bc50` for both BC and frozen IQL to align actor membership:
50 selected records, 50 roots, and 4,296 eligible chunks. The 800 × 16 budget is
12,800 replacement draws; it does not promise every eligible chunk is drawn.
Using `data/scene23_61` for the actor adds 11 failure records and changes the data
composition. That is a separate experiment even if the batch/update budget is
identical. Pretraining the critic on the 61-record package remains independent
of which verified package is subsequently used for the actor.

Use the following checks when running on a new server:

| Stage | Acceptance criterion |
| --- | --- |
| Download and verify | Every dataset SHA matches; model identity and tokenizer are available. |
| Environment and `--dry-run` | Exit 0, expected 8 visible GPUs and cu128 build, correct batches/updates. Resolve any minor-compatibility warning through frozen GPU smoke before a long run. |
| `smoke.yaml` | Four frozen updates on eight NCCL ranks; finite loss, correct batch counters, complete checkpoint, export, and summary `COMPLETED`. |
| `smoke_full.yaml` | Two full updates, finite loss, a complete rank-sharded checkpoint and export, `COMPLETED`; inspect host RSS and CUDA allocated/reserved peaks in `memory.jsonl` and summary. |
| `bc50_reproduce.yaml` | Exactly 800 actor updates / 12,800 draws; export loads and infers on the evaluation platform; assess paired outcomes on the original 50 openings against the 40/50 reference. |
| Critic, IQL from critic, then further experiments | Select the pretraining critic using held-out validation; verify all actor chunks are eligible, inherited normalization matches, and later IQL metrics are labeled training diagnostics. |

For reproduction, record per-opening outcomes and an acceptance tolerance for
BC50. A statistically inconclusive difference on only 50 openings does not by
itself establish equivalence. This code has not yet been evaluated on those openings.

Use `--nproc_per_node=1` with the same configuration to retain global batch and
optimizer update counts. Accumulation is inferred exactly; incompatible world
sizes or explicit accumulation settings fail before training. The configurations
also support 2 and 4 GPUs. Full-backbone BC is in `configs/bc_full.yaml`; full IQL
is in `configs/iql.yaml`. The recommended `iql_from_critic.yaml` uses frozen mode.
Full H20 memory fit has not yet been measured.

`--dry-run` for BC/IQL requires downloaded weights for identity verification;
having only this repository is not sufficient. For framework-only validation:

```bash
CUDA_VISIBLE_DEVICES='' torchrun --standalone --nproc_per_node=2 train.py \
  --config configs/smoke_critic_cpu.yaml --data data/scene23_61 --method critic --dry-run
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 torchrun --standalone --nproc_per_node=2 train.py \
  --config configs/smoke_critic_cpu.yaml --data data/scene23_61 --method critic
```

The first training run preprocesses NPZ recordings into versioned mmap `.npy`
arrays once on rank zero. Cache identity includes dataset bytes, normalization,
split, image orientation, horizon, reward/termination contract and implementation
hashes. Later runs verify and reuse it. Training workers read only mmap arrays.
Allow enough disk for original recordings plus caches; preprocessing time is
separate from the four-update server smoke and may take longer than a few minutes.
Cache generation and dataset hashing have console progress; multi-process workers
are used during training. Monitor `data_wait_seconds` to distinguish I/O from
compute time. Do not benchmark the first cache build as model throughput.

## 5. Exact batch and update meanings

```
global_batch_size = per_device_batch_size × world_size × gradient_accumulation_steps
```

All lengths, warmup intervals, validation intervals and checkpoint intervals use
**completed optimizer updates**, never microbatches. Each Q/V update comprises
one V optimizer step and one twin-Q optimizer step. IQL `optimizer_updates` is
the number of policy updates, with `critic.warmup_updates` added before them.
Logs expose both counters, cumulative microbatches and sample draws.

| Configuration | Global batch | Per GPU | Accumulation at 8 / 1 GPU | Policy updates | Q/V updates |
| --- | --- | --- | --- | --- | --- |
| BC50 | 16 | 2 | 1 / 8 | 800 | 0 |
| Critic | 256 | 32 | 1 / 8 | 0 | 15,000 |
| IQL | 64 | 2 | 4 / 32 | 5,000 | 17,500 |
| IQL from critic | 16 | 2 | 1 / 8 | 800 | 800 after pretraining |
| Frozen IQL | 16 | 2 | 1 / 8 | 800 | 13,300 |

BC50 uses all 50 roots, no holdout, 4,296 chunks, peak/final LR `3e-5/3e-6`,
80-update warmup, cosine decay, seed 7, and replacement sampling as in the
reference. It draws 12,800 samples. This matches the reported training budget;
DDP partitioning, random flow noise and numerical kernels mean it does not promise
bitwise equality or the original 40/50 evaluation result. Evaluate the returned
overlay on the same 50 openings to assess reproduction.

Other configurations shuffle global microbatches without duplicate rank padding;
the short epoch tail is dropped and reshuffled next epoch. Replacement sampling
can intentionally draw an index more than once, including across ranks.
ZeRO-1 partitions optimizer state by rank and parameter dtype. The policy is
replicated on each GPU. `training.checkpoint_format: consolidated` retains the
original full-optimizer checkpoint. All full-backbone configurations use
`sharded`: optimizer/RNG state is saved separately by rank, avoiding rank-zero
optimizer consolidation. Model weights are still saved once and can be large.

## 6. Reward, termination and critic diagnostics

The first release implements sparse reward only: every block gets `-1`, a
confirmed-success block gets `0`, no reward normalization, `gamma=0.99` **once per
block**. Success requires five consecutive successful primitive steps. Blocks
after confirmed success are excluded. `success_aware_recording_end_v1` keeps
`bootstrap_mask=1` at an unsuccessful recording-limit ending; confirmed success
sets it to zero. The final available observation provides the truncated
transition's bootstrap state. Unknown reward settings are rejected; a future
reward implementation must update the versioned contract and cache identity.

Image orientation is always vertical flip enabled and horizontal flip disabled.
Critic inputs use two cameras, eight proprioception coordinates, action chunks
of eight × seven, and a mask. Policy inputs retain raw actions/state for its own
OpenPI normalization. Critic normalization is fit only on training roots and
stored in the export; `data.normalization` can select an explicit JSON file.
Standalone critic initialization carries its original normalization into IQL.

Critic pretraining holdout is deterministic by root trajectory, including takeover
lineage, so related prefixes cannot cross splits. All shipped IQL configurations
use the full selected actor pool. Explicit IQL experiments with a positive holdout
fraction remain supported, but then the actor pool is smaller and is not aligned
with all-data BC50. The critic uses 128-pixel images and pad-4
random shifts, V → Q → target-Q updates, expectile 0.8, beta 3, and weights capped
at 20.
Both Q and V use **AdamW with weight decay 0.01**.
`critic.optimizer: adam` and an explicit weight decay can be used for
controlled comparisons. The actor optimizer is configured independently and is
set in each configuration. Actor gradient clipping is 1; critic/value clipping
is 10.

Validation reports Spearman(V, blocks remaining) on successful paths, weight
quantiles, fraction at 20, fraction below 0.1, ESS/N and successful-final-block
weight mass. With a real holdout, `validation` metrics select `critic_best.pt` by
minimizing finite validation Spearman. In all-data IQL, the metric group is
`train_diagnostic`; training scores do not create a best critic or populate a
best-validation result. `critic_last.pt` contains the final weights in either
case. Dry-run, provenance and summary expose `critic_evaluation` with its split,
sample count, holdout status and selection policy. Initialization provenance keeps
the source critic's metadata and artifact hashes and its historical evaluation.
An undefined metric is `null` and does not select a best checkpoint.
This progress correlation does not establish
Q's ability to rank alternative actions. Reusing validation to select a checkpoint
does not produce an independent test score. See [CRITIC.md](docs/CRITIC.md).

## 7. Outputs, safe stop and resume

Each run has resolved `config.yaml`, `provenance.json`, `normalization.json`,
`metrics.jsonl`, stage-wise `memory.jsonl`, TensorBoard events, atomic checkpoint
files with SHA sidecars,
and `summary.json`. Provenance records code commit/dirty state, dataset identity,
model identity, replay fingerprint, reward/termination rule and resolved batches.

```bash
tensorboard --logdir outputs --bind_all
tail -f outputs/critic/metrics.jsonl
# Deterministic safe stop: every rank observes this at an accumulation boundary.
touch outputs/critic/stop.request
# SIGTERM or Ctrl+C also request a boundary stop; avoid SIGKILL.
rm outputs/critic/stop.request
torchrun --standalone --nproc_per_node=8 train.py \
  --config configs/critic.yaml --data data/scene23_61 --method critic \
  --resume outputs/critic/checkpoints/update_00001000.pt
```

Use the checkpoint path written in `summary.json`. Restore in the same output
directory with the same world size, update schedule, global/per-device batch,
data/model identities and training settings. Exact resume restores optimizer,
target networks, rank RNG and sample position; worker prefetch does not advance
the saved position. A changed contract fails. Checkpoints use PyTorch optimizer
serialization: load only your own trusted files. The portable critic export
uses tensor-only loading and includes all input interpretation metadata.

For `sharded` checkpoints, `update_XXXXXXXX.pt` points to an immutable hidden
generation directory containing model weights, the committed SHA manifest and
one optimizer/RNG file per rank. Exact restore additionally checks Torch's full
version, dtype groups, ZeRO mode and the parameter-partition layout. Transfer the
**whole checkpoints directory including hidden generation directories and
symlinks**, for example with `rsync -a`; copying only the public `.pt` link cannot
restore training. Manual policy export reads only model weights from a committed
generation. Previous generations are retained; automatic garbage collection is
not implemented. Existing consolidated files remain a supported format, while
changes to defaults/configuration still invalidate an exact-resume contract.

`memory.jsonl` samples before updates, before/after checkpoint and after export.
`summary.json.memory_metrics` retains the maxima sampled in the current invocation:
rank-zero and maximum process peak RSS, aggregate current trainer RSS, and maximum
CUDA allocated/reserved peaks. Linux process high-water RSS catches transient
checkpoint allocations. These trainer counters exclude DataLoader worker RSS and
are not total host RAM use; CUDA allocator metrics exclude other processes and
some driver allocations. Inspect them with the server's host/GPU monitoring
before accepting the full memory smoke.

Resuming an older checkpoint archives prior run artifacts under `retired/`,
restores the selected best critic at that boundary, and truncates later metrics.
The original output directory is required. An explicit resume consumes a stale
`stop.request`. Use the stop file when large checkpoints need time to save:
an external scheduler or torchrun launcher may enforce a timeout after signals;
a forced kill can only retain the last fully published checkpoint.

## 8. Export an actor for evaluation

Training exports `outputs/<run>/policy/`. To evaluate using the compatible
collection platform, create a package with the base-model path as it exists on
the evaluation machine:

```bash
python scripts/export_policy.py \
  --checkpoint outputs/bc50_reproduce/checkpoints/update_00000800.pt \
  --output exports/pi05-bc50-reproduce \
  --platform-base-checkpoint /PATH/ON/EVALUATION/MACHINE/pi05_liberox_torch
tar -czf pi05-bc50-reproduce.tar.gz -C exports pi05-bc50-reproduce
```

Transfer the package and place its directory into the evaluation platform's
configured policy registry, then select it normally. Preserve the directory name:
it is the `policy_id`.
`policy.yaml` keeps the platform's exact schema-5 fields and compatibility hashing;
extra metadata goes in a sidecar. Do not hand-edit its base path or hashes.
The platform needs the identical base weights. Critic outputs are independent
Q/V artifacts, not platform policy overlays. Verify real-model load and inference
on the evaluation machine before assessing task performance.

## 9. Package and upload datasets

```bash
python scripts/package_dataset.py \
  --dataset-root /PATH/TO/platform/dataset-root \
  --dataset-id ds_20261018_132030_68ca5a3c --output /EXPORTS/bc50
python scripts/package_dataset.py \
  --dataset-root /PATH/TO/platform/dataset-root \
  --dataset-id ds_20261019_164826_1910b132 --output /EXPORTS/scene23_61
# Alternative selection: --run-ids RUN_ID_1 RUN_ID_2 ...
# Verification/plan only; no network mutation:
python scripts/upload_dataset.py /EXPORTS \
  --repo-id JMG-NTU123/liberox-scene23-l2-posttrain
```

To upload, add `--execute`. The script creates
a private dataset repository, rejects public destinations, verifies all source
hashes, and uses the Hub's resumable large-folder upload transport. It uploads only
manifest-listed files. Data and weights are excluded from Git. See the official
[download](https://huggingface.co/docs/huggingface_hub/guides/download) and
[upload](https://huggingface.co/docs/huggingface_hub/guides/upload) guides.

## 10. Verification and troubleshooting

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  python -m pytest -q
```

Tests cover configuration and update conversion, portable packaging/hash failure,
root splitting, cache construction, termination, augmentation/diagnostics,
checkpoint/resume, policy schema and CPU distributed gradient/optimizer behavior.
They do not require pi0.5 weights or a GPU. Run the policy smoke configurations
separately on the target GPU environment.

- **Out of memory:** use `frozen`, reduce `per_device_batch_size` and let `auto`
  increase accumulation while retaining global batch. Keep ZeRO enabled; it does
  not shard the replicated model. Save/resume checkpoints can use substantial
  CPU RAM for model serialization and, with consolidated checkpoints, optimizer
  consolidation. Full configurations use rank-local optimizer shards; this still
  requires a real H20 memory smoke. Start a new run if changing batch layout.
- **NCCL timeout:** verify the visible device count, run `configs/smoke.yaml`,
  set `NCCL_DEBUG=INFO`, and inspect all rank logs. Allow first-time cache/hash
  preprocessing to complete. If the server's transport is unavailable, diagnose
  its specific NCCL network/P2P configuration before a long run; increasing the
  timeout alone does not fix a dead rank.
- **Hash mismatch:** redownload the damaged file or recover a known-good package.
  Cache corruption is rejected/rebuilt from verified recordings. The bounded CRC
  reread workaround never bypasses CRC or SHA checks and never changes source
  bytes.
- **No best critic:** all-data IQL deliberately reports training diagnostics and
  does not select `critic_best.pt`. In held-out critic runs, Spearman may also be
  undefined for constant values or too few successful blocks. Inspect the split
  and diagnostics; the last checkpoint is retained independently.
- **Missing model on dry-run:** critic mode is self-contained; BC/IQL require the
  pinned converted base and identity files before their model preflight can pass.

PyTorch distribution behavior follows its official
[DDP](https://docs.pytorch.org/docs/2.7/generated/torch.nn.parallel.DistributedDataParallel.html)
and [distributed optimizer](https://docs.pytorch.org/docs/2.7/distributed.optim.html)
contracts.
