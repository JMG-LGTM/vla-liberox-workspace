# Optional memory settings for full pi0.5 BC

The full-batch contract stays unchanged: two ranks, per-rank batch32, global
batch64, gradient accumulation1, every backbone parameter trainable. The following
settings are explicit opt-ins; old configurations keep their previous defaults.

```yaml
model:
  backbone: full
  vision_gradient_checkpointing: true
training:
  zero_optimizer: true
  adamw_foreach: false
distributed:
  gradient_as_bucket_view: true
```

`gradient_as_bucket_view` lets DDP gradients share reduction-bucket storage rather
than retaining separate gradient and bucket allocations. It does not partition
parameters or gradients between cards and is not gradient accumulation. The size
and timing of any peak-memory reduction must be measured on the actual model.
The existing optimizer uses `zero_grad(set_to_none=True)` and does not detach
gradient views in place. Checkpoint save/resume keeps its original native actor
names and ZeRO optimizer shards.

`adamw_foreach: false` explicitly selects scalar-tensor AdamW operations rather
than CUDA's automatic foreach path, avoiding foreach's tensor-list intermediates.
It preserves the optimizer, schedule, betas, epsilon, weight decay and gradient
clipping settings. Scalar and foreach kernels may have hardware-dependent
rounding or throughput differences; no memory or speed guarantee is inferred from
the CPU tests. `null` retains the original PyTorch automatic selection.

The pinned OpenPI model's custom `gradient_checkpointing_enable()` already wraps
image embedding calls and its coupled Gemma layer computations. However, setting
only `vision_tower.gradient_checkpointing=True` does not set individual HF4.53
`SiglipEncoderLayer` flags. `vision_gradient_checkpointing: true` additionally
calls the existing HF vision tower's public API with
`gradient_checkpointing_kwargs={"use_reentrant": False}`. This recursively enables
the vision layers. It changes no pinned OpenPI or Transformers source files and
is accepted only for full-backbone training. Full training mode remains enabled;
validation temporarily switches to eval mode and restores the original flags.

CPU tests use the actual installed Transformers4.53.2 `SiglipVisionModel` with tiny
dimensions. They check equal forward losses and every corresponding gradient
with and without layer checkpointing, including nonzero vision gradients when
input pixels require no gradient. Two real Gloo ranks additionally exercise full
fake-policy training with real tiny vision layers, actual bucket-view gradients,
ZeRO AdamW foreach=False, held-out validation, schema5 export and fresh-process
sharded checkpoint resume. These tests do not establish whether real batch32 per
rank fits a 40GB A100. The real full-model GPU smoke remains required.
