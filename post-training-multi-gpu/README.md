# pi0.5 post-training and official IDQL twin-Q

This branch contains training code, tests, generic configuration examples and
usage documentation. The official IDQL entry point is `train_idql_official.py`.

## Entry points

| Entry point | Implementation |
| --- | --- |
| `train_idql_official.py` | Pinned authors' JAX twin-Q/V updates with explicit observation/action adapters. |
| `train.py --method bc` | Masked flow-matching behavior cloning for pi0.5. |
| `train.py --method iql` | Existing PyTorch IQL with advantage-weighted actor updates. |
| `train.py --method critic` | Existing custom PyTorch PixelIQL critic-only path. |

The official path uses the authors' Q/V core. Its vision, action-chunk and
external frozen-policy integration are adaptations, not a reproduction of
the paper's complete DDPM actor experiments. The PyTorch critic is a separate
implementation; its weights cannot be loaded as official JAX checkpoints.

## Code review

- [Official IDQL architecture, interfaces and usage](docs/OFFICIAL_IDQL.md)
- [Official training loop](src/post_training/official_idql/runner.py)
- [Official Q/V core](src/post_training/official_idql/core.py)
- [Fixed upstream source manifest](src/post_training/_vendor/idql_official/UPSTREAM.json)
- [pi0.5 environment and policy integration](docs/PI05.md)
- [Existing PyTorch critic](docs/CRITIC.md)
- [Root-group split and BC validation](docs/HAIZHOU_BC_VALIDATION.md)
- [Optional full-model memory settings](docs/FULL_MEMORY_OPTIONS.md)

## Official IDQL setup and input checks

Use a dedicated Python environment. The optional dependency group pins the CPU
JAX stack; GPU execution requires compatible JAX CUDA plugins and drivers.

```bash
python -m pip install -e '.[idql-official,test,plot]'
python train_idql_official.py \
  --config configs/official_idql/idql_official_pixels.yaml \
  --data /path/to/native-package --dry-run
```

Fill in the dataset, frozen membership, normalization and behavior-policy
identity required by the chosen configuration before training. The official
multi-device runner uses one process, not torchrun. Data packages, policy weights
and experiment outputs are supplied separately.

## Tests

Tests are under `tests/`. The official verification command is:

```bash
python tools/verify_official_idql.py --output /tmp/official_idql_verification.json
```

The multi-device CPU reference comparison has a known numerical-tolerance
issue to investigate. Its original test and assertions are retained; do not
assume a completely passing suite.
