# pi0.5 post-training and official IDQL twin-Q

## 代码导读（BC 与 IDQL）

Scene24 L1 实验（BC78 / BC v2 75 分，IDQL 73 / 76 分）用的就是这个文件夹里的代码。`src/post_training/` 下的路径简写为 `pt/`。

结构和平台的 `vla_rynn_iql` 一样：`pt/training.py` 是 BC 和 IQL 共用的训练流程；两种方法的区别写在 `pt/algorithms.py` 的 `BehaviorCloning` 和 `ImplicitQLearning` 两个类里；IQL 专用的代码在 `pt/iql/`。

### BC：π₀.₅ 行为克隆（多卡）

| 步骤 | 文件 · 函数 |
|---|---|
| 入口 | `train.py --method bc` |
| 读配置 | [`pt/config.py`](src/post_training/config.py) |
| 数据：按根轨迹划分训练/验证集，把轨迹切成 8 步动作块 | [`pt/data.py`](src/post_training/data.py)（`build_semi_mdp_chunks`、`replay_chunks`） |
| 缓存：图像、状态和动作的归一化，mmap 读取 | [`pt/replay.py`](src/post_training/replay.py)（`build_replay_cache`、`fit_normalization`、`CachedReplayDataset`） |
| 模型与损失：π₀.₅（OpenPI）前向，flow matching 损失只算真实执行的步 | [`pt/policy.py`](src/post_training/policy.py)（`Pi05Policy.forward`、`masked_flow_losses`） |
| BC 与 IQL 的差别：BC 不训 critic，损失直接取平均 | [`pt/algorithms.py`](src/post_training/algorithms.py)（`BehaviorCloning`） |
| 共用的训练流程：一步优化、验证集损失 | [`pt/training.py`](src/post_training/training.py)（`_train`、`optimization_step`、`evaluate_actor`） |
| 多卡：DDP、全局采样、ZeRO-1 | [`pt/distributed.py`](src/post_training/distributed.py) |
| 存档和导出成平台模型包 | [`pt/checkpoint.py`](src/post_training/checkpoint.py)、[`pt/export.py`](src/post_training/export.py) |
| 官方示教（只有观测和动作）加入 BC | [`pt/official_bc.py`](src/post_training/official_bc.py) |

### IQL（PyTorch 版，对照用，Scene24 实验中最好 37/50）

| 步骤 | 文件 · 函数 |
|---|---|
| 入口 | `train.py --method iql`（只训 critic：`--method critic`） |
| 与 BC 的差别：先更新 V、Q 和目标网络，再按优势权重加权 actor 损失 | [`pt/algorithms.py`](src/post_training/algorithms.py)（`ImplicitQLearning`、`CriticOnly`） |
| critic 网络（Q1、Q2、V，PixelIQL） | [`pt/iql/critic.py`](src/post_training/iql/critic.py) |
| 可选：官方示教辅助损失、用 BC 初始化 actor、30000→45000 续训 | [`pt/iql/actor_bc.py`](src/post_training/iql/actor_bc.py)、[`pt/iql/actor_init.py`](src/post_training/iql/actor_init.py)、[`pt/continuation.py`](src/post_training/continuation.py) |

### IDQL 训练：只训 Q/V，BC 冻结不训

| 步骤 | 文件 · 函数 |
|---|---|
| 入口（单进程，JAX pmap 多卡） | [`train_idql_official.py`](train_idql_official.py) |
| 训练主流程：准备数据、循环、验证、存档 | [`pt/official_idql/runner.py`](src/post_training/official_idql/runner.py)（`prepare`、`train`、`create_state`） |
| 输入适配：两路相机 128px 加机器人状态；8 步动作块拼成一个动作 | [`pt/official_idql/adapters.py`](src/post_training/official_idql/adapters.py) |
| Q/V 更新：先 V（expectile 0.8），再双 Q 和目标网络 | [`pt/official_idql/core.py`](src/post_training/official_idql/core.py)（`update`，调用作者的 `update_v` / `update_q`） |
| 作者原版 IDQL 代码（未修改，加载时校验） | [`pt/_vendor/idql_official/`](src/post_training/_vendor/idql_official/)（[`jaxrl5/agents/ddpm_iql/ddpm_iql_learner.py`](src/post_training/_vendor/idql_official/jaxrl5/agents/ddpm_iql/ddpm_iql_learner.py)） |
| 验证集检查、选用哪一步的 Q | [`pt/official_idql/diagnostics.py`](src/post_training/official_idql/diagnostics.py)、[`pairs.py`](src/post_training/official_idql/pairs.py)、[`selection.py`](src/post_training/official_idql/selection.py) |
| 导出打分器、给候选打分 | [`pt/official_idql/scoring.py`](src/post_training/official_idql/scoring.py)（`export_scorer`、`FrozenScorer.score_chunks`） |

### IDQL 推理：评测时 BC 出候选，Q 挑一个

| 步骤 | 文件 · 函数 |
|---|---|
| BC 出 4 个候选（第 0 个就是 BC 原本的输出） | [`idql_eval/runtime_core.py`](idql_eval/runtime_core.py)（`generate_pool`） |
| 用 min(Q1, Q2) 打分，执行分最高的那个 | [`idql_eval/runtime_core.py`](idql_eval/runtime_core.py)（`infer`） |
| 接平台评测（不改平台代码） | [`idql_eval/provider.py`](idql_eval/provider.py)、[`idql_eval/run_idql_evaluation.py`](idql_eval/run_idql_evaluation.py) |

详细说明见 [idql_eval/README.md](idql_eval/README.md)。IDQL 不经过 `pt/algorithms.py` 和 `pt/iql/`。

### 写死的地方

训练和 IDQL 的核心代码只认数据包，不针对某个场景。下面这些地方是写死的，代码里都有注释：

| 位置 | 写死的内容 | 换场景时 |
|---|---|---|
| `pt/official_bc.py`、`pt/iql/actor_bc.py` | 官方示教固定为 LIBERO-X 的第 1719、1720 两条（Scene24 L1，任务 264） | 不加官方示教就不受影响；要加别的任务的示教，需要把这几个值改成配置项 |
| `idql_eval/contract.py`、`provider.py`、`idql_worker.py`、`runtime_core.py` | 评测协议：每局 950 步，每次执行 8 步，20 Hz | 平台评测协议不同时要改这几处 |
| `idql_eval/gpu_smoke.py` | 小测用的 Scene24 L1 指令 | 换任务时连同保存的观测一起换 |
| `idql_eval/runtime_core.py`（`derive_seed`） | 种子名里的 `bc78-n4-v1` | 只是名字。改了会让所有候选变掉，和 73、76 分两次的结果对不上，建议保留 |
| `scripts/upload_dataset.py`、`download_dataset.py` | 默认 HuggingFace 仓库是最早 SCENE23 L2 的数据 | 用 `--repo-id` 指定 |
| `pt/data.py`（上传数据时生成的说明文字） | 标题写着 "SCENE23 L2" | 只影响 HuggingFace 上的说明文字。这个文件是 IDQL 打分器核对指纹的文件之一，所以没在代码里加注释 |


This folder contains training code, tests, generic configuration examples and
usage documentation. The official IDQL entry point is `train_idql_official.py`.

## Entry points

| Entry point | Implementation |
| --- | --- |
| `train_idql_official.py` | Pinned authors' JAX twin-Q/V updates with explicit observation/action adapters. |
| `train.py --method bc` | Masked flow-matching behavior cloning for pi0.5. |
| `train.py --method iql` | Existing PyTorch IQL with advantage-weighted actor updates. |
| `train.py --method critic` | Existing custom PyTorch PixelIQL critic-only path. |
| `idql_eval/` | IDQL at evaluation time: frozen BC proposes 4 candidates, the exported twin-Q picks one. See [idql_eval/README.md](idql_eval/README.md). |

The official path uses the authors' Q/V core. Its vision, action-chunk and
external frozen-policy integration are adaptations, not a reproduction of
the paper's complete DDPM actor experiments. The PyTorch critic is a separate
implementation; its weights cannot be loaded as official JAX checkpoints.

## Code review

- [Official IDQL architecture, interfaces and usage](docs/OFFICIAL_IDQL.md)
- [Official training loop](src/post_training/official_idql/runner.py)
- [Official Q/V core](src/post_training/official_idql/core.py)
- [Candidate generation and Q selection at evaluation time](idql_eval/runtime_core.py)
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
