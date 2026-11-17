# IDQL 推理（评测时的候选生成与 Q 挑选）

这里是 IDQL **评测时**用的代码：冻结的 BC 每次出 4 个候选动作块，Q 给每个打分，执行分最高的那个的前 8 步。Q 的**训练**在上一级的 `src/post_training/official_idql/`。

Scene24 L1 的两次 IDQL 平台评测用的都是这套代码：

| 评测 | BC | Q | 结果（100 次） |
|---|---|---|---|
| 11-09 | BC78 | 双 Q 第 12500 步（seed 7 划分） | 73 |
| 11-16 | BC v2 | 双 Q 第 15000 步（seed 5993 划分） | 76 |

同样 100 次的对照：base 68，BC78 75，BC v2 75。

## 和训练共用的部分

推理时 Q 相关的代码全部来自训练代码本身，不是另写的：

| 推理里做的事 | 代码 |
|---|---|
| 建 Q 网络、加载导出的打分器 | `official_idql/runner.py`（`create_state`）、`official_idql/scoring.py`（`load_scorer`） |
| 图像缩放到 128、状态归一化 | `replay.py`（`_pixels`、`normalize`） |
| 给候选打分 | `official_idql/scoring.py`（`score_chunks`） |

运行前会检查：上一级仓库里这 16 个源码文件，必须和训练这个 Q 时记录的 SHA-256 逐个一致，否则直接报错。

## 文件

| 文件 | 内容 |
|---|---|
| `runtime_core.py` | **核心，要改挑选方式就改这里**。`generate_pool`：BC 出 4 个候选；`infer`：打分并挑选 |
| `idql_worker.py` | 子进程：加载 BC（GPU）和 Q（CPU，JAX），收观测、回动作 |
| `provider.py` | 平台策略接口：配置里的那个 BC 交给 `idql_worker.py`，其他模型仍走平台原来的流程 |
| `run_idql_evaluation.py` | 在自己的进程里替换平台的策略接口，然后调用平台原有的评测循环（只读导入） |
| `contract.py` | 读配置；生成并校验"运行合同"：把 BC 文件、Q 文件、训练源码和本目录代码的 SHA-256 绑在一起 |
| `prepare_eval.py` | 准备一次评测：从平台已有的 BC 评测任务复制配置，生成运行合同 |
| `gpu_smoke.py` | 正式评测前的 GPU 小测（8 次决策，不跑仿真） |
| `run_eval.sh` | 正式评测：按配置跑各批，期间持有平台 GPU 锁 |
| `config.example.json` | 配置示例，填的是 76 分那次的值 |

## 每一步怎么做（`runtime_core.py`）

1. **出候选**（`generate_pool`）
   - 候选 0：和单独跑 BC 完全一样的那个输出，噪声也来自同一个随机流。
   - 候选 1–3：一次批量采样，噪声来自独立的随机流。
   - 随机种子从每局的种子派生（`derive_seed`），所以同一局可以完全复现。
2. **打分**（`infer`）：对每个候选的前 8 步算 Q1、Q2，分数取 min(Q1, Q2)。剩余步数不足 8 步时，用掩码只算剩下的步数。
3. **挑选**：取分数最高的那个（`np.argmax`），执行它的前 8 步。

## 想改的话

- **换挑选规则**（比如按分数抽样，或者分差不够大就用候选 0）：改 `infer` 里 `index = int(np.argmax(scores))` 这一行附近。
- **换候选个数**：要一起改这几处：
  - `generate_pool` 里的 1+3 以及噪声形状 `(3,10,32)`；
  - `infer` 里的 `(4,8)` 掩码和 `(4,)` 形状检查；
  - `contract.py` 里的 `SELECTOR['candidate_count']`；
  - `provider.py`、`idql_worker.py` 里的 `candidate_count == 4`。
- **换候选来源**（比如混入原始 π₀.₅ 的动作）：改 `generate_pool`，另外在 `idql_worker.py` 里加载第二个模型。

改了以后，旧的运行合同就对不上了，这是有意的：`prepare_eval.py` 会重新生成一份新的合同。

## 怎么跑

前提：
- 平台已经在跑；
- 配置里的 BC 已经注册在平台上，并且有对应的平台评测任务（用来复制任务、开局和种子）；
- Q 已经用 `train_idql_official.py` 训好并导出。

1. **写配置**：复制 `config.example.json`，填好本机路径。
   - `dependencies_dir`：只装了 gym 的私有目录。作者代码导入时需要 gym，这样就不用往 pi05 环境里装东西。
   - `platform_env_snapshot`：平台进程的环境变量（`/proc/<平台 pid>/environ` 的拷贝），让评测和平台任务的运行环境一样。
2. **准备评测**：
   ```bash
   cd idql_eval
   IDQL_EVAL_CONFIG=/path/config.json conda run -n vla-liberox python -B prepare_eval.py
   ```
3. **GPU 小测**：
   ```bash
   IDQL_EVAL_CONFIG=/path/config.json conda run -n pi05 python -B gpu_smoke.py \
     --runtime-contract <results_dir>/runtime_contract.json --observation-npz <观测.npz> --output <结果.json>
   ```
4. **正式评测**：`bash run_eval.sh /path/config.json`。每批 50 次，约 70 分钟。结果在 `<results_dir>/<批次名>/evaluation.json`。

**隔离**：平台代码一个文件都不改，也不注册新模型。结果只写在 `results_dir`，平台的评测列表里看不到。

## 和评测时副本的关系

之前每次评测都在实验目录里复制一份这些文件，并把路径、Q 步数、BC 身份等写死在代码里。这里把这些值全部移到了配置文件，推理逻辑本身没有改。

核对方法：用 76 分那次的 BC v2、Q15000 和同一份观测重跑 `gpu_smoke.py`。8 次决策的候选、噪声、Q1/Q2 分数和选中的序号，全部和当时的记录逐位相同。

**不包括**：11-15 那次单 Q（branch/v1 数据，75 分）的评测。那次的 Q 是用实验分支的代码训练的，本仓库里没有那套代码。
