# BC/IQL 分支审阅说明（2026-11-14）

`review/bc-iql-updates-20261114` 汇总初版 `main` 后的 BC、IQL 及共用训练改进，便于独立审阅。基础为 `main` 的 `5ba2f0abed68b570841b175d5b43f79e35a8223f`；保留截至 `1980210be3b7657dd199c31bb73277541171ab7d` 的七个功能提交，并单独移入 `2602816bdc13ea07e590607451de89f6fa0f7281` 的 actor BC CPU/DDP 测试修复。

训练入口仍为 `train.py --method bc|iql|critic`。IQL 和独立 critic 使用本项目原有的 PyTorch `PixelIQL`（twin Q、V、target Q）；本分支没有移入作者官方 IDQL 的 JAX Q/V、专用入口或 vendor 代码。BC 官方示教数据的适配与“官方 IDQL 算法实现”是不同功能。

拆分后的分支分工：

| 分支 | 角色 |
| --- | --- |
| `main` | 初版 BC/IQL 发布基线。 |
| `review/bc-iql-updates-20261114` | BC/IQL 更新及共享基础，审阅基座为 `main`。 |
| `review/official-idql-twinq-20261114` | 官方 IDQL，以上一分支为共享基座，审阅差异只包含官方 IDQL 层。 |
| `archive/custom-preference-critic-20261114` | 旧 custom preference critic 历史归档。 |

## 功能范围

| 功能 | 行为与入口 |
| --- | --- |
| 官方 BC 示教数据 | `scripts/attach_official_bc.py`、`official_bc.py` 适配固定来源的 observation/action 示教；保留真实动作 mask 和来源校验，完成最后一个 optimizer update 后仍可导出。 |
| Haizhou/root split 与 BC 验证 | `data.split_strategy: haizhou_hash_threshold` 按完整 root 的哈希阈值划分，父轨迹、接管分支及共享前缀同组。`training.bc_validation: true` 在 held-out cache 上输出固定种子的 flow loss；它不是 rollout 成功率。默认仍是原有 `hash_rank_count`。详见 [HAIZHOU_BC_VALIDATION.md](HAIZHOU_BC_VALIDATION.md)。 |
| BCinit | `model.actor_init_policy` 校验并读取完整 BC 导出包，为 full IQL 初始化 actor 参数；不继承 BC optimizer、RNG 或 IQL 更新计数。IQL resume 以自身 checkpoint 为准。 |
| IQL 的官方 BC 辅助损失 | `actor_bc.dataset_root` 启用独立、仅用于 actor 的官方示教池；native IQL 与官方 BC 各自取均值后加权，一次 forward/backward。官方示教不进入 Q/V，也不补造 reward、done 或 next state。示例两卡 global batch64 为 62 native + 2 official，Q/V 仍用 64 native。详见 [README_actor_bc.txt](../README_actor_bc.txt)。 |
| 全状态续训 | 原有同目录 `--resume` 恢复 actor、critic、optimizer、RNG、计数和日志边界。`training.continuation` 仅支持代码中绑定来源、SHA、两 rank 分片及学习率策略的 IQL 30000→45000 特例，不是任意延长训练的通用开关，也不同于 BCinit。 |
| 共用 critic 诊断 | held-out 评估增加 Q1/Q2 MSE、Q loss、V expectile loss、Q 分歧及 target 均值；按 transition 数量聚合，恢复训练模式与 RNG，不更新参数、梯度或 target。IQL 使用全量 actor 数据时，训练集诊断仍明确标注为 training diagnostic。 |
| 可选内存设置 | `distributed.gradient_as_bucket_view`、`training.adamw_foreach: false`、`model.vision_gradient_checkpointing` 均为显式选项；默认保留旧行为。CPU 检查不能证明实际 GPU 显存或吞吐，详见 [FULL_MEMORY_OPTIONS.md](FULL_MEMORY_OPTIONS.md)。 |

## 本次本地验证

验证在独立 worktree 中进行，使用已有虚拟环境，不安装新依赖。所有测试命令显式设置 `CUDA_VISIBLE_DEVICES=''`，只运行已有 CPU 测试；不加载生产模型或历史训练权重。

环境：Python 3.11.17、PyTorch 2.7.1+cu128、pytest 8.4.2、Transformers 4.53.2。虽然安装的是 CUDA wheel，测试通过以下环境设置屏蔽 GPU：

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/JMG/Projects/post-training-multi-gpu-official-idql/.venv/bin/python -m pytest -q \
  tests/test_official_bc.py tests/test_haizhou_bc_validation.py \
  tests/test_full_memory_options.py tests/test_actor_init.py tests/test_actor_bc.py \
  tests/test_continuation.py tests/test_critic_validation_losses.py \
  tests/test_training.py tests/test_iql_all_data.py tests/test_iql_all_data_ddp.py \
  tests/test_config.py tests/test_data_replay.py tests/test_data_policy_contract.py \
  tests/test_critic.py tests/test_critic_ddp.py tests/test_distributed.py \
  tests/test_optimizer_config.py tests/test_policy_export.py
```

工作目录为本分支仓库根目录。结果：**214 passed in 74.75s**，退出码 0，无失败或跳过；包含真实 Gloo 多进程、fake-policy/ZeRO 分片续训，以及实际 tiny HF vision layer 的 loss/gradient 检查。`git diff --check` 也通过。

本次验证只支持代码与小型测试模型的行为结论，不证明历史训练权重一致、真实 GPU 显存/吞吐或实际任务成功率一致。
