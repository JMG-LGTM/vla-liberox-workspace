# Q-planning

本模块从带终止标签的机器人轨迹训练动作块 Q，并在闭环中使用冻结策略生成候选。在仓库根目录运行 `python -m qplanning.<模块> --help` 查看参数。

## 配置与数据

[defaults.json](defaults.json) 定义每局最大步数、动作块长度、动作维度，以及存档、平台观测和策略输入的有序相机键。按该文件的格式给出实验配置，训练、缓存、离线评分和运行时使用同一份设置。适配其他环境时，平台策略也需要支持对应的动作和观测形状。

训练 manifest 为每条原始根轨迹提供 train/val 划分、成功标签、终止动作索引、记录路径和轨迹与观测的 SHA-256。动作归一化仅使用训练根轨迹。DINO 图像特征以 float16 缓存，进入 Q 网络时转为 float32。

## 训练与离线评分

- `qplanning.fit` 接收 `--manifest`、`--dino-dir`、`--cache`、`--output`、`--seed` 和可选的 `--config`。输出逐步训练日志、验证损失、恢复点、评分权重和按根轨迹计算的验证统计。`qplanning.smoke` 通过相同的数据入口验证训练和 EMA 更新。
- `qplanning.c_score` 接收 `--checkpoint`、`--action-stats`、`--candidates` 和 `--output`。候选 NPZ 提供按状态对齐的冻结视觉 `features` 和原始环境 `actions`，后者形状为 [状态, 候选数, 动作块长度, 动作维度]。可附二值 `labels` [状态, 候选数, 重复次数]。输出候选分数、状态内和跨状态的离散程度，以及有标签时的成对排序指标。

## 闭环选择

`qplanning.runtime.contract` 将平台评测配置与候选、接续策略 ID 对齐。当前协议由同一个策略在每个执行窗口结束后重新规划。`--candidate-count` 控制候选数，`--pick greedy` 使用期望 Q 最高的候选，`--pick random` 使用独立选择种子从相同候选池中均匀取样。附加候选使用通用噪声命名空间 `qplanning-n4-v1`。复跑已有实验时，用 `--noise-namespace` 传入该实验配置记录的值。

`qplanning.runtime.smoke` 核对固定噪声下的每个候选与原始策略推理是否一致。对于按 Q 选择的运行，外部评分复核配置提供参考权重、动作统计、状态、分数和容差；复核记录与本次运行契约绑定。`qplanning.runtime.run_eval` 在进入平台批处理循环前检查 smoke 证明。真实推理期间持有平台 GPU 锁。

## CPU 回归

`python -m qplanning.check_training` 覆盖数据窗口、输入形状、梯度规则、恢复和导出。`python -m qplanning.runtime.check_runtime` 检查候选生成、选择随机数流和 smoke 证明。特定实验的归档权重、数据和对照脚本由实验目录管理。

### 归一化动作块的加权规划

使用 `--pick weighted --candidate-count N --elite-count K --temperature T --flow-steps F` 可对 N 个候选的期望 Q 排序，从最高的 K 个计算 `softmax((Q - max(Q)) / T)`。使用 `--pick random_mean` 时，同一噪声流生成相同候选，独立的选择种子均匀、无放回地抽取 K 个，再等权平均。两个模式都先在策略的归一化空间平均完整动作块，再解码并裁剪环境动作。候选采样步数可以独立于已注册模型设置。

每次决策复用一个图像和指令前缀缓存，候选分别使用原生单样本去噪。加权模式对原始单位动作的前 H 步使用训练集统计归一化，复用一份视觉特征，批量计算所有候选的 Q。环境在最后一次决策只执行剩余步数。轨迹记录包括每槽 Q、最小值、最大值、均值、样本标准差、选中下标与权重。样本标准差小于 `1e-4` 的决策标记为退化。

`python -m qplanning.runtime.check_planning` 在 CPU 上测试完整动作块加权、选择随机数、共享前缀以及与 OpenPI 原生单条采样的等价性。GPU smoke 对每个实际候选逐项核对原生归一化输出与解码输出，核对本次 Q 的离线分数并独立重算聚合动作。可用 `--reference-proof` 提供已经验证的归档评分记录。

可在构建运行契约时用 `--score-max-abs-tolerance` 设置当前 Q 的运行时与离线评分最大绝对误差。逐项差的最大值小于或等于该值才通过，非有限分数直接失败。smoke 按状态记录两边的第一名、前 K 个候选的重合数和 Spearman 秩相关。第一名并列时用最小候选下标，前 K 个使用规划器的 CPU top-K 排序。Spearman 对并列分数取平均秩，常数向量和单候选记为空。候选少于 K 时，前 K 重合数记为空。加权校验继续使用运行时记录的 Q 值独立复算。归档评分使用归档本身的容差。
