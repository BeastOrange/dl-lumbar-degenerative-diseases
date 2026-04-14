# Collapse Recovery Design

**目标**

修复当前单目标 baseline 在验证集上塌缩为“永远预测多数类”的问题，让预训练 backbone 真正吃到有效输入，同时补足训练过程的可观测性和最小可调试闭环。

**问题定位**

- 当前验证集预测已经确认全部为 `0` 类，`accuracy` 与 `macro_f1` 完全等于“全预测多数类”的理论值。
- 当前输入是单通道灰度图，经随机初始化的 `1x1 Conv` 映射到 3 通道后送入预训练 ConvNeXt，这会削弱预训练权重收益。
- 当前图像只做了 `0-1` 归一化，没有补 ImageNet mean/std 标准化。
- 当前只有 loss 加权，没有 batch 级别重采样，也没有训练期的每类召回与预测分布诊断。

**方案选项**

1. 只加 `WeightedRandomSampler`  
   优点：改动最小。  
   缺点：如果输入链路本身不对，仍可能继续塌缩。

2. 修输入链路 + 重采样 + 诊断日志 + tiny-overfit  
   优点：同时解决“预训练没吃到”“类别不平衡仍被忽略”“训练不可观测”和“无法快速定位是否能学会”四个问题。  
   缺点：改动面比只加 sampler 更大，但仍然局限在训练边界。

3. 直接升级到多切片/多序列重构  
   优点：潜力更高。  
   缺点：范围过大，会把当前问题和数据方案升级混在一起，不适合现在。

**推荐**

采用方案 2。

**设计**

- 模型输入侧：
  - 当输入通道为 `1` 且目标 backbone 需要 `3` 通道时，直接复制灰度通道到 RGB，不再使用随机 `1x1 Conv`。
  - 当启用预训练时，对输入增加 ImageNet mean/std 标准化。
- 数据加载侧：
  - 新增训练集 `sampler_mode`，第一版支持 `balanced`，使用 `WeightedRandomSampler`。
  - 新增 tiny-overfit 模式，允许把训练/验证都绑定到同一小批样本上，快速判断模型是否能学穿。
- 训练诊断侧：
  - 在已有 `macro_f1`、`accuracy` 之外，新增每类 recall 和预测类别分布指标。
  - 保持现有 `metrics.csv`、`history.json`、`predictions.csv`、`run_summary.json` 输出兼容。

**风险**

- 重采样后 accuracy 可能下降，但如果少数类 recall 和 macro-F1 提升，这是预期行为。
- tiny-overfit 只用于调试，不能作为正式实验配置默认开启。
- 输入归一化行为变化后，历史 run 不能直接横向对比，需要按新基线重新评估。

**验证**

- 单元测试覆盖：
  - 单通道输入会被稳定复制到 3 通道
  - 预训练输入会进行 ImageNet 标准化
  - `balanced` sampler 会提升少数类采样权重
  - tiny-overfit 会让 train/val 共享同一小样本
  - 训练输出会包含每类 recall 与预测分布指标
- smoke 验证：
  - 合成数据下训练命令可跑通
  - `metrics.csv` 能读到新增指标
  - `predictions.csv` 和 `evaluate` 继续可用
