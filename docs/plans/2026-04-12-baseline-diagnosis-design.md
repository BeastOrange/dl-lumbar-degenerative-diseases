# Baseline Diagnosis Design

**目标**

把当前“单目标、快速可跑”的 baseline 升级成一个可诊断、可比较、对类别不平衡更稳的 baseline，为后续多目标扩展提供可信起点。

**范围**

- 继续使用当前单目标配置：`spinal_canal_stenosis_l4_l5`
- 保持现有训练/评估 CLI 不变
- 不引入新的训练框架
- 不扩展到多任务或 cross-validation 训练

**方案选项**

1. 只调配置  
   优点：改动最小。  
   缺点：只能看到指标变化，看不到为什么指标差，不足以解释当前 `macro_f1` 塌缩。

2. 只补诊断输出  
   优点：能快速确认多数类塌缩。  
   缺点：不能直接改善结果，仍然需要第二轮改训练逻辑。

3. 诊断输出 + 不平衡修正 + 更合理默认配置  
   优点：一轮同时解决“看不见问题”和“baseline 太弱”两个问题。  
   缺点：改动略多，但仍然局限在训练/配置边界内。

**推荐**

采用方案 3。

**设计**

- 在训练阶段为验证集保存 `predictions.csv`，包含 `y_true`、`y_pred` 和每类概率分数。
- 保持现有 `lumbar-cli evaluate` 行为不变，让它自动复用 `predictions.csv` 输出 confusion matrix / ROC。
- 在训练器中新增可选 `class_weight_mode`，第一版支持 `balanced`，按训练集类别频次构造 `CrossEntropyLoss(weight=...)`。
- 默认训练配置切到 `pretrained: true`、`epochs: 20`、`class_weight_mode: balanced`，但仍保留单目标实验边界。

**风险**

- 预训练权重首次下载会增加启动时间，需要依赖镜像缓存。
- class weight 可能降低 accuracy，但只要 macro-F1 和少数类召回提升，就是符合目标的。
- `predictions.csv` 绑定的是最佳验证 epoch，对当前诊断是合适的，但不是完整离线推理框架。

**验证**

- 训练测试应覆盖：class weight 生效、`predictions.csv` 落盘。
- CLI 训练完成后，应能直接运行 `lumbar-cli evaluate --run-dir ...` 生成图像。
- 新默认配置需要能正常跑完并保留原有 summary / history 产物。
