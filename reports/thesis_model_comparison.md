# 腰椎退行性疾病分类模型对比报告

**生成时间**: 2026-04-14
**数据集**: RSNA 2024 Lumbar Spine Degenerative Classification
**任务**: 脊柱椎管狭窄（L4/L5水平）三分类 — Normal/Mild (0) / Moderate (1) / Severe (2)
**训练配置**: 80/20 划分，BalancedWeightedRandomSampler，AdamW + CosineAnnealing，AMP，Early Stopping (patience=7)

---

## 1. 模型排名

| 排名 | 模型 | 最佳Epoch | **val_macro_f1** | val_accuracy | 过拟合程度 |
|:---:|------|:---:|:---:|:---:|:---:|
| 1 | **ConvNeXt-Tiny + CBAM + TTA(5x)** | 15 | **0.6748** | 0.833 | ⚠️ 中等 |
| 2 | ConvNeXt-Tiny + CBAM | 6 | 0.6685 | 0.800 | ⚠️ 中等 |
| 3 | DenseNet-121 + Dense Reuse | 10 | 0.6078 | 0.808 | ❌ 严重 |
| 4 | Swin Transformer + Hierarchical Fusion | 15 | 0.5231 | 0.717 | ✅ 轻微 |
| 5 | ResNet-101 3D + FeatureVolume3D | 6 | 0.4942 | 0.762 | ❌ 严重 |
| 6 | ViT-Base + Positional Encoding | 20 | 0.3415 | 0.544 | ❌ 严重 |

> **注**: val_macro_f1 = sklearn.metrics.f1_score(y_true, y_pred, average='macro')，在最佳Epoch的验证集上计算。

---

## 2. 每类召回率分析

| 模型 | Recall Class 0 (Normal/Mild) | Recall Class 1 (Moderate) | Recall Class 2 (Severe) |
|------|:---:|:---:|:---:|
| ConvNeXt-Tiny+CBAM | 0.859 | **0.723** | **0.529** |
| DenseNet-121+dense reuse | **0.943** | 0.447 | 0.353 |
| Swin Transformer | 0.828 | 0.340 | 0.412 |
| ResNet-101 3D | 0.919 | 0.106 | 0.451 |
| ViT-Base | 0.667 | 0.128 | 0.216 |

**关键发现**: 所有模型对Class 1 (Moderate) 和 Class 2 (Severe) 的召回率均较低，说明中等和重度病变是分类难点。ConvNeXt+CBAM在各类别上表现最均衡。

---

## 3. 架构设计对比

| 模型 | 核心模块 | 参数量级 | 特征维度 |
|------|---------|---------|---------|
| ConvNeXt-Tiny+CBAM | ConvNeXt Tiny + 通道/空间注意力 | ~28M | 768 |
| DenseNet-121+Dense Reuse | DenseNet-121 + 密集特征复用投影 | ~8M | 1024 |
| Swin Transformer | Swin-Tiny + 4阶段层级特征融合 | ~28M | 768 |
| ResNet-101 3D | ResNet-101 + 3D特征体构建 | ~42M | 2048 |
| ViT-Base | ViT-B/16 + 可学习位置编码 | ~86M | 768 |

---

## 4. 过拟合分析

| 模型 | Train F1 (最佳Epoch) | Val F1 (最佳Epoch) | Gap | 状态 |
|------|:---:|:---:|:---:|:---:|
| ConvNeXt-Tiny+CBAM | 0.888 | 0.669 | 0.219 | ⚠️ |
| DenseNet-121+dense reuse | 0.982 | 0.608 | 0.374 | ❌ |
| Swin Transformer | 0.715 | 0.523 | 0.192 | ✅ |
| ResNet-101 3D | 0.953 | 0.494 | 0.459 | ❌ |
| ViT-Base | 0.658 | 0.342 | 0.316 | ❌ |

**观察**:
- **Swin Transformer过拟合最轻**（gap=0.192），泛化能力最强
- **ResNet-101 3D过拟合最严重**（gap=0.459），3D卷积对小样本数据集过于庞大
- **DenseNet严重过拟合**，密集连接导致特征记忆
- **ViT-Base难以收敛**，ViT缺乏医学影像所需的局部特征归纳偏置

---

## 5. 生成的可视化文件

| 模型 | 混淆矩阵 | ROC曲线 | 训练曲线 |
|------|---------|---------|---------|
| ConvNeXt-Tiny+CBAM | `eval/convnext_tiny_cbam-20260414-101357/confusion_matrix.png` | `.../roc_ovr.png` | `.../training_history.png` |
| DenseNet-121+dense reuse | `eval/densenet121_dense_reuse-20260414-103000/confusion_matrix.png` | `.../roc_ovr.png` | `.../training_history.png` |
| Swin Transformer | `eval/swin_transformer-20260414-104734/confusion_matrix.png` | `.../roc_ovr.png` | `.../training_history.png` |
| ResNet-101 3D | `eval/resnet101_3d-20260414-113804/confusion_matrix.png` | `.../roc_ovr.png` | `.../training_history.png` |
| ViT-Base | `eval/vit_base_posenc-20260414-111046/confusion_matrix.png` | `.../roc_ovr.png` | `.../training_history.png` |

模型排名总图: `reports/figures/model_ranking.png`

---

## 6. 结论与建议

### 主要结论

1. **ConvNeXt-Tiny + CBAM为最佳单模型**（val_macro_f1=0.6685），CBAM注意力机制有效提升各类别召回率
2. **Swin Transformer泛化性最好**（过拟合gap最小），适合作为未来优化的基线
3. **ResNet-101 3D和ViT-Base不推荐**用于此任务（过拟合严重或难以收敛）
4. **所有模型在Moderate/Severe类别上表现欠佳**，需要重点优化少数类分类

### 优化方向

1. **测试时增强(TTA)** ✅ 已验证有效（+0.63% val_f1提升）
2. ~~增强数据增强（medium模式）~~ → 对预训练模型有害，不推荐
3. ~~多任务学习~~ → 梯度冲突导致坍缩，不推荐
4. ~~增强正则化（dropout/wd/ls）~~ → 未超越baseline

---

## 附录A: 增强实验结果

后续进行了两轮增强实验，验证数据增强和正则化策略：

### A1. ConvNeXt正则化增强实验

| 配置项 | baseline | regularization |
|-------|---------|----------------|
| dropout | 0.2 | **0.3** |
| weight_decay | 0.0001 | **0.0005** |
| label_smoothing | 0.0 | **0.05** |
| class_weight_mode | null | **balanced** |
| train_augment_mode | light | light |
| early_stopping_patience | 7 | **10** |

| 模型 | val_macro_f1 | 最佳Epoch | Train F1 | Gap |
|------|:---:|:---:|:---:|:---:|
| ConvNeXt-Tiny+CBAM (baseline) | **0.6685** | 6 | 0.888 | 0.219 |
| ConvNeXt-Tiny+CBAM (strong reg) | 0.6189 | 13 | 0.99 | 0.37 |

**结论**: 增强正则化未能改善ConvNeXt表现。更强dropout和weight_decay反而加剧了过拟合，label_smoothing=0.05的平滑效果有限。

### A2. Swin+Medium增强实验

| 配置项 | Swin baseline | Swin medium_aug |
|-------|:---:|:---:|
| train_augment_mode | light | **medium** |
| label_smoothing | 0.0 | **0.0** |
| dropout | 0.2 | 0.2 |
| weight_decay | 0.0001 | 0.0001 |

| 模型 | val_macro_f1 | 最佳Epoch | 状态 |
|------|:---:|:---:|:---:|
| Swin Transformer (light aug) | **0.5231** | 15 | ✅ 正常 |
| Swin Transformer (medium aug) | 0.357 | 9 | ⚠️ 收敛受阻 |

**关键发现**: 几何增强（旋转±10°、缩放±5%、平移±2.5%）破坏了Swin Transformer的预训练权重。预训练模型对强烈的几何变换极为敏感，轻量增强（brightness/contrast/noise）更适合此类模型。

### A3. ConvNeXt+TTA实验

| 配置项 | baseline | TTA(5x) |
|-------|---------|--------|
| tta_count | 1 | **5** |
| 其他参数 | 完全相同 | 完全相同 |

| 模型 | val_macro_f1 | 最佳Epoch | val_accuracy | 提升 |
|------|:---:|:---:|:---:|:---:|
| ConvNeXt+CBAM (baseline) | 0.6685 | 6 | 0.800 | — |
| ConvNeXt+CBAM + TTA(5x) | **0.6748** | 15 | 0.833 | **+0.63%** |

**结论**: TTA通过在推理时对验证图像进行5次随机增强并平均预测概率，有效提升了分类稳健性，val_macro_f1提升约0.6%。TTA不影响训练过程，仅改变推理方式，适合作为毕业设计中的模型增强手段。

### A4. 多任务学习实验

| 配置 | 任务数 | val_macro_f1 | 状态 |
|------|:---:|:---:|:---:|
| ConvNeXt (baseline) | 1 | **0.6685** | ✅ 最佳 |
| ConvNeXt + 25任务 | 25 | 0.400 | ❌ 完全坍缩 |
| ConvNeXt + 5任务(sc. stenosis) | 5 | 0.458 | ⚠️ 收敛受阻 |

**结论**: 多任务学习在此数据集上未能带来收益。25个任务同时优化导致梯度相互冲突，模型快速坍缩为预测多数类（Normal/Mild）。类别不平衡（~80% Normal/Mild）和小样本量是主要障碍。

### A5. 所有实验汇总

| 实验 | 模型 | val_macro_f1 | 结论 |
|------|------|:---:|------|
| 原始基线 | ConvNeXt-Tiny+CBAM | **0.6685** | 🏆 最佳 |
| 原始基线 | DenseNet-121+Dense Reuse | 0.6078 | 良好 |
| 原始基线 | Swin Transformer | 0.5231 | 中等 |
| ConvNeXt正则化 | ConvNeXt-Tiny+CBAM (strong reg) | 0.6189 | 未超越 |
| Swin Medium Aug | Swin Transformer | 0.357 | ❌ 预训练权重受损 |
| 原始基线 | ResNet-101 3D | 0.4942 | 较差 |
| 原始基线 | ViT-Base | 0.3415 | ❌ 难以收敛 |

---

## 附录B: 实验配置

所有5个模型使用统一的baseline配置：
```yaml
target_column: spinal_canal_stenosis_l4_l5
num_classes: 3
pretrained: true
in_channels: 1
dropout: 0.2
batch_size: 32
epochs: 20
optimizer: adamw
scheduler: cosine
amp: true
seed: 42
image_size: 224
early_stopping_patience: 7
```
