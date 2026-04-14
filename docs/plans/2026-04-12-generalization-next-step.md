# Generalization Next Step Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在已确认“模型可学习、正式验证已有提升”的基础上，压制过拟合并把当前改进版 baseline 固化成可复现实验配置。

**Architecture:** 下一阶段不再修输入链路，而是转向泛化优化。核心思路是把正式训练配置和 overfit 调试配置分开管理，在训练器中加入 early stopping，并用保守的数据增强去提升少数类召回，同时保持现有 `evaluate` 和 `predictions.csv` 链路不变。

**Tech Stack:** Python 3.11, PyTorch, torchvision, pandas, pytest

---

### Task 1: 固化正式基线与调试配置

**Files:**
- Create: `configs/train/baseline_v2.yaml`
- Create: `configs/train/overfit_debug.yaml`
- Modify: `configs/train/default.yaml`

**Step 1: Write the failing/config validation tests or checks**

- 断言正式配置包含当前验证有效的设置：`pretrained=true`、`class_weight_mode=balanced`、`sampler_mode=balanced`
- 断言 overfit 配置显式启用 `overfit_subset_size`

**Step 2: Run config check to verify current gap**

Run: `python -c "...yaml safe_load..."`
Expected: 当前默认配置尚未拆分正式训练与调试训练职责。

**Step 3: Write minimal implementation**

- 新建正式基线配置，固定本轮有效超参
- 新建 overfit 调试配置，专门用于可学习性验证
- `default.yaml` 指向正式基线思路，不再承担调试职责

**Step 4: Run checks**

Run: `python -c "...yaml safe_load..."`
Expected: 配置拆分完成且字段正确。

### Task 2: 加入 early stopping，防止长轮次过拟合

**Files:**
- Modify: `src/dl_lumbar_dd/train/config.py`
- Modify: `src/dl_lumbar_dd/train/trainer.py`
- Modify: `src/dl_lumbar_dd/train/commands.py`
- Test: `tests/train/test_trainer.py`

**Step 1: Write the failing test**

- 断言在 `early_stopping_patience` 达到后训练提前结束
- 断言最佳 checkpoint 与 `best_epoch` 仍保持一致

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/train/test_trainer.py -q`
Expected: FAIL，因为当前训练总会跑满全部 epoch。

**Step 3: Write minimal implementation**

- 在训练配置中加入 `early_stopping_patience`
- 以 `val_macro_f1` 作为停止指标
- 停止时保持现有 artifact 输出兼容

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/train/test_trainer.py -q`
Expected: PASS

### Task 3: 加入保守的数据增强，优先提升泛化

**Files:**
- Modify: `src/dl_lumbar_dd/train/data.py`
- Modify: `src/dl_lumbar_dd/data/dicom.py`
- Test: `tests/train/test_data.py`

**Step 1: Write the failing test**

- 断言训练集路径可开启增强，验证集路径保持确定性
- 断言增强不改变张量 shape 和标签

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/train/test_data.py -q`
Expected: FAIL，因为当前没有 train/val 区分的增强路径。

**Step 3: Write minimal implementation**

- 只添加低风险增强：亮度/对比度扰动、轻微仿射、轻度噪声
- 避免会破坏未来左右标签语义的激进翻转

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/train/test_data.py -q`
Expected: PASS

### Task 4: 做对照实验并确认新的正式基线

**Files:**
- Reuse existing training/evaluation commands only

**Step 1: Run 3 controlled experiments**

- 正式基线：当前有效配置
- 正式基线 + early stopping
- 正式基线 + early stopping + augmentation

**Step 2: Evaluate all runs**

Run: `lumbar-cli evaluate ...`
Run: `lumbar-cli compare ...`
Expected: 至少一个方案在 `val_macro_f1` 或少数类 recall 上优于当前 `0.5383` 基线。

**Step 3: Record winner**

- 记录最佳 run
- 记录对应配置
- 决定是否把它升级为新的默认训练配置
