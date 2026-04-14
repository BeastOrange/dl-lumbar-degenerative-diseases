# Baseline Diagnosis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让当前单目标 baseline 同时具备类别不平衡修正能力和可诊断输出，并把默认配置提升到更像正式实验的强度。

**Architecture:** 在不改 CLI 外部接口的前提下，把 class weight 和验证预测落盘收口到训练模块内部，让评估模块直接复用已有 `predictions.csv` 约定。默认配置仅调整现有 YAML，不新增新的训练入口。

**Tech Stack:** Python 3.11, PyTorch, torchvision, pandas, tqdm, pytest

---

### Task 1: 为训练新行为补失败测试

**Files:**
- Modify: `tests/train/test_trainer.py`

**Step 1: Write the failing test**

- 断言训练完成后存在 `predictions.csv`
- 断言 `predictions.csv` 至少包含 `y_true`、`y_pred`、`score_0`、`score_1`、`score_2`
- 断言 `class_weight_mode="balanced"` 时训练器的 loss 带权重

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/train/test_trainer.py -q`
Expected: FAIL，因为当前不会生成 `predictions.csv`，也没有 class weight 配置。

### Task 2: 实现 class weight 与验证预测落盘

**Files:**
- Modify: `src/dl_lumbar_dd/train/config.py`
- Modify: `src/dl_lumbar_dd/train/data.py`
- Modify: `src/dl_lumbar_dd/train/trainer.py`
- Modify: `src/dl_lumbar_dd/train/commands.py`

**Step 1: 添加最小实现**

- 在 `TrainingConfig` 中加入 `class_weight_mode`
- 让训练集 dataset 暴露标签索引，供 trainer 构造 balanced 权重
- 在 trainer 中用带权 `CrossEntropyLoss`
- 在最佳验证 epoch 落 `predictions.csv`

**Step 2: Run test to verify it passes**

Run: `uv run pytest tests/train/test_trainer.py -q`
Expected: PASS

### Task 3: 调整默认训练配置

**Files:**
- Modify: `configs/train/default.yaml`

**Step 1: 更新配置**

- `pretrained: true`
- `epochs: 20`
- `class_weight_mode: balanced`

**Step 2: Verify config is readable**

Run: `PYTHONPATH=src uv run python - <<'PY' ...`
Expected: 能正常读出更新后的 YAML 字段

### Task 4: 验证评估链路可以消费预测输出

**Files:**
- Reuse existing code path: `src/dl_lumbar_dd/eval/commands.py`

**Step 1: Run a smoke training on synthetic data**

Run: `PYTHONPATH=src uv run python - <<'PY' ...`
Expected: 训练完成并产出 `predictions.csv`

**Step 2: Report verification**

- 列出训练测试结果
- 列出 smoke 输出中的文件存在性
- 若无法完整跑 CLI 评估，则明确说明原因
