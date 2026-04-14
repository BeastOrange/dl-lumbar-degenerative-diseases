# Collapse Recovery Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修复当前 baseline 的多数类塌缩问题，并补齐可观测训练诊断与 tiny-overfit 调试闭环。

**Architecture:** 这次改动只落在模型输入适配、训练数据采样和训练指标输出三个边界内。输入链路负责让 pretrained backbone 吃到合理的 RGB/normalization，数据链路负责 balanced sampler 与 overfit 调试集，训练链路负责把每类表现写入现有指标产物。

**Tech Stack:** Python 3.11, PyTorch, torchvision, pandas, pytest

---

### Task 1: 为输入适配与训练诊断补失败测试

**Files:**
- Modify: `tests/train/test_trainer.py`
- Create: `tests/train/test_data.py`
- Create: `tests/models/test_registry.py`

**Step 1: Write the failing tests**

- 断言单通道输入会被复制成 3 通道，而不是经随机卷积投影
- 断言 `sampler_mode="balanced"` 时 train loader 使用加权采样
- 断言 tiny-overfit 模式会让 train/val 使用同一小样本集合
- 断言训练指标会写出每类 recall 和预测类别分布

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/models/test_registry.py tests/train/test_data.py tests/train/test_trainer.py -q`
Expected: FAIL，因为当前还没有这些行为。

### Task 2: 实现模型输入链路修复

**Files:**
- Modify: `src/dl_lumbar_dd/models/registry.py`

**Step 1: Write minimal implementation**

- 用稳定的灰度复制逻辑替代 `1x1 Conv` 单通道转三通道
- 在启用预训练时补 ImageNet normalization

**Step 2: Run targeted tests**

Run: `uv run pytest tests/models/test_registry.py -q`
Expected: PASS

### Task 3: 实现 balanced sampler 与 tiny-overfit 模式

**Files:**
- Modify: `src/dl_lumbar_dd/train/config.py`
- Modify: `src/dl_lumbar_dd/train/data.py`
- Modify: `src/dl_lumbar_dd/train/commands.py`
- Modify: `configs/train/default.yaml`

**Step 1: Write minimal implementation**

- 在配置中加入 `sampler_mode` 与 tiny-overfit 所需字段
- train dataloader 支持 `WeightedRandomSampler`
- tiny-overfit 模式下 train/val 绑定到同一训练子集

**Step 2: Run targeted tests**

Run: `uv run pytest tests/train/test_data.py -q`
Expected: PASS

### Task 4: 实现每类诊断指标

**Files:**
- Modify: `src/dl_lumbar_dd/train/metrics.py`
- Modify: `src/dl_lumbar_dd/train/trainer.py`

**Step 1: Write minimal implementation**

- 在训练和验证指标中加入每类 recall
- 加入预测类别分布指标
- 保持现有 `metrics.csv`、`history.json`、`predictions.csv` 输出兼容

**Step 2: Run targeted tests**

Run: `uv run pytest tests/train/test_trainer.py -q`
Expected: PASS

### Task 5: 做一次训练/评估 smoke 验证

**Files:**
- Reuse existing code path only

**Step 1: Run smoke**

Run: `PYTHONPATH=src uv run python scripts/...`
Expected: synthetic config 能完成训练，且 `metrics.csv` 包含新增指标。

**Step 2: Report**

- 汇总测试结果
- 汇总 smoke 结果
- 若通过，则同步到服务器并给出新的训练与 tiny-overfit 调试命令
