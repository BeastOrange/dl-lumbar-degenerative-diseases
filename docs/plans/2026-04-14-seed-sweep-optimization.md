# Seed Sweep Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在当前已恢复的 `sampler_only` 基线上，先用真正生效的随机种子搜索把 `val_macro_f1` 拉回 `0.69+`，若失败再做单变量的不平衡优化。

**Architecture:** 冻结当前稳定训练链路，不再同时改 backbone、split、sampler、loss。第一阶段只做 seed sweep，因为 seed 修复后这一步的性价比最高；第二阶段仅在 seed sweep 仍未达标时，增加一种更克制的损失函数实验，保持 `sampler_mode=balanced` 且 `class_weight_mode=null` 不变。

**Tech Stack:** Python 3.11, PyTorch 2.11/cu130, torchvision, timm, YAML configs, rsync, Ubuntu GPU server

---

### Task 1: 冻结基线并准备 seed sweep

**Files:**
- Modify: `/Users/orange/Desktop/OTHER/graduation_project/dl-lumbar-degenerative-diseases/configs/train/default.yaml`
- Create: `/Users/orange/Desktop/OTHER/graduation_project/dl-lumbar-degenerative-diseases/configs/train/experiments/default_seed_17.yaml`
- Create: `/Users/orange/Desktop/OTHER/graduation_project/dl-lumbar-degenerative-diseases/configs/train/experiments/default_seed_27.yaml`
- Create: `/Users/orange/Desktop/OTHER/graduation_project/dl-lumbar-degenerative-diseases/configs/train/experiments/default_seed_42.yaml`
- Create: `/Users/orange/Desktop/OTHER/graduation_project/dl-lumbar-degenerative-diseases/configs/train/experiments/default_seed_77.yaml`
- Create: `/Users/orange/Desktop/OTHER/graduation_project/dl-lumbar-degenerative-diseases/configs/train/experiments/default_seed_97.yaml`

**Step 1: 复制当前默认配置**

保留以下核心项不变：
- `class_weight_mode: null`
- `sampler_mode: balanced`
- `pretrained: true`
- `target_column: spinal_canal_stenosis_l4_l5`
- `seed`: 分别改为 `17 / 27 / 42 / 77 / 97`

**Step 2: 校验配置可读**

Run: `python3 -c "import yaml; [yaml.safe_load(open(p, encoding='utf-8')) for p in ['configs/train/default.yaml','configs/train/experiments/default_seed_17.yaml','configs/train/experiments/default_seed_27.yaml','configs/train/experiments/default_seed_42.yaml','configs/train/experiments/default_seed_77.yaml','configs/train/experiments/default_seed_97.yaml']]"`  
Expected: 无异常退出

**Step 3: 提交本地配置变更**

```bash
git add configs/train/default.yaml configs/train/experiments/
git commit -m "chore(train): 准备种子搜索配置"
```

### Task 2: 在新服务器执行 seed sweep

**Files:**
- Read: `/root/autodl-tmp/dl-lumbar-degenerative-diseases/configs/train/experiments/default_seed_17.yaml`
- Read: `/root/autodl-tmp/dl-lumbar-degenerative-diseases/configs/train/experiments/default_seed_27.yaml`
- Read: `/root/autodl-tmp/dl-lumbar-degenerative-diseases/configs/train/experiments/default_seed_42.yaml`
- Read: `/root/autodl-tmp/dl-lumbar-degenerative-diseases/configs/train/experiments/default_seed_77.yaml`
- Read: `/root/autodl-tmp/dl-lumbar-degenerative-diseases/configs/train/experiments/default_seed_97.yaml`
- Create: `/root/autodl-tmp/dl-lumbar-degenerative-diseases/artifacts/logs/train_seed_*.log`

**Step 1: 串行跑 5 个 seed**

Run:
- `bash scripts/run_training_foreground.sh configs/train/experiments/default_seed_17.yaml`
- `bash scripts/run_training_foreground.sh configs/train/experiments/default_seed_27.yaml`
- `bash scripts/run_training_foreground.sh configs/train/experiments/default_seed_42.yaml`
- `bash scripts/run_training_foreground.sh configs/train/experiments/default_seed_77.yaml`
- `bash scripts/run_training_foreground.sh configs/train/experiments/default_seed_97.yaml`

Expected: 每轮都生成 `run_summary.json`

**Step 2: 对每个 run 做评估**

Run: `./.venv/bin/lumbar-cli evaluate --run-dir artifacts/runs/<run_id> --output-root reports/figures/eval`  
Expected: 每个 run 都生成 `training_history.png`、`confusion_matrix.png`、`roc_ovr.png`

**Step 3: 汇总 best 指标**

记录每个 seed 的：
- `best val_macro_f1`
- `val_accuracy`
- `val_recall_class_1`
- `val_recall_class_2`
- `pred_rate_0 / 1 / 2`

**Step 4: 提交实验记录**

```bash
git add docs/plans/2026-04-14-seed-sweep-optimization.md
git commit -m "docs(train): 记录种子搜索计划"
```

### Task 3: 提升默认配置到最佳 seed

**Files:**
- Modify: `/Users/orange/Desktop/OTHER/graduation_project/dl-lumbar-degenerative-diseases/configs/train/default.yaml`

**Step 1: 选择最佳 seed**

规则：
- 第一优先：`val_macro_f1`
- 第二优先：`val_recall_class_1 + val_recall_class_2`
- 第三优先：`pred_rate_0` 不要继续恶化

**Step 2: 回写默认配置**

把 `default.yaml` 的 `seed` 改成最佳 seed。

**Step 3: 重新跑一次默认训练做确认**

Run: `bash scripts/run_training_foreground.sh configs/train/default.yaml`  
Expected: 确认最佳 seed 可复现，不是偶然单次峰值

**Step 4: 提交基线更新**

```bash
git add configs/train/default.yaml
git commit -m "fix(train): 固化最佳随机种子"
```

### Task 4: 若 seed sweep 仍未达到 0.69，则做单变量 loss 实验

**Files:**
- Modify: `/Users/orange/Desktop/OTHER/graduation_project/dl-lumbar-degenerative-diseases/src/dl_lumbar_dd/train/config.py`
- Modify: `/Users/orange/Desktop/OTHER/graduation_project/dl-lumbar-degenerative-diseases/src/dl_lumbar_dd/train/commands.py`
- Modify: `/Users/orange/Desktop/OTHER/graduation_project/dl-lumbar-degenerative-diseases/src/dl_lumbar_dd/train/trainer.py`
- Create: `/Users/orange/Desktop/OTHER/graduation_project/dl-lumbar-degenerative-diseases/tests/train/test_loss_config.py`
- Create: `/Users/orange/Desktop/OTHER/graduation_project/dl-lumbar-degenerative-diseases/configs/train/experiments/default_focal.yaml`

**Step 1: 写失败测试**

测试目标：
- `TrainingConfig` 支持 `loss_name`
- `Trainer` 支持 `cross_entropy` 与 `focal`
- `focal` 不依赖 `class_weight_mode=balanced`

**Step 2: 最小实现 focal loss**

约束：
- 默认仍是 `cross_entropy`
- 保持 `sampler_mode=balanced`
- 不重新打开 `class_weight_mode`

**Step 3: 跑定向测试**

Run: `uv run pytest tests/train/test_loss_config.py tests/train/test_trainer.py -q`  
Expected: PASS

**Step 4: 只跑一个 focal 配置**

Run: `bash scripts/run_training_foreground.sh configs/train/experiments/default_focal.yaml`  
Expected: 仅比较一个变量，避免重新进入多变量混战

### Task 5: 若 focal 仍未达标，则转入数据质量诊断

**Files:**
- Create: `/Users/orange/Desktop/OTHER/graduation_project/dl-lumbar-degenerative-diseases/scripts/analyze_zero_view_samples.py`
- Create: `/Users/orange/Desktop/OTHER/graduation_project/dl-lumbar-degenerative-diseases/tests/data/test_zero_view_analysis.py`

**Step 1: 统计空视图 / 缺序列样本**

目标：
- 按 `study_id`
- 按 `target_label`
- 按 split

**Step 2: 判断是否存在类条件数据污染**

若 `class 1/2` 的坏样本比例显著更高，再决定是否过滤或降权。

**Step 3: 停止继续盲试配置**

只有确认数据质量问题后，才进入下一轮代码修改。
