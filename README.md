# 基于深度学习的腰椎退行性病变影像分类系统

## 项目简介

基于 RSNA 2024 数据集，使用 5 种深度学习模型对腰椎 MRI 影像进行三分类（Normal/Mild、Moderate、Severe）。最佳模型 ConvNeXt-Tiny+CBAM 达到 F1=0.720，Severe 召回率 86.3%。

## 目录结构

```
├── src/dl_lumbar_dd/       # 核心源代码
│   ├── cli.py              #   命令行入口
│   ├── data/               #   数据加载、DICOM处理、预处理
│   ├── models/             #   5种模型架构 + 自定义模块
│   ├── train/              #   训练循环、数据集、指标计算
│   ├── eval/               #   评估、模型排名
│   └── visualization/      #   混淆矩阵、ROC曲线等绘图
├── configs/train/           # 训练配置（YAML）
│   ├── default.yaml        #   默认配置
│   └── thesis/             #   论文实验配置（70+组）
├── artifacts/               # 训练产物
│   ├── runs/               #   每次训练的指标、预测、历史记录
│   └── processed/          #   预处理后的数据划分文件
├── apps/streamlit/          # Streamlit 检测平台（6页面）
├── reports/                 # 论文报告 + 可视化图表
├── scripts/                 # 服务器同步、训练、模型下载脚本
└── tests/                   # 单元测试
```

## 环境搭建

### 前置要求

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)（Python 包管理器）

### 安装步骤

#### macOS / Linux

```bash
# 1. 安装 uv（如果未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 安装 Python 3.11
uv python install 3.11

# 3. 安装项目依赖
# macOS 下推荐使用非 editable 安装，避免 .pth 被系统标记为 hidden 后无法注入 src 路径
uv sync --extra dev --no-editable

# 4. 验证安装
uv run lumbar-cli healthcheck --target mac    # macOS
uv run lumbar-cli healthcheck --target linux  # Linux
```

#### Windows PowerShell

```powershell
# 1. 安装 uv（如果未安装）
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. 安装 Python 3.11
uv python install 3.11

# 3. 安装项目依赖
# 无 NVIDIA GPU 时使用 CPU 版本
uv sync --extra dev --extra windows-cpu

# 如果客户电脑装有 NVIDIA 驱动并希望启用 GPU，可改用：
# uv sync --extra dev --extra windows-gpu

# 4. 验证安装
uv run lumbar-cli healthcheck --target windows
```

## 常用命令

### 数据处理

```bash
# 数据探索分析（生成类别分布、缺失值等图表）
uv run lumbar-cli eda --dataset-root ./rsna-2024-lumbar-spine-degenerative-classification

# 数据预处理（生成训练/验证划分文件）
uv run lumbar-cli preprocess --dataset-root ./rsna-2024-lumbar-spine-degenerative-classification
```

Windows PowerShell:

```powershell
uv run lumbar-cli eda --dataset-root ".\rsna-2024-lumbar-spine-degenerative-classification"
uv run lumbar-cli preprocess --dataset-root ".\rsna-2024-lumbar-spine-degenerative-classification"
```

### 模型训练

```bash
# 使用指定配置训练模型（结果保存在 artifacts/runs/<run-id>/）
uv run lumbar-cli train --config configs/train/thesis/targeted_best_combo.yaml

# 3-fold 交叉验证集成训练
uv run lumbar-cli train --cv --config configs/train/thesis/phase_d_cv_ensemble.yaml
```

Windows PowerShell:

```powershell
uv run lumbar-cli train --config .\configs\train\thesis\targeted_best_combo.yaml
uv run lumbar-cli train --cv --config .\configs\train\thesis\phase_d_cv_ensemble.yaml
```

### 模型评估

```bash
# 评估单次运行（生成混淆矩阵、ROC曲线、训练历史图）
uv run lumbar-cli evaluate --run-dir artifacts/runs/<run-id>

# 对比所有运行并生成排名
uv run lumbar-cli compare --runs-root artifacts/runs --primary-metric val_macro_f1
```

Windows PowerShell:

```powershell
uv run lumbar-cli evaluate --run-dir .\artifacts\runs\<run-id>
uv run lumbar-cli compare --runs-root .\artifacts\runs --primary-metric val_macro_f1
```

### 检测平台

```bash
# 启动 Streamlit 平台（浏览器自动打开）
uv run streamlit run apps/streamlit_app.py
```

当前推荐入口为中文极简一键分析页。页面主路径为“点击开始分析 -> 系统自动扫描项目根目录中的数据集 -> 识别病例目录 -> 批量分析 -> 输出结果”。

Windows PowerShell:

```powershell
uv run streamlit run .\apps\streamlit_app.py
```

页面使用说明：

- 无需上传文件，数据来源固定为项目根目录下的 `rsna-2024-lumbar-spine-degenerative-classification/`
- 点击“开始分析”后，系统会自动扫描 `train_images/<study_id>/` 目录结构
- 系统会自动判断哪些目录属于单个病例并逐个分析
- 页面最终输出批量分析结果、状态统计和当前模型信息

### 测试

```bash
uv run pytest tests/ -q
```

Windows PowerShell:

```powershell
uv run pytest .\tests\ -q
```

## 常见问题

### `ModuleNotFoundError: No module named 'dl_lumbar_dd'`

如果执行 `uv run lumbar-cli ...` 时出现这个错误，通常是因为 macOS 上的 editable 安装生成了 `.pth` 文件，但该文件被系统标记为 `hidden`，Python 启动时会跳过它，导致 `src/dl_lumbar_dd` 没有加入 `sys.path`。

直接重新同步为非 editable 安装即可：

```bash
uv sync --extra dev --no-editable
```

如果你已经执行过一次 `uv sync --extra dev`，直接再运行上面的命令覆盖安装即可，然后重新执行：

```bash
uv run lumbar-cli healthcheck --target mac
uv run lumbar-cli eda --dataset-root ./rsna-2024-lumbar-spine-degenerative-classification
```

## 最佳模型

位于 `artifacts/runs/convnext_tiny_cbam-20260419-103601/`：

| 文件 | 说明 |
|------|------|
| `best.ckpt` | 模型权重（328MB） |
| `config.yaml` | 训练配置 |
| `metrics.csv` | 逐 epoch 指标 |
| `predictions.csv` | 验证集预测结果 |
| `history_train_metrics.png` | 训练曲线图 |

## 数据集

将 RSNA 2024 数据集放在项目根目录下：

```
rsna-2024-lumbar-spine-degenerative-classification/
├── train.csv
├── train_series_descriptions.csv
├── train_label_coordinates.csv
└── train_images/
```
