# Streamlit Minimal Inference Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 把当前 Streamlit 改成中文极简单页，只保留一组 DCM 上传和模型推理结果展示。

**Architecture:** 新增单文件 `apps/streamlit_app.py` 作为唯一推荐入口；底层新增最小 inference service，从最新可用 `best.ckpt` 加载模型，并把上传 DCM 自动组装成当前模型所需的三视图输入。旧研究页保留但不再作为主入口。

**Tech Stack:** Python 3.11, Streamlit, PyTorch, pydicom, OpenCV, pytest

---

### Task 1: 为 DCM 上传推理链写失败测试

**Files:**
- Create: `tests/inference/test_service.py`
- Modify: `tests/models/test_registry.py`

**Step 1: Write the failing test**

- 断言上传的一组 DCM 能按三类序列组装成 `(3, H, W)` 张量
- 断言 inference service 会选择最新带 `best.ckpt` 的 run
- 断言推理结果会解码成中文等级与概率
- 断言模型可在“保留预训练输入适配但不加载 backbone 默认权重”的模式下构建

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/inference/test_service.py tests/models/test_registry.py -q`
Expected: FAIL，因为当前没有 inference service，也不支持上述模型构建模式。

### Task 2: 实现最小 inference service

**Files:**
- Create: `src/dl_lumbar_dd/inference/__init__.py`
- Create: `src/dl_lumbar_dd/inference/service.py`
- Modify: `src/dl_lumbar_dd/data/dicom.py`
- Modify: `src/dl_lumbar_dd/models/registry.py`

**Step 1: 写最小实现**

- 在 `dicom.py` 增加上传 DCM 分组、排序、取中间切片、归一化逻辑
- 在 `service.py` 中实现：
  - 最新 checkpoint run 解析
  - 模型创建与权重加载
  - 上传文件推理
  - 结果解码
- 在模型注册处支持“保留预训练输入管线但不额外下载 backbone 权重”

**Step 2: Run test to verify it passes**

Run: `uv run pytest tests/inference/test_service.py tests/models/test_registry.py -q`
Expected: PASS

### Task 3: 实现中文极简单页 Streamlit

**Files:**
- Create: `apps/streamlit_app.py`

**Step 1: 新增单页入口**

- 中文标题
- 多文件 DCM 上传
- “开始推理”按钮
- 最小结果展示
- 中文异常提示

**Step 2: Smoke check**

Run: `uv run streamlit run apps/streamlit_app.py --server.headless true`
Expected: 应用能启动，不因导入或模型服务初始化而崩溃

### Task 4: 更新仓库说明并验证主入口

**Files:**
- Modify: `README.md`

**Step 1: 更新推荐启动命令**

- 把 Streamlit 运行入口改为 `apps/streamlit_app.py`
- 说明输入为“同一病例的一组 DCM 文件”

**Step 2: Final verification**

Run: `uv run pytest tests/inference/test_service.py tests/models/test_registry.py -q`
Run: `uv run python - <<'PY' ...`（定向调用 inference service）
Expected: 测试通过，服务可以对 mock DCM 组返回预测结果
