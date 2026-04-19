# Streamlit 极简推理页 Design

**目标**

把当前研究型多页面 Streamlit 看板收缩成一个中文、最小可交付的推理入口，只保留“一组 DCM 上传 -> 模型推理 -> 输出结果”。

**范围**

- 新增一个单文件 Streamlit 入口，作为后续推荐运行方式
- 保留现有研究看板代码，但不再作为主入口
- 新增最小 inference service，复用当前训练产物 `artifacts/runs/*/best.ckpt`
- 输入限定为“一组属于同一病例的 DCM 文件”
- 不新增训练、评估、可解释性、图表、工作区状态等页面内容

**方案选项**

1. 直接把现有 `apps/streamlit/Home.py` 改成单页  
   优点：表面改动少。  
   缺点：`pages/` 目录仍会被 Streamlit 自动识别，界面不会真正变成单页。

2. 新增独立单文件入口 `apps/streamlit_app.py`，旧研究页保留但不再推荐  
   优点：能真正做到极简单页，不需要删除旧文件，风险最低。  
   缺点：README 与启动命令需要同步更新。

3. 删除旧多页面结构，彻底替换  
   优点：仓库表面最干净。  
   缺点：触及文件删除，不适合当前边界。

**推荐**

采用方案 2。

**设计**

- 新入口：`apps/streamlit_app.py`
  - 中文标题与最小说明
  - `st.file_uploader(..., accept_multiple_files=True, type=["dcm"])`
  - 一个“开始推理”按钮
  - 成功时仅展示预测等级、目标任务、各类别概率
  - 失败时给出中文错误说明

- 新增 `src/dl_lumbar_dd/inference/service.py`
  - 负责查找最新可用 checkpoint
  - 读取对应 `config.yaml`
  - 构建模型并加载参数
  - 将上传的 DCM 按 `SeriesDescription` 归类为三视图
  - 输出单次推理结果

- 在 `src/dl_lumbar_dd/data/dicom.py` 增加上传 DCM 组装逻辑
  - 从内存字节读取 DICOM
  - 使用 `SeriesDescription` + 关键词规则归类到：
    - `Sagittal T1`
    - `Sagittal T2/STIR`
    - `Axial T2`
  - 每个序列按 `InstanceNumber` 排序并取中间切片
  - 继续复用现有归一化逻辑

- 模型加载兼容
  - 为避免推理时再次下载 torchvision 预训练权重，模型构建需要支持：
    - 保留训练时的 grayscale->RGB 与 ImageNet normalization 行为
    - 但不额外加载 backbone 默认权重

**风险**

- 当前模型是 study-level 三视图分类器，不是任意单张 DCM 分类器；因此上传文件必须尽量来自同一病例并包含三类序列。
- 如果上传文件缺失某个关键序列，应明确报错，而不是静默补零。
- 真实医院数据的 `SeriesDescription` 可能和 RSNA 命名不完全一致，因此归类规则需要做关键词兜底。

**验证**

- 单元测试覆盖：
  - 上传 DCM 归类与三视图张量构建
  - inference service 的 checkpoint 选择与结果解码
  - 关闭 backbone 默认权重加载时仍保留预训练输入适配
- 手工验证：
  - `uv run streamlit run apps/streamlit_app.py`
  - 至少确认应用可启动、可加载模型、可对一组 DCM 返回中文结果
