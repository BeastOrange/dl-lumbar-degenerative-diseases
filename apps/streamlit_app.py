from __future__ import annotations

import pandas as pd
import streamlit as st

from dl_lumbar_dd.data.dicom import DicomUpload
from dl_lumbar_dd.inference import StudyInferenceService


@st.cache_resource(show_spinner=False)
def load_service() -> StudyInferenceService:
    return StudyInferenceService.from_latest_run()


st.set_page_config(page_title="腰椎退变影像推理", page_icon="🩻", layout="centered")
st.title("腰椎退变影像推理")
st.caption("请上传同一个病例文件夹里的全部 DCM 文件，系统会自动完成推理。")

st.warning(
    "上传说明：请选择同一个病例文件夹中的全部 DCM 文件一起上传，不要混合不同病例，也不要手动随便挑几张。"
)

uploads = st.file_uploader(
    "上传 DCM 文件",
    type=["dcm"],
    accept_multiple_files=True,
    help="请选择同一个病例文件夹中的全部 DCM 文件，不要混传不同病例。",
)

run_disabled = not uploads
if st.button("开始推理", type="primary", disabled=run_disabled):
    try:
        with st.spinner("正在加载模型并执行推理..."):
            service = load_service()
            result = service.predict([DicomUpload(name=file.name, content=file.getvalue()) for file in uploads])
        st.success(f"推理结果：{result.target_name} -> {result.predicted_label}")
        probability_table = pd.DataFrame(
            {
                "等级": list(result.probabilities.keys()),
                "概率": [f"{value:.2%}" for value in result.probabilities.values()],
            }
        )
        st.table(probability_table)
        st.caption(f"当前模型：{result.run_dir.name}")
    except Exception as exc:  # pragma: no cover - Streamlit UI branch
        st.error(str(exc))
elif not uploads:
    st.info("请先选择同一个病例文件夹中的全部 DCM 文件。")
