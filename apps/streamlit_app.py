from __future__ import annotations

from dataclasses import asdict, is_dataclass
import inspect
from collections.abc import Iterable, Mapping
from numbers import Real
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from dl_lumbar_dd.inference import StudyInferenceService

RESULT_COLUMNS = ["study_id", "预测结果", "轻度/正常概率", "中度概率", "重度概率", "状态"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "rsna-2024-lumbar-spine-degenerative-classification"


@st.cache_resource(show_spinner=False)
def load_service() -> StudyInferenceService:
    return StudyInferenceService.from_latest_run()


def _discover_case_directories(dataset_root: Path) -> list[Path]:
    train_images_root = dataset_root / "train_images"
    if not train_images_root.exists():
        raise FileNotFoundError(f"未找到病例影像目录：{train_images_root}")

    case_dirs: list[Path] = []
    for study_dir in sorted(train_images_root.iterdir()):
        if not study_dir.is_dir():
            continue
        if not any(child.is_dir() for child in study_dir.iterdir()):
            continue
        case_dirs.append(study_dir)
    return case_dirs


def _validate_dataset_root(dataset_root: Path) -> list[Path]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"未找到数据集目录：{dataset_root}")
    if not (dataset_root / "train_series_descriptions.csv").exists():
        raise FileNotFoundError(f"未找到病例序列表：{dataset_root / 'train_series_descriptions.csv'}")
    case_dirs = _discover_case_directories(dataset_root)
    if not case_dirs:
        raise ValueError("未在数据集中识别到任何病例目录，请检查 train_images 下的数据结构。")
    return case_dirs


def _run_batch_analysis(service: StudyInferenceService) -> Any:
    method_names = (
        "predict_dataset",
        "analyze_project_dataset",
        "analyze_all_studies",
        "predict_project_dataset",
        "predict_all_studies",
        "predict_batch_from_project_dataset",
        "predict_batch",
        "batch_predict",
    )
    for method_name in method_names:
        method = getattr(service, method_name, None)
        if callable(method):
            return _invoke_batch_method(method)
    raise RuntimeError("当前模型服务尚未提供批量分析接口，请先对齐批量分析服务实现。")


def _invoke_batch_method(method: Any) -> Any:
    signature = inspect.signature(method)
    kwargs: dict[str, Any] = {}
    for name in signature.parameters:
        if name == "dataset_root":
            kwargs[name] = DEFAULT_DATASET_ROOT
        elif name == "project_root":
            kwargs[name] = PROJECT_ROOT
    return method(**kwargs)


def _normalize_results(payload: Any) -> pd.DataFrame:
    rows = _extract_rows(payload)
    if not rows:
        raise RuntimeError("批量分析服务返回为空，无法展示结果。")
    normalized = [_normalize_row(row) for row in rows]
    return pd.DataFrame(normalized, columns=RESULT_COLUMNS)


def _extract_rows(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, pd.DataFrame):
        return [record for record in payload.to_dict(orient="records") if isinstance(record, Mapping)]

    if isinstance(payload, Mapping):
        for key in ("results", "rows", "records", "items", "predictions"):
            candidate = payload.get(key)
            if candidate is None:
                continue
            try:
                rows = _extract_rows(candidate)
            except RuntimeError:
                continue
            else:
                if rows:
                    return rows
        return [_to_mapping(payload)]

    if isinstance(payload, Iterable) and not isinstance(payload, (str, bytes)):
        return [_to_mapping(item) for item in payload]

    raise RuntimeError("批量分析服务返回格式不受支持。")


def _to_mapping(item: Any) -> Mapping[str, Any]:
    if isinstance(item, Mapping):
        return item
    if is_dataclass(item):
        return asdict(item)
    if hasattr(item, "to_dict") and callable(item.to_dict):
        converted = item.to_dict()
        if isinstance(converted, Mapping):
            return converted
    if hasattr(item, "__dict__"):
        return vars(item)
    raise RuntimeError("批量分析结果项无法解析为字典。")


def _normalize_row(row: Mapping[str, Any]) -> dict[str, str]:
    probabilities = row.get("probabilities")
    if not isinstance(probabilities, Mapping):
        probabilities = {}

    prediction = _pick_first(row, "预测结果", "predicted_label", "prediction", "predicted_result", "label")
    status = _pick_first(row, "状态", "status", default="成功" if prediction else "未知")
    return {
        "study_id": str(_pick_first(row, "study_id", "studyId", "id", default="-")),
        "预测结果": str(prediction or "-"),
        "轻度/正常概率": _format_probability(
            _pick_probability(
                row,
                probabilities,
                "轻度/正常概率",
                "normal_mild_probability",
                "prob_normal_mild",
                "轻度/正常",
                "Normal/Mild",
                "normal_mild",
            )
        ),
        "中度概率": _format_probability(
            _pick_probability(row, probabilities, "中度概率", "moderate_probability", "prob_moderate", "中度", "Moderate", "moderate")
        ),
        "重度概率": _format_probability(
            _pick_probability(row, probabilities, "重度概率", "severe_probability", "prob_severe", "重度", "Severe", "severe")
        ),
        "状态": str(status),
    }


def _pick_first(row: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return default


def _pick_probability(row: Mapping[str, Any], probabilities: Mapping[str, Any], *keys: str) -> Any:
    value = _pick_first(row, *keys)
    if value is not None:
        return value
    return _pick_first(probabilities, *keys)


def _format_probability(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return "-"
        if stripped.endswith("%"):
            return stripped
        try:
            value = float(stripped)
        except ValueError:
            return stripped
    if isinstance(value, Real):
        number = float(value)
        if 0.0 <= number <= 1.0:
            return f"{number:.2%}"
        return f"{number:.2f}"
    return str(value)


def _is_success(status: str) -> bool:
    normalized = status.strip().lower()
    return normalized in {"成功", "完成", "ok", "success"}


def _is_skipped(status: str) -> bool:
    normalized = status.strip().lower()
    return normalized in {"已跳过", "跳过", "skipped", "skip"}


st.set_page_config(page_title="腰椎退变数据集一键分析", page_icon="🩻", layout="centered")
st.title("腰椎退变数据集一键分析")
st.caption("点击开始分析后，系统会自动扫描项目根目录中的数据集，识别病例目录并完成批量分析。")
st.caption(f"数据集目录：{DEFAULT_DATASET_ROOT}")

if st.button("开始分析", type="primary", use_container_width=True):
    status_panel = st.status("准备分析任务...", expanded=False)
    try:
        status_panel.update(label="扫描数据集目录中...", state="running")
        case_dirs = _validate_dataset_root(DEFAULT_DATASET_ROOT)

        status_panel.update(label="加载模型服务中...", state="running")
        service = load_service()

        status_panel.update(label=f"已识别 {len(case_dirs)} 个病例目录，正在执行批量分析...", state="running")
        result_table = _normalize_results(_run_batch_analysis(service))

        status_panel.update(label=f"分析完成，共输出 {len(result_table)} 条结果。", state="complete")
        success_count = sum(_is_success(str(item)) for item in result_table["状态"])
        skipped_count = sum(_is_skipped(str(item)) for item in result_table["状态"])
        failure_count = len(result_table) - success_count - skipped_count

        st.success("全量分析完成")
        col_case, col_total, col_success, col_failed = st.columns(4)
        col_case.metric("扫描到的病例目录", len(case_dirs))
        col_total.metric("输出结果数", len(result_table))
        col_success.metric("成功", success_count)
        col_failed.metric("跳过/失败", skipped_count + failure_count)
        st.caption(f"当前模型：{service.run_dir.name}")

        st.subheader("分析结果")
        st.dataframe(result_table, use_container_width=True, hide_index=True)
    except Exception as exc:  # pragma: no cover - Streamlit UI branch
        status_panel.update(label="分析失败", state="error")
        st.error(str(exc))
else:
    st.info("点击“开始分析”后，系统会自动扫描根目录数据集并输出结果。")
