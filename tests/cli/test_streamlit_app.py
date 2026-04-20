from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from dl_lumbar_dd.inference.service import BatchInferenceResult


class _FakeStatus:
    def update(self, **_: object) -> None:
        return None


class _FakeStreamlit(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("streamlit")

    def cache_resource(self, **_: object):
        def decorator(func):
            return func

        return decorator

    def set_page_config(self, **_: object) -> None:
        return None

    def title(self, *_: object, **__: object) -> None:
        return None

    def caption(self, *_: object, **__: object) -> None:
        return None

    def button(self, *_: object, **__: object) -> bool:
        return False

    def status(self, *_: object, **__: object) -> _FakeStatus:
        return _FakeStatus()

    def success(self, *_: object, **__: object) -> None:
        return None

    def columns(self, count: int):
        return [self for _ in range(count)]

    def metric(self, *_: object, **__: object) -> None:
        return None

    def subheader(self, *_: object, **__: object) -> None:
        return None

    def dataframe(self, *_: object, **__: object) -> None:
        return None

    def error(self, *_: object, **__: object) -> None:
        return None

    def info(self, *_: object, **__: object) -> None:
        return None


def _load_streamlit_app_module():
    module_name = "tests_streamlit_app_module"
    app_path = Path(__file__).resolve().parents[2] / "apps" / "streamlit_app.py"
    fake_streamlit = _FakeStreamlit()
    previous_streamlit = sys.modules.get("streamlit")
    spec = importlib.util.spec_from_file_location(module_name, app_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["streamlit"] = fake_streamlit
        spec.loader.exec_module(module)
    finally:
        if previous_streamlit is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = previous_streamlit
    return module


def test_discover_case_directories_reads_train_images_layout(tmp_path: Path) -> None:
    streamlit_app = _load_streamlit_app_module()
    dataset_root = tmp_path / "dataset"
    valid_case = dataset_root / "train_images" / "1001" / "2001"
    valid_case.mkdir(parents=True)
    (valid_case / "1.dcm").write_bytes(b"fake")
    (dataset_root / "train_images" / "README.txt").write_text("ignore", encoding="utf-8")
    (dataset_root / "train_images" / "1002").mkdir(parents=True)

    case_dirs = streamlit_app._discover_case_directories(dataset_root)

    assert [case_dir.name for case_dir in case_dirs] == ["1001"]


def test_validate_dataset_root_requires_series_table(tmp_path: Path) -> None:
    streamlit_app = _load_streamlit_app_module()
    dataset_root = tmp_path / "dataset"
    case_dir = dataset_root / "train_images" / "1001" / "2001"
    case_dir.mkdir(parents=True)
    (case_dir / "1.dcm").write_bytes(b"fake")

    try:
        streamlit_app._validate_dataset_root(dataset_root)
    except FileNotFoundError as exc:
        assert "train_series_descriptions.csv" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("expected FileNotFoundError when series table is missing")


def test_normalize_results_accepts_slots_dataclass_items() -> None:
    streamlit_app = _load_streamlit_app_module()
    payload = [
        BatchInferenceResult(
            study_id=1000,
            run_dir=Path("artifacts/runs/demo"),
            target_name="L4/L5 椎管狭窄",
            status="成功",
            predicted_index=1,
            predicted_label="中度",
            probabilities={
                "轻度/正常": 0.1,
                "中度": 0.8,
                "重度": 0.1,
            },
            error_message=None,
        )
    ]

    result_table = streamlit_app._normalize_results(payload)

    assert result_table.to_dict(orient="records") == [
        {
            "study_id": "1000",
            "预测结果": "中度",
            "轻度/正常概率": "10.00%",
            "中度概率": "80.00%",
            "重度概率": "10.00%",
            "状态": "成功",
        }
    ]
