from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "healthcheck.py"
SPEC = importlib.util.spec_from_file_location("healthcheck_script", SCRIPT_PATH)
healthcheck = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(healthcheck)


def test_evaluate_status_requires_core_commands(monkeypatch):
    payload = {
        "target": "linux",
        "commands": {
            "uv": True,
            "git": True,
            "rsync": True,
            "ssh": True,
            "python3": True,
            "nvidia-smi": False,
        },
        "paths": {
            "project_root": ".",
            "dataset_present": False,
            "artifacts_present": True,
            "streamlit_app_present": True,
        },
    }
    assert healthcheck.evaluate_status(payload) is True
    payload["commands"]["rsync"] = False
    assert healthcheck.evaluate_status(payload) is False


def test_gather_health_reports_streamlit_presence(monkeypatch):
    monkeypatch.setattr(healthcheck, "has_command", lambda name: name in {"uv", "git", "python3"})
    monkeypatch.setattr(healthcheck, "detect_cuda_hint", lambda: "unavailable")
    payload = healthcheck.gather_health("windows")
    assert payload["target"] == "windows"
    assert payload["commands"]["uv"] is True
    assert payload["commands"]["git"] is True
    assert payload["paths"]["streamlit_app_present"] is True
