#!/usr/bin/env python3
"""Cross-platform environment checks for local, Linux, and Windows targets."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "rsna-2024-lumbar-spine-degenerative-classification"


def has_command(name: str) -> bool:
    """Return whether a command is available on PATH."""
    return shutil.which(name) is not None


def detect_cuda_hint() -> str:
    """Detect CUDA support from common command-line hints."""
    if has_command("nvidia-smi"):
        return "nvidia-smi"
    if has_command("nvcc"):
        return "nvcc"
    return "unavailable"


def gather_health(target: str) -> dict[str, object]:
    """Collect a simple, serializable health snapshot."""
    system = platform.system().lower()
    return {
        "target": target,
        "host_system": system,
        "commands": {
            "uv": has_command("uv"),
            "git": has_command("git"),
            "rsync": has_command("rsync") if target != "windows" else True,
            "ssh": has_command("ssh") if target != "windows" else True,
            "python3": has_command("python3") or has_command("python"),
            "nvidia-smi": has_command("nvidia-smi"),
        },
        "paths": {
            "project_root": str(PROJECT_ROOT),
            "dataset_present": DATASET_DIR.exists(),
            "artifacts_present": (PROJECT_ROOT / "artifacts").exists(),
            "streamlit_app_present": (PROJECT_ROOT / "apps" / "streamlit" / "Home.py").exists(),
        },
        "cuda_hint": detect_cuda_hint(),
    }


def evaluate_status(payload: dict[str, object]) -> bool:
    """Return pass/fail for required commands on the selected target."""
    commands = payload["commands"]
    paths = payload["paths"]
    required_commands = ["uv", "git", "python3"]
    if payload["target"] in {"mac", "linux"}:
        required_commands.extend(["ssh", "rsync"])
    commands_ok = all(bool(commands[name]) for name in required_commands)
    paths_ok = bool(paths["artifacts_present"] and paths["streamlit_app_present"])
    return commands_ok and paths_ok


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Project environment healthcheck")
    parser.add_argument("target", choices=["mac", "linux", "windows"], help="Deployment target")
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()

    payload = gather_health(args.target)
    payload["ok"] = evaluate_status(payload)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Target: {args.target}")
        print(f"Host system: {payload['host_system']}")
        print(f"CUDA hint: {payload['cuda_hint']}")
        for key, value in payload["commands"].items():
            print(f"command:{key}={value}")
        for key, value in payload["paths"].items():
            print(f"path:{key}={value}")
        print(f"ok={payload['ok']}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
