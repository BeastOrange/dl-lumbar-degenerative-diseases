"""Runtime health checks for macOS/Linux/Windows targets."""

from __future__ import annotations

import platform
import shutil
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class CheckResult:
    """Represents one health-check item result."""

    name: str
    passed: bool
    detail: str


def _has_command(name: str) -> bool:
    return shutil.which(name) is not None


def run_healthcheck(target: str) -> dict[str, Any]:
    """Run environment checks for the requested target runtime."""
    target_norm = target.lower()
    checks: list[CheckResult] = [
        CheckResult("python3", _has_command("python3"), "Python runtime is required."),
        CheckResult("uv", _has_command("uv"), "uv package manager is required."),
    ]

    if target_norm in {"linux", "windows"}:
        checks.append(
            CheckResult(
                "nvidia-smi",
                _has_command("nvidia-smi"),
                "GPU check; optional when CPU fallback is allowed.",
            )
        )

    system_name = platform.system().lower()
    target_map = {"mac": "darwin", "linux": "linux", "windows": "windows"}
    expected = target_map.get(target_norm)
    if expected is None:
        raise ValueError(f"Unsupported target: {target}")

    checks.append(
        CheckResult(
            "platform-match",
            system_name == expected,
            f"Current host is '{system_name}', target is '{expected}'.",
        )
    )

    passed = all(check.passed for check in checks if check.name != "nvidia-smi")
    return {
        "target": target_norm,
        "host": system_name,
        "passed": passed,
        "checks": [asdict(check) for check in checks],
    }
