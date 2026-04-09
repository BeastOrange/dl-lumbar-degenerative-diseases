"""Filesystem helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create directory if missing and return as Path."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def write_json(data: dict[str, Any], path: str | Path) -> None:
    """Write JSON with UTF-8 and pretty indentation."""
    destination = Path(path)
    ensure_dir(destination.parent)
    destination.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
