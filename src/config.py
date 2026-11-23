"""Load config.yaml once and share it across the pipeline."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Read the YAML config. CONFIG_PATH env var overrides the default location."""
    path = Path(path or os.environ.get("CONFIG_PATH", DEFAULT_CONFIG))
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(path: str | os.PathLike) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def flatten(d: dict, parent: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten nested config so it can be logged as MLflow params."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            out.update(flatten(v, key, sep))
        else:
            out[key] = v
    return out
