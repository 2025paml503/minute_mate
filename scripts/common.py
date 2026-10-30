"""Shared paths, dataset loading, and experiment setup."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
DEFAULT_MODEL = os.environ.get("MINUTEMATE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


def ensure_project_dirs() -> None:
    """Create the output directories used by data and experiment scripts."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


def project_path(path: str | Path) -> Path:
    """Resolve relative paths from the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def read_csv(
    path: str | Path,
    required_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Load a project CSV and report missing files or required columns clearly."""
    resolved = project_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Required CSV file does not exist: {resolved}")
    frame = pd.read_csv(resolved)
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{resolved} is missing required columns: {missing}")
    return frame


def write_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a data frame beneath the project root, creating parent directories."""
    resolved = project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(resolved, index=False)
    return resolved