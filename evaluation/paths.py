"""Shared filesystem paths for VisualMem evaluation modules."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = str(PROJECT_ROOT)
DEFAULT_INPUT = str(PROJECT_ROOT / "visualmem_data" / "data.json")
