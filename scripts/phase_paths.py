"""Storage paths shared by Phase 4B–6B analysis and replay scripts."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

try:
    from .runtime_paths import storage_root
except ImportError:  # Running a script directly from scripts/.
    from runtime_paths import storage_root

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def storage_path(*parts: str) -> Path:
    """Resolve phase artifacts beneath TRANSNAS_STORAGE_ROOT on every host."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return storage_root(PROJECT_ROOT).joinpath(*parts).resolve()
