"""Cross-platform path and cache defaults for Task-Aware-TransNAS scripts."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path, PureWindowsPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WINDOWS_STORAGE_ROOT = PureWindowsPath("D:/Task-Aware-TransNAS")


def _system_name(platform: str | None) -> str:
    return platform or os.name


def _windows_relative_parts(value: str) -> tuple[str, ...] | None:
    windows_path = PureWindowsPath(value)
    if not windows_path.drive:
        return None
    parts = windows_path.parts[1:]
    if parts and parts[0].casefold() == "task-aware-transnas":
        parts = parts[1:]
    return tuple(parts)


def storage_root(
    project_root: Path | None = None,
    *,
    platform: str | None = None,
) -> Path:
    """Return the run-data root, preserving D: on Windows and using the repo on Linux."""
    root = Path(project_root or PROJECT_ROOT).expanduser().resolve()
    system = _system_name(platform)
    configured = os.environ.get("TRANSNAS_STORAGE_ROOT")
    if configured:
        configured = os.path.expandvars(os.path.expanduser(configured))
        windows_parts = _windows_relative_parts(configured)
        if system != "nt" and windows_parts is not None:
            return (root.joinpath(*windows_parts) if windows_parts else root / "runs").resolve()
        candidate = Path(configured)
        return (candidate if candidate.is_absolute() else root / candidate).resolve()
    if system == "nt":
        return Path(str(WINDOWS_STORAGE_ROOT))
    return (root / "runs").resolve()


def data_root(
    project_root: Path | None = None,
    *,
    platform: str | None = None,
) -> Path:
    """Return the raw/processed data root, separate from generated run artifacts."""
    root = Path(project_root or PROJECT_ROOT).expanduser().resolve()
    system = _system_name(platform)
    configured = os.environ.get("TRANSNAS_DATA_ROOT")
    if configured:
        configured = os.path.expandvars(os.path.expanduser(configured))
        windows_parts = _windows_relative_parts(configured)
        if system != "nt" and windows_parts is not None:
            if windows_parts and windows_parts[0].casefold() == "data":
                windows_parts = windows_parts[1:]
            return (root / "data").joinpath(*windows_parts).resolve()
        candidate = Path(configured)
        return (candidate if candidate.is_absolute() else root / candidate).resolve()
    return (root / "data").resolve()


def resolve_runtime_path(
    value: str | os.PathLike[str],
    *,
    project_root: Path | None = None,
    platform: str | None = None,
) -> Path:
    """Resolve native absolute, project-relative, or legacy D: paths for this host."""
    root = Path(project_root or PROJECT_ROOT).expanduser().resolve()
    raw = os.path.expandvars(os.path.expanduser(os.fspath(value)))
    windows_parts = _windows_relative_parts(raw)
    if _system_name(platform) != "nt" and windows_parts is not None:
        if windows_parts and windows_parts[0].casefold() == "data":
            return data_root(root, platform=platform).joinpath(*windows_parts[1:]).resolve()
        return storage_root(root, platform=platform).joinpath(*windows_parts).resolve()
    candidate = Path(raw)
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def resolve_config_paths(
    config: dict,
    *,
    project_root: Path | None = None,
    platform: str | None = None,
) -> dict:
    """Return a config copy with known filesystem fields normalized for this host."""
    resolved = deepcopy(config)
    dataset = resolved.get("dataset", {})
    for key in (
        "source_repo",
        "source_data",
        "processed_dir",
        "task_stream_manifest",
        "manifest_report_path",
        "manifest_project_path",
    ):
        if dataset.get(key):
            dataset[key] = str(
                resolve_runtime_path(dataset[key], project_root=project_root, platform=platform)
            )
    storage = resolved.get("storage", {})
    for key in ("root", "checkpoints", "results"):
        if storage.get(key):
            storage[key] = str(
                resolve_runtime_path(storage[key], project_root=project_root, platform=platform)
            )
    return resolved


def configure_cache_environment(project_root: Path | None = None) -> dict[str, str]:
    """Set absent model/data caches under the selected storage root and return them."""
    root = storage_root(project_root)
    defaults = {
        "HF_HOME": root / "cache" / "huggingface",
        "HF_HUB_CACHE": root / "cache" / "huggingface" / "hub",
        "HF_DATASETS_CACHE": root / "cache" / "huggingface" / "datasets",
        "TORCH_HOME": root / "cache" / "torch",
    }
    result: dict[str, str] = {}
    for key, value in defaults.items():
        os.environ.setdefault(key, str(value))
        result[key] = os.environ[key]
    return result
