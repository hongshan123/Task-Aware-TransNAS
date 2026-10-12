"""Small, testable adapters around the unmodified official PaRSP repository."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = Path(
    os.environ.get("PARSP_OFFICIAL_ROOT", str(PROJECT_ROOT.parent / "PaRSP-official"))
).resolve()
STRICT_NULLSPACE_RCOND = 1e-12


def load_official_parsp() -> dict[str, Any]:
    """Import upstream methods from the separately cloned, immutable checkout."""
    if not (OFFICIAL_ROOT / "parsp" / "methods" / "localization.py").is_file():
        raise FileNotFoundError(f"Official PaRSP source is unavailable: {OFFICIAL_ROOT}")
    path = str(OFFICIAL_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    return {
        "localization": importlib.import_module("parsp.methods.localization"),
        "masking": importlib.import_module("parsp.methods.masking"),
        "projection": importlib.import_module("parsp.methods.projection"),
        "wrapper": importlib.import_module("parsp.models.wrapper"),
        "utils": importlib.import_module("parsp.methods.utils"),
    }


def official_repo_metadata() -> dict[str, str]:
    """Return the checked-out upstream commit without altering its source."""
    import subprocess

    commit = subprocess.check_output(
        ["git", "-C", str(OFFICIAL_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(OFFICIAL_ROOT), "status", "--porcelain"], text=True
    )
    if status.strip():
        raise RuntimeError("Official PaRSP checkout is dirty; refusing to claim an unchanged reproduction")
    return {"repository": "https://github.com/JinhuiBot/PaRSP", "commit": commit, "working_tree": "clean"}


def map_query_channels_to_kv_channels(
    indices: list[int] | tuple[int, ...],
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> list[int]:
    """Map selected Q/O channels to shared K/V channels under GQA."""
    if min(num_query_heads, num_kv_heads, head_dim) <= 0:
        raise ValueError("Head counts and head_dim must be positive")
    if num_query_heads % num_kv_heads:
        raise ValueError("GQA requires num_query_heads to be divisible by num_kv_heads")
    group_size = num_query_heads // num_kv_heads
    mapped: set[int] = set()
    for raw_index in indices:
        index = int(raw_index)
        if index < 0 or index >= num_query_heads * head_dim:
            raise IndexError(f"Query channel {index} is outside [0, {num_query_heads * head_dim})")
        query_head, channel = divmod(index, head_dim)
        kv_head = query_head // group_size
        mapped.add(kv_head * head_dim + channel)
    return sorted(mapped)


def _null_projector(covariance: torch.Tensor, rcond: float) -> torch.Tensor:
    # C4's audited residual threshold is tighter than the official PaRSP
    # projector cutoff. Resolve its nullspace in float64 and retain only much
    # lower-variance directions for the post-update safety correction.
    covariance = torch.as_tensor(covariance, dtype=torch.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix")
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    largest = max(float(eigenvalues[-1].item()), 0.0)
    if largest == 0.0:
        return torch.eye(covariance.shape[0], dtype=torch.float64)
    null = eigenvalues <= rcond * largest
    if not bool(null.any()):
        return torch.zeros_like(covariance)
    basis = eigenvectors[:, null]
    return basis @ basis.T


def strict_project_update(
    delta: torch.Tensor,
    mask: torch.Tensor | None,
    covariance: torch.Tensor,
    mask_axis: int = 1,
    rcond: float = STRICT_NULLSPACE_RCOND,
) -> torch.Tensor:
    """Project an actual linear-weight update into the mask/null-space intersection.

    `covariance` is the uncentered input covariance, proportional to X Xᵀ.
    For input-column masks (axis 1), the null space is computed after restricting
    both the update and covariance to active columns. This prevents right
    projection from repopulating excluded columns.
    """
    if delta.ndim != 2:
        raise ValueError("strict projection expects a two-dimensional linear weight update")
    if mask_axis not in (0, 1):
        raise ValueError("mask_axis must be 0, 1, or omitted by passing mask=None")
    if covariance.shape != (delta.shape[1], delta.shape[1]):
        raise ValueError(
            f"covariance shape {tuple(covariance.shape)} does not match linear input width {delta.shape[1]}"
        )
    covariance = covariance.to(device=delta.device, dtype=torch.float64)
    source_dtype = delta.dtype
    source = delta.to(dtype=torch.float64)
    result = torch.zeros_like(source)

    if mask is None:
        active = torch.arange(delta.shape[0] if mask_axis == 0 else delta.shape[1], device=delta.device)
    else:
        flat_mask = mask.to(device=delta.device, dtype=torch.bool).reshape(-1)
        expected = delta.shape[mask_axis]
        if flat_mask.numel() != expected:
            raise ValueError(f"mask length {flat_mask.numel()} does not match axis {mask_axis} size {expected}")
        active = torch.nonzero(flat_mask, as_tuple=False).reshape(-1)
    if active.numel() == 0:
        return result.to(dtype=source_dtype)

    if mask_axis == 0 and mask is not None:
        result[active, :] = source[active, :] @ _null_projector(covariance, rcond)
    elif mask_axis == 1 and mask is not None:
        restricted_covariance = covariance.index_select(0, active).index_select(1, active)
        restricted_projector = _null_projector(restricted_covariance, rcond)
        result[:, active] = source.index_select(1, active) @ restricted_projector
    else:
        projector = _null_projector(covariance, rcond)
        if mask_axis == 0:
            result[active, :] = source[active, :] @ projector
        else:
            result[:, active] = source[:, active] @ projector
    return result.to(dtype=source_dtype)


def load_phase7a_splits(
    manifest_path: str | Path, task_ids: list[str]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    """Load only train/dev records and reject any request for another split."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    leakage = manifest.get("leakage_check", {})
    if not isinstance(leakage, dict) or leakage.get("passed") is not True:
        raise RuntimeError("Phase 7A requires a passing train/dev/test ID-disjointness audit")
    infos = {row["task_id"]: row for row in manifest.get("tasks", [])}
    if set(task_ids) != set(infos):
        raise ValueError("Requested Phase 7A task IDs differ from the prepared manifest")

    def read_split(split: str) -> dict[str, list[dict[str, Any]]]:
        if split not in {"train", "dev"}:
            raise ValueError("Phase 7A opens train and dev splits only")
        output = {}
        for task_id in task_ids:
            info = infos[task_id]
            path = Path(info["split_files"][split])
            if not path.is_absolute():
                path = manifest_path.parent / path
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(rows) != int(info["counts"][split]):
                raise ValueError(f"Split row count mismatch for {task_id}/{split}")
            if any(row.get("split") != split for row in rows):
                raise ValueError(f"Unexpected record split in {task_id}/{split}")
            output[task_id] = rows
        return output

    return read_split("train"), read_split("dev")


def load_phase7a_task_split(
    manifest_path: str | Path,
    task_id: str,
    split: str,
    access_log: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Open one task/split on demand so future records stay unread during training."""
    if split not in {"train", "dev"}:
        raise ValueError("Phase 7A opens train and dev splits only")
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    leakage = manifest.get("leakage_check", {})
    if not isinstance(leakage, dict) or leakage.get("passed") is not True:
        raise RuntimeError("Phase 7A requires a passing train/dev/test ID-disjointness audit")
    info = next((row for row in manifest.get("tasks", []) if row.get("task_id") == task_id), None)
    if info is None:
        raise KeyError(f"Task {task_id} is absent from the prepared manifest")
    path = Path(info["split_files"][split])
    if not path.is_absolute():
        path = manifest_path.parent / path
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(info["counts"][split]):
        raise ValueError(f"Split row count mismatch for {task_id}/{split}")
    if any(row.get("split") != split for row in rows):
        raise ValueError(f"Unexpected record split in {task_id}/{split}")
    if access_log is not None:
        access_log.append(str(path.resolve()))
    return rows
