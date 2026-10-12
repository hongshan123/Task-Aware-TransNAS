from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_phase7a_pilot import _load_completed_condition_summary  # noqa: E402


def test_completed_condition_summary_is_loaded_without_rerunning(tmp_path):
    assert _load_completed_condition_summary(tmp_path) is None

    expected = {"condition": "C1", "seed": 10, "constraint_violation_detected": True}
    (tmp_path / "metrics_training.json").write_text(json.dumps(expected), encoding="utf-8")

    assert _load_completed_condition_summary(tmp_path) == expected
