from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_phase7a import (  # noqa: E402
    gradient_dot_delta,
    matched_control,
    positive_risk_part,
    rank_correlations,
    select_discordant_components,
)


def test_component_risk_is_gradient_dot_actual_parameter_delta():
    gradient = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    delta = torch.tensor([[0.5, 1.0], [-1.0, 2.0]])

    assert gradient_dot_delta(gradient, delta) == pytest.approx(3.5)


def test_positive_risk_part_does_not_hide_signed_loss_improvement():
    assert positive_risk_part(0.75) == pytest.approx(0.75)
    assert positive_risk_part(-0.25) == pytest.approx(0.0)


def test_rank_correlations_use_components_with_actual_updates_only():
    rows = [
        {"importance": 0.1, "risk": 1.0, "update_norm": 1.0},
        {"importance": 0.9, "risk": 2.0, "update_norm": 1.0},
        {"importance": 100.0, "risk": -100.0, "update_norm": 0.0},
    ]

    result = rank_correlations(rows)

    assert result["n"] == 2
    assert result["spearman_rho"] == pytest.approx(1.0)


def test_discordant_component_selection_finds_high_importance_low_risk_and_low_importance_high_risk():
    rows = [
        {"component_id": "a", "importance": 0.9, "risk": 0.01, "update_norm": 1.0, "parameter_count": 10},
        {"component_id": "b", "importance": 0.8, "risk": 0.02, "update_norm": 1.0, "parameter_count": 10},
        {"component_id": "c", "importance": 0.1, "risk": 0.8, "update_norm": 1.0, "parameter_count": 10},
        {"component_id": "d", "importance": 0.2, "risk": 0.7, "update_norm": 1.0, "parameter_count": 10},
    ]

    high_important_low_risk, low_important_high_risk = select_discordant_components(rows)

    assert high_important_low_risk["component_id"] == "a"
    assert low_important_high_risk["component_id"] == "c"


def test_control_matches_parameter_count_and_closest_update_norm():
    candidates = [
        {"component_id": "excluded", "parameter_count": 10, "update_norm": 3.0},
        {"component_id": "wrong-size", "parameter_count": 11, "update_norm": 2.0},
        {"component_id": "near", "parameter_count": 10, "update_norm": 2.8},
        {"component_id": "far", "parameter_count": 10, "update_norm": 1.0},
    ]

    control = matched_control(candidates, target=candidates[0], exclude_ids={"excluded"}, seed=10)

    assert control["component_id"] == "near"
    assert control["parameter_count"] == candidates[0]["parameter_count"]
