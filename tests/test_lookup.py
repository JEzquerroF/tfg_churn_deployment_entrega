"""Tests unitarios de la lógica OOF/live + drift detection (Fase 1.5)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.lookup import apply_oof_lookup_with_drift  # noqa: E402


def _make_mock_registry(oof_df: pd.DataFrame) -> MagicMock:
    """Devuelve un ModelRegistry mock cuyo lookup_oof retorna el DF dado."""
    registry = MagicMock()
    registry.lookup_oof.return_value = oof_df
    return registry


def test_drift_below_threshold_serves_oof():
    """Si |live - oof| <= threshold, sirve OOF y marca source=oof_stable."""
    live = pd.DataFrame({"user_id": ["A"], "churn_prob_30d": [0.55]})
    oof = pd.DataFrame({"user_id": ["A"], "churn_prob_30d_oof": [0.50]})
    registry = _make_mock_registry(oof)

    result = apply_oof_lookup_with_drift(
        live, registry, targets=["churn_30d"], drift_threshold=0.10
    )

    assert result.loc[0, "churn_30d_source"] == "oof_stable"
    assert result.loc[0, "churn_30d_final"] == pytest.approx(0.50)
    assert result.loc[0, "churn_30d_delta"] == pytest.approx(0.05)


def test_drift_above_threshold_serves_live():
    """Si |live - oof| > threshold, sirve live y marca source=live_drift."""
    live = pd.DataFrame({"user_id": ["B"], "churn_prob_30d": [0.30]})
    oof = pd.DataFrame({"user_id": ["B"], "churn_prob_30d_oof": [0.70]})
    registry = _make_mock_registry(oof)

    result = apply_oof_lookup_with_drift(
        live, registry, targets=["churn_30d"], drift_threshold=0.10
    )

    assert result.loc[0, "churn_30d_source"] == "live_drift"
    assert result.loc[0, "churn_30d_final"] == pytest.approx(0.30)
    assert result.loc[0, "churn_30d_delta"] == pytest.approx(-0.40)


def test_new_user_no_oof_serves_live():
    """Si no hay OOF (NaN), sirve live y marca source=live_new."""
    live = pd.DataFrame({"user_id": ["C"], "churn_prob_30d": [0.85]})
    oof = pd.DataFrame({"user_id": ["C"], "churn_prob_30d_oof": [np.nan]})
    registry = _make_mock_registry(oof)

    result = apply_oof_lookup_with_drift(
        live, registry, targets=["churn_30d"], drift_threshold=0.10
    )

    assert result.loc[0, "churn_30d_source"] == "live_new"
    assert result.loc[0, "churn_30d_final"] == pytest.approx(0.85)
    assert pd.isna(result.loc[0, "churn_30d_delta"])


def test_drift_threshold_boundary_exact():
    """Si |delta| es exactamente igual al threshold, se considera stable."""
    live = pd.DataFrame({"user_id": ["D"], "churn_prob_30d": [0.60]})
    oof = pd.DataFrame({"user_id": ["D"], "churn_prob_30d_oof": [0.50]})
    registry = _make_mock_registry(oof)

    result = apply_oof_lookup_with_drift(
        live, registry, targets=["churn_30d"], drift_threshold=0.10
    )

    # |delta| = 0.10, threshold = 0.10. NO supera estrictamente.
    assert result.loc[0, "churn_30d_source"] == "oof_stable"


def test_multiple_targets():
    """3 targets simultáneos, comportamientos distintos por target."""
    live = pd.DataFrame({
        "user_id": ["E"],
        "churn_prob_7d": [0.20],   # estable vs OOF 0.18 (|delta|=0.02)
        "churn_prob_14d": [0.55],  # drift  vs OOF 0.20 (|delta|=0.35)
        "churn_prob_30d": [0.70],  # estable vs OOF 0.72 (|delta|=0.02)
    })
    oof = pd.DataFrame({
        "user_id": ["E"],
        "churn_prob_7d_oof": [0.18],
        "churn_prob_14d_oof": [0.20],
        "churn_prob_30d_oof": [0.72],
    })
    registry = _make_mock_registry(oof)

    result = apply_oof_lookup_with_drift(
        live,
        registry,
        targets=["churn_7d", "churn_14d", "churn_30d"],
        drift_threshold=0.10,
    )

    assert result.loc[0, "churn_7d_source"] == "oof_stable"
    assert result.loc[0, "churn_14d_source"] == "live_drift"
    assert result.loc[0, "churn_30d_source"] == "oof_stable"
