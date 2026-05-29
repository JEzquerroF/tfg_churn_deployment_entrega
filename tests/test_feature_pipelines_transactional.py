"""Tests unitarios de los pipelines transaccionales (Fase 2.2)."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import (  # noqa: E402
    feature_pipeline_arena,
    feature_pipeline_currency,
    feature_pipeline_fights,
)
from pipeline._stats import binge_index, gini, shannon, simpson  # noqa: E402
from pipeline.pipeline_context import PipelineContext  # noqa: E402


REFERENCE = date(2026, 4, 4)


def _make_ctx(tmp_path: Path) -> PipelineContext:
    return PipelineContext(raw_csvs_dir=tmp_path, reference_date=REFERENCE)


# === Stats helpers ===


def test_shannon_uniform_vs_concentrated():
    """Shannon: uniforme da log2(N), concentrado da 0."""
    # 4 categorías uniformes → log2(4) = 2.0
    assert shannon([10, 10, 10, 10]) == pytest.approx(2.0)
    # Una sola categoría → 0
    assert shannon([100]) == pytest.approx(0.0)


def test_simpson_uniform_vs_concentrated():
    """Simpson: 1 - 1/N en uniforme; 0 en concentrado."""
    # 4 uniformes → 1 - 4*(1/4)^2 = 0.75
    assert simpson([5, 5, 5, 5]) == pytest.approx(0.75)
    assert simpson([100]) == pytest.approx(0.0)


def test_gini_uniform_vs_concentrated():
    """Gini sobre distribución uniforme → ~0, distribución sesgada → mayor."""
    assert gini([10, 10, 10, 10]) == pytest.approx(0.0, abs=1e-9)
    # Distribución sesgada: 1 valor grande
    assert gini([1, 1, 1, 97]) > 0.5


def test_binge_index_basic():
    """binge = max / median. Si median=0, NaN."""
    assert binge_index({"d1": 10, "d2": 10, "d3": 10}) == pytest.approx(1.0)
    assert binge_index({"d1": 30, "d2": 10, "d3": 10}) == pytest.approx(3.0)
    assert pd.isna(binge_index({}))
    # Median == 0 → NaN
    assert pd.isna(binge_index({"d1": 0, "d2": 0, "d3": 10}))


# === Currency pipeline ===


def test_currency_missing_csv_returns_nan_features(tmp_path: Path):
    """Si currency_transactions.csv no existe, todas las features quedan NaN."""
    ctx = _make_ctx(tmp_path)
    sample = pd.Series(["a" * 24, "b" * 24])
    out = feature_pipeline_currency.compute(ctx, sample)
    expected_cols = {
        "user_id",
        "entropy_currency_concept", "gini_currency_concept",
        "n_distinct_concepts_used", "entropy_currency_type",
        "pct_days_active_currency_30d", "weekend_pct_currency",
        "binge_index_currency",
        "currency_pct_inflow", "currency_pct_outflow",
    }
    assert set(out.columns) == expected_cols
    assert len(out) == 2
    for col in expected_cols - {"user_id"}:
        assert out[col].isna().all(), f"{col} debería ser NaN sin CSV"


def test_currency_aggregator_correct_features(tmp_path: Path):
    """Sintético: 1 usuario con 5 tx en 2 concepts distintos, 1 día."""
    user_id = "a" * 24
    tx_data = pd.DataFrame({
        "user_id": [user_id] * 5,
        "concept": ["bought", "bought", "bought", "sold", "sold"],
        "currency": ["gold", "gold", "gems", "gold", "gems"],
        "quantity": [10, 20, 5, -100, -50],
        "created_at": pd.to_datetime(["2026-04-03T10:00:00Z"] * 5),
    })
    tx_data.to_csv(tmp_path / "currency_transactions.csv", index=False)

    ctx = _make_ctx(tmp_path)
    sample = pd.Series([user_id])
    out = feature_pipeline_currency.compute(ctx, sample)

    row = out.iloc[0]
    # 2 concepts (bought, sold), counts {bought:3, sold:2} → Shannon ~0.97
    assert row["n_distinct_concepts_used"] == 2
    assert row["entropy_currency_concept"] == pytest.approx(shannon([3, 2]))
    # currency: gold=3, gems=2 → similar
    assert row["entropy_currency_type"] == pytest.approx(shannon([3, 2]))
    # 1 día único, max 30 → pct = 1/30
    assert row["pct_days_active_currency_30d"] == pytest.approx(1 / 30)
    # No es weekend (2026-04-03 viernes)
    assert row["weekend_pct_currency"] == pytest.approx(0.0)
    # quantity [10,20,5,-100,-50] → 3 inflow, 2 outflow de 5 (eje perfil_oro)
    assert row["currency_pct_inflow"] == pytest.approx(3 / 5)
    assert row["currency_pct_outflow"] == pytest.approx(2 / 5)


# === Fights pipeline ===


def test_fights_missing_csv_returns_nan(tmp_path: Path):
    ctx = _make_ctx(tmp_path)
    sample = pd.Series(["x" * 24])
    out = feature_pipeline_fights.compute(ctx, sample, char_to_user={})
    assert set(out.columns) == {
        "user_id", "entropy_fights_type", "binge_index_fights",
        "fights_pct_pvp", "fights_pct_won",
    }
    assert out["entropy_fights_type"].isna().all()
    assert out["binge_index_fights"].isna().all()
    assert out["fights_pct_pvp"].isna().all()
    assert out["fights_pct_won"].isna().all()


def test_fights_build_char_to_user():
    """char_to_user filtra IDs no canónicos y mapea bien."""
    chars = pd.DataFrame({
        "_id":     ["ObjectId('" + "a" * 24 + "')", "garbage", "b" * 24],
        "user_id": ["c" * 24,                       "d" * 24,  None],
    })
    mapping = feature_pipeline_fights.build_char_to_user(chars)
    assert "a" * 24 in mapping
    assert mapping["a" * 24] == "c" * 24
    # 'garbage' no aplica → no debe estar
    assert len(mapping) == 1


# === Arena pipeline ===


def test_arena_missing_csv_returns_zero(tmp_path: Path):
    """Sin arena_log.csv → is_arena_player = 0 para todos."""
    ctx = _make_ctx(tmp_path)
    sample = pd.Series(["u" * 24, "v" * 24])
    out = feature_pipeline_arena.compute(ctx, sample, char_to_user={})
    assert (out["is_arena_player"] == 0).all()


def test_arena_marks_active_users(tmp_path: Path):
    """Sintético: 1 attacker que es char_id mapeado a user_id en sample."""
    char_id = "c" * 24
    user_id = "u" * 24
    ar = pd.DataFrame({
        "attacker_id": [char_id, char_id],
        "time_attacked": [
            int(pd.Timestamp("2026-04-03T12:00:00Z").timestamp()),
            int(pd.Timestamp("2026-04-04T08:00:00Z").timestamp()),
        ],
    })
    ar.to_csv(tmp_path / "arena_log.csv", index=False)

    ctx = _make_ctx(tmp_path)
    out = feature_pipeline_arena.compute(
        ctx,
        sample_user_ids=pd.Series([user_id, "w" * 24]),
        char_to_user={char_id: user_id},
    )
    assert out.loc[out["user_id"] == user_id, "is_arena_player"].iloc[0] == 1
    assert out.loc[out["user_id"] == "w" * 24, "is_arena_player"].iloc[0] == 0


# === Filtro temporal ===


def test_currency_temporal_filter_excludes_old_data(tmp_path: Path):
    """Datos fuera de la ventana de 30d se excluyen."""
    user_id = "a" * 24
    # 1 tx dentro de ventana (2026-04-03) + 1 fuera (2025-01-01)
    tx = pd.DataFrame({
        "user_id": [user_id, user_id],
        "concept": ["bought", "bought"],
        "currency": ["gold", "gold"],
        "quantity": [10, 20],
        "created_at": pd.to_datetime([
            "2026-04-03T10:00:00Z",  # dentro de ventana 30d (REF=2026-04-04)
            "2025-01-01T10:00:00Z",  # FUERA de ventana
        ]),
    })
    tx.to_csv(tmp_path / "currency_transactions.csv", index=False)

    ctx = _make_ctx(tmp_path)
    out = feature_pipeline_currency.compute(ctx, pd.Series([user_id]))
    # Solo debería haber 1 día activo (el de dentro de ventana)
    assert out.iloc[0]["pct_days_active_currency_30d"] == pytest.approx(1 / 30)
