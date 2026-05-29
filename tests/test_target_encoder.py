"""Tests para pipeline.target_encoder.TargetEncoderMappings."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.target_encoder import TargetEncoderMappings  # noqa: E402

MAPPINGS_PATH = ROOT / "models" / "churn" / "v2_rf_L22_2026-05-19" / "target_encoder_mappings.json"


def test_target_encoder_loads_from_json() -> None:
    """El JSON generado se parsea bien y contiene la estructura esperada."""
    assert MAPPINGS_PATH.exists(), f"falta {MAPPINGS_PATH}: ejecuta generate_target_encoder_mappings.py"
    te = TargetEncoderMappings.from_json(MAPPINGS_PATH)

    assert te.smoothing == 10.0
    assert te.missing_sentinel == "__missing__"
    assert set(te.cat_cols) == {
        "country",
        "has_user_rated_app",
        "user_store_where_published",
        "device_primary_platform",
    }
    assert set(te.per_target.keys()) == {"churn_7d", "churn_14d", "churn_30d"}

    for target, cfg in te.per_target.items():
        assert "global_mean" in cfg, f"{target}: falta global_mean"
        assert 0.0 < cfg["global_mean"] < 1.0, (
            f"{target}: global_mean fuera de (0,1): {cfg['global_mean']}"
        )
        assert "mappings" in cfg
        for col in te.cat_cols:
            assert col in cfg["mappings"], f"{target}: falta mapping para {col}"
            assert len(cfg["mappings"][col]) > 0, f"{target}.{col}: mapping vacío"


def test_target_encoder_transform_known_values() -> None:
    """transform() devuelve floats y los valores coinciden con las mappings."""
    te = TargetEncoderMappings.from_json(MAPPINGS_PATH)

    # Construimos un DataFrame pequeño con valores que sabemos que están en mappings
    # (paises comunes + has_user_rated_app True/False)
    df = pd.DataFrame({
        "country": ["Spain", "Brazil", "Germany"],
        "has_user_rated_app": [True, False, True],
        "user_store_where_published": ["google_play", "app_store", "google_play"],
        "device_primary_platform": ["android", "ios", "android"],
        "other_feature": [1.0, 2.0, 3.0],  # numérica, no debe tocarse
    })

    out = te.transform(df, target="churn_30d")

    # Cat_cols deben ser float ahora
    for col in te.cat_cols:
        assert out[col].dtype == np.float64, f"{col}: dtype={out[col].dtype}"
        assert out[col].notna().all(), f"{col}: NaN tras transform"

    # other_feature debe permanecer intacta
    assert (out["other_feature"] == df["other_feature"]).all()

    # Verificar contra las mappings: los valores en out[col] deben coincidir
    cfg = te.per_target["churn_30d"]
    for col in te.cat_cols:
        col_mapping = cfg["mappings"][col]
        for i, raw_value in enumerate(df[col]):
            raw_str = "__missing__" if pd.isna(raw_value) else str(raw_value)
            expected = col_mapping.get(raw_str, cfg["global_mean"])
            assert abs(out[col].iloc[i] - expected) < 1e-9, (
                f"{col} row {i}: raw='{raw_str}' got {out[col].iloc[i]}, expected {expected}"
            )


def test_target_encoder_handles_unseen() -> None:
    """Valores no vistos en mappings reciben global_mean."""
    te = TargetEncoderMappings.from_json(MAPPINGS_PATH)
    cfg = te.per_target["churn_14d"]
    global_mean_14d = cfg["global_mean"]

    df = pd.DataFrame({
        "country": ["Mordor", "Wakanda", "Spain"],  # 2 paises ficticios + 1 real
        "has_user_rated_app": [True, False, True],
        "user_store_where_published": ["google_play"] * 3,
        "device_primary_platform": ["android"] * 3,
    })

    out = te.transform(df, target="churn_14d")

    # Mordor y Wakanda no están en mappings → global_mean
    country_map = cfg["mappings"]["country"]
    assert "Mordor" not in country_map
    assert "Wakanda" not in country_map
    assert abs(out["country"].iloc[0] - global_mean_14d) < 1e-9
    assert abs(out["country"].iloc[1] - global_mean_14d) < 1e-9

    # Spain SÍ está
    assert "Spain" in country_map
    assert abs(out["country"].iloc[2] - country_map["Spain"]) < 1e-9


def test_target_encoder_handles_nan() -> None:
    """NaN en cat_cols → mapped con missing_sentinel ('__missing__')."""
    te = TargetEncoderMappings.from_json(MAPPINGS_PATH)
    cfg = te.per_target["churn_7d"]

    df = pd.DataFrame({
        "country": [None, "Spain", np.nan],
        "has_user_rated_app": [True, False, None],
        "user_store_where_published": [None, "google_play", "app_store"],
        "device_primary_platform": ["android", None, "ios"],
    })

    out = te.transform(df, target="churn_7d")

    # NaN convertidos a sentinel y mapeados (o global_mean si sentinel no está en mapping)
    for col in te.cat_cols:
        assert out[col].notna().all(), f"{col}: NaN tras transform debe estar resuelto"

    # country tiene 2 NaN (None y np.nan). Verificar contra el sentinel mapping
    country_map = cfg["mappings"]["country"]
    if "__missing__" in country_map:
        expected_nan_country = country_map["__missing__"]
    else:
        expected_nan_country = cfg["global_mean"]
    assert abs(out["country"].iloc[0] - expected_nan_country) < 1e-9
    assert abs(out["country"].iloc[2] - expected_nan_country) < 1e-9


def test_target_encoder_unknown_target_raises() -> None:
    """Pedir un target inexistente lanza KeyError."""
    te = TargetEncoderMappings.from_json(MAPPINGS_PATH)
    df = pd.DataFrame({
        "country": ["Spain"],
        "has_user_rated_app": [True],
        "user_store_where_published": ["google_play"],
        "device_primary_platform": ["android"],
    })
    with pytest.raises(KeyError, match="churn_999d"):
        te.transform(df, target="churn_999d")
