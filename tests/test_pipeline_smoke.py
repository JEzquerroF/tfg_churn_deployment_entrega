"""
Test smoke del pipeline end-to-end (post-Fase 1.5).

Ejecuta el CLI `scripts/predict.py` sobre `tests/sample_data/` (subset
coherente de 100 usuarios reales del TFG) y verifica el nuevo schema:

- predictions.csv (capa negocio) con 20 cols: dual (live/oof/delta/source/final
  por target) + archetype_n1/name + contramedida_primaria_cod/label + user_id.
  El antiguo sub_archetype_n2 (Nivel 2 HDBSCAN) se eliminó.
- predictions_full.csv (capa técnica) con 28 cols: lo anterior + has_tier2 +
  los 7 ejes de gusto.
- contramedida primaria con cobertura 100% (0 jugadores en CM00).
- summary.json con `per_target` + countermeasure_distribution.
- _run_metadata.json con stage_6_status ∈ {"ok", "skipped"}
- CERO columnas binarizadas (sin risk_level, priority, will_churn, label)
- segmentation.csv NO existe (binarización vive en el frontend D1b)

NO usa xfail dinámico: el pipeline retorna exit=0 siempre que las stages
[1/6]-[5/6] funcionen, gracias al try/except en [6/6].
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLE_DATA = ROOT / "tests" / "sample_data"
CHURN_TARGETS = ("churn_7d", "churn_14d", "churn_30d")
VALID_SOURCES = {"oof_stable", "live_drift", "live_new"}
# Cols que NO deben aparecer en predictions.csv (filosofía F1.5: no binarizar).
FORBIDDEN_BINARIZED_COLS = {
    "risk_level", "priority", "label", "will_churn",
    "suggested_action", "segment",
}


@pytest.fixture(scope="module")
def pipeline_output(tmp_path_factory) -> Path:
    """Ejecuta predict.main() una vez por módulo y devuelve el output dir."""
    output_dir = tmp_path_factory.mktemp("smoke_output")

    import scripts.predict as predict

    original_argv = sys.argv
    sys.argv = [
        "predict.py",
        "--input", str(SAMPLE_DATA),
        "--output", str(output_dir),
        "--config", str(ROOT / "config" / "pipeline_config.yaml"),
    ]
    try:
        exit_code = predict.main()
    finally:
        sys.argv = original_argv

    # F1.5: el CLI debe terminar OK aunque [6/6] falle (try/except interno).
    assert exit_code == 0, f"predict.main() falló con exit={exit_code}"
    return output_dir


def test_outputs_exist(pipeline_output: Path) -> None:
    """predictions.csv, predictions_full.csv, summary.json, _run_metadata.json existen."""
    assert (pipeline_output / "predictions.csv").exists()
    assert (pipeline_output / "predictions_full.csv").exists()
    assert (pipeline_output / "summary.json").exists()
    assert (pipeline_output / "_run_metadata.json").exists()
    assert not (pipeline_output / "segmentation.csv").exists(), (
        "segmentation.csv NO debería existir (binarización en frontend)"
    )
    # Intermediates deben haber sido limpiados tras escritura final
    intermediates = list(pipeline_output.glob("_intermediate_*.parquet"))
    assert intermediates == [], f"intermediates no borrados: {intermediates}"


def test_predictions_shape_and_cols(pipeline_output: Path) -> None:
    """20 cols (capa negocio), 100 filas. Sin sub_archetype_n2, con contramedida."""
    df = pd.read_csv(pipeline_output / "predictions.csv")
    assert len(df) == 100, f"esperaba 100 filas, hay {len(df)}"

    expected_cols = {
        "user_id", "archetype_n1", "archetype_name",
        "contramedida_primaria_cod", "contramedida_primaria_label",
    }
    for t in CHURN_TARGETS:
        for kind in ("live", "oof", "delta", "source", "final"):
            expected_cols.add(f"{t}_{kind}")
    assert set(df.columns) == expected_cols, (
        f"diff cols: extra={set(df.columns) - expected_cols}, "
        f"falta={expected_cols - set(df.columns)}"
    )
    assert len(df.columns) == 20, f"esperaba 20 cols, hay {len(df.columns)}"
    # El Nivel 2 se eliminó: sub_archetype_n2 NO debe aparecer
    assert "sub_archetype_n2" not in df.columns


def test_predictions_full_has_seven_axes(pipeline_output: Path) -> None:
    """predictions_full.csv (capa técnica) tiene los 7 ejes + has_tier2 + contramedida."""
    df = pd.read_csv(pipeline_output / "predictions_full.csv")
    assert len(df) == 100
    seven_axes = {
        "clase_favorita", "estilo_build", "monetizacion", "perfil_enhance",
        "perfil_coleccion", "pvp_perfil", "perfil_oro",
    }
    assert seven_axes.issubset(set(df.columns)), (
        f"faltan ejes: {seven_axes - set(df.columns)}"
    )
    assert "has_tier2" in df.columns
    assert "contramedida_primaria_cod" in df.columns
    assert "sub_archetype_n2" not in df.columns

    # Los 5 ejes universales se calculan para el 100% (sin NaN)
    for eje in ("clase_favorita", "estilo_build", "monetizacion",
                "perfil_enhance", "perfil_coleccion"):
        assert df[eje].notna().all(), f"eje universal {eje} tiene NaN"


def test_countermeasure_coverage_100(pipeline_output: Path) -> None:
    """Cobertura de contramedidas = 100%: 0 jugadores sin asignar (CM00)."""
    df = pd.read_csv(pipeline_output / "predictions.csv")
    cods = df["contramedida_primaria_cod"]
    assert cods.notna().all(), "hay jugadores sin contramedida (NaN)"
    assert (cods == "CM00").sum() == 0, "hay jugadores en CM00 (sin asignar)"
    # Todos los códigos asignados son del catálogo CM01..CM12
    valid = {f"CM{i:02d}" for i in range(1, 13)}
    assert set(cods.unique()).issubset(valid), (
        f"códigos fuera del catálogo: {set(cods.unique()) - valid}"
    )


def test_no_binarization_cols(pipeline_output: Path) -> None:
    """Ninguna col binarizada (risk_level, priority, etc.) debe aparecer."""
    df = pd.read_csv(pipeline_output / "predictions.csv")
    leaked = FORBIDDEN_BINARIZED_COLS & set(df.columns)
    assert leaked == set(), (
        f"cols binarizadas presentes (filosofía F1.5: NO binarizar): {leaked}"
    )


def test_finals_in_unit_interval(pipeline_output: Path) -> None:
    """Por cada target, *_final ∈ [0, 1] y sin NaN."""
    df = pd.read_csv(pipeline_output / "predictions.csv")
    for t in CHURN_TARGETS:
        col = f"{t}_final"
        assert df[col].between(0.0, 1.0).all(), (
            f"{col} fuera de [0,1]: min={df[col].min()}, max={df[col].max()}"
        )
        assert df[col].notna().all(), f"{col} tiene NaN"


def test_sources_valid_values(pipeline_output: Path) -> None:
    """Por cada target, *_source ∈ {oof_stable, live_drift, live_new}."""
    df = pd.read_csv(pipeline_output / "predictions.csv")
    for t in CHURN_TARGETS:
        col = f"{t}_source"
        actual = set(df[col].dropna().unique())
        assert actual.issubset(VALID_SOURCES), (
            f"{col} tiene valores no válidos: {actual - VALID_SOURCES}"
        )


def test_oof_consistency_with_source(pipeline_output: Path) -> None:
    """live_new ⇔ oof es NaN. oof_stable o live_drift ⇒ oof no NaN."""
    df = pd.read_csv(pipeline_output / "predictions.csv")
    for t in CHURN_TARGETS:
        source_col = f"{t}_source"
        oof_col = f"{t}_oof"
        is_new = df[source_col] == "live_new"
        # Para live_new, OOF debe ser NaN
        assert df.loc[is_new, oof_col].isna().all(), (
            f"hay live_new con oof no NaN en {t}"
        )
        # Para los que NO son live_new, OOF debe estar
        assert df.loc[~is_new, oof_col].notna().all(), (
            f"hay oof_stable/live_drift con oof NaN en {t}"
        )


def test_archetype_cols_skipped_or_valid(pipeline_output: Path) -> None:
    """
    Si stage_6=skipped: archetype_n1=NaN, archetype_name='N/A'.
    Si stage_6=ok: archetype_n1 ∈ [0,5], archetype_name ∈ nombres del YAML.
    """
    df = pd.read_csv(pipeline_output / "predictions.csv")
    meta = json.loads((pipeline_output / "_run_metadata.json").read_text())
    status = meta["stage_6_status"]
    assert status in ("ok", "skipped"), f"stage_6_status inesperado: {status}"

    if status == "skipped":
        assert df["archetype_n1"].isna().all(), "archetype_n1 debería ser NaN si stage_6=skipped"
        name_col = df["archetype_name"]
        all_na_or_text = (name_col.isna() | (name_col == "N/A")).all()
        assert all_na_or_text, (
            f"archetype_name debería ser NaN o 'N/A' si stage_6=skipped, "
            f"valores únicos: {name_col.unique()}"
        )
        df_raw = pd.read_csv(pipeline_output / "predictions.csv", keep_default_na=False)
        assert (df_raw["archetype_name"] == "N/A").all(), (
            "leído con keep_default_na=False, archetype_name debe ser 'N/A'"
        )
    else:
        assert df["archetype_n1"].dropna().between(0, 5).all(), (
            f"archetype_n1 fuera de [0,5]: {df['archetype_n1'].unique()}"
        )
        # Confirma que archetype_name NO lleva emoji
        for name in df["archetype_name"].dropna().unique():
            assert not any(ord(c) > 0x1F000 for c in str(name)), (
                f"archetype_name '{name}' contiene emoji; debe ser limpio"
            )


def test_summary_json_schema(pipeline_output: Path) -> None:
    """summary.json tiene las claves nuevas de F1.5."""
    summary = json.loads((pipeline_output / "summary.json").read_text())
    for key in ("n_users", "churn_targets", "drift_threshold", "stage_6_status", "per_target"):
        assert key in summary, f"falta '{key}' en summary.json"

    assert summary["n_users"] == 100
    assert summary["drift_threshold"] == pytest.approx(0.10)
    assert set(summary["churn_targets"]) == set(CHURN_TARGETS)
    assert summary["stage_6_status"] in ("ok", "skipped")

    # per_target con source_distribution, final_mean/median, delta_stats por cada target
    for t in CHURN_TARGETS:
        assert t in summary["per_target"], f"falta {t} en per_target"
        per = summary["per_target"][t]
        assert "source_distribution" in per
        assert "final_mean" in per
        assert "final_median" in per
        assert "delta_stats" in per
        # source_distribution solo contiene claves válidas
        for src in per["source_distribution"]:
            assert src in VALID_SOURCES, f"source inválida en summary: {src}"

    # NO debe haber priority_distribution / risk_distribution (filosofía F1.5)
    for forbidden in ("priority_distribution", "risk_distribution"):
        assert forbidden not in summary, (
            f"'{forbidden}' no debería estar en summary.json (binarización en frontend)"
        )


def test_run_metadata_schema(pipeline_output: Path) -> None:
    """_run_metadata.json incluye stage_6_status + drift_threshold + traceback si aplica."""
    meta = json.loads((pipeline_output / "_run_metadata.json").read_text())
    for key in (
        "started_at", "elapsed_seconds", "reference_date", "cutoff_date",
        "n_users", "churn_targets", "drift_threshold", "stage_6_status",
        "schemas_identified", "models",
    ):
        assert key in meta, f"falta '{key}' en _run_metadata.json"
    assert meta["n_users"] == 100
    assert meta["stage_6_status"] in ("ok", "skipped")
    if meta["stage_6_status"] == "skipped":
        assert meta.get("stage_6_error"), (
            "stage_6_status=skipped pero falta stage_6_error con traceback"
        )

    # Los 9 schemas del expected_schema.yaml deben haber sido identificados
    assert set(meta["schemas_identified"]) == {
        "users", "characters", "devices",
        "iaps_consumables", "iaps_subscriptions",
        "daily_rewards", "user_items", "user_items_collection",
        "support_feedback",
    }
