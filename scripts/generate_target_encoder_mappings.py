"""
generate_target_encoder_mappings.py — reconstruye las mappings de target
encoding del RF L22 v1 del TFG y las serializa como JSON declarativo.

One-shot (similar a prepare_models.py). Solo se ejecuta cuando se hace swap
de modelo o cambia el sample de training.

Reproduce EXACTAMENTE la lógica de `target_encode_cv` en
`tfg/04_estudio_validacion/final_models/production/scripts/data_prep_production.py`:
  - Split 70/15/15 estratificado, seed=42
  - Para val/test/inference: stats_full sobre TRAIN con smoothing=10
  - NaN → '__missing__'
  - Valores no vistos → global_mean = y_train.mean()

Output:
    models/churn/v2_rf_L22_2026-05-19/target_encoder_mappings.json

Uso:
    .venv/bin/python scripts/generate_target_encoder_mappings.py

Dependencias (lectura, NO escritura):
    /Users/jezquerro/Documents/tfg/04_estudio_validacion/data/masters/L22/master_L22_v1_conservative.parquet
    /Users/jezquerro/Documents/tfg/04_estudio_validacion/data/samples/sample_user_ids_L22.parquet
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# Constantes idénticas al training del TFG
RANDOM_SEED = 42
SMOOTHING = 10.0
MISSING_SENTINEL = "__missing__"
CAT_COLS = [
    "country",
    "has_user_rated_app",
    "user_store_where_published",
    "device_primary_platform",
]
TARGETS = ("churn_7d", "churn_14d", "churn_30d")

# Rutas TFG (lectura)
TFG_ROOT = Path("/Users/jezquerro/Documents/tfg")
MASTER_L22 = TFG_ROOT / "04_estudio_validacion/data/masters/L22/master_L22_v1_conservative.parquet"
SAMPLE_L22 = TFG_ROOT / "04_estudio_validacion/data/samples/sample_user_ids_L22.parquet"

# Output (deployment)
DEPLOYMENT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = DEPLOYMENT_ROOT / "models/churn/v2_rf_L22_2026-05-19/target_encoder_mappings.json"


def load_master_with_churn_7d() -> pd.DataFrame:
    """Carga master L22 v1 y mergea churn_7d desde el sample (master no lo tiene)."""
    master = pd.read_parquet(MASTER_L22)
    sample = pd.read_parquet(SAMPLE_L22)
    master["user_id"] = master["user_id"].astype(str)
    sample["user_id"] = sample["user_id"].astype(str)

    if "churn_7d" not in master.columns:
        master = master.merge(sample[["user_id", "churn_7d"]], on="user_id", how="left")
        if master["churn_7d"].isna().any():
            raise RuntimeError(
                "NaN en churn_7d tras merge: sample y master no alineados por user_id"
            )
    return master


def reproduce_train_indices(y: list, n_rows: int) -> list:
    """
    Replica el split 70/15/15 estratificado seed=42 de data_prep_production.
    Devuelve los índices (sobre el master entero) del TRAIN.

    El split solo depende de (n_rows, y, random_state, test_size). Pasamos
    los índices [0..n_rows] como X dummy — sklearn los devuelve permutados.
    """
    indices_all = list(range(n_rows))
    idx_temp, _idx_test, y_temp, _y_test = train_test_split(
        indices_all,
        y,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    idx_train, _idx_val, _y_train, _y_val = train_test_split(
        idx_temp,
        y_temp,
        test_size=0.176,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )
    return idx_train


def compute_mappings_for_target(master: pd.DataFrame, target: str) -> dict:
    """Calcula las mappings smoothed para las 4 cat_cols sobre el TRAIN del target."""
    y = master[target].astype(int).values
    train_idx = reproduce_train_indices(y, len(master))
    train_subset = master.iloc[train_idx]
    y_train = train_subset[target].astype(int).values
    global_mean = float(y_train.mean())

    per_target = {"global_mean": global_mean, "mappings": {}}
    for col in CAT_COLS:
        # Cadena de transformación IDÉNTICA al training:
        #   .astype(object).fillna('__missing__').astype(str)
        train_col = train_subset[col].astype(object).fillna(MISSING_SENTINEL).astype(str)
        stats = (
            pd.DataFrame({"col": train_col.values, "y": y_train})
            .groupby("col")["y"]
            .agg(["mean", "count"])
        )
        stats["smoothed"] = (
            stats["mean"] * stats["count"] + global_mean * SMOOTHING
        ) / (stats["count"] + SMOOTHING)
        mapping = {str(k): float(v) for k, v in stats["smoothed"].to_dict().items()}
        per_target["mappings"][col] = mapping
        logger.info(
            "    %s.%s: %d claves (sample: %s)",
            target,
            col,
            len(mapping),
            list(mapping.items())[:2],
        )

    logger.info("  global_mean(%s) = %.6f", target, global_mean)
    return per_target


def main() -> int:
    if not MASTER_L22.exists():
        logger.error("❌ Master L22 no encontrado: %s", MASTER_L22)
        return 1
    if not SAMPLE_L22.exists():
        logger.error("❌ Sample L22 no encontrado: %s", SAMPLE_L22)
        return 1
    if not OUTPUT_PATH.parent.exists():
        logger.error(
            "❌ Directorio de modelo no existe: %s\n"
            "  (Ejecuta el Bloque 2 antes: copia de artefactos)",
            OUTPUT_PATH.parent,
        )
        return 1

    logger.info("=" * 70)
    logger.info("GENERATE TARGET ENCODER MAPPINGS — RF L22 v1")
    logger.info("=" * 70)
    logger.info("  Master:    %s", MASTER_L22)
    logger.info("  Sample:    %s", SAMPLE_L22)
    logger.info("  Output:    %s", OUTPUT_PATH)
    logger.info("  Seed:      %d", RANDOM_SEED)
    logger.info("  Smoothing: %s", SMOOTHING)
    logger.info("  Cat_cols:  %s", CAT_COLS)
    logger.info("=" * 70)

    logger.info("\nCargando master L22 + mergeando churn_7d…")
    master = load_master_with_churn_7d()
    logger.info("  master shape: %s", master.shape)
    logger.info("  targets disponibles: %s", [t for t in TARGETS if t in master.columns])

    mappings: dict = {
        "smoothing": SMOOTHING,
        "cat_cols": CAT_COLS,
        "missing_sentinel": MISSING_SENTINEL,
        "source": {
            "master": str(MASTER_L22),
            "sample": str(SAMPLE_L22),
            "random_seed": RANDOM_SEED,
            "split_test_size_outer": 0.15,
            "split_test_size_inner": 0.176,
        },
        "per_target": {},
    }

    for target in TARGETS:
        logger.info("\n[%s] Calculando mappings…", target)
        mappings["per_target"][target] = compute_mappings_for_target(master, target)

    OUTPUT_PATH.write_text(json.dumps(mappings, indent=2, ensure_ascii=False))
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    logger.info("\n" + "=" * 70)
    logger.info("✅ MAPPINGS GENERADAS")
    logger.info("=" * 70)
    logger.info("  Output: %s (%.1f KB)", OUTPUT_PATH, size_kb)

    logger.info("\nResumen claves por (target, col):")
    for target in TARGETS:
        gm = mappings["per_target"][target]["global_mean"]
        logger.info("  %s (global_mean=%.4f):", target, gm)
        for col in CAT_COLS:
            n = len(mappings["per_target"][target]["mappings"][col])
            logger.info("    · %s: %d claves", col, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
