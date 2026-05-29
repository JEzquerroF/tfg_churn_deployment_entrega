"""
generate_gustos_thresholds.py — Calcula y persiste el p75 de `chars.level`
del training del modelo de gustos N1.

Replica EXACTAMENTE el cálculo del notebook 04b (cell `b03_chars_props`):
    p75 = chars_df['level'].quantile(0.75)
sobre el sample del TFG (`sample_user_ids_gustos.parquet`, 114,412 users).

Persistir este valor en `models/gustos_nivel1/v1_kmeans_k6_2026-05-12/thresholds.json`
permite a la inferencia usar el MISMO threshold que se usó en training,
garantizando que `pct_chars_high_level` sea consistente entre sample del
TFG y sample del cliente.

One-shot: se ejecuta solo cuando se hace swap del modelo de gustos.

Uso:
    python scripts/generate_gustos_thresholds.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


TFG_ROOT = Path("/Users/jezquerro/Documents/tfg")
SAMPLE_GUSTOS = TFG_ROOT / "data" / "data_qc_gustos" / "sample_user_ids_gustos.parquet"
CHARACTERS_CSV = TFG_ROOT / "data" / "data_raw" / "characters.csv"

DEPLOYMENT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (
    DEPLOYMENT_ROOT / "models" / "gustos_nivel1" / "v1_kmeans_k6_2026-05-12"
    / "thresholds.json"
)

_OID_RE = re.compile(r"[0-9a-f]{24}")


def _clean_uid(value):
    if pd.isna(value):
        return None
    m = _OID_RE.search(str(value))
    return m.group(0) if m else None


def main() -> int:
    if not SAMPLE_GUSTOS.exists():
        logger.error("❌ Sample no encontrado: %s", SAMPLE_GUSTOS)
        return 1
    if not CHARACTERS_CSV.exists():
        logger.error("❌ Characters CSV no encontrado: %s", CHARACTERS_CSV)
        return 1
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("GENERATE GUSTOS THRESHOLDS")
    logger.info("=" * 70)
    logger.info("  Sample:     %s", SAMPLE_GUSTOS)
    logger.info("  Characters: %s", CHARACTERS_CSV)
    logger.info("  Output:     %s", DEFAULT_OUTPUT)

    sample = pd.read_parquet(SAMPLE_GUSTOS)
    sample_ids = set(sample["user_id"].astype(str))
    logger.info("\nSample N: %d", len(sample_ids))

    logger.info("Cargando characters.csv (filtrado al sample)…")
    chars = pd.read_csv(CHARACTERS_CSV, usecols=["user_id", "level"], low_memory=False)
    chars["user_id"] = chars["user_id"].apply(_clean_uid)
    chars = chars[chars["user_id"].isin(sample_ids)].copy()
    logger.info("  N chars del sample: %d", len(chars))

    p75 = float(chars["level"].quantile(0.75))
    p50 = float(chars["level"].quantile(0.50))
    p90 = float(chars["level"].quantile(0.90))
    n = int(len(chars))
    logger.info("\np50: %.4f", p50)
    logger.info("p75: %.4f  ← persistido", p75)
    logger.info("p90: %.4f", p90)

    payload = {
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "sample_path": str(SAMPLE_GUSTOS),
            "characters_path": str(CHARACTERS_CSV),
            "n_users_in_sample": len(sample_ids),
            "n_chars_in_sample": n,
        },
        "chars_level_p75": p75,
        "chars_level_p50": p50,
        "chars_level_p90": p90,
        "notes": (
            "p75 calculado sobre chars del sample de gustos (114,412 users del TFG). "
            "Usado por `pct_chars_high_level` en feature_pipeline_derived.py "
            "para mantener consistencia con el training del modelo gustos_nivel1."
        ),
    }
    DEFAULT_OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info("\n✓ %s escrito", DEFAULT_OUTPUT.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
