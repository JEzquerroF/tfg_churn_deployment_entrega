"""
lookup.py — combina predicciones live + OOF con detección de drift.

Para cada usuario y cada target:
  - Si NO hay OOF (jugador nuevo)             → final = live, source = 'live_new'
  - Si HAY OOF y |live - OOF| <= threshold    → final = OOF,  source = 'oof_stable'
  - Si HAY OOF y |live - OOF| >  threshold    → final = live, source = 'live_drift'

El output incluye SIEMPRE: live, oof (NaN si nuevo), delta (NaN si nuevo),
source, final por cada target.

El sistema NO binariza. La conversión a 0/1 y el risk_level se aplican
downstream (frontend D1b) según los umbrales que el usuario configure.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import pandas as pd
import yaml

from pipeline.model_loader import ModelRegistry

logger = logging.getLogger(__name__)


def _target_suffix(target: str) -> str:
    """`churn_30d` -> `30d`."""
    return target.split("_", 1)[1]


def apply_oof_lookup_with_drift(
    predictions_live: pd.DataFrame,
    registry: ModelRegistry,
    targets: Sequence[str],
    drift_threshold: float = 0.10,
) -> pd.DataFrame:
    """
    Combina predicciones live + OOF con detección de drift por usuario y target.

    Args:
        predictions_live: DataFrame con cols `user_id` + `churn_prob_{suffix}`
            para cada target.
        registry: con OOF lookup cargado (puede devolver NaN para usuarios sin OOF).
        targets: lista de targets (ej. ['churn_7d', 'churn_14d', 'churn_30d']).
        drift_threshold: umbral absoluto sobre `|live - oof|`. Por encima del
            umbral, la OOF se considera obsoleta y se sirve la live.

    Returns:
        DataFrame con columnas (por cada target):
          - {target}_live:   predicción del modelo en vivo
          - {target}_oof:    OOF guardada (NaN si jugador nuevo)
          - {target}_delta:  live - oof (NaN si jugador nuevo)
          - {target}_source: 'oof_stable' | 'live_drift' | 'live_new'
          - {target}_final:  decisión aplicada (OOF si estable, live en otro caso)
        Plus la columna `user_id` original.

    Convención de boundary:
        |delta| == threshold se considera ESTABLE (sirve OOF). Solo
        `> threshold` triggea drift.
    """
    targets = list(targets)
    oof_lookup = registry.lookup_oof(predictions_live["user_id"])

    # Renombrar cols live `churn_prob_{suffix}` → `{target}_live` para evitar
    # colisiones con las cols oof que también empiezan por `churn_prob_`.
    live_renamed = predictions_live.rename(
        columns={f"churn_prob_{_target_suffix(t)}": f"{t}_live" for t in targets}
    )
    # OOF: `churn_prob_{suffix}_oof` → `{target}_oof`.
    oof_renamed = oof_lookup.rename(
        columns={f"churn_prob_{_target_suffix(t)}_oof": f"{t}_oof" for t in targets}
    )

    merged = live_renamed.merge(oof_renamed, on="user_id", how="left")

    for t in targets:
        live_col = f"{t}_live"
        oof_col = f"{t}_oof"
        delta_col = f"{t}_delta"
        source_col = f"{t}_source"
        final_col = f"{t}_final"

        if live_col not in merged.columns:
            logger.warning("Falta %s en predicciones live, skipping target %s", live_col, t)
            continue
        if oof_col not in merged.columns:
            # OOF lookup no devolvió esta col → todos los usuarios sin OOF.
            merged[oof_col] = np.nan

        # Delta: live - oof (NaN propagated)
        merged[delta_col] = merged[live_col] - merged[oof_col]

        # Clasificación per-usuario
        has_oof = merged[oof_col].notna()
        abs_delta = merged[delta_col].abs()
        stable = has_oof & (abs_delta <= drift_threshold)
        drift = has_oof & (abs_delta > drift_threshold)
        new_user = ~has_oof

        merged[source_col] = np.select(
            [stable, drift, new_user],
            ["oof_stable", "live_drift", "live_new"],
            default="unknown",
        )
        # Final: OOF si estable, live en cualquier otro caso (drift o new)
        merged[final_col] = np.where(stable, merged[oof_col], merged[live_col])

    # Logging de distribución por target
    for t in targets:
        source_col = f"{t}_source"
        if source_col in merged.columns:
            counts = merged[source_col].value_counts().to_dict()
            logger.info(
                "%s: stable=%d, drift=%d, new=%d",
                t,
                counts.get("oof_stable", 0),
                counts.get("live_drift", 0),
                counts.get("live_new", 0),
            )

    return merged


def load_drift_threshold(thresholds_path: Union[str, Path]) -> float:
    """Lee drift_threshold de thresholds.yaml. Default 0.10 si falta."""
    with open(thresholds_path) as f:
        cfg = yaml.safe_load(f) or {}
    return float(cfg.get("drift_detection", {}).get("threshold", 0.10))


def apply_oof_lookup(*args, **kwargs):
    """
    DEPRECATED (Fase 1.5): la lógica antigua sustituía live por OOF de forma
    incondicional cuando había OOF para el usuario. Esto enmascaraba cualquier
    drift de comportamiento del jugador desde el training original.

    Usar `apply_oof_lookup_with_drift` que devuelve live + oof + delta + source
    + final por target y aplica una regla de drift configurable.
    """
    warnings.warn(
        "apply_oof_lookup está deprecada desde Fase 1.5. "
        "Usar apply_oof_lookup_with_drift.",
        DeprecationWarning,
        stacklevel=2,
    )
    raise DeprecationWarning(
        "apply_oof_lookup deprecated, usa apply_oof_lookup_with_drift"
    )
