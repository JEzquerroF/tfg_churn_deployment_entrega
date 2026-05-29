"""
crossing.py — cruce churn × arquetipo + segmentación por prioridad.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def cross_and_segment(
    predictions: pd.DataFrame,
    archetypes: pd.DataFrame,
    thresholds_path: Union[str, Path],
    archetypes_path: Union[str, Path],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Une predictions + archetypes y aplica reglas de segmentación.

    Args:
        predictions: user_id, churn_prob_14d, churn_prob_30d, source
        archetypes: user_id, archetype_n1
        thresholds_path: ruta a thresholds.yaml
        archetypes_path: ruta a archetypes.yaml

    Returns:
        (predictions_full, segmentation):
        - predictions_full: user_id, churn_prob_14d, churn_prob_30d, archetype_n1,
          archetype_name, source
        - segmentation: user_id, risk_level, segment, priority, label
    """
    with open(thresholds_path) as f:
        thresholds = yaml.safe_load(f)
    with open(archetypes_path) as f:
        archetypes_cfg = yaml.safe_load(f)

    full = predictions.merge(archetypes, on="user_id", how="inner")

    archetypes_n1 = archetypes_cfg["archetypes_n1"]
    full["archetype_name"] = full["archetype_n1"].apply(
        lambda x: archetypes_n1.get(int(x), {}).get("name", "Unknown")
    )

    def assign_risk(prob: float) -> str:
        for name, bounds in thresholds["risk_levels"].items():
            if bounds["min"] <= prob < bounds["max"]:
                return name
        return "very_low"

    full["risk_level"] = full["churn_prob_30d"].apply(assign_risk)

    def apply_priority(row):
        risk = row["risk_level"]
        archetype = int(row["archetype_n1"])
        for rule in thresholds["priority_rules"]:
            cond = rule["if"]
            risk_match = (
                cond["risk"] == risk
                if isinstance(cond["risk"], str)
                else risk in cond["risk"]
            )
            arch_match = True
            if "archetypes" in cond:
                arch_match = archetype in cond["archetypes"]
            if risk_match and arch_match:
                return pd.Series({
                    "priority": rule["priority"],
                    "label": rule["label"],
                })
        return pd.Series({"priority": "P4", "label": "Monitorización pasiva"})

    priority_cols = full.apply(apply_priority, axis=1)

    segmentation = pd.DataFrame({
        "user_id": full["user_id"],
        "risk_level": full["risk_level"],
        "segment": full["archetype_name"],
        "priority": priority_cols["priority"],
        "label": priority_cols["label"],
    })

    # Incluir dinámicamente todas las cols churn_prob_* (e.g. 7d/14d/30d en RF L22)
    prob_cols = [c for c in predictions.columns if c.startswith("churn_prob_")]
    predictions_full = full[
        ["user_id"]
        + prob_cols
        + ["archetype_n1", "archetype_name", "source"]
    ]

    logger.info("Segmentación generada: %d usuarios", len(segmentation))
    logger.info("Distribución de prioridad: %s", segmentation["priority"].value_counts().to_dict())

    return predictions_full, segmentation
