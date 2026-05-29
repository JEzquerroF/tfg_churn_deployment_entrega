"""
prediction.py — aplica los modelos de churn al master.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import pandas as pd

from pipeline.model_loader import ModelRegistry

logger = logging.getLogger(__name__)


def _target_suffix(target: str) -> str:
    """`churn_30d` -> `30d`."""
    return target.split("_", 1)[1]


def predict_churn_for_all_users(
    master: pd.DataFrame,
    registry: ModelRegistry,
    targets: Optional[Sequence[str]] = None,
    apply_calibration: bool = True,
) -> pd.DataFrame:
    """
    Aplica los modelos de churn (todos los targets configurados) sobre el master.

    Args:
        master: DataFrame con user_id + features esperadas por el modelo
        registry: ModelRegistry con los churn targets cargados
        targets: lista de targets a predecir (e.g. ['churn_7d', 'churn_14d', 'churn_30d']).
            Si None, se usan todos los churn_* presentes en `registry._artifacts`.
        apply_calibration: si True y existe calibrador, lo aplica.

    Returns:
        DataFrame con cols: user_id, churn_prob_<suffix> por target, source.
        source = 'live_model'; en lookup.py se sustituye por 'oof' donde aplique.
    """
    if "user_id" not in master.columns:
        raise ValueError("master debe tener columna 'user_id'")

    if targets is None:
        targets = [t for t in registry._artifacts if t.startswith("churn_")]
        if not targets:
            raise ValueError("No hay targets de churn cargados en el registry")

    user_ids = master["user_id"].copy()
    X = master.drop(columns=["user_id"])

    out = pd.DataFrame({"user_id": user_ids})
    for target in targets:
        logger.info("Prediciendo %s sobre %d usuarios…", target, len(X))
        probs = registry.predict_churn(X, target=target, apply_calibration=apply_calibration)
        out[f"churn_prob_{_target_suffix(target)}"] = probs

    out["source"] = "live_model"
    return out
