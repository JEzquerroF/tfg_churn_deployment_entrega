"""
target_encoder.py — aplicación declarativa de target encoding.

Lee mappings desde JSON serializado en
`models/<modelo>/target_encoder_mappings.json` y las aplica a un DataFrame
en inference.

La serialización es JSON puro (no pickle). La lógica de aplicación vive
aquí, no embebida en el artifact serializado, lo cual hace la transformación
auditable y diffeable.

Estructura del JSON esperado:

    {
        "smoothing": 10.0,
        "cat_cols": ["country", "has_user_rated_app", ...],
        "missing_sentinel": "__missing__",
        "per_target": {
            "churn_7d": {
                "global_mean": 0.1234,
                "mappings": {
                    "country": {"Spain": 0.13, "Brazil": 0.21, ...},
                    "has_user_rated_app": {"True": 0.10, "False": 0.18, ...},
                    ...
                }
            },
            "churn_14d": {...},
            "churn_30d": {...}
        }
    }

La cadena de transformación replica EXACTAMENTE la del training del TFG
(`data_prep_production.py:target_encode_cv`):

    X[col].astype(object).fillna('__missing__').astype(str).map(mapping).fillna(global_mean)

Cualquier cambio en el orden o tipo intermedio puede hacer que keys del
mapping no coincidan con los valores en inference (ej. "True" vs "true").
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TargetEncoderMappings:
    """Mappings de target encoding reconstruidas desde el train del TFG."""

    smoothing: float
    cat_cols: list[str]
    missing_sentinel: str
    per_target: dict[str, dict]  # target → {global_mean, mappings: {col: {cat: val}}}

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "TargetEncoderMappings":
        with open(path) as f:
            data = json.load(f)
        return cls(
            smoothing=data["smoothing"],
            cat_cols=list(data["cat_cols"]),
            missing_sentinel=data["missing_sentinel"],
            per_target=data["per_target"],
        )

    def transform(self, X: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Aplica el target encoding a las cat_cols del DataFrame para el target dado.

        Devuelve un DataFrame con las mismas columnas, pero con cat_cols convertidas
        de string/object a float (su mean encoding).
        """
        if target not in self.per_target:
            raise KeyError(
                f"Target '{target}' no tiene mappings. "
                f"Disponibles: {list(self.per_target.keys())}"
            )

        cfg = self.per_target[target]
        global_mean = cfg["global_mean"]
        col_mappings = cfg["mappings"]

        X_out = X.copy()
        for col in self.cat_cols:
            if col not in X_out.columns:
                logger.warning("cat_col '%s' no en X, skipping", col)
                continue
            X_out[col] = (
                X_out[col]
                .astype(object)
                .fillna(self.missing_sentinel)
                .astype(str)
                .map(col_mappings[col])
                .fillna(global_mean)
                .astype(float)
            )
        return X_out

    def n_mappings(self, target: str, col: str) -> int:
        """Número de claves en el mapping de (target, col). Útil para sanity checks."""
        return len(self.per_target[target]["mappings"][col])
