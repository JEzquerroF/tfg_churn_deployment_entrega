"""
clustering.py — asigna el arquetipo N1 (KMeans K=6) a todos los jugadores.

El antiguo Nivel 2 (HDBSCAN de sub-arquetipos) se eliminó: nunca fue
operacional en producción. La capa de personalización ahora es el perfilado
de gustos (pipeline/perfilado.py), que se aplica tras el clustering N1.
"""

from __future__ import annotations

import logging

import pandas as pd

from pipeline.model_loader import ModelRegistry

logger = logging.getLogger(__name__)


def assign_archetypes(
    master_tier1: pd.DataFrame,
    registry: ModelRegistry,
) -> pd.DataFrame:
    """
    Asigna arquetipo N1 (KMeans) a todos los jugadores.

    Args:
        master_tier1: DataFrame con user_id + features Tier 1 (todos los usuarios).
        registry: ModelRegistry con gustos_nivel1 cargado.

    Returns:
        DataFrame: user_id, archetype_n1.
    """
    user_ids = master_tier1["user_id"].copy()
    X_tier1 = master_tier1.drop(columns=["user_id"])

    logger.info("Asignando arquetipo N1 (KMeans) sobre %d usuarios…", len(X_tier1))
    archetypes_n1 = registry.assign_archetype_n1(X_tier1)

    return pd.DataFrame({
        "user_id": user_ids,
        "archetype_n1": archetypes_n1,
    })
