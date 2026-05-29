"""
Master builder.

Dos masters separados (decisión Fase 2.2):
- `master_churn`: 50 cols, replica `02z_build_master_table.ipynb` del TFG.
  Lo consume el modelo de churn (RF L22 v1).
- `master_gustos`: 78 cols (master_churn + 28 features de gustos). Lo
  consume el modelo de clustering KMeans K=6.

`build_both_masters(ctx, sample, cutoff)` los construye en un solo pase
de I/O sobre los CSVs estables (los 8 dominios de churn) más los 3 CSVs
transaccionales (currency, fights, arena) que solo necesita gustos.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd

from .pipeline_context import PipelineContext
from . import feature_pipeline_users as fp_users
from . import feature_pipeline_characters as fp_chars
from . import feature_pipeline_devices as fp_devices
from . import feature_pipeline_iaps as fp_iaps
from . import feature_pipeline_rewards as fp_rewards
from . import feature_pipeline_items as fp_items
from . import feature_pipeline_collection as fp_coll
from . import feature_pipeline_feedback as fp_feedback
from . import feature_pipeline_currency as fp_currency
from . import feature_pipeline_fights as fp_fights
from . import feature_pipeline_arena as fp_arena
from . import feature_pipeline_derived as fp_derived

logger = logging.getLogger(__name__)


def load_all_raw_csvs(
    ctx: PipelineContext,
    sample_user_ids: Optional[set] = None,
) -> Dict[str, object]:
    """
    Carga todos los CSVs raw una sola vez. Para items y collection (~2GB cada uno)
    pasa sample_user_ids para pre-filtrar en streaming.
    """
    return {
        'users': fp_users.load_users_raw(ctx),
        'characters': fp_chars.load_characters_raw(ctx),
        'devices': fp_devices.load_devices_raw(ctx),
        'iaps': fp_iaps.load_iaps_raw(ctx),
        'rewards': fp_rewards.load_rewards_raw(ctx),
        'items': fp_items.load_items_raw(ctx, sample_user_ids=sample_user_ids),
        'collection': fp_coll.load_collection_raw(ctx, sample_user_ids=sample_user_ids),
        'feedback': fp_feedback.load_feedback_raw(ctx),
    }


def build_master_churn(
    ctx: PipelineContext,
    sample_user_ids: pd.DataFrame,
    cutoff_date: datetime,
    raw_dfs: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """
    Construye el master de CHURN (50 cols): los 8 dominios del TFG.

    DEPLOYMENT: en training del TFG el sample contenía user_id,
    days_since_last_login, player_lifespan_days, has_corrupted_dates,
    churn_14d, churn_30d. En deployment el sample NO trae churn_14d/30d
    (son el target a predecir, no labels). El reorden de target_cols al
    final del DataFrame (líneas finales) es tolerante a su ausencia
    (filtra por presencia en master.columns).
    """
    if raw_dfs is None:
        sample_ids = set(sample_user_ids['user_id'].astype(str))
        raw_dfs = load_all_raw_csvs(ctx, sample_user_ids=sample_ids)

    # Base = sample con renames
    base = sample_user_ids.copy()
    rename_sample = {
        'days_since_last_login': 'user_days_since_last_login',
        'player_lifespan_days': 'user_player_lifespan_days',
        'has_corrupted_dates': 'user_has_corrupted_dates',
    }
    base = base.rename(columns={k: v for k, v in rename_sample.items() if k in base.columns})

    base['user_id'] = base['user_id'].astype(str)
    base = base.set_index('user_id')

    # Features de cada dominio (indexed por user_id)
    feats_users = fp_users.compute_users_features(sample_user_ids, cutoff_date, raw_dfs.get('users'))
    feats_chars = fp_chars.compute_characters_features(sample_user_ids, cutoff_date, raw_dfs.get('characters'))
    feats_devices = fp_devices.compute_devices_features(sample_user_ids, cutoff_date, raw_dfs.get('devices'))
    feats_iaps = fp_iaps.compute_iaps_features(sample_user_ids, cutoff_date, raw_dfs.get('iaps'))
    feats_rewards = fp_rewards.compute_rewards_features(sample_user_ids, cutoff_date, raw_dfs.get('rewards'))
    feats_items = fp_items.compute_items_features(sample_user_ids, cutoff_date, raw_dfs.get('items'))
    feats_coll = fp_coll.compute_collection_features(sample_user_ids, cutoff_date, raw_dfs.get('collection'))
    feats_feedback = fp_feedback.compute_feedback_features(sample_user_ids, cutoff_date, raw_dfs.get('feedback'))

    # Join todos sobre base
    master = base.copy()
    for feats, name in [
        (feats_users, 'users'),
        (feats_chars, 'characters'),
        (feats_devices, 'devices'),
        (feats_iaps, 'iaps'),
        (feats_rewards, 'rewards'),
        (feats_items, 'items'),
        (feats_coll, 'collection'),
        (feats_feedback, 'feedback'),
    ]:
        if feats is None or feats.empty:
            continue
        n_before = len(master)
        # Asegurar índice consistente
        feats = feats.copy()
        feats.index = feats.index.astype(str)
        feats.index.name = 'user_id'
        # Evitar colisiones de columnas con base
        dup_cols = [c for c in feats.columns if c in master.columns]
        if dup_cols:
            feats = feats.drop(columns=dup_cols)
        master = master.join(feats, how='left')
        assert len(master) == n_before, f"JOIN {name} cambió shape: {n_before} → {len(master)}"

    # Reordenar: user_id (index→col) primero, targets al final
    master = master.reset_index()
    target_cols = [c for c in ['churn_14d', 'churn_30d'] if c in master.columns]
    other_cols = [c for c in master.columns if c not in (['user_id'] + target_cols)]
    master = master[['user_id'] + other_cols + target_cols]

    return master


# Alias retrocompatible para call sites antiguos.
build_master = build_master_churn


def _extend_to_gustos(
    master_churn: pd.DataFrame,
    sample_user_ids: pd.DataFrame,
    ctx: PipelineContext,
    raw_dfs: Dict[str, object],
) -> pd.DataFrame:
    """
    Extiende `master_churn` con las 28 features que el modelo gustos_nivel1
    espera (Fase 2.2). Devuelve el master de GUSTOS.

    El char_to_user mapping se construye una sola vez aquí (a partir de
    raw_dfs['characters']) y se reutiliza en fights + arena.
    """
    sample_ids_series = sample_user_ids["user_id"].astype(str)

    # Construir char_to_user para fights + arena
    char_to_user = fp_fights.build_char_to_user(raw_dfs["characters"])

    # 1) Features derivadas (sin nuevos CSVs)
    logger.info("[gustos] computing derived features…")
    feats_derived = fp_derived.compute(
        master_churn=master_churn,
        sample_user_ids=sample_ids_series,
        ctx=ctx,
        raw_dfs=raw_dfs,
    )

    # 2) Currency (CSV nuevo, ~574 MB)
    logger.info("[gustos] computing currency features…")
    feats_currency = fp_currency.compute(ctx, sample_ids_series)

    # 3) Fights (CSV ENORME, ~28 GB) — cuello de botella
    logger.info("[gustos] computing fights features…")
    feats_fights = fp_fights.compute(ctx, sample_ids_series, char_to_user)

    # 4) Arena (CSV pequeño, ~71 MB)
    logger.info("[gustos] computing arena features…")
    feats_arena = fp_arena.compute(ctx, sample_ids_series, char_to_user)

    # Merge incremental sobre master_churn evitando colisiones de columnas
    master = master_churn.copy()
    master["user_id"] = master["user_id"].astype(str)
    for feats_df, label in [
        (feats_derived, "derived"),
        (feats_currency, "currency"),
        (feats_fights, "fights"),
        (feats_arena, "arena"),
    ]:
        if feats_df is None or feats_df.empty:
            logger.warning("[gustos] %s sin features (df vacío)", label)
            continue
        feats_df = feats_df.copy()
        feats_df["user_id"] = feats_df["user_id"].astype(str)
        # Drop cols ya presentes en master (excepto user_id, key del merge)
        dup = [c for c in feats_df.columns if c in master.columns and c != "user_id"]
        if dup:
            feats_df = feats_df.drop(columns=dup)
        n_before = len(master)
        master = master.merge(feats_df, on="user_id", how="left")
        assert len(master) == n_before, (
            f"MERGE {label} cambió shape: {n_before} → {len(master)}"
        )
        logger.info(
            "[gustos] +%d cols tras merge %s → master %s",
            feats_df.shape[1] - 1,  # menos user_id
            label,
            master.shape,
        )

    return master


def build_master_gustos(
    ctx: PipelineContext,
    sample_user_ids: pd.DataFrame,
    cutoff_date: datetime,
    raw_dfs: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """
    Construye el master de GUSTOS (78 cols): churn + 28 features de gustos.

    Si `raw_dfs` no se pasa, los carga internamente. El cliente típico
    usa `build_both_masters(...)` para evitar I/O duplicado.
    """
    if raw_dfs is None:
        sample_ids = set(sample_user_ids["user_id"].astype(str))
        raw_dfs = load_all_raw_csvs(ctx, sample_user_ids=sample_ids)

    master_churn = build_master_churn(ctx, sample_user_ids, cutoff_date, raw_dfs=raw_dfs)
    return _extend_to_gustos(master_churn, sample_user_ids, ctx, raw_dfs)


def build_both_masters(
    ctx: PipelineContext,
    sample_user_ids: pd.DataFrame,
    cutoff_date: datetime,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construye `master_churn` y `master_gustos` en un solo pase de I/O.

    Recomendado para `predict.py`: evita cargar dos veces los CSVs estables.

    Returns:
        (master_churn, master_gustos)
    """
    sample_ids = set(sample_user_ids["user_id"].astype(str))
    raw_dfs = load_all_raw_csvs(ctx, sample_user_ids=sample_ids)
    master_churn = build_master_churn(ctx, sample_user_ids, cutoff_date, raw_dfs=raw_dfs)
    master_gustos = _extend_to_gustos(master_churn, sample_user_ids, ctx, raw_dfs)
    return master_churn, master_gustos
