"""Feature pipeline para items — replica de 02m_user_items.ipynb (12 features items_*)."""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .pipeline_context import PipelineContext


_OID_RE = re.compile(r"ObjectId\(?'?([a-f0-9]+)'?\)?")

USECOLS = [
    'user_id', 'item_definition_excel_id',
    'c_base_critical_chance', 'c_base_attack_enhanced', 'c_base_defense_enhanced',
    'enhance_level', 'tempering_level', 'max_enhance_level',
    'updated_at', 'created_at',
]


def _clean_uid(uid):
    if pd.isna(uid):
        return None
    s = str(uid)
    m = _OID_RE.search(s)
    if m:
        return m.group(1)
    return s


def load_items_raw(
    ctx: PipelineContext,
    sample_user_ids: Optional[set] = None,
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Carga user_items.csv. ~2GB; lectura chunked con pre-filtrado opcional al sample.
    """
    if path is None:
        path = ctx.raw_csvs_dir / "user_items.csv"
    chunks = []
    for chunk in pd.read_csv(path, usecols=USECOLS, chunksize=500_000, low_memory=False):
        chunk['user_id'] = chunk['user_id'].apply(_clean_uid)
        chunk = chunk.dropna(subset=['user_id'])
        if sample_user_ids is not None:
            chunk = chunk[chunk['user_id'].astype(str).isin(sample_user_ids)]
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(columns=USECOLS + ['created_dt'])

    df = pd.concat(chunks, ignore_index=True)
    df['created_dt'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True).dt.tz_localize(None)
    return df


def compute_items_features(
    sample_user_ids: pd.DataFrame,
    cutoff_date: datetime,
    raw_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    sample_ids = set(sample_user_ids['user_id'].astype(str))
    cutoff_ts = pd.Timestamp(cutoff_date).tz_localize(None) if pd.Timestamp(cutoff_date).tz else pd.Timestamp(cutoff_date)

    if raw_df is None:
        df = load_items_raw(sample_user_ids=sample_ids)
    else:
        df = raw_df[raw_df['user_id'].astype(str).isin(sample_ids)].copy()
        if 'created_dt' not in df.columns:
            df['created_dt'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True).dt.tz_localize(None)

    df = df[df['created_dt'].notna() & (df['created_dt'] <= cutoff_ts)].copy()

    if df.empty:
        empty_idx = pd.Index(sample_user_ids['user_id'].astype(str).values, name='user_id')
        return pd.DataFrame({
            'items_total_instances': 0,
            'items_unique_definitions': 0,
            'items_total_enhance_invested': 0,
            'items_days_since_last_item': 9999,
        }, index=empty_idx).astype('int32')

    # Agregaciones
    agg = df.groupby('user_id').agg(
        items_total_instances=('item_definition_excel_id', 'size'),
        items_unique_definitions=('item_definition_excel_id', 'nunique'),
        items_mean_enhance_level=('enhance_level', 'mean'),
        items_max_enhance_level=('enhance_level', 'max'),
        items_total_enhance_invested=('enhance_level', 'sum'),
        items_mean_attack=('c_base_attack_enhanced', 'mean'),
        items_mean_defense=('c_base_defense_enhanced', 'mean'),
        items_mean_critical=('c_base_critical_chance', 'mean'),
        _last_item_at=('created_dt', 'max'),
        _first_item_at=('created_dt', 'min'),
    )

    # ratios
    agg['items_attack_defense_ratio'] = np.where(
        agg['items_mean_defense'].notna() & (agg['items_mean_defense'] > 0),
        agg['items_mean_attack'] / agg['items_mean_defense'],
        np.nan,
    )
    agg['items_redundancy_ratio'] = np.where(
        agg['items_unique_definitions'] > 0,
        agg['items_total_instances'] / agg['items_unique_definitions'],
        np.nan,
    )

    # *_days_ago al cutoff
    days_last = (cutoff_ts - agg['_last_item_at']).dt.total_seconds() / 86400
    days_first = (cutoff_ts - agg['_first_item_at']).dt.total_seconds() / 86400
    agg['items_days_since_last_item'] = days_last.where(agg['_last_item_at'].notna(), 9999).round(0).astype('int32')
    agg['items_first_item_days_ago'] = days_first.round(0).astype('Int32')
    agg.loc[agg['items_first_item_days_ago'] < 0, 'items_first_item_days_ago'] = pd.NA

    features = agg.drop(columns=['_first_item_at', '_last_item_at'])

    # Reindex al sample
    features = features.reindex(sample_user_ids['user_id'].astype(str).values)

    # Fillna: counts → 0, ratios/means → NaN, days_since → 9999, first_days_ago → NaN
    for c in ['items_total_instances', 'items_unique_definitions', 'items_total_enhance_invested']:
        features[c] = features[c].fillna(0).astype('int32')

    features['items_days_since_last_item'] = features['items_days_since_last_item'].fillna(9999).astype('int32')

    # Float32 para means/ratios
    for c in ['items_mean_enhance_level', 'items_max_enhance_level', 'items_mean_attack',
              'items_mean_defense', 'items_mean_critical', 'items_attack_defense_ratio',
              'items_redundancy_ratio']:
        features[c] = features[c].astype('float32')

    return features
