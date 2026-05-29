"""Feature pipeline para items collection — replica de 02l_user_items_collection.ipynb (7 features coll_*)."""

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .pipeline_context import PipelineContext


_OID_RE = re.compile(r"ObjectId\(?'?([a-f0-9]+)'?\)?")

USECOLS = ['user_id', 'item_definition_excel_id', 'updated_at', 'created_at']


def _clean_uid(uid):
    if pd.isna(uid):
        return None
    s = str(uid)
    m = _OID_RE.search(s)
    if m:
        return m.group(1)
    return s


def load_collection_raw(
    ctx: PipelineContext,
    sample_user_ids: Optional[set] = None,
    path: Optional[Path] = None,
) -> pd.DataFrame:
    if path is None:
        path = ctx.raw_csvs_dir / "user_items_collection.csv"
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


def compute_collection_features(
    sample_user_ids: pd.DataFrame,
    cutoff_date: datetime,
    raw_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    sample_ids = set(sample_user_ids['user_id'].astype(str))
    cutoff_ts = pd.Timestamp(cutoff_date).tz_localize(None) if pd.Timestamp(cutoff_date).tz else pd.Timestamp(cutoff_date)
    cutoff_30d = cutoff_ts - timedelta(days=30)
    cutoff_90d = cutoff_ts - timedelta(days=90)

    if raw_df is None:
        df = load_collection_raw(sample_user_ids=sample_ids)
    else:
        df = raw_df[raw_df['user_id'].astype(str).isin(sample_ids)].copy()
        if 'created_dt' not in df.columns:
            df['created_dt'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True).dt.tz_localize(None)

    df = df[df['created_dt'].notna() & (df['created_dt'] <= cutoff_ts)].copy()

    if df.empty:
        empty_idx = pd.Index(sample_user_ids['user_id'].astype(str).values, name='user_id')
        return pd.DataFrame({
            'coll_total_items': 0,
            'coll_collection_span_days': 0,
            'coll_days_since_last_item': 9999,
            'coll_items_last_30d': 0,
            'coll_items_last_90d': 0,
            'coll_unique_families': 0,
        }, index=empty_idx)

    df['_family'] = (df['item_definition_excel_id'] // 100).astype('int32')
    df['_in_30d'] = (df['created_dt'] >= cutoff_30d)
    df['_in_90d'] = (df['created_dt'] >= cutoff_90d)

    agg = df.groupby('user_id').agg(
        coll_total_items=('item_definition_excel_id', 'size'),
        _first_item_at=('created_dt', 'min'),
        _last_item_at=('created_dt', 'max'),
        coll_items_last_30d=('_in_30d', 'sum'),
        coll_items_last_90d=('_in_90d', 'sum'),
        coll_unique_families=('_family', 'nunique'),
    )

    span = (agg['_last_item_at'] - agg['_first_item_at']).dt.total_seconds() / 86400
    agg['coll_collection_span_days'] = span.fillna(0).round(0).astype('int32')

    days_last = (cutoff_ts - agg['_last_item_at']).dt.total_seconds() / 86400
    agg['coll_days_since_last_item'] = days_last.where(agg['_last_item_at'].notna(), 9999).round(0).astype('int32')

    days_first = (cutoff_ts - agg['_first_item_at']).dt.total_seconds() / 86400
    agg['coll_first_item_days_ago'] = days_first.round(0).astype('Int32')
    agg.loc[agg['coll_first_item_days_ago'] < 0, 'coll_first_item_days_ago'] = pd.NA

    features = agg.drop(columns=['_first_item_at', '_last_item_at'])
    features = features.reindex(sample_user_ids['user_id'].astype(str).values)

    for c in ['coll_total_items', 'coll_collection_span_days', 'coll_days_since_last_item',
              'coll_items_last_30d', 'coll_items_last_90d']:
        features[c] = features[c].fillna(0 if c != 'coll_days_since_last_item' else 9999).astype('int32')
    features['coll_unique_families'] = features['coll_unique_families'].fillna(0).astype('int16')

    return features
