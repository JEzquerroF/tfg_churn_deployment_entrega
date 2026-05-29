"""Feature pipeline para devices — replica de 02g_devices.ipynb (9 features)."""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .pipeline_context import PipelineContext


_OID_RE = re.compile(r"ObjectId\(?'?([a-f0-9]+)'?\)?")


def _clean_uid(uid):
    if pd.isna(uid):
        return None
    s = str(uid)
    m = _OID_RE.search(s)
    if m:
        return m.group(1)
    return s


def load_devices_raw(ctx: PipelineContext, path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        path = ctx.raw_csvs_dir / "devices.csv"
    df = pd.read_csv(path, low_memory=False)
    df['user_id'] = df['user_id'].apply(_clean_uid)
    df = df.dropna(subset=['user_id']).copy()
    df['created_dt'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True).dt.tz_localize(None)
    df['updated_dt'] = pd.to_datetime(df['updated_at'], errors='coerce', utc=True).dt.tz_localize(None)
    return df


def compute_devices_features(
    sample_user_ids: pd.DataFrame,
    cutoff_date: datetime,
    raw_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if raw_df is None:
        raw_df = load_devices_raw()

    sample_ids = set(sample_user_ids['user_id'].astype(str))
    cutoff_ts = pd.Timestamp(cutoff_date).tz_localize(None) if pd.Timestamp(cutoff_date).tz else pd.Timestamp(cutoff_date)

    df = raw_df[raw_df['user_id'].astype(str).isin(sample_ids)].copy()
    # Records pre-cutoff (por created_dt; updated_dt se usa para last_active)
    df = df[df['created_dt'].notna() & (df['created_dt'] <= cutoff_ts)].copy()
    # Clip updated_dt al cutoff (no se puede saber post-cutoff)
    df['updated_dt_clip'] = df['updated_dt'].clip(upper=cutoff_ts)

    # Grupo A: volumen
    grp_a = df.groupby('user_id').agg(
        device_records_total=('_id' if '_id' in df.columns else 'user_id', 'size'),
        device_unique_models=('device_model', 'nunique'),
        device_platform_count=('platform', 'nunique'),
    )

    # Grupo B: platform flags
    users_android = set(df.loc[df['platform'] == 'android', 'user_id'].astype(str))
    users_ios = set(df.loc[df['platform'] == 'ios', 'user_id'].astype(str))

    grp_b = pd.DataFrame(index=grp_a.index)
    grp_b['device_has_android'] = grp_b.index.astype(str).isin(users_android).astype(int)
    grp_b['device_has_ios'] = grp_b.index.astype(str).isin(users_ios).astype(int)
    grp_b['device_is_multi_platform'] = (
        (grp_b['device_has_android'] == 1) & (grp_b['device_has_ios'] == 1)
    ).astype(int)

    # primary_platform: el de más records, alfabético en empate
    pcounts = df.groupby(['user_id', 'platform']).size().reset_index(name='_n')
    pcounts = pcounts.sort_values(['user_id', '_n', 'platform'], ascending=[True, False, True])
    primary = pcounts.drop_duplicates(subset='user_id', keep='first').set_index('user_id')['platform']
    grp_b['device_primary_platform'] = primary

    # Grupo C: temporales
    grp_c = df.groupby('user_id').agg(
        _first_seen_dt=('created_dt', 'min'),
        _last_active_dt=('updated_dt_clip', 'max'),
    )
    days_first = (cutoff_ts - grp_c['_first_seen_dt']).dt.total_seconds() / 86400
    grp_c['device_first_seen_days_ago'] = days_first.round(0).astype('Int32')
    grp_c.loc[grp_c['device_first_seen_days_ago'] < 0, 'device_first_seen_days_ago'] = pd.NA

    days_last = (cutoff_ts - grp_c['_last_active_dt']).dt.total_seconds() / 86400
    grp_c['device_days_since_last_active'] = days_last.clip(lower=0).round(0).astype('int64')
    grp_c = grp_c.drop(columns=['_first_seen_dt', '_last_active_dt'])

    # Combinar
    features = pd.concat([grp_a, grp_b, grp_c], axis=1)

    # Reindex al sample
    features = features.reindex(sample_user_ids['user_id'].astype(str).values)

    # Fillna
    int_zero_cols = ['device_records_total', 'device_unique_models', 'device_platform_count',
                     'device_has_android', 'device_has_ios', 'device_is_multi_platform']
    for c in int_zero_cols:
        features[c] = features[c].fillna(0).astype('int64')

    features['device_primary_platform'] = features['device_primary_platform'].fillna('none').astype(str)
    features['device_days_since_last_active'] = features['device_days_since_last_active'].fillna(9999).astype('int64')
    # device_first_seen_days_ago se queda como Int32 nullable

    return features
