"""
Feature pipeline para users — replica parametrizada de 02a_users.ipynb del TFG.

A diferencia del notebook original, el filtrado del sample ya está hecho en
sample_generation.py. Aquí solo extraemos features estáticas/casi-estáticas
del CSV de users y las parametrizamos por cutoff_date.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .pipeline_context import PipelineContext


_OID_RE = re.compile(r"ObjectId\(?'?([a-f0-9]+)'?\)?")


def _extract_user_id(value):
    if pd.isna(value):
        return None
    s = str(value)
    m = _OID_RE.search(s)
    if m:
        return m.group(1)
    if len(s) == 24 and all(c in '0123456789abcdef' for c in s):
        return s
    return None


def load_users_raw(ctx: PipelineContext, path: Optional[Path] = None) -> pd.DataFrame:
    """Carga users.csv y devuelve dataframe con user_id extraído y fechas parseadas."""
    if path is None:
        path = ctx.raw_csvs_dir / "users.csv"
    df = pd.read_csv(path, low_memory=False)

    _id_col = '_id' if '_id' in df.columns else df.columns[0]
    df['user_id'] = df[_id_col].apply(_extract_user_id)

    if 'created_at' in df.columns:
        df['created_at_dt'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True).dt.tz_localize(None)
    else:
        df['created_at_dt'] = pd.NaT

    if 'last_login_date' in df.columns:
        df['last_login_dt'] = pd.to_datetime(df['last_login_date'], unit='s', errors='coerce')
    else:
        df['last_login_dt'] = pd.NaT

    if 'updated_at' in df.columns:
        df['updated_at_dt'] = pd.to_datetime(df['updated_at'], errors='coerce', utc=True).dt.tz_localize(None)
    else:
        df['updated_at_dt'] = pd.NaT

    return df.dropna(subset=['user_id'])


def compute_users_features(
    sample_user_ids: pd.DataFrame,
    cutoff_date: datetime,
    raw_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Features de users (estáticas del CSV + days_ago al cutoff).

    Devuelve DataFrame indexed por user_id (no incluye user_id como columna).
    """
    if raw_df is None:
        raw_df = load_users_raw()

    sample_ids = set(sample_user_ids['user_id'].astype(str))
    df = raw_df[raw_df['user_id'].isin(sample_ids)].copy()

    # Columnas pass-through (KEEP_AS_IS)
    keep_as_is = ['country', 'language', 'is_google_play', 'tutorial_done',
                  'has_user_rated_app', 'focus']

    # Columnas a prefijar con user_
    rename_map = {
        'current_character': 'user_current_character',
        'current_session': 'user_current_session',
        'dark_steel': 'user_dark_steel',
        'gold': 'user_gold',
        'gems': 'user_gems',
        'runes': 'user_runes',
        'game_version': 'user_game_version',
        'last_completed_tutorial_block': 'user_last_completed_tutorial_block',
        'num_logins': 'user_num_logins',
        'store_where_published': 'user_store_where_published',
        'template_item_stats_augment_update_done': 'user_template_item_stats_augment_update_done',
        'created_at': 'user_created_at',
        'updated_at': 'user_updated_at',
        'last_login_date': 'user_last_login_date',
    }

    cols_present = [c for c in keep_as_is + list(rename_map.keys()) if c in df.columns]
    out = df.set_index('user_id')[cols_present + ['created_at_dt', 'last_login_dt']].copy()
    out = out.rename(columns=rename_map)

    # *_days_ago al cutoff (Int32 nullable)
    cutoff_ts = pd.Timestamp(cutoff_date)
    if hasattr(cutoff_ts, 'tz') and cutoff_ts.tz is not None:
        cutoff_ts = cutoff_ts.tz_localize(None)

    days_created = (cutoff_ts - out['created_at_dt']).dt.total_seconds() / 86400
    out['user_created_at_days_ago'] = days_created.round(0).astype('Int32')
    # post-cutoff → NaN
    out.loc[out['user_created_at_days_ago'] < 0, 'user_created_at_days_ago'] = pd.NA

    days_login = (cutoff_ts - out['last_login_dt']).dt.total_seconds() / 86400
    out['user_last_login_days_ago'] = days_login.round(0).astype('Int32')
    out.loc[out['user_last_login_days_ago'] < 0, 'user_last_login_days_ago'] = pd.NA

    # Drop helper datetime cols
    out = out.drop(columns=['created_at_dt', 'last_login_dt'])

    # Filtrar al sample (orden no garantizado)
    out = out.reindex(sample_user_ids['user_id'].astype(str).values)

    return out
