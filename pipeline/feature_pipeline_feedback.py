"""Feature pipeline para support_feedback — replica de 02i_support_feedback.ipynb (6 features feedback_*)."""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .pipeline_context import PipelineContext


_OID_RE = re.compile(r"ObjectId\(?'?([a-f0-9]+)'?\)?")

# Mapping explícito de feedback_type → categoría (5 categorías)
FEEDBACK_CATEGORY_MAP = {
    'VERY_FUN': 'positive', 'LOVE_ITEMS': 'positive', 'GREAT_VISUALS': 'positive',
    'LIKE': 'positive', 'FEM_CHARACTERS': 'positive', 'CHARACTER_LOOKS_GOOD': 'positive',
    'LOVE_GAME': 'positive', 'BETTER_THAN_OTHERS': 'positive', 'NICE_COMMUNITY': 'positive',
    'TOO_MANY_HITS': 'negative', 'DIFFICULT_GAME': 'negative', 'DONT_LIKE': 'negative',
    'OBTAIN_ITEMS_EASIER': 'negative', 'BAD_VISUALS': 'negative', 'TOO_EASY': 'negative',
    'OTHER_BUGS': 'bug', 'CHARACTER_BUGS': 'bug', 'GAMEPLAY_BUGS': 'bug',
    'CRASHES': 'bug', 'VISUAL_BUGS': 'bug',
    'MEMBERSHIP_EXPENSIVE': 'monetization', 'CANT_PAY': 'monetization',
    'WANT_REFUND': 'monetization', 'PAY_TO_WIN': 'monetization',
    'SUGGESTION': 'neutral',
}


def _clean_uid(uid):
    if pd.isna(uid):
        return None
    s = str(uid)
    m = _OID_RE.search(s)
    if m:
        return m.group(1)
    return s


def load_feedback_raw(ctx: PipelineContext, path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        path = ctx.raw_csvs_dir / "support_user_feedback_by_type.csv"
    df = pd.read_csv(path, low_memory=False)
    df['user_id'] = df['user_id'].apply(_clean_uid)
    df = df.dropna(subset=['user_id']).copy()
    df['created_dt'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True).dt.tz_localize(None)
    df['feedback_category'] = df['feedback_type'].map(FEEDBACK_CATEGORY_MAP).fillna('neutral')
    return df


def compute_feedback_features(
    sample_user_ids: pd.DataFrame,
    cutoff_date: datetime,
    raw_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if raw_df is None:
        raw_df = load_feedback_raw()

    sample_ids = set(sample_user_ids['user_id'].astype(str))
    cutoff_ts = pd.Timestamp(cutoff_date).tz_localize(None) if pd.Timestamp(cutoff_date).tz else pd.Timestamp(cutoff_date)

    df = raw_df[raw_df['user_id'].astype(str).isin(sample_ids)].copy()
    df = df[df['created_dt'].notna() & (df['created_dt'] <= cutoff_ts)].copy()

    empty_idx = pd.Index(sample_user_ids['user_id'].astype(str).values, name='user_id')

    if df.empty:
        return pd.DataFrame({
            'feedback_total': 0,
            'feedback_has_any': 0,
            'feedback_n_negative': 0,
            'feedback_n_positive': 0,
            'feedback_n_monetization': 0,
            'feedback_days_since_last': 9999,
        }, index=empty_idx).astype('int64')

    # Volumen
    grp_a = df.groupby('user_id').agg(feedback_total=('feedback_type', 'size'))
    grp_a['feedback_has_any'] = (grp_a['feedback_total'] > 0).astype('int64')

    # Sentimiento (categorías)
    neg = df[df['feedback_category'].isin(['negative', 'bug'])].groupby('user_id').size().rename('feedback_n_negative')
    pos = df[df['feedback_category'] == 'positive'].groupby('user_id').size().rename('feedback_n_positive')
    mon = df[df['feedback_category'] == 'monetization'].groupby('user_id').size().rename('feedback_n_monetization')

    grp_b = pd.concat([neg, pos, mon], axis=1)

    # Temporal
    last_dt = df.groupby('user_id')['created_dt'].max()
    days_since = ((cutoff_ts - last_dt).dt.total_seconds() / 86400).clip(lower=0).round(0).astype('int64')
    grp_c = pd.DataFrame({'feedback_days_since_last': days_since})

    features = pd.concat([grp_a, grp_b, grp_c], axis=1)
    features = features.reindex(empty_idx)

    int_zero_cols = ['feedback_total', 'feedback_has_any', 'feedback_n_negative',
                     'feedback_n_positive', 'feedback_n_monetization']
    for c in int_zero_cols:
        features[c] = features[c].fillna(0).astype('int64')
    features['feedback_days_since_last'] = features['feedback_days_since_last'].fillna(9999).astype('int64')

    return features
