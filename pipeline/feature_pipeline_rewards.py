"""Feature pipeline para daily rewards — replica de 02e_user_daily_rewards.ipynb (10 features)."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .pipeline_context import PipelineContext


def load_rewards_raw(ctx: PipelineContext, path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        path = ctx.raw_csvs_dir / "user_daily_rewards.csv"
    df = pd.read_csv(path, low_memory=False)
    df['created_dt'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True).dt.tz_localize(None)
    return df


def compute_rewards_features(
    sample_user_ids: pd.DataFrame,
    cutoff_date: datetime,
    raw_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if raw_df is None:
        raw_df = load_rewards_raw()

    sample_ids = set(sample_user_ids['user_id'].astype(str))
    cutoff_ts = pd.Timestamp(cutoff_date).tz_localize(None) if pd.Timestamp(cutoff_date).tz else pd.Timestamp(cutoff_date)

    df = raw_df[raw_df['user_id'].astype(str).isin(sample_ids)].copy()
    df = df[df['created_dt'].notna() & (df['created_dt'] <= cutoff_ts)].copy()

    if df.empty:
        empty_idx = pd.Index(sample_user_ids['user_id'].astype(str).values, name='user_id')
        cols = {
            'reward_records_total': 0.0,
            'reward_sets_completed_max': 0.0,
            'reward_current_day_max': 0.0,
            'reward_claim_rate': 0.0,
            'reward_has_premium': 0.0,
            'reward_has_ad': 0.0,
            'reward_premium_days_max': 0.0,
            'reward_ad_days_max': 0.0,
        }
        out = pd.DataFrame(cols, index=empty_idx).astype('float64')
        out['reward_first_created_days_ago'] = pd.array([pd.NA] * len(out), dtype='Int32')
        out['reward_last_claim_days_ago'] = pd.array([pd.NA] * len(out), dtype='Int32')
        return out

    # Grupo A: volumen
    grp_a = df.groupby('user_id').agg(
        reward_records_total=('set', 'size'),
        reward_sets_completed_max=('sets_completed', 'max'),
        reward_current_day_max=('current_day', 'max'),
    )
    claimed = df[df['last_claimed_reward_day'] > 0].groupby('user_id')['last_claimed_reward_day'].max()
    available = df.groupby('user_id')['current_day'].max()
    grp_a['reward_claim_rate'] = (claimed / available.replace(0, np.nan)).clip(0, 1).fillna(0).round(2)

    # Grupo B: monetización
    has_premium = df.groupby('user_id')['last_claimed_premium_reward_day'].apply(lambda x: int((x > 0).any())).rename('reward_has_premium')
    has_ad = df.groupby('user_id')['last_claimed_ad_reward_day'].apply(lambda x: int((x > 0).any())).rename('reward_has_ad')
    premium_max = df[df['last_claimed_premium_reward_day'] > 0].groupby('user_id')['last_claimed_premium_reward_day'].max().rename('reward_premium_days_max')
    ad_max = df[df['last_claimed_ad_reward_day'] > 0].groupby('user_id')['last_claimed_ad_reward_day'].max().rename('reward_ad_days_max')

    grp_b = pd.concat([has_premium, has_ad, premium_max, ad_max], axis=1)

    # Grupo C: temporales
    first_created = df.groupby('user_id')['created_dt'].min()
    days_first = (cutoff_ts - first_created).dt.total_seconds() / 86400
    first_days_ago = days_first.round(0).astype('Int32').rename('reward_first_created_days_ago')

    last_claim_unix = df[df['last_claimed_reward_time'] > 0].groupby('user_id')['last_claimed_reward_time'].max()
    last_claim_dt = pd.to_datetime(last_claim_unix, unit='s', errors='coerce')
    days_last = (cutoff_ts - last_claim_dt).dt.total_seconds() / 86400
    last_days_ago = days_last.round(0).astype('Int32').rename('reward_last_claim_days_ago')

    grp_c = pd.concat([first_days_ago, last_days_ago], axis=1)

    features = pd.concat([grp_a, grp_b, grp_c], axis=1)
    features = features.reindex(sample_user_ids['user_id'].astype(str).values)

    # Fillna
    numeric_zero_cols = ['reward_records_total', 'reward_sets_completed_max', 'reward_current_day_max',
                         'reward_claim_rate', 'reward_has_premium', 'reward_has_ad',
                         'reward_premium_days_max', 'reward_ad_days_max']
    for c in numeric_zero_cols:
        features[c] = features[c].fillna(0).astype('float64')

    # *_days_ago: nullable Int32, mantener NaN para users sin reward
    for c in ('reward_first_created_days_ago', 'reward_last_claim_days_ago'):
        features.loc[features[c] < 0, c] = pd.NA

    return features
