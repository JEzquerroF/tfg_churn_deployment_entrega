"""Feature pipeline para IAPs — replica de 02f_iaps.ipynb (20 features iap_*)."""

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

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


def load_iaps_raw(
    ctx: PipelineContext,
    consumables_path: Optional[Path] = None,
    subscriptions_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (consumables, subscriptions) limpiados."""
    if consumables_path is None:
        consumables_path = ctx.raw_csvs_dir / "processed_consumables_iaps.csv"
    if subscriptions_path is None:
        subscriptions_path = ctx.raw_csvs_dir / "processed_subscriptions_iaps.csv"

    cons = pd.read_csv(consumables_path, low_memory=False)
    cons['user_id'] = cons['user_id'].apply(_clean_uid)
    cons = cons.dropna(subset=['user_id']).copy()
    cons['purchase_dt'] = pd.to_datetime(cons['purchase_time'], unit='s', errors='coerce')

    subs = pd.read_csv(subscriptions_path, low_memory=False)
    subs['user_id'] = subs['user_id'].apply(_clean_uid)
    subs = subs.dropna(subset=['user_id']).copy()
    subs['purchase_dt'] = pd.to_datetime(subs['purchase_time'], unit='s', errors='coerce')
    subs['end_dt'] = pd.to_datetime(subs['end_date'], unit='s', errors='coerce')

    return cons, subs


def compute_iaps_features(
    sample_user_ids: pd.DataFrame,
    cutoff_date: datetime,
    raw_df: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None,
) -> pd.DataFrame:
    if raw_df is None:
        cons, subs = load_iaps_raw()
    else:
        cons, subs = raw_df

    sample_ids = set(sample_user_ids['user_id'].astype(str))
    cutoff_ts = pd.Timestamp(cutoff_date).tz_localize(None) if pd.Timestamp(cutoff_date).tz else pd.Timestamp(cutoff_date)
    cutoff_unix = int(cutoff_ts.timestamp())
    unix_7d = cutoff_unix - 7 * 86400
    unix_30d = cutoff_unix - 30 * 86400
    unix_90d = cutoff_unix - 90 * 86400

    # Filtrar consumables: sample + purchase pre-cutoff
    cons = cons[cons['user_id'].astype(str).isin(sample_ids)].copy()
    cons = cons[cons['purchase_dt'].notna() & (cons['purchase_dt'] <= cutoff_ts)].copy()

    # Filtrar subscriptions: sample + purchase pre-cutoff (end_dt PUEDE ser post-cutoff)
    subs = subs[subs['user_id'].astype(str).isin(sample_ids)].copy()
    subs = subs[subs['purchase_dt'].notna() & (subs['purchase_dt'] <= cutoff_ts)].copy()

    # ===== Grupo A: consumables volumen =====
    grp_a = cons.groupby('user_id').agg(
        iap_consumables_count=('product_id', 'size'),
        iap_consumables_unique_products=('product_id', 'nunique'),
    )
    grp_a['iap_has_consumables'] = (grp_a['iap_consumables_count'] > 0).astype('int64')

    # gems packs (prefix matching)
    gems_mask = cons['product_id'].astype(str).str.startswith('gems')
    gems_cnt = cons[gems_mask].groupby('user_id').size().rename('iap_gems_packs_count')
    grp_a = grp_a.join(gems_cnt, how='left')

    # days since last consumable
    last_cons_dt = cons.groupby('user_id')['purchase_dt'].max()
    days_since_cons = ((cutoff_ts - last_cons_dt).dt.total_seconds() / 86400).clip(lower=0).round(0)
    grp_a['iap_consumables_days_since_last'] = days_since_cons.astype('int64')

    # ===== Grupo B: consumables temporales (first dt) =====
    first_cons_dt = cons.groupby('user_id')['purchase_dt'].min()
    days_first_cons = (cutoff_ts - first_cons_dt).dt.total_seconds() / 86400
    grp_b = pd.DataFrame({'iap_first_consumable_days_ago': days_first_cons.round(0).astype('Int32')})
    grp_b.loc[grp_b['iap_first_consumable_days_ago'] < 0, 'iap_first_consumable_days_ago'] = pd.NA

    # ===== Grupo C: subscriptions volumen/tipo =====
    grp_c = subs.groupby('user_id').agg(
        iap_subscriptions_count=('product_id', 'size'),
    )
    grp_c['iap_has_subscription_ever'] = (grp_c['iap_subscriptions_count'] > 0).astype('int64')

    for prod, col in [('subs_monthly', 'iap_has_monthly'),
                      ('subs_3_months', 'iap_has_quarterly'),
                      ('subs_12_months', 'iap_has_annual')]:
        users_prod = set(subs.loc[subs['product_id'] == prod, 'user_id'].astype(str))
        grp_c[col] = grp_c.index.astype(str).isin(users_prod).astype('int64')

    # ===== Grupo D: subscriptions state =====
    max_end = subs.groupby('user_id')['end_date'].max()
    grp_d = pd.DataFrame(index=grp_c.index)
    grp_d['iap_is_subscription_active'] = (max_end.reindex(grp_d.index) >= cutoff_unix).astype('int64').fillna(0).astype('int64')
    grp_d['iap_subscription_active_last_7d'] = (max_end.reindex(grp_d.index) >= unix_7d).astype('int64').fillna(0).astype('int64')

    # days_since_subscription_end: 0 si activa, ≥1 si caducada, 9999 si nunca
    def _days_since_end(end_unix):
        if pd.isna(end_unix):
            return 9999
        if end_unix >= cutoff_unix:
            return 0
        days = (cutoff_unix - end_unix) / 86400
        return max(1, int(days))

    grp_d['iap_days_since_subscription_end'] = max_end.reindex(grp_d.index).apply(_days_since_end).astype('int64')

    # trial_only: TODAS las subs del user son is_trial==1
    if 'is_trial' in subs.columns:
        trial_mean = subs.groupby('user_id')['is_trial'].mean()
        grp_d['iap_trial_only'] = (trial_mean.reindex(grp_d.index) == 1.0).astype('int64').fillna(0).astype('int64')
    else:
        grp_d['iap_trial_only'] = 0

    # ===== Grupo E: subs temporales (first) =====
    first_sub_dt = subs.groupby('user_id')['purchase_dt'].min()
    days_first_sub = (cutoff_ts - first_sub_dt).dt.total_seconds() / 86400
    grp_e = pd.DataFrame({'iap_first_subscription_days_ago': days_first_sub.round(0).astype('Int32')})
    grp_e.loc[grp_e['iap_first_subscription_days_ago'] < 0, 'iap_first_subscription_days_ago'] = pd.NA

    # ===== Grupo F: combinadas =====
    users_cons = set(cons['user_id'].astype(str).unique())
    users_sub_ever = set(subs['user_id'].astype(str).unique())
    users_payer = users_cons | users_sub_ever

    users_active_sub = set(subs.loc[subs['end_date'] >= cutoff_unix, 'user_id'].astype(str))
    last_cons_unix = cons.groupby('user_id')['purchase_time'].max()
    users_cons_recent = set(last_cons_unix[last_cons_unix >= unix_30d].index.astype(str))
    users_current = users_active_sub | users_cons_recent

    cons_30d = set(cons.loc[cons['purchase_time'] >= unix_30d, 'user_id'].astype(str))
    sub_30d = set(subs.loc[subs['purchase_time'] >= unix_30d, 'user_id'].astype(str))
    cons_90d = set(cons.loc[cons['purchase_time'] >= unix_90d, 'user_id'].astype(str))
    sub_90d = set(subs.loc[subs['purchase_time'] >= unix_90d, 'user_id'].astype(str))

    all_users_idx = pd.Index(sample_user_ids['user_id'].astype(str).values, name='user_id')
    grp_f = pd.DataFrame(index=all_users_idx)
    grp_f['iap_is_payer'] = grp_f.index.isin(users_payer).astype('int64')
    grp_f['iap_is_current_payer'] = grp_f.index.isin(users_current).astype('int64')
    grp_f['iap_paid_last_30d'] = grp_f.index.isin(cons_30d | sub_30d).astype('int64')
    grp_f['iap_paid_last_90d'] = grp_f.index.isin(cons_90d | sub_90d).astype('int64')

    # ===== Combinar =====
    features = grp_f.join([grp_a, grp_b, grp_c, grp_d, grp_e], how='left')

    # Fillna por tipo
    int_zero_cols = [c for c in features.columns
                     if c not in ('iap_first_consumable_days_ago', 'iap_first_subscription_days_ago')]
    sentinel_cols = ('iap_consumables_days_since_last', 'iap_days_since_subscription_end')

    for c in int_zero_cols:
        if c in sentinel_cols:
            features[c] = features[c].fillna(9999).astype('int64')
        else:
            features[c] = features[c].fillna(0).astype('int64')

    return features
