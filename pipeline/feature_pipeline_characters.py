"""
Feature pipeline para characters — replica parametrizada de 02c_characters.ipynb.

23 features prefijadas char_*. Aggregación por user_id con filtro pre-cutoff.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .pipeline_context import PipelineContext


COSMETIC_COLS = ['hairIcon', 'beardIcon', 'eyebrowsIcon', 'faceScarIcon',
                 'facePaintIcon', 'torsoScarIcon', 'torsoPaintIcon']
EQUIP_COLS = ['rHandEqObj', 'lHandEqObj', 'helmEqObj', 'chestEqObj',
              'handsEqObj', 'legsEqObj']


def load_characters_raw(ctx: PipelineContext, path: Optional[Path] = None) -> pd.DataFrame:
    """Carga characters.csv y parsea fechas. Filtra NPCs."""
    if path is None:
        path = ctx.raw_csvs_dir / "characters.csv"
    df = pd.read_csv(path, low_memory=False)

    # Filtrar NPCs (~269 filas)
    if 'is_npc' in df.columns:
        df = df[~df['is_npc'].fillna(False).astype(bool)].copy()

    df['created_dt'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True).dt.tz_localize(None)
    df['updated_dt'] = pd.to_datetime(df['updated_at'], errors='coerce', utc=True).dt.tz_localize(None)

    return df


def compute_characters_features(
    sample_user_ids: pd.DataFrame,
    cutoff_date: datetime,
    raw_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if raw_df is None:
        raw_df = load_characters_raw()

    sample_ids = set(sample_user_ids['user_id'].astype(str))
    cutoff_ts = pd.Timestamp(cutoff_date).tz_localize(None) if pd.Timestamp(cutoff_date).tz else pd.Timestamp(cutoff_date)

    df = raw_df[raw_df['user_id'].astype(str).isin(sample_ids)].copy()
    df = df[df['created_dt'].notna() & (df['created_dt'] <= cutoff_ts)].copy()

    # Asegurar cols equip y cosmetic existen
    for col in COSMETIC_COLS + EQUIP_COLS:
        if col not in df.columns:
            df[col] = 0

    # Score cosmético por char y slots equipados
    df['_cosmetic_score'] = (df[COSMETIC_COLS].fillna(0) > 0).sum(axis=1)
    df['_equip_slots_filled'] = df[EQUIP_COLS].notna().sum(axis=1)

    # Main character: max(level) → desempate por max(experience)
    df_sorted = df.sort_values(['user_id', 'level', 'experience'], ascending=[True, False, False])
    main_chars = df_sorted.drop_duplicates(subset='user_id', keep='first').set_index('user_id')

    # Grupo A: agregaciones simples
    grp_count = df.groupby('user_id').agg(
        char_count=('level', 'size'),
        char_classes_unique=('class', 'nunique'),
        char_level_mean=('level', 'mean'),
        char_experience_mean=('experience', 'mean'),
    )
    grp_count['char_has_multiple'] = (grp_count['char_count'] > 1).astype(int)

    # Grupo B: del main char
    feat_main = pd.DataFrame(index=main_chars.index)
    feat_main['char_class_main'] = main_chars['class'].astype('Int64').fillna(0).astype('int64')
    feat_main['char_level_max'] = main_chars['level']
    feat_main['char_experience_max'] = main_chars['experience']
    feat_main['char_attack_max'] = main_chars.get('c_total_attack', np.nan)
    feat_main['char_defense_max'] = main_chars.get('c_total_defense', np.nan)
    feat_main['char_attack_defense_sum_max'] = main_chars.get('c_total_attack_defense_sum', np.nan)
    feat_main['char_critical_chance_max'] = main_chars.get('c_total_critical_chance', 0)
    feat_main['char_critical_chance_max'] = feat_main['char_critical_chance_max'].fillna(0).astype('int64')
    feat_main['char_equip_slots_filled_max'] = main_chars['_equip_slots_filled'].fillna(0).astype('int64')
    feat_main['char_cosmetic_score'] = main_chars['_cosmetic_score'].fillna(0).astype('int64')
    feat_main['char_talent_total_max'] = main_chars.get('total_talent_points', np.nan)
    feat_main['char_talent_spent_max'] = main_chars.get('spent_talent_points', np.nan)

    feat_main['char_talent_pct_spent'] = (
        feat_main['char_talent_spent_max']
        / feat_main['char_talent_total_max'].replace(0, np.nan)
    ).fillna(0).round(2)

    # Grupo C: any-char aggregations
    grp_any = df.groupby('user_id').agg(
        char_arena_category_max=('arena_category', 'max'),
        char_medals_max=('medals', 'max'),
    )
    grp_any['char_has_arena'] = df.groupby('user_id')['arena_category'].apply(
        lambda x: int(x.notna().any())
    )
    grp_any['char_arena_category_max'] = grp_any['char_arena_category_max'].fillna(0).astype('float64')
    grp_any['char_medals_max'] = grp_any['char_medals_max'].fillna(0).astype('float64')

    # is_customized: any char tiene is_customized=True
    if 'is_customized' in df.columns:
        custom_any = df.groupby('user_id')['is_customized'].apply(
            lambda x: int(x.astype(str).str.lower().isin(['true', '1', 'yes']).any())
        ).rename('char_is_customized')
    else:
        custom_any = pd.Series(0, index=df['user_id'].unique(), name='char_is_customized')

    # *_days_ago al cutoff (Int32 nullable)
    grp_dates = df.groupby('user_id').agg(
        _first_created_dt=('created_dt', 'min'),
        _last_updated_dt=('updated_dt', 'max'),
    )
    days_first = (cutoff_ts - grp_dates['_first_created_dt']).dt.total_seconds() / 86400
    days_last = (cutoff_ts - grp_dates['_last_updated_dt']).dt.total_seconds() / 86400
    grp_dates['char_first_created_days_ago'] = days_first.round(0).astype('Int32')
    grp_dates['char_last_updated_days_ago'] = days_last.round(0).astype('Int32')
    # Las que están post-cutoff (no debería pasar por el filtro), o NaN
    for c in ('char_first_created_days_ago', 'char_last_updated_days_ago'):
        grp_dates.loc[grp_dates[c] < 0, c] = pd.NA
    grp_dates = grp_dates.drop(columns=['_first_created_dt', '_last_updated_dt'])

    # Combinar todo
    features = pd.concat([grp_count, feat_main, grp_any, custom_any, grp_dates], axis=1)

    # Reindex al sample completo
    features = features.reindex(sample_user_ids['user_id'].astype(str).values)

    # Fillna numéricos con 0 (excepto *_days_ago que son nullable)
    fill_zero_cols = [c for c in features.columns
                      if c not in ('char_first_created_days_ago', 'char_last_updated_days_ago',
                                   'char_attack_max', 'char_defense_max',
                                   'char_attack_defense_sum_max',
                                   'char_talent_total_max', 'char_talent_spent_max',
                                   'char_experience_max', 'char_experience_mean',
                                   'char_level_max', 'char_level_mean')]
    for c in fill_zero_cols:
        features[c] = features[c].fillna(0)

    # Para num cols que pueden quedar NaN (no en sample con chars)
    for c in ['char_attack_max', 'char_defense_max', 'char_attack_defense_sum_max',
              'char_talent_total_max', 'char_talent_spent_max',
              'char_experience_max', 'char_experience_mean',
              'char_level_max', 'char_level_mean']:
        if c in features.columns:
            features[c] = features[c].fillna(0)

    # Dtypes
    int_cols = ['char_count', 'char_classes_unique', 'char_has_multiple', 'char_class_main',
                'char_critical_chance_max', 'char_equip_slots_filled_max', 'char_cosmetic_score',
                'char_has_arena', 'char_is_customized']
    for c in int_cols:
        if c in features.columns:
            features[c] = features[c].astype('int64')

    return features
