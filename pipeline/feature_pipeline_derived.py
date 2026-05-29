"""
feature_pipeline_derived.py — Features derivadas sin CSVs nuevos.

Las 18 features que el modelo `gustos_nivel1` espera pero el master_churn
NO produce, calculables sobre los DataFrames `raw_dfs` que `master_builder`
ya carga.

Réplica fiel del TFG (notebooks 04b, 04c, 04d, 04f, 04g) según el informe
de Fase 2.1. Importante:
- Usa REFERENCE_DATE (no cutoff_date) como en el TFG para gustos.
- p75 de chars.level lee desde `models/gustos_nivel1/.../thresholds.json`
  (persistido del training). Fallback a recálculo con warning si falta.
- Para IAPs y collection, recalcula las ventanas `_at_ref` desde scratch
  porque las del master_churn están medidas contra `cutoff`, no `REFERENCE`.

FEATURES GENERADAS (18):
  Sobre users.csv:
    - user_account_age_days
  Sobre user_items_collection.csv:
    - coll_age_days, coll_items_recent_30d, coll_items_recent_90d
    - coll_pct_recent_30d, coll_pct_recent_60d
  Sobre user_daily_rewards.csv:
    - reward_register_age_days, reward_pct_premium_track
    - claim_consistency_60d
  Sobre IAPs:
    - iap_paid_last_90d_at_ref, iap_n_distinct_products
  Sobre characters.csv:
    - pct_chars_arena_active, pct_chars_high_level
  Sobre user_items.csv:
    - pct_items_critical_build, pct_items_high_enhance
    - entropy_items_family, simpson_items_family
    - items_creation_velocity_30d
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from pipeline._stats import shannon, simpson
from pipeline.pipeline_context import PipelineContext

logger = logging.getLogger(__name__)


HIGH_ENHANCE_THRESHOLD = 5    # del TFG 04b: enhance_level >= 5
QUANTILE_CHARS_LEVEL = 0.75   # p75 del TFG 04b


def _reference_ts(ctx: PipelineContext) -> pd.Timestamp:
    """REFERENCE_DATE del ctx como Timestamp UTC."""
    ts = pd.Timestamp(ctx.reference_date)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts


def load_chars_level_p75(thresholds_path: Path) -> Optional[float]:
    """
    Lee el p75 de `chars.level` persistido del training. Devuelve None si
    no existe (fallback será recalcular con warning).
    """
    if not thresholds_path.exists():
        return None
    try:
        cfg = json.loads(thresholds_path.read_text())
        v = cfg.get("chars_level_p75")
        return float(v) if v is not None else None
    except Exception as e:
        logger.warning("error leyendo thresholds %s: %s", thresholds_path, e)
        return None


def _user_account_age_days(users_raw: pd.DataFrame, ref: pd.Timestamp, sample: pd.Series) -> dict:
    """`REFERENCE - created_at_dt` en días por user_id."""
    if "created_at_dt" not in users_raw.columns:
        users_raw = users_raw.copy()
        users_raw["created_at_dt"] = pd.to_datetime(
            users_raw.get("created_at"), errors="coerce", utc=True
        )
    # Asegurar tz UTC para comparación
    series = users_raw.set_index("user_id")["created_at_dt"]
    if series.dt.tz is None:
        series = series.dt.tz_localize("UTC")
    age = (ref - series).dt.days
    return age.to_dict()


def _collection_features(coll_raw: pd.DataFrame, ref: pd.Timestamp, sample: pd.Series) -> dict:
    """5 features de collection: age, recent 30d/90d (count), pct 30d/60d."""
    if coll_raw is None or coll_raw.empty:
        return {}
    df = coll_raw.copy()
    if "created_dt" not in df.columns:
        df["created_dt"] = pd.to_datetime(df.get("created_at"), errors="coerce", utc=True)
    if df["created_dt"].dt.tz is None:
        df["created_dt"] = df["created_dt"].dt.tz_localize("UTC")
    df = df.dropna(subset=["created_dt"])

    grp = df.groupby("user_id")
    first_dt = grp["created_dt"].min()
    total = grp.size()

    age_days = ((ref - first_dt).dt.days).to_dict()

    coll_30d = df[df["created_dt"] >= ref - pd.Timedelta(days=30)].groupby("user_id").size().to_dict()
    coll_60d = df[df["created_dt"] >= ref - pd.Timedelta(days=60)].groupby("user_id").size().to_dict()
    coll_90d = df[df["created_dt"] >= ref - pd.Timedelta(days=90)].groupby("user_id").size().to_dict()

    pct_30d = {u: coll_30d.get(u, 0) / total[u] for u in total.index if total[u] > 0}
    pct_60d = {u: coll_60d.get(u, 0) / total[u] for u in total.index if total[u] > 0}

    return {
        "coll_age_days": age_days,
        "coll_items_recent_30d": coll_30d,
        "coll_items_recent_90d": coll_90d,
        "coll_pct_recent_30d": pct_30d,
        "coll_pct_recent_60d": pct_60d,
    }


def _rewards_features(rewards_raw: pd.DataFrame, ref: pd.Timestamp) -> dict:
    """3 features de rewards: register_age, pct_premium_track, claim_consistency_60d."""
    if rewards_raw is None or rewards_raw.empty:
        return {}
    df = rewards_raw.copy()
    if "created_dt" not in df.columns:
        df["created_dt"] = pd.to_datetime(df.get("created_at"), errors="coerce", utc=True)
    if df["created_dt"].dt.tz is None:
        df["created_dt"] = df["created_dt"].dt.tz_localize("UTC")

    # Snapshot: si hay duplicados por user, tail(1) por updated_at más reciente.
    if "updated_at" in df.columns:
        df["updated_dt"] = pd.to_datetime(df["updated_at"], errors="coerce", utc=True)
        df = df.sort_values("updated_dt").groupby("user_id").tail(1)
    df = df.set_index("user_id")

    age_days = ((ref - df["created_dt"]).dt.days).to_dict()

    # reward_pct_premium_track = premium / (premium + ad), NaN si denom=0
    premium = df.get("last_claimed_premium_reward_day", pd.Series(dtype="float")).fillna(0)
    ad = df.get("last_claimed_ad_reward_day", pd.Series(dtype="float")).fillna(0)
    denom = premium + ad
    pct_premium = (premium / denom.replace(0, np.nan)).to_dict()

    # claim_consistency_60d (proxy 04d): ((sets_completed*30 + last_claimed_reward_day) clip 60) / 60
    sets_completed = df.get("sets_completed", pd.Series(dtype="float")).fillna(0)
    last_claimed = df.get("last_claimed_reward_day", pd.Series(dtype="float")).fillna(0)
    proxy = ((sets_completed * 30 + last_claimed).clip(upper=60) / 60).clip(upper=1.0).to_dict()

    return {
        "reward_register_age_days": age_days,
        "reward_pct_premium_track": pct_premium,
        "claim_consistency_60d": proxy,
    }


def _iaps_features(iaps_raw: Any, ref: pd.Timestamp) -> dict:
    """2 features de IAPs: paid_last_90d_at_ref, n_distinct_products."""
    if iaps_raw is None:
        return {}
    # raw_dfs['iaps'] es tupla (cons, subs)
    cons, subs = iaps_raw
    # purchase_dt ya está parseado en feature_pipeline_iaps.load_iaps_raw
    cons = cons.copy()
    subs = subs.copy()

    # Asegurar tz UTC
    for df in (cons, subs):
        if "purchase_dt" not in df.columns:
            df["purchase_dt"] = pd.to_datetime(df.get("purchase_time"), unit="s", utc=True, errors="coerce")
        else:
            # Coercer a UTC si está naive
            if pd.api.types.is_datetime64_dtype(df["purchase_dt"]) and df["purchase_dt"].dt.tz is None:
                df["purchase_dt"] = df["purchase_dt"].dt.tz_localize("UTC")

    cutoff_90d = ref - pd.Timedelta(days=90)
    paid_cons = set(cons.loc[cons["purchase_dt"] >= cutoff_90d, "user_id"].astype(str).unique())
    paid_subs = set(subs.loc[subs["purchase_dt"] >= cutoff_90d, "user_id"].astype(str).unique())
    paid_90d = paid_cons | paid_subs

    # n_distinct_products: concat cons+subs y nunique
    prod = pd.concat(
        [cons[["user_id", "product_id"]], subs[["user_id", "product_id"]]],
        ignore_index=True,
    )
    n_distinct = prod.groupby("user_id")["product_id"].nunique().to_dict()

    return {
        "iap_paid_last_90d_at_ref": paid_90d,  # SET (uso isin downstream)
        "iap_n_distinct_products": n_distinct,
    }


def _chars_features(
    chars_raw: pd.DataFrame, p75_level: Optional[float]
) -> tuple[dict, float]:
    """
    2 features de characters: pct_chars_arena_active, pct_chars_high_level.

    Devuelve (dict_features, p75_efectivo) — el p75 efectivo es el persistido
    si llegó, o el calculado on-the-fly como fallback.
    """
    if chars_raw is None or chars_raw.empty:
        return {}, float("nan")
    df = chars_raw[["user_id", "arena_category", "level"]].copy()
    df["user_id"] = df["user_id"].astype(str)

    total_chars = df.groupby("user_id").size()
    arena_active = df[df["arena_category"] > 0].groupby("user_id").size()

    if p75_level is None:
        p75_effective = float(df["level"].quantile(QUANTILE_CHARS_LEVEL))
        logger.warning(
            "thresholds.json no encontrado; usando p75 calculado on-the-fly "
            "(%.2f). Esto puede divergir del modelo entrenado.",
            p75_effective,
        )
    else:
        p75_effective = float(p75_level)
        logger.info("Usando p75 persistido del training: %.2f", p75_effective)

    high = df[df["level"] >= p75_effective].groupby("user_id").size()

    pct_arena = (arena_active / total_chars).to_dict()
    pct_high = (high / total_chars).to_dict()
    return (
        {
            "pct_chars_arena_active": pct_arena,
            "pct_chars_high_level": pct_high,
        },
        p75_effective,
    )


def _items_features(items_raw: pd.DataFrame, ref: pd.Timestamp) -> dict:
    """5 features de items: pct_critical, pct_high_enhance, entropy/simpson family, velocity.

    Optimizado para 1M+ users: usa iteración sobre groupby (O(N)) en lugar de
    lookup por usuario sobre MultiIndex (O(N²)).
    """
    if items_raw is None or items_raw.empty:
        return {}
    df = items_raw.copy()
    df["user_id"] = df["user_id"].astype(str)

    total = df.groupby("user_id").size()
    crit = df[df["c_base_critical_chance"] > 0].groupby("user_id").size()
    he = df[df["enhance_level"] >= HIGH_ENHANCE_THRESHOLD].groupby("user_id").size()

    pct_crit = (crit / total).to_dict()
    pct_he = (he / total).to_dict()

    # Entropy / Simpson: iteración directa sobre groupby (O(N), no O(N²))
    entropy_dict: dict[str, float] = {}
    simpson_dict: dict[str, float] = {}
    for uid, series in df.groupby("user_id")["item_definition_excel_id"]:
        counts = series.value_counts().values
        if len(counts) > 0:
            entropy_dict[uid] = shannon(counts)
            simpson_dict[uid] = simpson(counts)

    # items_creation_velocity_30d
    if "created_dt" not in df.columns:
        df["created_dt"] = pd.to_datetime(df.get("created_at"), errors="coerce", utc=True)
    if df["created_dt"].dt.tz is None:
        df["created_dt"] = df["created_dt"].dt.tz_localize("UTC")
    recent = df[df["created_dt"] >= ref - pd.Timedelta(days=30)]
    velocity = recent.groupby("user_id").size().to_dict()

    return {
        "pct_items_critical_build": pct_crit,
        "pct_items_high_enhance": pct_he,
        "entropy_items_family": entropy_dict,
        "simpson_items_family": simpson_dict,
        "items_creation_velocity_30d": velocity,
    }


def compute(
    master_churn: pd.DataFrame,
    sample_user_ids: pd.Series,
    ctx: PipelineContext,
    raw_dfs: dict,
    thresholds_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Calcula las 18 features derivadas.

    Args:
        master_churn: master ya construido por master_builder.build_master.
            (Hoy no se usa directamente — todo se calcula desde raw_dfs —
            pero queda en la signature por consistencia futura.)
        sample_user_ids: Series con user_ids canónicos.
        ctx: contexto del pipeline (provee `reference_date`).
        raw_dfs: dict con cargas crudas. Claves: users, characters, devices,
            iaps, rewards, items, collection, feedback.
        thresholds_path: ruta opcional a `thresholds.json` con `chars_level_p75`.
            Si None, busca en `models/gustos_nivel1/<version>/thresholds.json`.

    Returns:
        DataFrame con user_id + 18 features.
    """
    t0 = time.time()
    ref = _reference_ts(ctx)
    user_ids = sample_user_ids.astype(str)
    out = pd.DataFrame({"user_id": user_ids})

    # Threshold p75 (persistido del training)
    if thresholds_path is None:
        # Búsqueda por defecto
        candidate = Path(__file__).resolve().parent.parent / "models" / "gustos_nivel1"
        if candidate.exists():
            versions = sorted([d for d in candidate.iterdir() if d.is_dir()])
            if versions:
                thresholds_path = versions[-1] / "thresholds.json"
    p75_level = load_chars_level_p75(thresholds_path) if thresholds_path else None

    # 1) users
    logger.info("[derived] users…")
    if "users" in raw_dfs and raw_dfs["users"] is not None:
        out["user_account_age_days"] = out["user_id"].map(
            _user_account_age_days(raw_dfs["users"], ref, sample_user_ids)
        )
    else:
        out["user_account_age_days"] = np.nan

    # 2) collection
    logger.info("[derived] collection…")
    coll_feats = _collection_features(raw_dfs.get("collection"), ref, sample_user_ids)
    for name, mp in coll_feats.items():
        out[name] = out["user_id"].map(mp)
        # los counts (recent_30d/90d) son enteros: fillna 0
        if name in ("coll_items_recent_30d", "coll_items_recent_90d"):
            out[name] = out[name].fillna(0)

    # 3) rewards
    logger.info("[derived] rewards…")
    rw_feats = _rewards_features(raw_dfs.get("rewards"), ref)
    for name, mp in rw_feats.items():
        out[name] = out["user_id"].map(mp)

    # 4) IAPs
    logger.info("[derived] iaps…")
    iap_feats = _iaps_features(raw_dfs.get("iaps"), ref)
    # iap_paid_last_90d_at_ref es un set, los demás son dicts
    if "iap_paid_last_90d_at_ref" in iap_feats:
        out["iap_paid_last_90d_at_ref"] = out["user_id"].isin(
            iap_feats["iap_paid_last_90d_at_ref"]
        ).astype(int)
    if "iap_n_distinct_products" in iap_feats:
        out["iap_n_distinct_products"] = out["user_id"].map(
            iap_feats["iap_n_distinct_products"]
        ).fillna(0).astype(int)

    # 5) characters
    logger.info("[derived] characters…")
    chars_feats, p75_effective = _chars_features(raw_dfs.get("characters"), p75_level)
    for name, mp in chars_feats.items():
        out[name] = out["user_id"].map(mp).fillna(0)

    # 6) items
    logger.info("[derived] items…")
    items_feats = _items_features(raw_dfs.get("items"), ref)
    for name, mp in items_feats.items():
        out[name] = out["user_id"].map(mp)
        if name == "items_creation_velocity_30d":
            out[name] = out[name].fillna(0)

    logger.info("[derived] %d cols generadas en %.1fs", out.shape[1] - 1, time.time() - t0)
    return out
