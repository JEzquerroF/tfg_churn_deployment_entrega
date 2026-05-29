"""
feature_pipeline_currency.py — Features sobre currency_transactions.csv.

Réplica fiel del TFG (notebooks 04b cell b05_currency_props, 04c cell
c04_currency_div, 04d cell d04_curr_temp). Combina la lógica de los 3
notebooks en un SOLO pase chunked sobre el CSV (~574 MB en el dataset
del TFG).

Verificación de coherencia (Fase 2.2 Bloque 0): el CSV del TFG cubre
EXACTAMENTE 30 días (2026-03-04 → 2026-04-04, terminando en REFERENCE_DATE).
Por eso el TFG no aplica filtro temporal en el código. En deployment, para
defendernos de clientes que entreguen histórico más largo, aplicamos filtro
explícito `ts >= REFERENCE - 30d`.

FEATURES GENERADAS (9):
  - entropy_currency_concept      (Shannon sobre concept counts)
  - gini_currency_concept         (Gini sobre concept counts)
  - n_distinct_concepts_used      (count concepts únicos)
  - entropy_currency_type         (Shannon sobre currency counts)
  - pct_days_active_currency_30d  (min(len(distinct_days), 30) / 30)
  - weekend_pct_currency          (weekend tx / total tx)
  - binge_index_currency          (max(daily_counts) / median(daily_counts))
  - currency_pct_inflow           (tx con quantity>0 / total)  [eje perfil_oro]
  - currency_pct_outflow          (tx con quantity<0 / total)  [eje perfil_oro]

Las 7 primeras alimentan el preprocessor de gustos_nivel1. Las 2 últimas
(inflow/outflow) son Tier 2 y alimentan el eje perfil_oro del perfilado de
gustos (réplica del TFG notebook 04b cell b05_currency_props).
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional, Set

import numpy as np
import pandas as pd

from pipeline._stats import binge_index, gini, shannon
from pipeline.pipeline_context import PipelineContext

logger = logging.getLogger(__name__)


_OID_RE = re.compile(r"[0-9a-f]{24}")
TEMPORAL_WINDOW_DAYS = 30  # divisor del nombre `_30d`; coincide con período del CSV TFG


def _clean_uid(value) -> Optional[str]:
    """Extrae el hex de 24 chars de un string ObjectId, o None."""
    if pd.isna(value):
        return None
    m = _OID_RE.search(str(value))
    return m.group(0) if m else None


class CurrencyAggregator:
    """
    Agregador que escanea currency_transactions.csv una vez y mantiene los
    counts necesarios para las 7 features de currency. Replica los pickles
    `_currency_agg_concept.pkl`, `_currency_agg_currency.pkl`, etc., del TFG
    pero en memoria.
    """

    def __init__(self) -> None:
        self.concept_counts: dict[str, dict] = {}   # user_id -> {concept: n}
        self.currency_counts: dict[str, dict] = {}  # user_id -> {currency: n}
        self.daily_counts: dict[str, dict] = {}     # user_id -> {date: n}
        self.weekend_n: dict[str, int] = {}
        self.total_n: dict[str, int] = {}
        self.days_active: dict[str, Set] = {}        # user_id -> set(dates)
        self.inflow_n: dict[str, int] = {}           # user_id -> tx con quantity>0
        self.outflow_n: dict[str, int] = {}          # user_id -> tx con quantity<0

    def feed_chunk(self, chunk: pd.DataFrame) -> None:
        """Acumula un chunk preprocesado (con user_id limpio + ts datetime)."""
        if chunk.empty:
            return
        for uid, g in chunk.groupby("user_id"):
            # concepts
            cc = self.concept_counts.setdefault(uid, {})
            for c, n in g["concept"].value_counts().items():
                cc[c] = cc.get(c, 0) + int(n)
            # currencies
            curc = self.currency_counts.setdefault(uid, {})
            for c, n in g["currency"].value_counts().items():
                curc[c] = curc.get(c, 0) + int(n)
            # totals + weekend
            self.total_n[uid] = self.total_n.get(uid, 0) + len(g)
            self.weekend_n[uid] = self.weekend_n.get(uid, 0) + int(g["is_weekend"].sum())
            # inflow / outflow (signo de quantity) — eje perfil_oro
            self.inflow_n[uid] = self.inflow_n.get(uid, 0) + int((g["quantity"] > 0).sum())
            self.outflow_n[uid] = self.outflow_n.get(uid, 0) + int((g["quantity"] < 0).sum())
            # days
            dset = self.days_active.setdefault(uid, set())
            dset.update(g["day"].unique())
            # daily counts
            dc = self.daily_counts.setdefault(uid, {})
            for d, n in g.groupby("day").size().items():
                dc[d] = dc.get(d, 0) + int(n)


def load_currency_raw(
    ctx: PipelineContext,
    sample_user_ids: Set[str],
    path: Optional[Path] = None,
    chunksize: int = 1_000_000,
) -> CurrencyAggregator:
    """
    Escanea currency_transactions.csv por chunks, filtra por sample y
    ventana temporal, y acumula los counts en un CurrencyAggregator.

    Args:
        ctx: contexto con `raw_csvs_dir`, `reference_date`.
        sample_user_ids: set de user_ids canónicos (hex24) del sample.
        path: override opcional del path (si no, usa `ctx.raw_csvs_dir`).
        chunksize: tamaño de chunk para `pd.read_csv`.

    Returns:
        CurrencyAggregator con los counts acumulados.
    """
    if path is None:
        path = ctx.raw_csvs_dir / "currency_transactions.csv"

    if not path.exists():
        logger.warning(
            "currency: CSV no encontrado en %s — features quedarán NaN para todos los users",
            path,
        )
        return CurrencyAggregator()

    reference_ts = pd.Timestamp(ctx.reference_date)
    if reference_ts.tz is None:
        reference_ts = reference_ts.tz_localize("UTC")
    window_start = reference_ts - pd.Timedelta(days=TEMPORAL_WINDOW_DAYS)

    logger.info(
        "currency: leyendo %s (filtro temporal: ts >= %s)",
        path.name,
        window_start.date(),
    )
    t0 = time.time()
    agg = CurrencyAggregator()
    n_scanned = 0
    n_in_window = 0
    for chunk in pd.read_csv(
        path,
        usecols=["user_id", "concept", "currency", "quantity", "created_at"],
        chunksize=chunksize,
        low_memory=False,
    ):
        n_scanned += len(chunk)
        chunk["user_id"] = chunk["user_id"].apply(_clean_uid)
        chunk = chunk[chunk["user_id"].isin(sample_user_ids)]
        if chunk.empty:
            continue
        chunk["ts"] = pd.to_datetime(chunk["created_at"], errors="coerce", utc=True)
        chunk = chunk.dropna(subset=["ts"])
        # Filtro temporal explícito: ventana de 30d terminando en REFERENCE
        chunk = chunk[(chunk["ts"] >= window_start) & (chunk["ts"] <= reference_ts)]
        if chunk.empty:
            continue
        n_in_window += len(chunk)
        chunk["day"] = chunk["ts"].dt.date
        chunk["is_weekend"] = chunk["ts"].dt.dayofweek >= 5
        agg.feed_chunk(chunk)

    elapsed = time.time() - t0
    logger.info(
        "currency: %d filas escaneadas, %d en ventana 30d, %d users con datos (%.1fs)",
        n_scanned,
        n_in_window,
        len(agg.total_n),
        elapsed,
    )
    return agg


def compute_currency_features(
    agg: CurrencyAggregator,
    sample_user_ids: pd.Series,
) -> pd.DataFrame:
    """
    Calcula las 7 features de currency a partir del aggregator.

    Returns:
        DataFrame con cols: user_id + 7 features.
        Usuarios sin datos en el aggregator quedan NaN (los rellenará el
        SimpleImputer del preprocessor de gustos_nivel1).
    """
    user_ids = sample_user_ids.astype(str)

    # 1) entropy_currency_concept
    ent_concept = {u: shannon(d.values()) for u, d in agg.concept_counts.items()}

    # 2) gini_currency_concept
    g_concept = {u: gini(d.values()) for u, d in agg.concept_counts.items()}

    # 3) n_distinct_concepts_used
    n_concepts = {u: len(d) for u, d in agg.concept_counts.items()}

    # 4) entropy_currency_type
    ent_type = {u: shannon(d.values()) for u, d in agg.currency_counts.items()}

    # 5) pct_days_active_currency_30d — min(len, 30) / 30
    pct_days = {
        u: min(len(dset), TEMPORAL_WINDOW_DAYS) / TEMPORAL_WINDOW_DAYS
        for u, dset in agg.days_active.items()
    }

    # 6) weekend_pct_currency — weekend_n / total_n
    weekend_pct = {
        u: agg.weekend_n.get(u, 0) / total
        for u, total in agg.total_n.items()
        if total > 0
    }

    # 7) binge_index_currency
    binge = {u: binge_index(d) for u, d in agg.daily_counts.items()}

    # 8) currency_pct_inflow / 9) currency_pct_outflow (eje perfil_oro, Tier 2)
    pct_inflow = {
        u: agg.inflow_n.get(u, 0) / total
        for u, total in agg.total_n.items()
        if total > 0
    }
    pct_outflow = {
        u: agg.outflow_n.get(u, 0) / total
        for u, total in agg.total_n.items()
        if total > 0
    }

    out = pd.DataFrame({"user_id": user_ids})
    out["entropy_currency_concept"] = out["user_id"].map(ent_concept)
    out["gini_currency_concept"] = out["user_id"].map(g_concept)
    out["n_distinct_concepts_used"] = out["user_id"].map(n_concepts)
    out["entropy_currency_type"] = out["user_id"].map(ent_type)
    out["pct_days_active_currency_30d"] = out["user_id"].map(pct_days)
    out["weekend_pct_currency"] = out["user_id"].map(weekend_pct)
    out["binge_index_currency"] = out["user_id"].map(binge)
    out["currency_pct_inflow"] = out["user_id"].map(pct_inflow)
    out["currency_pct_outflow"] = out["user_id"].map(pct_outflow)
    return out


def compute(
    ctx: PipelineContext,
    sample_user_ids: pd.Series,
    path: Optional[Path] = None,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """
    Pipeline completo: lectura + acumulación + cálculo de features.

    Args:
        ctx: contexto del pipeline.
        sample_user_ids: Series de user_ids (canónicos, hex24).
        path: override opcional.
        chunksize: tamaño de chunk.

    Returns:
        DataFrame con user_id + 7 features de currency.
    """
    sample_set = set(sample_user_ids.astype(str))
    agg = load_currency_raw(ctx, sample_set, path=path, chunksize=chunksize)
    return compute_currency_features(agg, sample_user_ids)
