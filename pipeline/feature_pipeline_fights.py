"""
feature_pipeline_fights.py — Features sobre fights_log.csv.

Réplica fiel del TFG (notebook 04b cell b06_fights_props para fight type
counts, 04c cell c05_fights_div para entropy, 04d cell d05_fights_temp
para binge index).

`fights_log.csv` es ENORME (~28 GB en el dataset del TFG, 1.3M filas
pero filas muy largas por la col `actions_log`). Lectura por chunks
obligatoria. Solo cargamos las cols mínimas necesarias:
  - player_id (char_id, hay que mapearlo a user_id vía char_to_user)
  - fight_type
  - fight_winner  (1.0 = gana el jugador) → eje pvp_perfil
  - start_time (unix s)

Verificación de coherencia (Bloque 0): el CSV cubre EXACTAMENTE 30 días.
Aplicamos filtro temporal explícito `ts >= REFERENCE - 30d` en deployment.

FEATURES GENERADAS (4):
  - entropy_fights_type    (Shannon sobre fight_type counts)
  - binge_index_fights     (max(daily_counts) / median(daily_counts))
  - fights_pct_pvp         (fight_type=='FIGHT_TYPE_ARENA' / total)  [eje pvp_perfil]
  - fights_pct_won         (fight_winner==1.0 / total)               [eje pvp_perfil]

Las 2 primeras alimentan el preprocessor de gustos_nivel1. Las 2 últimas
(pvp/won) son Tier 2 y alimentan el eje pvp_perfil del perfilado de gustos
(réplica del TFG notebook 04b cell b06_fights_props).
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional, Set

import pandas as pd

from pipeline._stats import binge_index, shannon
from pipeline.pipeline_context import PipelineContext

logger = logging.getLogger(__name__)


_OID_RE = re.compile(r"[0-9a-f]{24}")
TEMPORAL_WINDOW_DAYS = 30
CHUNK_LOG_INTERVAL = 5_000_000   # log progreso cada N filas escaneadas


def _clean_uid(value) -> Optional[str]:
    if pd.isna(value):
        return None
    m = _OID_RE.search(str(value))
    return m.group(0) if m else None


def build_char_to_user(characters_df: pd.DataFrame) -> dict[str, str]:
    """
    Construye el mapping char_id -> user_id desde characters.csv.
    Necesario porque fights_log.player_id es el char_id, no el user_id.

    Si `characters_df` ya tiene cols `_id` y `user_id` limpias (hex24),
    se usan directamente. Si no, se aplica `_clean_uid` a ambas.
    """
    if "char_id" in characters_df.columns:
        char_col = "char_id"
    else:
        char_col = "_id"

    df = characters_df[[char_col, "user_id"]].copy()
    df["char_id_clean"] = df[char_col].apply(_clean_uid)
    df["user_id_clean"] = df["user_id"].apply(_clean_uid)
    df = df.dropna(subset=["char_id_clean", "user_id_clean"])
    mapping = dict(zip(df["char_id_clean"], df["user_id_clean"]))
    logger.info("char_to_user mapping: %d entradas", len(mapping))
    return mapping


class FightsAggregator:
    """
    Aggregador para fights_log. Mantiene type_counts y daily_counts por
    usuario. Réplica de los pickles `_fights_type_counts.pkl` y la
    estructura `per_day_f` del TFG.
    """

    def __init__(self) -> None:
        self.type_counts: dict[str, dict] = {}    # user_id -> {fight_type: n}
        self.daily_counts: dict[str, dict] = {}   # user_id -> {date: n}
        self.total_n: dict[str, int] = {}
        self.pvp_n: dict[str, int] = {}           # user_id -> fights ARENA (PvP)
        self.won_n: dict[str, int] = {}           # user_id -> fights ganados

    def feed_chunk(self, chunk: pd.DataFrame) -> None:
        if chunk.empty:
            return
        has_winner = "fight_winner" in chunk.columns
        for uid, g in chunk.groupby("user_id"):
            self.total_n[uid] = self.total_n.get(uid, 0) + len(g)
            # type counts
            tc = self.type_counts.setdefault(uid, {})
            for ft, n in g["fight_type"].value_counts().items():
                tc[ft] = tc.get(ft, 0) + int(n)
            # daily counts
            dc = self.daily_counts.setdefault(uid, {})
            for d, n in g.groupby("day").size().items():
                dc[d] = dc.get(d, 0) + int(n)
            # pvp (ARENA) + won — eje pvp_perfil
            self.pvp_n[uid] = self.pvp_n.get(uid, 0) + int(
                (g["fight_type"] == "FIGHT_TYPE_ARENA").sum()
            )
            if has_winner:
                self.won_n[uid] = self.won_n.get(uid, 0) + int((g["fight_winner"] == 1.0).sum())


def load_fights_raw(
    ctx: PipelineContext,
    sample_user_ids: Set[str],
    char_to_user: dict[str, str],
    path: Optional[Path] = None,
    chunksize: int = 500_000,
) -> FightsAggregator:
    """
    Escanea fights_log.csv por chunks. Mapea player_id (char_id) a user_id,
    filtra por sample + ventana temporal, acumula counts.
    """
    if path is None:
        path = ctx.raw_csvs_dir / "fights_log.csv"

    if not path.exists():
        logger.warning(
            "fights: CSV no encontrado en %s — features quedarán NaN para todos los users",
            path,
        )
        return FightsAggregator()

    reference_ts = pd.Timestamp(ctx.reference_date)
    if reference_ts.tz is None:
        reference_ts = reference_ts.tz_localize("UTC")
    window_start = reference_ts - pd.Timedelta(days=TEMPORAL_WINDOW_DAYS)

    file_size_gb = path.stat().st_size / (1024 ** 3)
    logger.info(
        "fights: leyendo %s (~%.1f GB; filtro temporal: ts >= %s)",
        path.name,
        file_size_gb,
        window_start.date(),
    )
    if file_size_gb > 10:
        logger.warning(
            "fights: CSV grande (%.1f GB). Estimación de tiempo: ~%.0f minutos.",
            file_size_gb,
            file_size_gb * 0.3,
        )

    t0 = time.time()
    agg = FightsAggregator()
    n_scanned = 0
    n_in_window = 0
    last_log_at = 0

    for chunk in pd.read_csv(
        path,
        usecols=["player_id", "fight_type", "fight_winner", "start_time"],
        chunksize=chunksize,
        low_memory=False,
    ):
        n_scanned += len(chunk)
        chunk["player_id"] = chunk["player_id"].apply(_clean_uid)
        chunk["user_id"] = chunk["player_id"].map(char_to_user)
        chunk = chunk.dropna(subset=["user_id"])
        chunk = chunk[chunk["user_id"].isin(sample_user_ids)]
        if chunk.empty:
            continue

        chunk["ts"] = pd.to_datetime(chunk["start_time"], unit="s", errors="coerce", utc=True)
        chunk = chunk.dropna(subset=["ts"])
        chunk = chunk[(chunk["ts"] >= window_start) & (chunk["ts"] <= reference_ts)]
        if chunk.empty:
            continue

        n_in_window += len(chunk)
        chunk["day"] = chunk["ts"].dt.date
        agg.feed_chunk(chunk)

        # Logging periódico
        if n_scanned - last_log_at >= CHUNK_LOG_INTERVAL:
            elapsed = time.time() - t0
            logger.info(
                "fights: %d filas escaneadas en %.1fs (%d en ventana, %d users)",
                n_scanned,
                elapsed,
                n_in_window,
                len(agg.total_n),
            )
            last_log_at = n_scanned

    elapsed = time.time() - t0
    logger.info(
        "fights: %d filas total, %d en ventana 30d, %d users con datos (%.1fs)",
        n_scanned,
        n_in_window,
        len(agg.total_n),
        elapsed,
    )
    return agg


def compute_fights_features(
    agg: FightsAggregator,
    sample_user_ids: pd.Series,
) -> pd.DataFrame:
    """
    Calcula las 2 features de fights desde el aggregator.
    """
    user_ids = sample_user_ids.astype(str)

    ent_type = {u: shannon(d.values()) for u, d in agg.type_counts.items()}
    binge = {u: binge_index(d) for u, d in agg.daily_counts.items()}

    # pct_pvp / pct_won (eje pvp_perfil, Tier 2)
    pct_pvp = {
        u: agg.pvp_n.get(u, 0) / total
        for u, total in agg.total_n.items()
        if total > 0
    }
    pct_won = {
        u: agg.won_n.get(u, 0) / total
        for u, total in agg.total_n.items()
        if total > 0
    }

    out = pd.DataFrame({"user_id": user_ids})
    out["entropy_fights_type"] = out["user_id"].map(ent_type)
    out["binge_index_fights"] = out["user_id"].map(binge)
    out["fights_pct_pvp"] = out["user_id"].map(pct_pvp)
    out["fights_pct_won"] = out["user_id"].map(pct_won)
    return out


def compute(
    ctx: PipelineContext,
    sample_user_ids: pd.Series,
    char_to_user: dict[str, str],
    path: Optional[Path] = None,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """
    Pipeline completo: scan + aggregator + features.

    Args:
        ctx: contexto.
        sample_user_ids: Series con user_ids canónicos hex24.
        char_to_user: dict construido con `build_char_to_user()` antes.
        path: override opcional.
        chunksize: tamaño de chunk (default 500k para fights — más pequeño
            que currency porque las filas son grandes).
    """
    sample_set = set(sample_user_ids.astype(str))
    agg = load_fights_raw(ctx, sample_set, char_to_user, path=path, chunksize=chunksize)
    return compute_fights_features(agg, sample_user_ids)
