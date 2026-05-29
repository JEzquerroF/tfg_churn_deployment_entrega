"""
feature_pipeline_arena.py — Features sobre arena_log.csv.

Réplica del TFG (notebooks 04e cell e05_arena y 04f cell f03).

`arena_log.csv` es relativamente pequeño (~71 MB). El TFG NO aplica filtro
temporal en el código pero el CSV ya cubre exactamente 30 días terminando
en REFERENCE_DATE (verificado en Bloque 0). En deployment aplicamos filtro
explícito por seguridad.

FEATURES GENERADAS (1, en num_low del preprocessor):
  - is_arena_player  (binario: user_id aparece como attacker en arena_log)
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional, Set

import pandas as pd

from pipeline.pipeline_context import PipelineContext

logger = logging.getLogger(__name__)


_OID_RE = re.compile(r"[0-9a-f]{24}")
TEMPORAL_WINDOW_DAYS = 30


def _clean_uid(value) -> Optional[str]:
    if pd.isna(value):
        return None
    m = _OID_RE.search(str(value))
    return m.group(0) if m else None


def load_arena_users(
    ctx: PipelineContext,
    sample_user_ids: Set[str],
    char_to_user: dict[str, str],
    path: Optional[Path] = None,
) -> Set[str]:
    """
    Lee arena_log.csv, mapea attacker_id (char_id) → user_id, filtra por
    sample + ventana temporal, y devuelve el SET de user_ids con al menos
    1 combate en arena.

    `arena_log.csv` cabe en memoria (~71 MB), no necesita chunking.
    """
    if path is None:
        path = ctx.raw_csvs_dir / "arena_log.csv"

    if not path.exists():
        logger.warning(
            "arena: CSV no encontrado en %s — is_arena_player = 0 para todos",
            path,
        )
        return set()

    reference_ts = pd.Timestamp(ctx.reference_date)
    if reference_ts.tz is None:
        reference_ts = reference_ts.tz_localize("UTC")
    window_start = reference_ts - pd.Timedelta(days=TEMPORAL_WINDOW_DAYS)

    logger.info(
        "arena: leyendo %s (filtro temporal: ts >= %s)",
        path.name,
        window_start.date(),
    )
    t0 = time.time()
    df = pd.read_csv(
        path,
        usecols=["attacker_id", "time_attacked"],
        low_memory=False,
    )
    df["attacker_id"] = df["attacker_id"].apply(_clean_uid)
    df["user_id"] = df["attacker_id"].map(char_to_user)
    df = df.dropna(subset=["user_id"])
    df = df[df["user_id"].isin(sample_user_ids)]

    # Filtro temporal — time_attacked es unix seconds en el TFG
    df["ts"] = pd.to_datetime(df["time_attacked"], unit="s", errors="coerce", utc=True)
    df = df.dropna(subset=["ts"])
    df = df[(df["ts"] >= window_start) & (df["ts"] <= reference_ts)]

    arena_users = set(df["user_id"].unique())
    elapsed = time.time() - t0
    logger.info(
        "arena: %d combates en ventana, %d users con arena (%.1fs)",
        len(df),
        len(arena_users),
        elapsed,
    )
    return arena_users


def compute_arena_features(
    arena_users: Set[str],
    sample_user_ids: pd.Series,
) -> pd.DataFrame:
    """
    Marca cada usuario del sample con 1 si tiene al menos 1 combate en
    arena, 0 si no.
    """
    user_ids = sample_user_ids.astype(str)
    out = pd.DataFrame({"user_id": user_ids})
    out["is_arena_player"] = out["user_id"].isin(arena_users).astype(int)
    return out


def compute(
    ctx: PipelineContext,
    sample_user_ids: pd.Series,
    char_to_user: dict[str, str],
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """Pipeline completo arena."""
    sample_set = set(sample_user_ids.astype(str))
    arena_users = load_arena_users(ctx, sample_set, char_to_user, path=path)
    return compute_arena_features(arena_users, sample_user_ids)
