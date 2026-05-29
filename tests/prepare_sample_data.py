"""
Genera un subset coherente de los CSVs del TFG para tests.

Toma las primeras N filas de users.csv y filtra todas las demás tablas a los
user_ids de ese subset. Garantiza coherencia referencial: si user X aparece
en users.csv, sus characters/devices/iaps/etc. también aparecen.

NO se ejecuta como test (no es test_*.py). Solo se invoca a mano para
(re)generar tests/sample_data/.

Uso:
    /Users/jezquerro/Documents/tfg/.venv/bin/python tests/prepare_sample_data.py [--n 100]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

_OID_RE = re.compile(r"ObjectId\(?'?([a-f0-9]+)'?\)?")


def _extract_uid(value):
    if pd.isna(value):
        return None
    s = str(value)
    m = _OID_RE.search(s)
    if m:
        return m.group(1)
    if len(s) == 24 and all(c in "0123456789abcdef" for c in s):
        return s
    return None


DATA_RAW = Path("/Users/jezquerro/Documents/tfg/data/data_raw")
OUTPUT_DIR = Path(__file__).resolve().parent / "sample_data"

# Map schema canónico → fichero CSV en data_raw (igual que el deployment espera).
# Los 3 transaccionales (currency, fights, arena) se añadieron en Fase 2.2.
# fights y arena se filtran por player_id/attacker_id (char_id), no user_id.
CSV_FILES = {
    "users": "users.csv",
    "characters": "characters.csv",
    "devices": "devices.csv",
    "iaps_consumables": "processed_consumables_iaps.csv",
    "iaps_subscriptions": "processed_subscriptions_iaps.csv",
    "daily_rewards": "user_daily_rewards.csv",
    "user_items": "user_items.csv",
    "user_items_collection": "user_items_collection.csv",
    "support_feedback": "support_user_feedback_by_type.csv",
    "currency_transactions": "currency_transactions.csv",
    # fights + arena se filtran aparte (char_id en vez de user_id)
}

# CSVs cuyo identificador es char_id (no user_id directo).
CSV_FILES_BY_CHAR_ID = {
    "fights_log": ("fights_log.csv", "player_id"),
    "arena_log":  ("arena_log.csv",  "attacker_id"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100, help="Número de usuarios en el subset")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Leyendo primeras {args.n} filas de users.csv…")
    users = pd.read_csv(DATA_RAW / "users.csv", nrows=args.n, low_memory=False)
    print(f"  N users en subset: {len(users)}")

    sample_uids = set(users["_id"].apply(_extract_uid).dropna())
    print(f"  N user_ids limpios distintos: {len(sample_uids)}")

    users.to_csv(OUTPUT_DIR / "users.csv", index=False)
    print(f"  guardado users.csv: {len(users)} filas")

    for schema, csv_name in CSV_FILES.items():
        if schema == "users":
            continue

        out_path = OUTPUT_DIR / csv_name
        src = DATA_RAW / csv_name
        size_mb = src.stat().st_size / 1024 / 1024
        print(f"\nFiltrando {csv_name} ({size_mb:.1f}MB)…")

        chunks_kept = []
        rows_scanned = 0
        for chunk in pd.read_csv(src, chunksize=200_000, low_memory=False):
            rows_scanned += len(chunk)
            chunk = chunk.copy()
            chunk["_clean_uid"] = chunk["user_id"].apply(_extract_uid)
            kept = chunk[chunk["_clean_uid"].isin(sample_uids)].drop(columns=["_clean_uid"])
            if len(kept) > 0:
                chunks_kept.append(kept)

        if chunks_kept:
            result = pd.concat(chunks_kept, ignore_index=True)
        else:
            # CSV vacío con solo headers (preserva schema)
            result = pd.read_csv(src, nrows=0, low_memory=False)

        result.to_csv(out_path, index=False)
        print(f"  escaneadas {rows_scanned} filas → guardado {csv_name}: {len(result)} filas")

    # === CSVs con char_id como FK (fights_log, arena_log) ===
    # Construir set de char_ids del sample desde characters.csv ya guardado.
    chars_local = pd.read_csv(OUTPUT_DIR / "characters.csv", usecols=["_id"], low_memory=False)
    sample_char_ids = set(chars_local["_id"].apply(_extract_uid).dropna())
    print(f"\nN char_ids del sample (para fights/arena): {len(sample_char_ids)}")

    for schema, (csv_name, fk_col) in CSV_FILES_BY_CHAR_ID.items():
        src = DATA_RAW / csv_name
        if not src.exists():
            print(f"  ⚠ {csv_name} no existe en data_raw, saltando")
            continue
        out_path = OUTPUT_DIR / csv_name
        size_mb = src.stat().st_size / 1024 / 1024
        print(f"\nFiltrando {csv_name} ({size_mb:.1f}MB) por {fk_col}…")

        chunks_kept = []
        rows_scanned = 0
        for chunk in pd.read_csv(src, chunksize=200_000, low_memory=False):
            rows_scanned += len(chunk)
            chunk = chunk.copy()
            chunk["_clean_char_id"] = chunk[fk_col].apply(_extract_uid)
            kept = chunk[chunk["_clean_char_id"].isin(sample_char_ids)].drop(columns=["_clean_char_id"])
            if len(kept) > 0:
                chunks_kept.append(kept)

        if chunks_kept:
            result = pd.concat(chunks_kept, ignore_index=True)
        else:
            result = pd.read_csv(src, nrows=0, low_memory=False)

        result.to_csv(out_path, index=False)
        print(f"  escaneadas {rows_scanned} filas → guardado {csv_name}: {len(result)} filas")

    print(f"\n✓ sample_data preparado en {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
