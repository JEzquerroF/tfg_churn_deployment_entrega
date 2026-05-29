"""Tests para pipeline.validation.CsvValidator."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.validation import CsvValidator  # noqa: E402

SCHEMA_PATH = ROOT / "config" / "expected_schema.yaml"


def _write_csv(path: Path, columns: list[str], n_rows: int = 3) -> None:
    """Crea un CSV con las columnas dadas y filas de placeholders."""
    df = pd.DataFrame({col: [f"v_{i}" for i in range(n_rows)] for col in columns})
    df.to_csv(path, index=False)


def _schema_columns(schema_name: str) -> list[str]:
    """Devuelve required + optional declaradas en expected_schema.yaml.

    Construye CSVs sintéticos realistas (matching el shape del CSV del TFG),
    no cherry-picks que penalizan el Jaccard.
    """
    cfg = yaml.safe_load(SCHEMA_PATH.read_text())
    s = cfg["schemas"][schema_name]
    return (s.get("required_columns") or []) + (s.get("optional_columns") or [])


def test_identify_three_csvs(tmp_path: Path) -> None:
    """
    Directorio con 3 CSVs:
      - users.csv válido (cols del schema users)
      - characters.csv válido
      - random.csv con cols aleatorias (debería quedar como unrecognized)

    Asserta:
      1. users y characters se identifican correctamente
      2. 1 unrecognized (random.csv)
      3. faltan 7 schemas (devices, iaps_*, daily_rewards, items, collection, feedback)
      4. is_valid=True porque los 7 faltantes son severity 'warn' (no 'fail')
    """
    _write_csv(tmp_path / "users.csv", _schema_columns("users"))
    _write_csv(tmp_path / "characters.csv", _schema_columns("characters"))
    _write_csv(
        tmp_path / "random.csv",
        ["foo", "bar", "baz", "qux", "lorem", "ipsum"],
    )

    validator = CsvValidator(SCHEMA_PATH)
    result = validator.validate_directory(tmp_path)

    assert "users" in result.identifications, (
        f"users no identificado. identifications={list(result.identifications.keys())}"
    )
    assert "characters" in result.identifications, (
        f"characters no identificado. identifications={list(result.identifications.keys())}"
    )
    assert result.identifications["users"].file_path.name == "users.csv"
    assert result.identifications["characters"].file_path.name == "characters.csv"

    assert len(result.unrecognized_files) == 1, (
        f"se esperaba 1 unrecognized, hay {len(result.unrecognized_files)}: "
        f"{[u.file_path.name for u in result.unrecognized_files]}"
    )
    assert result.unrecognized_files[0].file_path.name == "random.csv"

    expected_missing = {
        "devices", "iaps_consumables", "iaps_subscriptions",
        "daily_rewards", "user_items", "user_items_collection",
        "support_feedback",
    }
    assert set(result.missing_schemas) == expected_missing, (
        f"missing_schemas: {result.missing_schemas}"
    )

    assert result.is_valid is True, f"is_valid=False, errors={result.errors}"
    assert len(result.errors) == 0
    assert len(result.warnings) >= 7, (
        f"se esperaban >=7 warnings (los schemas warn faltantes), hay {len(result.warnings)}"
    )


def test_missing_required_csv_users_fails(tmp_path: Path) -> None:
    """Si users.csv no aparece pero sí characters, is_valid=False (severity 'fail' en YAML)."""
    _write_csv(tmp_path / "characters.csv", _schema_columns("characters"))

    validator = CsvValidator(SCHEMA_PATH)
    result = validator.validate_directory(tmp_path)

    assert "characters" in result.identifications, (
        f"characters no identificado: {result.identifications}"
    )
    assert result.is_valid is False
    assert any("users" in e for e in result.errors), f"errors={result.errors}"


def test_empty_directory(tmp_path: Path) -> None:
    """Directorio sin CSVs → is_valid=False con error claro."""
    validator = CsvValidator(SCHEMA_PATH)
    result = validator.validate_directory(tmp_path)
    assert result.is_valid is False
    assert len(result.errors) == 1
    assert "No se encontraron CSVs" in result.errors[0]
