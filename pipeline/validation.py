"""
validation.py — validación y auto-identificación de CSVs del cliente.

El cliente sube CSVs (drag & drop o upload de ZIP) sin garantías de:
  - Nombres exactos (puede llamarse 'users.csv' o 'tabla_usuarios.csv')
  - Orden
  - Que estén todos los esperados

Este módulo:
  1. Itera por todos los .csv del directorio
  2. Para cada uno, lee solo el header
  3. Calcula Jaccard similarity entre las columnas del CSV y las required+optional
     de cada schema
  4. Asigna el schema con mayor similarity (si supera el umbral mínimo)
  5. Valida columnas requeridas, tipos de fecha, etc.
  6. Devuelve un ValidationResult con qué CSV es qué, qué falta, y warnings
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass
class CsvIdentification:
    """Resultado de identificar UN csv."""

    file_path: Path
    identified_as: Optional[str]            # 'users', 'characters', ..., o None si unrecognized
    match_score: float                      # Jaccard similarity
    columns_found: list[str]
    missing_required: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Resultado completo de la validación de un directorio."""

    input_dir: Path
    identifications: dict[str, CsvIdentification]   # schema_name → CsvIdentification
    unrecognized_files: list[CsvIdentification]
    missing_schemas: list[str]                       # Schemas required que no aparecieron
    is_valid: bool                                   # True si pipeline puede correr
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class CsvValidator:
    def __init__(self, schema_config_path: str | Path):
        with open(schema_config_path) as f:
            self.config = yaml.safe_load(f)
        self.schemas = self.config["schemas"]
        self.identification_cfg = self.config["identification"]
        self.fallback = self.config["fallback"]["on_missing_required_csv"]

    def validate_directory(self, input_dir: str | Path) -> ValidationResult:
        """Identifica y valida todos los CSVs en un directorio."""
        input_dir = Path(input_dir)
        if not input_dir.exists():
            return ValidationResult(
                input_dir=input_dir,
                identifications={},
                unrecognized_files=[],
                missing_schemas=list(self.schemas.keys()),
                is_valid=False,
                errors=[f"Directorio no existe: {input_dir}"],
            )

        csv_files = sorted(input_dir.glob("*.csv"))
        if not csv_files:
            return ValidationResult(
                input_dir=input_dir,
                identifications={},
                unrecognized_files=[],
                missing_schemas=list(self.schemas.keys()),
                is_valid=False,
                errors=[f"No se encontraron CSVs en {input_dir}"],
            )

        identifications: dict[str, CsvIdentification] = {}
        unrecognized: list[CsvIdentification] = []

        for csv_path in csv_files:
            ident = self._identify_csv(csv_path)
            if ident.identified_as is None:
                unrecognized.append(ident)
                continue

            if ident.identified_as in identifications:
                # Dos CSVs identificados como el mismo schema → conflicto, gana el de mayor score
                existing = identifications[ident.identified_as]
                if ident.match_score > existing.match_score:
                    unrecognized.append(existing)
                    identifications[ident.identified_as] = ident
                else:
                    unrecognized.append(ident)
            else:
                identifications[ident.identified_as] = ident

        # Detectar schemas que faltan + decidir si es fatal según fallback
        present = set(identifications.keys())
        all_required = set(self.schemas.keys())
        missing = sorted(all_required - present)

        errors: list[str] = []
        warnings: list[str] = []
        for schema in missing:
            severity = self.fallback.get(schema, "warn")
            msg = f"CSV de schema '{schema}' no encontrado"
            if severity == "fail":
                errors.append(msg)
            else:
                warnings.append(msg)

        # Validar columnas requeridas en los que SÍ se identificaron
        for schema_name, ident in identifications.items():
            schema = self.schemas[schema_name]
            required = set(schema.get("required_columns", []) or [])
            found = set(ident.columns_found)
            ident.missing_required = sorted(required - found)
            if ident.missing_required:
                errors.append(
                    f"CSV identificado como '{schema_name}' ({ident.file_path.name}) "
                    f"le faltan columnas requeridas: {ident.missing_required}"
                )

        return ValidationResult(
            input_dir=input_dir,
            identifications=identifications,
            unrecognized_files=unrecognized,
            missing_schemas=missing,
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _identify_csv(self, csv_path: Path) -> CsvIdentification:
        """Lee solo el header del CSV y lo matchea contra los schemas conocidos."""
        try:
            cols = list(pd.read_csv(csv_path, nrows=0).columns)
        except Exception as e:
            return CsvIdentification(
                file_path=csv_path,
                identified_as=None,
                match_score=0.0,
                columns_found=[],
                warnings=[f"Error leyendo CSV: {e}"],
            )

        col_set = set(cols)
        best_match: tuple[Optional[str], float] = (None, 0.0)

        for schema_name, schema in self.schemas.items():
            required = set(schema.get("required_columns", []) or [])
            optional = set(schema.get("optional_columns", []) or [])
            relevant = required | optional
            if not relevant:
                continue
            intersection = col_set & relevant
            union = col_set | relevant
            jaccard = len(intersection) / len(union) if union else 0.0
            # Bonus: si todas las required están en el CSV → +0.1
            if required and required.issubset(col_set):
                jaccard = min(1.0, jaccard + 0.1)
            if jaccard > best_match[1]:
                best_match = (schema_name, jaccard)

        threshold = self.identification_cfg["min_match_score"]
        warn_below = self.identification_cfg["warn_below"]

        if best_match[1] < threshold:
            return CsvIdentification(
                file_path=csv_path,
                identified_as=None,
                match_score=best_match[1],
                columns_found=cols,
            )

        warnings: list[str] = []
        if best_match[1] < warn_below:
            warnings.append(f"Match débil ({best_match[1]:.2f}) con schema '{best_match[0]}'")

        return CsvIdentification(
            file_path=csv_path,
            identified_as=best_match[0],
            match_score=best_match[1],
            columns_found=cols,
            warnings=warnings,
        )
