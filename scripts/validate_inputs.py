"""
validate_inputs.py — valida un directorio de CSVs del cliente.

Uso:
    python scripts/validate_inputs.py --input /path/to/csvs/

Sale con código 0 si todo OK, 1 si hay errores.
Imprime un informe legible en stdout con qué CSV es qué, qué falta, y warnings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Añadir el repo al path para que `pipeline.*` funcione cuando se llama
# directamente con `python scripts/validate_inputs.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.validation import CsvValidator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directorio con CSVs")
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config" / "expected_schema.yaml",
        help="Ruta al expected_schema.yaml (default: config/expected_schema.yaml)",
    )
    args = parser.parse_args()

    validator = CsvValidator(args.schema)
    result = validator.validate_directory(args.input)

    print("=" * 70)
    print("VALIDACIÓN DE INPUTS")
    print(f"  Directorio: {result.input_dir}")
    print("=" * 70)

    if result.identifications:
        print("\n✓ CSVs identificados:")
        for schema_name, ident in result.identifications.items():
            warn_flag = " ⚠" if ident.warnings else ""
            print(f"  · {schema_name}: {ident.file_path.name} (match={ident.match_score:.2f}){warn_flag}")
            for w in ident.warnings:
                print(f"      ⚠ {w}")

    if result.unrecognized_files:
        print("\n? CSVs no identificados:")
        for ident in result.unrecognized_files:
            print(f"  · {ident.file_path.name} (mejor match: {ident.match_score:.2f})")

    if result.missing_schemas:
        print("\n✗ Schemas requeridos no encontrados:")
        for schema in result.missing_schemas:
            print(f"  · {schema}")

    if result.warnings:
        print("\n⚠ Warnings:")
        for w in result.warnings:
            print(f"  · {w}")

    if result.errors:
        print("\n❌ Errores:")
        for e in result.errors:
            print(f"  · {e}")

    print()
    if result.is_valid:
        print("✅ Inputs válidos. Pipeline puede ejecutarse.")
        return 0
    print("❌ Inputs inválidos. Corrige los errores antes de ejecutar el pipeline.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
