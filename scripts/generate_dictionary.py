"""
generate_dictionary.py — Genera diccionario.json para integración del cliente.

El diccionario traduce códigos numéricos (archetype_n1/n2 codes, source codes)
a significados legibles + IDs de contramedidas estables. Permite al cliente
integrar los outputs en su código de juego sin depender de strings con
emojis o sufijos cambiantes.

Uso CLI (standalone):
    python scripts/generate_dictionary.py --output /path/to/diccionario.json

Uso programático (desde predict.py):
    from scripts.generate_dictionary import build_dictionary
    diccionario = build_dictionary(repo_root)
    with open(output_dir / "diccionario.json", "w", encoding="utf-8") as f:
        json.dump(diccionario, f, indent=2, ensure_ascii=False)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

# Añadir repo al path para imports relativos cuando se ejecute como script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Valores posibles de cada eje de gusto (contrato estable de integración).
EJE_VALUES = {
    "clase_favorita": ["Caballero", "Asesino", "Campeón"],
    "estilo_build": ["tanque", "equilibrado", "agresivo"],
    "monetizacion": ["trial_no_convertido", "pagador", "no_pagador_ads", "no_pagador"],
    "perfil_enhance": ["inversor", "no_inversor"],
    "perfil_coleccion": ["coleccionista", "funcional"],
    "pvp_perfil": ["pvp_frustrado", "pve_focus", "null"],
    "perfil_oro": ["acumulador", "gastador", "neutro", "null"],
}


def build_dictionary(repo_root: Path) -> dict:
    """
    Construye el diccionario completo en formato dict (serializable a JSON).

    Args:
        repo_root: ruta al repo de deployment.

    Returns:
        dict con claves: version, generated_at, model_version,
        schema_description, archetype_n1, ejes_gusto, countermeasures,
        source_codes.
    """
    with open(repo_root / "config" / "archetypes.yaml") as f:
        archetypes_cfg = yaml.safe_load(f)

    with open(repo_root / "config" / "_active_models.yaml") as f:
        active_models = yaml.safe_load(f)

    churn_version = active_models["churn"]["active_version"]

    # archetype_n1 (los 6 clusters del KMeans).
    # Claves del JSON como string (compat con consumidores JS/web), pero `code`
    # numérico dentro del payload para uso programático.
    archetype_n1 = {}
    for cluster_id, cfg in archetypes_cfg["archetypes_n1"].items():
        archetype_n1[str(cluster_id)] = {
            "code": int(cluster_id),
            "name": cfg["name"],
            "business_priority": cfg.get("business_priority", "media"),
        }

    # countermeasures: catálogo de 12 (CM01-CM12) con id estable, label,
    # mecánica, prioridad y disparador. Reemplaza al antiguo set de 6 genéricas
    # por arquetipo. El cliente mapea el cod (CM0X) a funciones de su juego.
    contramedidas = sorted(
        archetypes_cfg.get("contramedidas", []),
        key=lambda cm: cm.get("prioridad", 99),
    )
    countermeasures = {
        cm["id"]: {
            "cod": cm["id"],
            "label": cm["label"],
            "prioridad": cm.get("prioridad"),
            "mecanica": cm.get("mecanica", ""),
            "disparador": " ".join(str(cm.get("disparador", "")).split()),
        }
        for cm in contramedidas
    }

    # ejes_gusto: los 7 ejes de perfilado con sus valores posibles. El eje es
    # una columna de predictions_full.csv; el cliente lo usa para segmentar.
    ejes_gusto = {}
    for eje, eje_cfg in archetypes_cfg.get("ejes_gusto", {}).items():
        ejes_gusto[eje] = {
            "fuente": eje_cfg.get("fuente", ""),
            "requiere_tier2": bool(eje_cfg.get("requiere_tier2", False)),
            "valores": EJE_VALUES.get(eje, []),
        }

    # Source codes producidos por la lógica de drift (Fase 1.5).
    source_codes = {
        "oof_stable": {
            "description": (
                "Predicción histórica del entrenamiento. Los datos del jugador "
                "no han cambiado lo suficiente como para invalidar la OOF."
            ),
            "recommended_action": "Confiar en la predicción.",
        },
        "live_drift": {
            "description": (
                "Predicción actualizada. El jugador ha cambiado significativamente "
                "respecto al entrenamiento (delta > threshold)."
            ),
            "recommended_action": "Atender al cambio. La predicción nueva refleja "
                                  "mejor su estado actual.",
        },
        "live_new": {
            "description": (
                "Predicción nueva. El jugador no estaba en el sample de "
                "entrenamiento del modelo."
            ),
            "recommended_action": "Confiar en la predicción nueva.",
        },
    }

    return {
        "version": "2.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": churn_version,
        "schema_description": (
            "Diccionario de códigos para integración cliente. Los códigos de "
            "arquetipo (archetype_n1.code) y de contramedida (CM01-CM12) son "
            "estables e inmutables — el cliente los mapea a funciones de su "
            "juego. La columna contramedida_primaria_cod de predictions.csv "
            "referencia una de las 12 contramedidas. Los 7 ejes de gusto "
            "(predictions_full.csv) describen el perfil del jugador."
        ),
        "archetype_n1": archetype_n1,
        "ejes_gusto": ejes_gusto,
        "countermeasures": countermeasures,
        "source_codes": source_codes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Ruta para diccionario.json")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    args = parser.parse_args()

    diccionario = build_dictionary(args.repo_root)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(diccionario, f, indent=2, ensure_ascii=False)

    print(f"Diccionario generado: {args.output}")
    print(f"  Modelo: {diccionario['model_version']}")
    print(f"  Arquetipos N1: {len(diccionario['archetype_n1'])}")
    print(f"  Ejes de gusto: {len(diccionario['ejes_gusto'])}")
    print(f"  Contramedidas: {len(diccionario['countermeasures'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
