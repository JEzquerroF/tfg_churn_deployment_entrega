"""
prepare_models.py — [DEPRECATED desde Fase 1, 2026-05-19].

Este script fue el "one-shot installer" de iter1, cuando el modelo activo era el
PLACEHOLDER CatBoost L32 v3 (con calibrador isotonic para churn_30d). Copiaba
los modelos desde `tfg/data/data_models/` al deployment y generaba metadata.

Tras el swap a Random Forest L22 v1 (Fase 1, modelo final del TFG):
  - La ruta del modelo en el TFG cambió:
      tfg/data/data_models/                       (placeholder CB L32)
      tfg/06_modelos_publicados/churn/v1_rf_L22_2026-05-19/  (modelo final RF)
  - El formato cambió (.cbm → .pkl dict sklearn)
  - El RF NO tiene calibrador (rechazo de calibración isotonic, está naturalmente
    bien calibrado)
  - Hay un paso adicional: regenerar target_encoder_mappings.json

El flujo actual para swapear modelo es manual y se documenta en el handoff:
  1. Crear dir destino bajo `models/churn/v<N>_<algo>_<fecha>/`
  2. Copiar .pkl + metrics + OOF parquet
  3. Generar feature_list.json + metadata.yaml (snippet python en el handoff)
  4. Ejecutar `scripts/generate_target_encoder_mappings.py` si el modelo
     usa target encoding (RF L22 v1 lo usa)
  5. Apuntar `config/_active_models.yaml` a la nueva versión

Este script sigue siendo válido si se quiere RE-INSTALAR el CB L32 placeholder
legacy (rollback de emergencia). NO se ha actualizado para soportar el formato
RF — su uso queda restringido al modelo antiguo.

Uso (legacy):
    python scripts/prepare_models.py --tfg-root /Users/jezquerro/Documents/tfg

Opciones:
    --dry-run     Solo muestra qué se copiaría, no copia nada
    --overwrite   Sobreescribe si ya existen modelos en destino
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


CHURN_VERSION = "v1_catboost_2026-05-09"
GUSTOS_N1_VERSION = "v1_kmeans_k6_2026-05-12"
# El Nivel 2 (HDBSCAN) se eliminó: la personalización ahora es el perfilado
# de gustos (pipeline/perfilado.py), que no usa modelo entrenado.


def get_file_mapping(tfg_root: Path, deployment_root: Path) -> list[tuple[Path, Path]]:
    return [
        (
            tfg_root / "data/data_models/final_model_catboost_churn_14d.cbm",
            deployment_root / f"models/churn/{CHURN_VERSION}/model_14d.cbm",
        ),
        (
            tfg_root / "data/data_models/final_model_catboost_churn_30d.cbm",
            deployment_root / f"models/churn/{CHURN_VERSION}/model_30d.cbm",
        ),
        (
            tfg_root / "data/data_models/final_calibrator_churn_30d.joblib",
            deployment_root / f"models/churn/{CHURN_VERSION}/calibrator_30d.joblib",
        ),
        (
            tfg_root / "data/data_models/predictions_full_sample_oof.parquet",
            deployment_root / "data/pretrained_user_predictions.parquet",
        ),
        (
            tfg_root / "data/data_models_gustos/nivel1_kmeans_k6.joblib",
            deployment_root / f"models/gustos_nivel1/{GUSTOS_N1_VERSION}/model.joblib",
        ),
        (
            tfg_root / "data/data_models_gustos/preprocessor_tier1.joblib",
            deployment_root / f"models/gustos_nivel1/{GUSTOS_N1_VERSION}/preprocessor.joblib",
        ),
    ]


def generate_churn_metadata(deployment_root: Path) -> None:
    from catboost import CatBoostClassifier

    churn_dir = deployment_root / f"models/churn/{CHURN_VERSION}"

    model = CatBoostClassifier()
    model.load_model(str(churn_dir / "model_14d.cbm"))
    feature_names = list(model.feature_names_)

    feature_list_payload = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "extraction_method": "catboost.feature_names_",
        "extracted_at": datetime.now().isoformat(),
    }
    (churn_dir / "feature_list.json").write_text(
        json.dumps(feature_list_payload, indent=2)
    )
    logger.info("  · feature_list.json (%d features)", len(feature_names))

    metadata = {
        "model_name": "CatBoost L32 v3 (TFG cerrado)",
        "version": CHURN_VERSION,
        "algorithm": "CatBoost",
        "sample_config": {
            "label": "L32",
            "cutoff": 120,
            "spike": 7,
            "min_logins": 5,
            "n_users": 25200,
        },
        "cleanup_version": "v3_aggressive",
        "targets": {
            "churn_14d": {"auc_test": 0.847, "auc_cv": 0.852, "calibration": None},
            "churn_30d": {"auc_test": 0.795, "auc_cv": 0.802, "calibration": "isotonic"},
        },
        "trained_at": "2026-05-09",
        "status": "quarantine_placeholder",
        "notes": (
            "Modelo del TFG original. Pendiente swap a L22 v1 cuando termine "
            "el screening extendido. Sirve como placeholder hasta entonces."
        ),
    }
    (churn_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))
    logger.info("  · metadata.yaml")


def generate_gustos_n1_metadata(deployment_root: Path) -> None:
    import joblib

    n1_dir = deployment_root / f"models/gustos_nivel1/{GUSTOS_N1_VERSION}"

    preprocessor = joblib.load(n1_dir / "preprocessor.joblib")
    feature_names = _extract_feature_names_from_preprocessor(preprocessor, fallback_name="tier1")

    feature_list_payload = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "extraction_method": "preprocessor.get_feature_names_out() o feature_names_in_",
        "extracted_at": datetime.now().isoformat(),
        "note": (
            "Si el modelo se entrenó con `feature_names_in_` originales, "
            "aquí se serializan ESOS nombres (no las features transformadas)."
        ),
    }
    (n1_dir / "feature_list.json").write_text(json.dumps(feature_list_payload, indent=2))
    logger.info("  · feature_list.json (%d features)", len(feature_names))

    metadata = {
        "model_name": "KMeans K=6 — Arquetipos identitarios",
        "version": GUSTOS_N1_VERSION,
        "algorithm": "KMeans",
        "n_clusters": 6,
        "tier": "tier1",
        "training_sample": "114412 jugadores activos últimos 60 días",
        "master_version": "v3_aggressive",
        "metrics": {
            "silhouette": 0.353,
            "davies_bouldin": 1.56,
        },
        "distribution_typical": {
            0: {"name": "🌱 Recién Llegado Explorador", "pct": 6.5},
            1: {"name": "⚔️ Jugador Establecido Activo", "pct": 18.3},
            2: {"name": "👑 Hardcore End-Game", "pct": 5.2},
            3: {"name": "🎯 Veterano Especializado", "pct": 4.6},
            4: {"name": "💤 Casual Dormido", "pct": 58.8},
            5: {"name": "🔧 Veterano Inversor", "pct": 6.5},
        },
        "trained_at": "2026-05-12",
        "status": "production",
    }
    (n1_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True))
    logger.info("  · metadata.yaml")


def _extract_feature_names_from_preprocessor(preprocessor, fallback_name: str) -> list[str]:
    if hasattr(preprocessor, "feature_names_in_"):
        return list(preprocessor.feature_names_in_)

    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            return list(preprocessor.get_feature_names_out())
        except Exception as e:
            logger.warning("get_feature_names_out falló: %s", e)

    if hasattr(preprocessor, "transformers_"):
        names = []
        for _name, _transformer, columns in preprocessor.transformers_:
            if isinstance(columns, list):
                names.extend(columns)
        if names:
            return names

    logger.warning(
        "No se pudo extraer feature_names del preprocessor de %s. "
        "feature_list.json quedará vacío y el pipeline puede fallar al alinear columnas.",
        fallback_name,
    )
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tfg-root",
        type=Path,
        required=True,
        help="Ruta a la raíz del proyecto TFG (ej. /Users/jezquerro/Documents/tfg)",
    )
    parser.add_argument(
        "--deployment-root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Ruta al repo de deployment (por defecto: directorio padre de este script)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Solo lista, no copia")
    parser.add_argument("--overwrite", action="store_true", help="Sobreescribe si ya existe")
    args = parser.parse_args()

    tfg_root = args.tfg_root.expanduser().resolve()
    deployment_root = args.deployment_root.expanduser().resolve()

    if not tfg_root.exists():
        logger.error("❌ TFG root no existe: %s", tfg_root)
        return 1

    logger.info("=" * 70)
    logger.info("PREPARE MODELS")
    logger.info("  TFG root:        %s", tfg_root)
    logger.info("  Deployment root: %s", deployment_root)
    logger.info("  Modo:            %s", "DRY-RUN" if args.dry_run else "COPIA REAL")
    logger.info("=" * 70)

    mapping = get_file_mapping(tfg_root, deployment_root)
    missing_sources = [src for src, _ in mapping if not src.exists()]
    if missing_sources:
        logger.error("❌ Faltan ficheros origen en el TFG:")
        for m in missing_sources:
            logger.error("    %s", m)
        return 1
    logger.info("✓ %d ficheros origen verificados", len(mapping))

    logger.info("\n[1/2] Copiando artefactos…")
    for src, dst in mapping:
        if dst.exists() and not args.overwrite:
            logger.info("  ⚠ %s ya existe (usar --overwrite para reemplazar)", dst.relative_to(deployment_root))
            continue
        if args.dry_run:
            logger.info("  [dry] %s → %s", src.name, dst.relative_to(deployment_root))
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            logger.info("  ✓ %s → %s", src.name, dst.relative_to(deployment_root))

    if args.dry_run:
        logger.info("\n[dry-run] saltando generación de metadata. Re-ejecuta sin --dry-run.")
        return 0

    logger.info("\n[2/2] Generando metadata.yaml + feature_list.json…")
    logger.info("\n  Churn (CatBoost):")
    generate_churn_metadata(deployment_root)

    logger.info("\n  Gustos Nivel 1 (KMeans):")
    generate_gustos_n1_metadata(deployment_root)

    logger.info("\n" + "=" * 70)
    logger.info("✅ MODELOS PREPARADOS")
    logger.info("=" * 70)
    logger.info("\nSiguiente paso: verificar carga con")
    logger.info("    python -c \"from pipeline.model_loader import ModelRegistry; \\")
    logger.info("                 r = ModelRegistry.from_config('config/_active_models.yaml'); \\")
    logger.info("                 r.load_all(); print(r.summary())\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
