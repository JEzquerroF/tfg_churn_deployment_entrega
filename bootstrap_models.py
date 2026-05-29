"""
bootstrap_models.py — descarga los modelos pesados desde HF Hub si faltan.

Por qué existe:
    HF Spaces rechaza ficheros >10 MiB sin LFS. Los modelos serializados
    del TFG (RF L22 .pkl y HDBSCAN .joblib) los superan. La solución es
    subir esos artefactos a un HF Models repo aparte y descargarlos en
    runtime, manteniendo el repo del Space ligero (<10 MiB/archivo).

Diseño:
    - Idempotente: si el fichero ya existe en local, skip.
    - No toca `pipeline/`: el ModelRegistry los carga del filesystem
      como siempre, sin saber que vienen de fuera.
    - Se invoca desde `app.py` al arranque del Space.

Uso desde Python:
    from bootstrap_models import ensure_models_present
    ensure_models_present()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent

# Repo HF Models donde viven los artefactos pesados.
# Pública, sin token. Si en el futuro pasa a privada, añadir
# HF_TOKEN env var al Space y pasarla a hf_hub_download(token=...).
HF_MODELS_REPO_ID = "JEzquerroF/tfg-churn-models"
HF_MODELS_REVISION = "main"

# Mapa filename_en_repo → ruta local (relativa a REPO_ROOT).
# El filename plano evita tener que replicar la jerarquía de carpetas
# del repo de modelos en el HF Models repo.
# El modelo HDBSCAN de Nivel 2 (gustos_n2_hdbscan_model.joblib) ya NO se
# descarga: el Nivel 2 se eliminó (sustituido por el perfilado de gustos,
# que no usa modelo entrenado, solo reglas sobre features). El fichero sigue
# existiendo en el HF Models repo pero es inerte.
REMOTE_FILES: dict[str, str] = {
    "model_7d.pkl":  "models/churn/v2_rf_L22_2026-05-19/model_7d.pkl",
    "model_14d.pkl": "models/churn/v2_rf_L22_2026-05-19/model_14d.pkl",
    "model_30d.pkl": "models/churn/v2_rf_L22_2026-05-19/model_30d.pkl",
    "gustos_n1_kmeans_model.joblib":
        "models/gustos_nivel1/v1_kmeans_k6_2026-05-12/model.joblib",
    "pretrained_user_predictions.parquet":
        "data/pretrained_user_predictions.parquet",
}


def _resolve(path_str: str) -> Path:
    return (REPO_ROOT / path_str).resolve()


def ensure_models_present(
    repo_id: str = HF_MODELS_REPO_ID,
    revision: str = HF_MODELS_REVISION,
    token: Optional[str] = None,
) -> list[Path]:
    """
    Garantiza que los artefactos pesados están en su ruta local.

    Returns:
        Lista de paths descargados (vacía si todos ya estaban).
    """
    downloaded: list[Path] = []
    missing: list[tuple[str, Path]] = []

    for filename, local_rel in REMOTE_FILES.items():
        local = _resolve(local_rel)
        if local.exists() and local.stat().st_size > 0:
            logger.debug("Modelo ya en local: %s", local)
            continue
        missing.append((filename, local))

    if not missing:
        logger.info("Todos los modelos remotos ya están en local.")
        return downloaded

    # Importación tardía: huggingface_hub solo se necesita si hay descargas.
    from huggingface_hub import hf_hub_download

    logger.info(
        "Descargando %d modelo(s) desde %s (revision=%s)...",
        len(missing), repo_id, revision,
    )

    for filename, local in missing:
        local.parent.mkdir(parents=True, exist_ok=True)
        logger.info("  · %s → %s", filename, local)
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=str(local.parent),
            token=token,
        )
        # hf_hub_download devuelve la ruta donde escribió. Si el nombre
        # remoto difiere del nombre local, renombramos.
        downloaded_path = Path(downloaded_path)
        if downloaded_path.name != local.name:
            downloaded_path.rename(local)
        downloaded.append(local)

    logger.info("Descarga de modelos completada: %d ficheros.", len(downloaded))
    return downloaded


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = ensure_models_present()
    if paths:
        print(f"Descargados {len(paths)} modelos:")
        for p in paths:
            print(f"  - {p}")
    else:
        print("Todos los modelos ya estaban en local.")
