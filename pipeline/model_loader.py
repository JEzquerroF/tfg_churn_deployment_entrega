"""
ModelRegistry polimórfico.

Carga modelos heterogéneos (sklearn RF en dict, CatBoost, KMeans) detrás de
una interfaz unificada. El pipeline NO sabe qué algoritmo hay detrás — solo
llama a `registry.predict_churn(X)` o `registry.assign_archetype_n1(X)`.

Nota: el antiguo Nivel 2 (HDBSCAN de sub-arquetipos) se eliminó. La capa de
personalización ahora es el perfilado de gustos (pipeline/perfilado.py).

Esto permite que al hacer swap de modelos (ej. CatBoost L32 placeholder →
Random Forest L22 v1 final del TFG), sólo cambia el YAML
`_active_models.yaml` y el `ModelRegistry` carga la nueva versión sin tocar
el código del pipeline.

Formatos de churn soportados:
- `model_<target_suffix>.pkl` — dict sklearn {model, feature_cols, cat_cols, ...}
- `model_<target_suffix>.cbm` — CatBoostClassifier nativo (legacy)

Si junto al modelo hay `target_encoder_mappings.json`, se carga como
`TargetEncoderMappings` y se aplica en `predict_churn` antes del predict.
Si hay `preprocessor.joblib`, también se aplica (en este orden:
target_encoder → preprocessor → predict_proba → calibrator).

Carga:
    registry = ModelRegistry.from_config('config/_active_models.yaml')
    registry.load_all()

Uso:
    probs_30d = registry.predict_churn(df_features, target='churn_30d')
    clusters = registry.assign_archetype_n1(df_features_tier1)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml

from pipeline.target_encoder import TargetEncoderMappings

logger = logging.getLogger(__name__)


@dataclass
class ModelArtifact:
    """Un modelo cargado en memoria + sus metadatos."""

    name: str
    version: str
    path: Path
    model: Any
    preprocessor: Optional[Any]
    calibrator: Optional[Any]
    feature_list: list[str]
    metadata: dict
    target_encoder: Optional[TargetEncoderMappings] = None
    cat_cols: list[str] = field(default_factory=list)


class ModelRegistry:
    """
    Registro central de los modelos activos.

    Carga todos los modelos al inicio (no lazy) para detectar errores temprano.
    La asignación de versiones se lee de `_active_models.yaml` — no se hard-codea.
    """

    def __init__(self, config: dict, base_dir: Path) -> None:
        self._config = config
        self._base_dir = base_dir
        self._artifacts: dict[str, ModelArtifact] = {}
        self._oof_lookup: Optional[pd.DataFrame] = None

    @classmethod
    def from_config(cls, config_path: str | Path, base_dir: str | Path = ".") -> "ModelRegistry":
        config_path = Path(config_path)
        base_dir = Path(base_dir)
        with config_path.open() as f:
            config = yaml.safe_load(f)
        return cls(config=config, base_dir=base_dir)

    def load_all(self) -> None:
        logger.info("Cargando modelos activos…")
        self._load_churn_models()
        self._load_gustos_n1()
        self._load_oof_lookup()
        logger.info("Modelos cargados: %s", list(self._artifacts.keys()))

    # ============================================================
    # Carga de modelos de churn (polimórfica)
    # ============================================================

    def _load_churn_models(self) -> None:
        churn_cfg = self._config["churn"]
        churn_dir = self._base_dir / churn_cfg["path"]
        version = churn_cfg["active_version"]

        # Target encoder compartido por los N targets (uno por dir de modelo).
        # Cargado una sola vez para reutilizarlo en cada artifact.
        target_encoder = self._load_target_encoder_if_present(churn_dir)

        # Preprocessor genérico compartido por los N targets (opcional).
        preprocessor = None
        preprocessor_file = churn_dir / "preprocessor.joblib"
        if preprocessor_file.exists():
            preprocessor = joblib.load(preprocessor_file)
            logger.info("  · Preprocessor cargado para churn")

        for target in churn_cfg["targets"]:
            target_suffix = target.split("_")[1]  # "7d", "14d", "30d"
            pkl_path = churn_dir / f"model_{target_suffix}.pkl"
            cbm_path = churn_dir / f"model_{target_suffix}.cbm"

            if pkl_path.exists():
                model, feature_list, cat_cols, metadata_extra = self._load_sklearn_artifact(pkl_path)
            elif cbm_path.exists():
                model, feature_list, cat_cols, metadata_extra = self._load_catboost_model(cbm_path)
            else:
                raise FileNotFoundError(
                    f"Modelo {target} no encontrado: ni {pkl_path} ni {cbm_path}"
                )

            # Calibrador (opcional, por target). RF L22 v1 no usa.
            calibrator = None
            calibrator_file = churn_dir / f"calibrator_{target_suffix}.joblib"
            if calibrator_file.exists():
                calibrator = joblib.load(calibrator_file)
                logger.info("  · Calibrator cargado para %s", target)

            metadata = self._load_metadata(churn_dir)
            metadata.update(metadata_extra)

            self._artifacts[target] = ModelArtifact(
                name=target,
                version=version,
                path=churn_dir,
                model=model,
                preprocessor=preprocessor,
                calibrator=calibrator,
                feature_list=feature_list,
                metadata=metadata,
                target_encoder=target_encoder,
                cat_cols=cat_cols,
            )
            logger.info(
                "  · %s (%s) cargado [%d features, cat_cols=%s]",
                target,
                version,
                len(feature_list),
                "ninguna" if not cat_cols else len(cat_cols),
            )

    @staticmethod
    def _load_sklearn_artifact(
        pkl_path: Path,
    ) -> Tuple[Any, list[str], list[str], dict]:
        """
        Carga un artifact sklearn serializado como dict.

        Estructura esperada (del TFG RF L22 v1):
            {model, feature_cols, cat_cols, best_params?, target?, sample?, cleanup?}

        Devuelve: (model, feature_list, cat_cols, metadata_extra)
        """
        artifact = joblib.load(pkl_path)
        if not isinstance(artifact, dict):
            raise ValueError(
                f"Artifact en {pkl_path} no es dict, es {type(artifact).__name__}"
            )
        if "model" not in artifact:
            raise KeyError(f"Artifact en {pkl_path} no tiene clave 'model'")

        model = artifact["model"]
        feature_list = list(artifact.get("feature_cols", []))
        cat_cols = list(artifact.get("cat_cols", []))
        metadata_extra = {
            "best_params": artifact.get("best_params"),
            "sample": artifact.get("sample"),
            "cleanup": artifact.get("cleanup"),
            "cat_cols": cat_cols,
            "format": "sklearn_pkl_dict",
        }
        return model, feature_list, cat_cols, metadata_extra

    @staticmethod
    def _load_catboost_model(
        cbm_path: Path,
    ) -> Tuple[Any, list[str], list[str], dict]:
        """
        Carga un modelo CatBoost serializado en formato .cbm nativo.

        Devuelve: (model, feature_list, cat_cols, metadata_extra)
        """
        from catboost import CatBoostClassifier

        model = CatBoostClassifier()
        model.load_model(str(cbm_path))
        feature_list = list(model.feature_names_)
        cat_idx = (
            model.get_cat_feature_indices()
            if hasattr(model, "get_cat_feature_indices")
            else []
        )
        cat_cols = [feature_list[i] for i in cat_idx]
        metadata_extra = {"cat_cols": cat_cols, "format": "catboost_cbm"}
        return model, feature_list, cat_cols, metadata_extra

    @staticmethod
    def _load_target_encoder_if_present(model_dir: Path) -> Optional[TargetEncoderMappings]:
        encoder_path = model_dir / "target_encoder_mappings.json"
        if not encoder_path.exists():
            return None
        te = TargetEncoderMappings.from_json(encoder_path)
        logger.info(
            "  · TargetEncoder cargado: %d cat_cols, %d targets",
            len(te.cat_cols),
            len(te.per_target),
        )
        return te

    # ============================================================
    # Carga de modelos de gustos
    # ============================================================

    @staticmethod
    def _unwrap_gustos_artifact(raw: Any) -> Any:
        """
        El joblib del modelo de gustos del TFG es un dict
        `{'model': KMeans, 'labels': ..., 'user_ids': ...}`.
        Extrae el estimador real. Si el joblib ya es el estimador, lo
        devuelve sin tocar.
        """
        if isinstance(raw, dict) and "model" in raw:
            return raw["model"]
        return raw

    def _load_gustos_n1(self) -> None:
        cfg = self._config["gustos_nivel1"]
        nivel1_dir = self._base_dir / cfg["path"]

        model_raw = joblib.load(nivel1_dir / "model.joblib")
        model = self._unwrap_gustos_artifact(model_raw)
        preprocessor = joblib.load(nivel1_dir / "preprocessor.joblib")
        feature_list = self._load_feature_list(nivel1_dir)
        metadata = self._load_metadata(nivel1_dir)

        self._artifacts["gustos_nivel1"] = ModelArtifact(
            name="gustos_nivel1",
            version=cfg["active_version"],
            path=nivel1_dir,
            model=model,
            preprocessor=preprocessor,
            calibrator=None,
            feature_list=feature_list,
            metadata=metadata,
        )
        logger.info(
            "  · gustos_nivel1 (%s) cargado [%s]",
            cfg["active_version"],
            type(model).__name__,
        )

    # ============================================================
    # OOF lookup
    # ============================================================

    def _load_oof_lookup(self) -> None:
        oof_cfg = self._config.get("oof_lookup", {})
        if not oof_cfg.get("enabled", False):
            logger.info("  · OOF lookup deshabilitado")
            return

        oof_path = self._base_dir / oof_cfg["path"]
        if not oof_path.exists():
            logger.warning("  · OOF lookup habilitado pero parquet no encontrado: %s", oof_path)
            return

        self._oof_lookup = pd.read_parquet(oof_path)
        logger.info("  · OOF lookup cargado: %d jugadores", len(self._oof_lookup))

    @staticmethod
    def _load_feature_list(model_dir: Path) -> list[str]:
        feature_list_path = model_dir / "feature_list.json"
        if not feature_list_path.exists():
            logger.warning("feature_list.json no encontrado en %s", model_dir)
            return []
        with feature_list_path.open() as f:
            payload = json.load(f)
        return payload.get("feature_names", [])

    @staticmethod
    def _load_metadata(model_dir: Path) -> dict:
        metadata_path = model_dir / "metadata.yaml"
        if not metadata_path.exists():
            return {}
        with metadata_path.open() as f:
            return yaml.safe_load(f) or {}

    # ============================================================
    # Inference
    # ============================================================

    def predict_churn(
        self, X: pd.DataFrame, target: str = "churn_30d", apply_calibration: bool = True
    ) -> np.ndarray:
        if target not in self._artifacts:
            raise KeyError(
                f"Modelo {target} no cargado. Disponibles: {list(self._artifacts.keys())}"
            )

        art = self._artifacts[target]
        X_ordered = self._align_columns(X, art.feature_list, model_name=target)

        # Target encoding declarativo (RF L22 v1): convierte cat_cols string a float.
        if art.target_encoder is not None:
            X_ordered = art.target_encoder.transform(X_ordered, target=target)

        # Preprocessor genérico (opcional).
        if art.preprocessor is not None:
            X_ordered = art.preprocessor.transform(X_ordered)

        probs = art.model.predict_proba(X_ordered)[:, 1]

        if apply_calibration and art.calibrator is not None:
            probs = art.calibrator.transform(probs)

        return probs

    def assign_archetype_n1(self, X_tier1: pd.DataFrame) -> np.ndarray:
        art = self._artifacts["gustos_nivel1"]
        X_ordered = self._align_columns(X_tier1, art.feature_list, model_name="gustos_nivel1")
        X_scaled = art.preprocessor.transform(X_ordered)
        return art.model.predict(X_scaled)

    def lookup_oof(self, user_ids: pd.Series) -> pd.DataFrame:
        """
        Devuelve un DataFrame con cols canónicas
        `churn_prob_{7d,14d,30d}_oof`.

        El parquet del RF L22 v1 trae las cols con nombres `p_churn_<target>`
        + targets reales (`y_churn_*`) y metadata del split (`split`, `sample`,
        `cleanup`, `algorithm`). Aquí renombramos a la nomenclatura canónica del
        deployment y descartamos lo demás.
        """
        cols_canonical = [
            "user_id",
            "churn_prob_7d_oof",
            "churn_prob_14d_oof",
            "churn_prob_30d_oof",
        ]
        if self._oof_lookup is None:
            return pd.DataFrame(
                {
                    "user_id": user_ids,
                    "churn_prob_7d_oof": np.nan,
                    "churn_prob_14d_oof": np.nan,
                    "churn_prob_30d_oof": np.nan,
                }
            )

        # Rename múltiple para soportar ambos schemas (legacy CB + nuevo RF).
        rename_map = {
            # Schema RF L22 v1 (actual)
            "p_churn_7d": "churn_prob_7d_oof",
            "p_churn_14d": "churn_prob_14d_oof",
            "p_churn_30d": "churn_prob_30d_oof",
            # Schema CB L32 (legacy, mantener compat por si se vuelve atrás)
            "prob_churn_7d_oof": "churn_prob_7d_oof",
            "prob_churn_14d_oof": "churn_prob_14d_oof",
            "prob_churn_30d_oof": "churn_prob_30d_oof",
        }
        oof = self._oof_lookup.rename(columns=rename_map)
        result = pd.DataFrame({"user_id": user_ids}).merge(
            oof[[c for c in cols_canonical if c in oof.columns]],
            on="user_id",
            how="left",
        )
        return result

    @staticmethod
    def _align_columns(
        X: pd.DataFrame, expected: list[str], model_name: str
    ) -> pd.DataFrame:
        if not expected:
            logger.warning("Sin feature_list para %s, pasando X sin reordenar", model_name)
            return X

        missing = set(expected) - set(X.columns)
        if missing:
            raise ValueError(
                f"Faltan columnas esperadas por {model_name}: {sorted(missing)[:10]}"
                f"{' …' if len(missing) > 10 else ''}"
            )
        return X[expected]

    def summary(self) -> dict:
        return {
            name: {
                "version": art.version,
                "n_features": len(art.feature_list),
                "has_calibrator": art.calibrator is not None,
                "has_preprocessor": art.preprocessor is not None,
                "has_target_encoder": art.target_encoder is not None,
                "cat_cols": art.cat_cols,
                "metadata": art.metadata,
            }
            for name, art in self._artifacts.items()
        }

    def __repr__(self) -> str:
        loaded = list(self._artifacts.keys())
        return f"<ModelRegistry loaded={loaded}>"
