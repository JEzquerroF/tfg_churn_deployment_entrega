"""
predict.py — pipeline end-to-end de inferencia.

Uso:
    python scripts/predict.py \\
        --input /path/to/csvs/ \\
        --output /path/to/results/ \\
        --config config/pipeline_config.yaml

Etapas:
    1. Validar inputs (CsvValidator)
    2. Cargar modelos (ModelRegistry)
    3. Generar features + master table
    4. Predecir churn live (3 targets)         → persiste _intermediate_*.parquet
    5. OOF lookup con drift detection          → persiste _intermediate_*.parquet
    6. Asignar arquetipo N1 + perfilado de gustos (7 ejes + contramedida
       primaria del catálogo de 12). try/except: si falla, archetype_*/perfil=NaN.
    7. Escribir outputs: predictions.csv, predictions_full.csv, summary.json,
       diccionario.json, _run_metadata.json

Filosofía:
    - El sistema NO binariza. Entrega probabilidades + descripciones.
    - Persistencia incremental: cada stage escribe su artifact apenas termina.
    - Si stage [6/6] falla, el run NO aborta — se generan outputs sin arquetipo.
    - La binarización a 0/1 y el risk_level viven en el frontend (D1b).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# Añadir repo al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# crossing.cross_and_segment: pausa hasta D2 (informes). El frontend D1b
# genera risk_level/priority dinámicamente vía sliders. NO se importa aquí.
from pipeline import clustering, lookup, master_builder, perfilado, prediction  # noqa: E402
from pipeline.feature_pipeline_users import _extract_user_id  # noqa: E402
from pipeline.lookup import apply_oof_lookup_with_drift, load_drift_threshold  # noqa: E402
from pipeline.model_loader import ModelRegistry  # noqa: E402
from pipeline.pipeline_context import PipelineContext  # noqa: E402
from pipeline.validation import CsvValidator  # noqa: E402
from scripts.generate_dictionary import build_dictionary  # noqa: E402


def setup_logging(log_dir: Path, verbose: bool = False) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"predict_{timestamp}.log"

    handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
    return log_file


def _target_suffix(target: str) -> str:
    return target.split("_", 1)[1]


def _per_target_summary(predictions_dual: pd.DataFrame, targets: list[str]) -> dict:
    """Distribución de _source + final mean/median + delta stats por target."""
    out: dict = {}
    for t in targets:
        source_col = f"{t}_source"
        final_col = f"{t}_final"
        delta_col = f"{t}_delta"
        if source_col not in predictions_dual.columns:
            continue

        source_dist = predictions_dual[source_col].value_counts().to_dict()
        final_mean = float(predictions_dual[final_col].mean())
        final_median = float(predictions_dual[final_col].median())

        delta_nonan = predictions_dual[delta_col].dropna()
        delta_stats = (
            {
                "mean": float(delta_nonan.mean()),
                "std": float(delta_nonan.std()),
                "min": float(delta_nonan.min()),
                "max": float(delta_nonan.max()),
                "n_observed": int(len(delta_nonan)),
            }
            if len(delta_nonan) > 0
            else {"n_observed": 0}
        )

        out[t] = {
            "source_distribution": source_dist,
            "final_mean": final_mean,
            "final_median": final_median,
            "delta_stats": delta_stats,
        }
    return out


def _apply_archetype_name(
    df: pd.DataFrame, archetypes_yaml_path: Path
) -> pd.DataFrame:
    """Añade columna `archetype_name` resolviendo `archetype_n1` contra el YAML."""
    with open(archetypes_yaml_path) as f:
        arch_cfg = (yaml.safe_load(f) or {}).get("archetypes_n1", {}) or {}

    def _name(x):
        if pd.isna(x):
            return "N/A"
        try:
            return arch_cfg.get(int(x), {}).get("name", "Unknown")
        except (TypeError, ValueError):
            return "Unknown"

    df = df.copy()
    df["archetype_name"] = df["archetype_n1"].apply(_name)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directorio con CSVs del cliente")
    parser.add_argument("--output", type=Path, required=True, help="Directorio para outputs")
    parser.add_argument("--config", type=Path, default=Path("config/pipeline_config.yaml"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    log_dir = args.output / "logs"
    log_file = setup_logging(log_dir, verbose=args.verbose)
    logger = logging.getLogger(__name__)

    t_start = time.time()
    logger.info("=" * 70)
    logger.info("PIPELINE DE INFERENCIA")
    logger.info("  Input:  %s", args.input)
    logger.info("  Output: %s", args.output)
    logger.info("  Log:    %s", log_file)
    logger.info("=" * 70)

    repo_root = Path(__file__).resolve().parent.parent

    try:
        # === 1. VALIDAR INPUTS ===
        logger.info("[1/6] Validando inputs…")
        validator = CsvValidator(repo_root / "config" / "expected_schema.yaml")
        val_result = validator.validate_directory(args.input)
        if not val_result.is_valid:
            logger.error("❌ Inputs inválidos:")
            for e in val_result.errors:
                logger.error("  · %s", e)
            return 1
        logger.info("✓ %d schemas identificados", len(val_result.identifications))

        # === 2. CARGAR MODELOS ===
        logger.info("[2/6] Cargando modelos…")
        active_models_path = repo_root / "config" / "_active_models.yaml"
        with open(active_models_path) as f:
            active_models = yaml.safe_load(f)
        churn_targets = list(active_models["churn"]["targets"])
        logger.info("  targets de churn configurados: %s", churn_targets)

        registry = ModelRegistry.from_config(active_models_path, base_dir=repo_root)
        registry.load_all()

        # === 3. GENERAR FEATURES + MASTER ===
        logger.info("[3/6] Generando features + master table…")

        users_path = val_result.identifications["users"].file_path
        users_df = pd.read_csv(users_path, low_memory=False)
        last_login_dt = pd.to_datetime(
            users_df["last_login_date"], unit="s", errors="coerce", utc=True
        ).dt.tz_localize(None)
        created_at_dt = pd.to_datetime(
            users_df["created_at"], errors="coerce", utc=True
        ).dt.tz_localize(None)
        reference_date = last_login_dt.max().date()
        logger.info("  reference_date derivada: %s", reference_date)

        sample_cfg = config.get("sample", {}) or {}
        ctx = PipelineContext(
            raw_csvs_dir=args.input,
            reference_date=reference_date,
            cutoff_days=sample_cfg.get("cutoff_days", 90),
            spike_days=sample_cfg.get("spike_days", 7),
            min_logins=sample_cfg.get("min_logins", 2),
        )
        cutoff_date = datetime.combine(ctx.cutoff_date, datetime.min.time())
        logger.info("  cutoff_date (reference - %dd): %s", ctx.cutoff_days, cutoff_date.date())

        # Derivaciones del sample (sample_generation.py:188-191 del TFG).
        cutoff_ts = pd.Timestamp(cutoff_date)
        last_login_clipped = last_login_dt.clip(upper=cutoff_ts)
        player_lifespan_days = (
            (last_login_clipped - created_at_dt).dt.total_seconds() / 86400
        ).round(0)
        days_since_last_login = (
            (cutoff_ts - last_login_dt).dt.total_seconds() / 86400
        ).round(0).clip(lower=0)
        has_corrupted_dates = (player_lifespan_days < 0) | player_lifespan_days.isna()

        sample_user_ids = pd.DataFrame({
            "user_id": users_df["_id"].apply(_extract_user_id),
            "player_lifespan_days": player_lifespan_days.astype("Int64"),
            "days_since_last_login": days_since_last_login.astype("Int64"),
            "has_corrupted_dates": has_corrupted_dates,
        }).dropna(subset=["user_id"]).reset_index(drop=True)

        n_corrupted = int(sample_user_ids["has_corrupted_dates"].sum())
        plf = sample_user_ids["player_lifespan_days"]
        median_lifespan = int(plf.median()) if plf.notna().any() else "N/A"
        logger.info(
            "  Sample derivado: %d usuarios, %d con fechas corruptas, "
            "lifespan median=%s días",
            len(sample_user_ids),
            n_corrupted,
            median_lifespan,
        )
        logger.info("  N usuarios a predecir: %d", len(sample_user_ids))

        # Fase 2.2: dos masters separados (churn + gustos) en UN SOLO pase de I/O.
        # master_churn (50 cols) alimenta predict_churn; master_gustos (78 cols)
        # alimenta el clustering N1.
        master_churn, master_gustos = master_builder.build_both_masters(
            ctx=ctx,
            sample_user_ids=sample_user_ids,
            cutoff_date=cutoff_date,
        )
        logger.info(
            "  master_churn:  %d filas × %d cols",
            master_churn.shape[0], master_churn.shape[1],
        )
        logger.info(
            "  master_gustos: %d filas × %d cols",
            master_gustos.shape[0], master_gustos.shape[1],
        )

        # === 4. PREDECIR CHURN ===
        logger.info("[4/6] Prediciendo churn (%d targets)…", len(churn_targets))
        predictions_live = prediction.predict_churn_for_all_users(
            master_churn, registry, targets=churn_targets
        )
        # Persistencia incremental — si el resto falla, esto se conserva
        live_path = args.output / "_intermediate_predictions_live.parquet"
        predictions_live.to_parquet(live_path, index=False)
        logger.info("  ✓ Persisted intermediate predictions_live (%d filas) → %s",
                    len(predictions_live), live_path.name)

        # === 5. OOF LOOKUP CON DRIFT DETECTION ===
        drift_threshold = load_drift_threshold(repo_root / "config" / "thresholds.yaml")
        logger.info(
            "[5/6] Aplicando OOF lookup con drift detection (threshold=%.2f)…",
            drift_threshold,
        )
        predictions_dual = apply_oof_lookup_with_drift(
            predictions_live=predictions_live,
            registry=registry,
            targets=churn_targets,
            drift_threshold=drift_threshold,
        )
        dual_path = args.output / "_intermediate_predictions_dual.parquet"
        predictions_dual.to_parquet(dual_path, index=False)
        logger.info("  ✓ Persisted intermediate predictions_dual (%d filas) → %s",
                    len(predictions_dual), dual_path.name)

        # === 6. ARQUETIPO N1 + PERFILADO DE GUSTOS ===
        # (try/except: si falla, run continúa sin arquetipo/perfil)
        logger.info("[6/6] Asignando arquetipo N1 + perfilado de gustos…")
        stage_6_status = "ok"
        stage_6_error = None
        archetypes = None
        taste = None
        try:
            # master_gustos tiene las 78 cols que KMeans N1 espera + las features
            # universales y Tier 2 que el perfilado de gustos consume.
            archetypes = clustering.assign_archetypes(master_gustos, registry)
            logger.info("  · Arquetipo N1 asignado a %d usuarios", len(archetypes))

            taste = perfilado.profile_and_assign(
                master_gustos=master_gustos,
                archetypes_df=archetypes,
                repo_root=repo_root,
            )
            n_t2 = int((taste["has_tier2"] == 1).sum())
            cov = (taste["contramedida_primaria_cod"] != "CM00").mean() * 100
            logger.info(
                "  · Perfilado: %d usuarios, %d con Tier 2, cobertura contramedidas %.1f%%",
                len(taste), n_t2, cov,
            )
        except Exception as e:
            logger.error("[6/6] Falló: %s. Continuando sin arquetipo/perfil.", e)
            stage_6_status = "skipped"
            stage_6_error = repr(e)[:500]

        # === ENSAMBLAJE FINAL + ESCRITURA ===
        archetypes_yaml = repo_root / "config" / "archetypes.yaml"
        eje_cols = perfilado.EJES
        taste_cols = ["has_tier2"] + eje_cols + [
            "contramedida_primaria_cod", "contramedida_primaria_label",
        ]

        if archetypes is not None:
            final = predictions_dual.merge(archetypes, on="user_id", how="left")
            final = _apply_archetype_name(final, archetypes_yaml)
            if taste is not None:
                final = final.merge(
                    taste[["user_id"] + taste_cols], on="user_id", how="left"
                )
        else:
            # Schema estable: las cols de arquetipo/perfil existen siempre, NaN si no hay
            final = predictions_dual.copy()
            final["archetype_n1"] = pd.NA
            final["archetype_name"] = "N/A"
            for c in taste_cols:
                final[c] = pd.NA

        # --- Capa TÉCNICA (predictions_full.csv): todo el detalle ---
        full_cols = ["user_id"]
        for t in churn_targets:
            full_cols.extend(
                [f"{t}_live", f"{t}_oof", f"{t}_delta", f"{t}_source", f"{t}_final"]
            )
        full_cols.extend(["archetype_n1", "archetype_name", "has_tier2"])
        full_cols.extend(eje_cols)
        full_cols.extend(["contramedida_primaria_cod", "contramedida_primaria_label"])
        predictions_full = final[[c for c in full_cols if c in final.columns]]
        predictions_full.to_csv(args.output / "predictions_full.csv", index=False)
        logger.info(
            "  · predictions_full.csv (%d filas × %d cols)", *predictions_full.shape
        )

        # --- Capa de NEGOCIO (predictions.csv): churn + arquetipo + contramedida ---
        business_cols = ["user_id"]
        for t in churn_targets:
            business_cols.extend(
                [f"{t}_live", f"{t}_oof", f"{t}_delta", f"{t}_source", f"{t}_final"]
            )
        business_cols.extend([
            "archetype_n1", "archetype_name",
            "contramedida_primaria_cod", "contramedida_primaria_label",
        ])
        final_business = final[[c for c in business_cols if c in final.columns]]
        final_business.to_csv(args.output / "predictions.csv", index=False)
        logger.info("  · predictions.csv (%d filas × %d cols)", *final_business.shape)
        final = final_business

        # === SUMMARY.JSON (sin priority/risk binarizados) ===
        per_target = _per_target_summary(predictions_dual, churn_targets)
        summary = {
            "n_users": int(len(final)),
            "churn_targets": churn_targets,
            "drift_threshold": drift_threshold,
            "stage_6_status": stage_6_status,
            "per_target": per_target,
        }
        if archetypes is not None and "archetype_name" in final.columns:
            summary["archetype_distribution"] = final["archetype_name"].value_counts().to_dict()
        if taste is not None:
            cm = taste["contramedida_primaria_cod"]
            summary["countermeasure_distribution"] = cm.value_counts().to_dict()
            summary["countermeasure_coverage_pct"] = round(
                100 * (cm != "CM00").mean(), 2
            )
            summary["n_tier2"] = int((taste["has_tier2"] == 1).sum())
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))

        # === _RUN_METADATA.JSON ===
        elapsed = time.time() - t_start
        run_metadata = {
            "started_at": datetime.fromtimestamp(t_start).isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "reference_date": str(reference_date),
            "cutoff_date": str(cutoff_date.date()),
            "n_users": int(len(final)),
            "churn_targets": churn_targets,
            "drift_threshold": drift_threshold,
            "stage_6_status": stage_6_status,
            "stage_6_error": stage_6_error,
            "schemas_identified": list(val_result.identifications.keys()),
            "models": registry.summary(),
            "input_dir": str(args.input),
            "output_dir": str(args.output),
        }
        (args.output / "_run_metadata.json").write_text(
            json.dumps(run_metadata, indent=2, default=str)
        )

        # === DICCIONARIO.JSON (integración cliente) ===
        diccionario = build_dictionary(repo_root)
        dict_path = args.output / "diccionario.json"
        with open(dict_path, "w", encoding="utf-8") as f:
            json.dump(diccionario, f, indent=2, ensure_ascii=False)
        logger.info(
            "  ✓ diccionario.json generado (modelo %s, %d arquetipos)",
            diccionario["model_version"],
            len(diccionario["archetype_n1"]),
        )

        # Borrar intermediates: el predictions.csv final ya está en disco
        for p in args.output.glob("_intermediate_*.parquet"):
            p.unlink()
        logger.info("  ✓ Intermediates borrados")

        logger.info("=" * 70)
        logger.info("✅ PIPELINE COMPLETADO en %.1fs (stage_6=%s)", elapsed, stage_6_status)
        logger.info("  predictions.csv:    %d filas × %d cols", *final.shape)
        logger.info("  diccionario.json:   ✓")
        logger.info("  summary.json:       ✓")
        logger.info("  _run_metadata.json: ✓")
        logger.info("=" * 70)
        return 0

    except Exception as e:
        logger.exception("❌ Pipeline falló: %s", e)
        # Marcar el run con un fichero de error para inspección post-mortem
        try:
            (args.output / "_pipeline_error.txt").write_text(
                f"{datetime.now().isoformat()}\n{repr(e)}\n\n{traceback.format_exc()}"
            )
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
