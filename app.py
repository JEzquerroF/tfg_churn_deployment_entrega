"""
app.py — UI Streamlit para el sistema de predicción de churn.

Wrapper de presentación encima del pipeline CLI existente. No contiene
lógica de modelado: solo orquesta upload, ejecución, visualización y
descarga. El backend (`scripts/predict.py`) se ejecuta como subprocess y
no se importa directamente.

Uso local:
    streamlit run app.py

Deploy (HF Spaces):
    se ejecuta automáticamente al conectar el repo.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml


# ============================================================================
#  CONFIGURACIÓN
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent
ARCHETYPES_YAML = REPO_ROOT / "config" / "archetypes.yaml"
ACTIVE_MODELS_YAML = REPO_ROOT / "config" / "_active_models.yaml"
PREDICT_SCRIPT = REPO_ROOT / "scripts" / "predict.py"

# Identificadores canónicos de los CSVs (el backend identifica por columnas, no
# por nombre, pero la UI los lista para guiar al usuario).
EXPECTED_CSVS_REQUIRED = [
    "users.csv",
    "characters.csv",
    "devices.csv",
    "processed_consumables_iaps.csv",
    "processed_subscriptions_iaps.csv",
    "user_daily_rewards.csv",
    "user_items.csv",
    "user_items_collection.csv",
    "support_user_feedback_by_type.csv",
]
EXPECTED_CSVS_TRANSACTIONAL = [
    "currency_transactions.csv",
    "fights_log.csv",
    "arena_log.csv",
]

# Timeout duro para el subprocess del pipeline. 20 min es generoso: el smoke
# completo (1.16M usuarios) tarda ~8m30s.
PIPELINE_TIMEOUT_SECONDS = 20 * 60

# Límite del demo público: usuarios por ejecución. Solo afecta a la UI;
# el backend no impone este límite.
USER_LIMIT = 10_000

# Threshold para "alto riesgo" en KPIs, leído de config/thresholds.yaml.
DEFAULT_HIGH_RISK_THRESHOLD = 0.65

CUSTOM_CSS = """
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stButton button { width: 100%; }
    .stDataFrame { font-size: 0.9rem; }
</style>
"""


# ============================================================================
#  CONFIG LOADERS (cacheados)
# ============================================================================


@st.cache_data
def load_active_models_config() -> dict:
    with open(ACTIVE_MODELS_YAML) as f:
        return yaml.safe_load(f) or {}


@st.cache_data
def load_high_risk_threshold() -> float:
    """Lee el threshold de 'high' de config/thresholds.yaml. Default 0.65."""
    path = REPO_ROOT / "config" / "thresholds.yaml"
    if not path.exists():
        return DEFAULT_HIGH_RISK_THRESHOLD
    try:
        cfg = yaml.safe_load(path.read_text()) or {}
        return float(cfg.get("risk_levels", {}).get("high", {}).get("min", DEFAULT_HIGH_RISK_THRESHOLD))
    except Exception:
        return DEFAULT_HIGH_RISK_THRESHOLD


# ============================================================================
#  HEADER
# ============================================================================


def render_header() -> None:
    st.set_page_config(
        page_title="Predicción de Churn — TFG",
        layout="wide",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.title("Sistema de predicción de churn")
    st.markdown(
        "*Análisis automático de probabilidad de churn y segmentación de jugadores "
        "para juegos free-to-play mobile.*"
    )
    st.markdown("---")


# ============================================================================
#  PASO 1 — UPLOAD
# ============================================================================


def _reset_input_dir() -> Path:
    """Crea (o reusa) un dir temp para los CSVs subidos, limpio."""
    if "input_dir" not in st.session_state:
        st.session_state.input_dir = tempfile.mkdtemp(prefix="churn_input_")
    input_dir = Path(st.session_state.input_dir)
    for f in input_dir.glob("*.csv"):
        try:
            f.unlink()
        except OSError:
            pass
    return input_dir


def _check_user_count(input_dir: Path) -> tuple[bool, int]:
    """
    Cuenta usuarios en users.csv para aplicar el límite del demo público.

    Returns:
        (can_continue, n_users). Si no encuentra users.csv, devuelve (True, 0)
        para no bloquear (la validación del backend lo capturará después).
    """
    users_csv = input_dir / "users.csv"
    if not users_csv.exists():
        candidates = list(input_dir.glob("*sers*.csv"))
        if not candidates:
            return True, 0
        users_csv = candidates[0]

    try:
        with open(users_csv, "rb") as f:
            n_lines = sum(1 for _ in f)
        n_users = max(n_lines - 1, 0)  # -1 por header
    except Exception:
        return True, 0

    return n_users <= USER_LIMIT, n_users


def render_upload_section() -> Optional[Path]:
    """Devuelve el path con los CSVs guardados, o None si no hay upload."""
    st.header("1. Subir datos del juego")
    st.markdown(
        "Sube los CSVs exportados de la base de datos. Los nombres de los "
        "ficheros no importan — el sistema los identifica automáticamente por "
        "sus columnas."
    )

    with st.expander("CSVs esperados", expanded=False):
        st.markdown("**Obligatorios:**")
        for csv in EXPECTED_CSVS_REQUIRED:
            st.markdown(f"- `{csv}`")
        st.markdown("**Recomendados (transaccionales, mejoran la segmentación):**")
        for csv in EXPECTED_CSVS_TRANSACTIONAL:
            st.markdown(f"- `{csv}` (debe cubrir los últimos 30 días)")

    uploaded_files = st.file_uploader(
        "Arrastra los CSVs aquí",
        type=["csv"],
        accept_multiple_files=True,
        help="Los CSVs se procesan localmente. Ningún dato sale del servidor.",
    )

    if not uploaded_files:
        return None

    st.success(f"{len(uploaded_files)} ficheros recibidos")

    input_dir = _reset_input_dir()
    for uploaded_file in uploaded_files:
        dst = input_dir / uploaded_file.name
        with open(dst, "wb") as f:
            f.write(uploaded_file.getbuffer())

    return input_dir


# ============================================================================
#  PASO 2 — PIPELINE
# ============================================================================


def _validate_inputs_pre_run(input_dir: Path) -> list[str]:
    """
    Comprueba que están los CSVs obligatorios. Devuelve lista de nombres
    faltantes (vacío si todo OK). El backend hace identificación por columnas
    igualmente; esto es solo una guía amistosa para el usuario.
    """
    present = {p.name for p in input_dir.glob("*.csv")}
    missing = [name for name in EXPECTED_CSVS_REQUIRED if name not in present]
    return missing


def run_pipeline(input_dir: Path) -> Optional[Path]:
    """
    Ejecuta scripts/predict.py contra el directorio de input.

    Devuelve el path al directorio de output, o None si falla.
    """
    # Validación pre-run informativa
    missing = _validate_inputs_pre_run(input_dir)
    if missing:
        st.warning(
            "Atención: faltan los siguientes CSVs obligatorios por nombre. El "
            "validador del backend los detectará por columnas, pero si los CSVs "
            "tienen nombres no estándar puede que algunos no se identifiquen:\n\n"
            + "\n".join(f"- `{m}`" for m in missing)
        )

    output_dir = Path(tempfile.mkdtemp(prefix="churn_output_"))

    cmd = [
        sys.executable,
        str(PREDICT_SCRIPT),
        "--input", str(input_dir),
        "--output", str(output_dir),
    ]

    progress_bar = st.progress(0, text="Iniciando pipeline…")
    log_container = st.empty()

    stage_progress = {
        "[1/6]": 5,
        "[2/6]": 10,
        "[3/6]": 50,
        "[4/6]": 80,
        "[5/6]": 85,
        "[6/6]": 95,
    }

    log_lines: list[str] = []
    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        start_ts = pd.Timestamp.utcnow()
        for line in process.stdout:  # type: ignore[union-attr]
            log_lines.append(line.rstrip())
            log_container.code("\n".join(log_lines[-15:]), language="bash")

            for stage, pct in stage_progress.items():
                if stage in line:
                    progress_bar.progress(pct, text=f"Procesando {stage}…")
                    break

            # Timeout duro
            elapsed = (pd.Timestamp.utcnow() - start_ts).total_seconds()
            if elapsed > PIPELINE_TIMEOUT_SECONDS:
                process.kill()
                st.error(
                    f"El pipeline tardó más de {PIPELINE_TIMEOUT_SECONDS // 60} "
                    f"minutos y se abortó. Revisa el log:"
                )
                st.code("\n".join(log_lines[-30:]), language="bash")
                return None

        process.wait(timeout=10)

        if process.returncode != 0:
            st.error(f"El pipeline falló con código {process.returncode}. Últimas líneas del log:")
            st.code("\n".join(log_lines[-30:]), language="bash")
            return None

        progress_bar.progress(100, text="Pipeline completado")
        return output_dir

    except subprocess.TimeoutExpired:
        if process is not None:
            process.kill()
        st.error("Timeout esperando el cierre del subprocess.")
        st.code("\n".join(log_lines[-30:]), language="bash")
        return None
    except Exception as e:
        if process is not None:
            try:
                process.kill()
            except Exception:
                pass
        st.error(f"Error ejecutando pipeline: {e}")
        if log_lines:
            st.code("\n".join(log_lines[-30:]), language="bash")
        return None


# ============================================================================
#  PASO 3 — RESULTADOS
# ============================================================================


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Error leyendo {path.name}: {e}")
        return None


def render_kpis(df: pd.DataFrame) -> None:
    st.subheader("Métricas generales")
    n_users = len(df)
    threshold = load_high_risk_threshold()

    top1, top2 = st.columns(2)
    top1.metric("Total jugadores analizados", f"{n_users:,}")
    if "archetype_n1" in df.columns:
        n_arch = int(df["archetype_n1"].nunique(dropna=True))
        top2.metric("Arquetipos distintos detectados", f"{n_arch} / 6")
    else:
        top2.metric("Arquetipos distintos detectados", "—")

    st.markdown(f"**Alto riesgo de churn** (probabilidad ≥ {threshold:.2f})")
    cols = st.columns(3)
    for col, horizon in zip(cols, ("7d", "14d", "30d")):
        final_col = f"churn_{horizon}_final"
        if final_col in df.columns:
            try:
                n_high_risk = int((df[final_col] >= threshold).sum())
                pct = (100 * n_high_risk / n_users) if n_users > 0 else 0.0
                col.metric(
                    f"Alto riesgo {horizon}",
                    f"{n_high_risk:,}",
                    f"{pct:.1f}% del total",
                    delta_color="off",
                )
            except Exception:
                col.metric(f"Alto riesgo {horizon}", "—")
        else:
            col.metric(f"Alto riesgo {horizon}", "—")


def render_archetype_chart(df: pd.DataFrame) -> None:
    st.subheader("Distribución por arquetipo")
    if "archetype_name" not in df.columns or df["archetype_name"].isna().all():
        st.info("No hay arquetipos asignados en este run (stage [6/6] no completó).")
        return

    n_users = len(df)
    counts = df["archetype_name"].value_counts(dropna=True).reset_index()
    counts.columns = ["Arquetipo", "Jugadores"]
    counts["Porcentaje"] = (100 * counts["Jugadores"] / n_users).round(2)

    fig = px.bar(
        counts,
        x="Jugadores",
        y="Arquetipo",
        orientation="h",
        text="Porcentaje",
        color="Jugadores",
        color_continuous_scale="Blues",
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(
        xaxis_title="Número de jugadores",
        yaxis_title="",
        showlegend=False,
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_high_risk_by_horizon(df: pd.DataFrame) -> None:
    st.subheader("Jugadores en alto riesgo por horizonte temporal")
    threshold = load_high_risk_threshold()
    horizons = ["7d", "14d", "30d"]

    n_users = len(df)
    rows = []
    for h in horizons:
        col = f"churn_{h}_final"
        if col in df.columns:
            n_hr = int((df[col] >= threshold).sum())
            pct = (100 * n_hr / n_users) if n_users > 0 else 0.0
            rows.append({
                "Horizonte": h,
                "Jugadores": n_hr,
                "Etiqueta": f"{n_hr:,} ({pct:.1f}%)",
            })

    if not rows:
        st.info("No hay columnas de churn para construir el gráfico.")
        return

    counts = pd.DataFrame(rows)
    fig = px.bar(
        counts,
        x="Horizonte",
        y="Jugadores",
        text="Etiqueta",
        color="Jugadores",
        color_continuous_scale="Blues",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Horizonte temporal",
        yaxis_title="Jugadores en alto riesgo",
        showlegend=False,
        height=400,
    )
    fig.update_xaxes(categoryorder="array", categoryarray=horizons)
    st.plotly_chart(fig, use_container_width=True)


def render_results(output_dir: Path) -> None:
    st.header("2. Resultados")

    predictions_path = output_dir / "predictions.csv"
    if not predictions_path.exists():
        st.error("No se generó `predictions.csv`. Algo falló en el pipeline.")
        return

    df = _safe_read_csv(predictions_path)
    if df is None:
        return

    render_kpis(df)
    render_archetype_chart(df)
    render_high_risk_by_horizon(df)
    st.caption(
        "El detalle por jugador (con contramedida recomendada) está en el Excel "
        "operacional y en predictions.csv — descárgalos abajo."
    )


# ============================================================================
#  PASO 4 — DESCARGAS
# ============================================================================


def render_downloads(output_dir: Path) -> None:
    st.header("3. Descargar resultados")

    # === 1 + 2. Entregables principales: PDF + Excel (destacados arriba) ===
    st.subheader("Informes (entregables principales)")
    col_pdf, col_xlsx = st.columns(2)

    with col_pdf:
        if st.button("Generar informe ejecutivo (PDF)", type="primary",
                     use_container_width=True, key="btn_pdf"):
            with st.spinner("Generando PDF…"):
                try:
                    from reports.pdf_generator import generate_pdf_report
                    pdf_path = output_dir / "informe_ejecutivo.pdf"
                    generate_pdf_report(
                        predictions_path=output_dir / "predictions.csv",
                        summary_path=output_dir / "summary.json",
                        output_path=pdf_path,
                    )
                    st.session_state.pdf_path = str(pdf_path)
                    st.success("PDF generado")
                except Exception as e:
                    st.error(f"Error generando PDF: {e}")

        pdf_path_str = st.session_state.get("pdf_path")
        if pdf_path_str and Path(pdf_path_str).exists():
            with open(pdf_path_str, "rb") as f:
                st.download_button(
                    label="Descargar informe_ejecutivo.pdf",
                    data=f.read(),
                    file_name="informe_ejecutivo.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="dl_pdf",
                )

    with col_xlsx:
        if st.button("Generar informe operacional (Excel)", type="primary",
                     use_container_width=True, key="btn_xlsx"):
            with st.spinner("Generando Excel…"):
                try:
                    from reports.excel_generator import generate_excel_report
                    xlsx_path = output_dir / "informe_operacional.xlsx"
                    generate_excel_report(
                        predictions_path=output_dir / "predictions.csv",
                        diccionario_path=output_dir / "diccionario.json",
                        output_path=xlsx_path,
                        summary_path=output_dir / "summary.json",
                    )
                    st.session_state.xlsx_path = str(xlsx_path)
                    st.success("Excel generado")
                except Exception as e:
                    st.error(f"Error generando Excel: {e}")

        xlsx_path_str = st.session_state.get("xlsx_path")
        if xlsx_path_str and Path(xlsx_path_str).exists():
            with open(xlsx_path_str, "rb") as f:
                st.download_button(
                    label="Descargar informe_operacional.xlsx",
                    data=f.read(),
                    file_name="informe_operacional.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="dl_xlsx",
                )

    # === 3-6. Archivos de datos crudos (orden: summary, diccionario, predictions, metadata) ===
    st.markdown("---")
    st.subheader("Datos crudos")
    files = [
        ("predictions.csv", "Predicciones (CSV)"),
        ("summary.json", "Resumen (JSON)"),
    ]
    cols = st.columns(len(files))
    for (filename, label), col in zip(files, cols):
        path = output_dir / filename
        if path.exists():
            try:
                with open(path, "rb") as f:
                    data = f.read()
                col.download_button(
                    label=label,
                    data=data,
                    file_name=filename,
                    mime="application/octet-stream",
                    use_container_width=True,
                )
            except Exception as e:
                col.error(f"Error leyendo {filename}: {e}")
        else:
            col.markdown(f"_(no generado: `{filename}`)_")


# ============================================================================
#  MAIN
# ============================================================================


@st.cache_resource
def _bootstrap_remote_models() -> None:
    """
    Descarga los modelos pesados desde HF Hub si faltan (en HF Spaces los
    .pkl/.joblib >10 MiB no caben en el repo del Space sin LFS). En local
    es no-op porque los ficheros ya están.

    @st.cache_resource garantiza una sola ejecución por proceso Streamlit.
    """
    from bootstrap_models import ensure_models_present
    ensure_models_present()


@st.cache_data
def _interpretation_dict_bytes() -> bytes:
    """Genera (una vez por proceso) el diccionario de interpretación estático."""
    import tempfile
    from reports.interpretation_dict import generate_interpretation_dict

    tmp = Path(tempfile.gettempdir()) / "diccionario_interpretacion.xlsx"
    generate_interpretation_dict(REPO_ROOT, tmp)
    return tmp.read_bytes()


def render_interpretation_dict_section() -> None:
    st.subheader("Diccionario de interpretación")
    st.markdown(
        "Antes de empezar, puedes descargar la **guía de interpretación de "
        "resultados**: explica cómo leer la tabla de predicciones, qué significa "
        "cada arquetipo y cada contramedida, y los códigos del sistema. No necesitas "
        "subir datos para obtenerla."
    )
    try:
        st.download_button(
            label="Descargar guía de interpretación (Excel)",
            data=_interpretation_dict_bytes(),
            file_name="diccionario_interpretacion.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_interp_dict",
        )
    except Exception as e:
        st.error(f"No se pudo generar la guía de interpretación: {e}")
    st.markdown("---")


def main() -> None:
    render_header()

    # Guía de interpretación: siempre disponible, no requiere subir datos ni modelos.
    render_interpretation_dict_section()

    try:
        _bootstrap_remote_models()
    except Exception as e:
        st.error(
            "No se han podido descargar los modelos desde Hugging Face Hub. "
            "El sistema no puede arrancar.\n\n"
            f"Error: `{e}`"
        )
        st.stop()

    input_dir = render_upload_section()
    if input_dir is None:
        st.info(
            "Arrastra los CSVs del juego para comenzar el análisis. Los "
            "archivos se procesan en local y no se almacenan después de la sesión."
        )
        _render_footer()
        return

    can_continue, n_users = _check_user_count(input_dir)
    if not can_continue:
        st.error(
            f"Tu archivo `users.csv` contiene **{n_users:,} usuarios**, "
            f"pero el demo público está limitado a **{USER_LIMIT:,} usuarios** "
            "por ejecución.\n\n"
            "Para procesar datasets completos, despliega el sistema en tu "
            "propia infraestructura (código disponible en el repositorio)."
        )
        _render_footer()
        return
    if n_users > 0:
        st.info(f"{n_users:,} usuarios detectados. Listo para procesar.")

    st.markdown("---")
    button_label = (
        f"Procesar {n_users:,} usuarios"
        if n_users > 0
        else "Procesar datos"
    )
    if st.button(button_label, type="primary", use_container_width=True):
        # Limpia output previo si existe
        if "output_dir" in st.session_state:
            try:
                shutil.rmtree(st.session_state.output_dir, ignore_errors=True)
            except Exception:
                pass
            del st.session_state["output_dir"]

        with st.spinner("Procesando…"):
            output_dir = run_pipeline(input_dir)

        if output_dir is None:
            _render_footer()
            return

        st.session_state.output_dir = str(output_dir)
        st.success("Procesamiento completado")

    if "output_dir" in st.session_state:
        st.markdown("---")
        output_dir = Path(st.session_state.output_dir)
        if output_dir.exists():
            render_results(output_dir)
            st.markdown("---")
            render_downloads(output_dir)

    _render_footer()


def _render_footer() -> None:
    """Footer discreto con versión del modelo y límite del demo."""
    try:
        cfg = load_active_models_config()
        churn_version = cfg.get("churn", {}).get("active_version", "—")
        gustos_version = cfg.get("gustos", {}).get("active_version", "—")
    except Exception:
        churn_version = "—"
        gustos_version = "—"

    st.markdown("---")
    st.caption(
        f"Modelos: churn `{churn_version}` · segmentación `{gustos_version}` · "
        f"Límite demo: {USER_LIMIT:,} usuarios/ejecución · "
        "TFG — Sistema de predicción de churn"
    )


if __name__ == "__main__":
    main()
