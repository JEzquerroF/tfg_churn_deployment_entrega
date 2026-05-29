"""
pdf_generator.py — Genera el informe ejecutivo PDF.

Recibe los outputs del pipeline (predictions.csv + summary.json +
diccionario.json) y produce un PDF de 1-2 páginas con:
  - Header con metadata del run
  - 3 KPIs principales
  - Gráfico de distribución por arquetipo
  - Tabla resumen: % churn por arquetipo
  - Footer con notas metodológicas

NO incluye narrativa generada por LLM. Es informe tabular + visual,
defendible y robusto sin dependencias de API externa.

NOTA macOS: weasyprint depende de libs de sistema (pango/cairo/gobject)
instaladas vía homebrew. Configuramos `DYLD_FALLBACK_LIBRARY_PATH` antes
del import si estamos en macOS y la ruta de homebrew existe — esto evita
que el cliente tenga que activar el venv con esa variable manualmente.
"""

from __future__ import annotations

# === Shim macOS: DYLD_FALLBACK_LIBRARY_PATH para weasyprint ===
# DEBE ir ANTES de `import weasyprint`. La variable se hereda en subprocesos
# pero Python no la usa para libs cargadas tras setear os.environ — sin
# embargo el ctypes loader sí la lee si aún no se ha cargado la lib.
import os as _os
import sys as _sys
from pathlib import Path as _Path

if _sys.platform == "darwin":
    _brew_lib = "/opt/homebrew/lib"
    if _Path(_brew_lib).exists():
        _existing = _os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        if _brew_lib not in _existing:
            _os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (
                f"{_brew_lib}:{_existing}" if _existing else _brew_lib
            )
# === fin shim ===

import base64
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # sin display server
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Template
from weasyprint import HTML

logger = logging.getLogger(__name__)


HIGH_RISK_THRESHOLD = 0.65  # default del config/thresholds.yaml (nivel 'high')


PDF_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Informe de churn — {{ run_date }}</title>
<style>
    @page {
        size: A4;
        margin: 1.5cm 2cm;
        @bottom-right { content: "Página " counter(page) " de " counter(pages); font-size: 9pt; color: #666; }
    }
    body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; color: #333; }
    h1 { color: #1f3a68; font-size: 22pt; margin-bottom: 0; }
    .subtitle { color: #666; font-size: 11pt; margin-top: 0; margin-bottom: 1.5em; }
    .kpi-row { display: flex; justify-content: space-between; gap: 1em; margin: 1.5em 0; }
    .kpi {
        flex: 1;
        background: #f0f4f9;
        border-left: 4px solid #1f3a68;
        padding: 0.8em 1em;
        border-radius: 3px;
    }
    .kpi-label { font-size: 9pt; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-value { font-size: 22pt; font-weight: bold; color: #1f3a68; margin-top: 0.2em; }
    .kpi-sub { font-size: 9pt; color: #888; }
    h2 { color: #1f3a68; font-size: 14pt; border-bottom: 1px solid #ddd; padding-bottom: 0.3em; margin-top: 1.8em; }
    img.chart { max-width: 100%; height: auto; margin: 1em 0; }
    table { width: 100%; border-collapse: collapse; margin: 1em 0; font-size: 10pt; }
    th, td { padding: 0.5em 0.7em; text-align: left; border-bottom: 1px solid #ddd; }
    th { background: #1f3a68; color: white; font-weight: 600; }
    tr:nth-child(even) { background: #f9fafc; }
    .risk-high { color: #c0392b; font-weight: bold; }
    .risk-medium { color: #e67e22; }
    .risk-low { color: #27ae60; }
    .footer {
        margin-top: 2em;
        padding-top: 1em;
        border-top: 1px solid #ddd;
        font-size: 8pt;
        color: #999;
        text-align: center;
    }
</style>
</head>
<body>

<h1>Informe de predicción de churn</h1>
<p class="subtitle">
    Análisis ejecutado el {{ run_date }} · Modelo {{ model_version }} ·
    {{ n_users_formatted }} jugadores analizados
</p>

<div class="kpi-row">
    <div class="kpi">
        <div class="kpi-label">Jugadores analizados</div>
        <div class="kpi-value">{{ n_users_formatted }}</div>
        <div class="kpi-sub">total en este análisis</div>
    </div>
    <div class="kpi">
        <div class="kpi-label">Alto riesgo de churn (14d)</div>
        <div class="kpi-value">{{ n_high_risk_formatted }}</div>
        <div class="kpi-sub">{{ pct_high_risk }}% del total</div>
    </div>
    <div class="kpi">
        <div class="kpi-label">Arquetipos detectados</div>
        <div class="kpi-value">{{ n_archetypes }}/6</div>
        <div class="kpi-sub">segmentos identificados</div>
    </div>
</div>

<h2>Alto riesgo de churn por horizonte temporal</h2>
<table>
    <thead>
        <tr>
            <th>Horizonte</th>
            <th style="text-align:right">Jugadores en alto riesgo</th>
            <th style="text-align:right">% del total</th>
        </tr>
    </thead>
    <tbody>
        {% for row in high_risk_rows %}
        <tr>
            <td>{{ row.horizon }}</td>
            <td style="text-align:right">{{ row.n_users }}</td>
            <td style="text-align:right">{{ row.pct }}%</td>
        </tr>
        {% endfor %}
    </tbody>
</table>

<h2>Distribución por arquetipo</h2>
<img class="chart" src="data:image/png;base64,{{ chart_archetypes_b64 }}" alt="Distribución arquetipos"/>

<h2>Riesgo de churn por arquetipo</h2>
<table>
    <thead>
        <tr>
            <th>Arquetipo</th>
            <th style="text-align:right">Jugadores</th>
            <th style="text-align:right">% del total</th>
            <th style="text-align:right">Prob. media (14d)</th>
            <th style="text-align:right">Alto riesgo (14d)</th>
        </tr>
    </thead>
    <tbody>
        {% for row in archetype_table %}
        <tr>
            <td>{{ row.name }}</td>
            <td style="text-align:right">{{ row.n_users }}</td>
            <td style="text-align:right">{{ row.pct }}%</td>
            <td style="text-align:right">{{ row.mean_prob }}</td>
            <td style="text-align:right" class="{{ row.risk_class }}">{{ row.pct_high_risk }}%</td>
        </tr>
        {% endfor %}
    </tbody>
</table>

{% if countermeasure_table %}
<h2>Contramedidas recomendadas</h2>
<p class="subtitle">
    Acción primaria asignada a cada jugador por reglas de prelación sobre su
    perfil de gustos (catálogo de 12 contramedidas). Cobertura {{ cm_coverage }}%.
</p>
<table>
    <thead>
        <tr>
            <th>Código</th>
            <th>Acción recomendada</th>
            <th style="text-align:right">Jugadores</th>
            <th style="text-align:right">% del total</th>
        </tr>
    </thead>
    <tbody>
        {% for row in countermeasure_table %}
        <tr>
            <td>{{ row.cod }}</td>
            <td>{{ row.label }}</td>
            <td style="text-align:right">{{ row.n_users }}</td>
            <td style="text-align:right">{{ row.pct }}%</td>
        </tr>
        {% endfor %}
    </tbody>
</table>
{% endif %}

<h2>Notas metodológicas</h2>
<ul>
    <li>Modelo: {{ model_version }} (Random Forest sobre sample L22, 33,598 usuarios de entrenamiento)</li>
    <li>Horizonte de referencia: 14 días desde fecha de análisis</li>
    <li>Umbral "alto riesgo": probabilidad de churn ≥ {{ high_risk_threshold }}</li>
    <li>Predicciones servidas con detección de drift OOF/live (threshold 0.10)</li>
    <li>Contramedidas asignadas por perfilado de gustos (7 ejes, catálogo de 12)</li>
</ul>

<div class="footer">
    Informe generado automáticamente por el sistema de predicción de churn TFG.
    Para integración programática, consulte el diccionario de códigos
    (diccionario.json) entregado junto con este informe.
</div>

</body>
</html>
"""


def _create_archetype_chart_b64(df: pd.DataFrame) -> str:
    """Crea el gráfico de barras horizontales y lo devuelve como base64 PNG."""
    counts = df["archetype_name"].value_counts().sort_values()

    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=120)
    n = len(counts)
    colors = plt.cm.Blues([0.4 + 0.6 * i / max(n, 1) for i in range(n)])
    bars = ax.barh(counts.index, counts.values, color=colors)

    total = counts.sum()
    for bar, value in zip(bars, counts.values):
        pct = 100 * value / total if total else 0
        ax.text(
            bar.get_width() + total * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Número de jugadores")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
    ax.set_xlim(0, max(counts.values) * 1.15 if len(counts) else 1)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _build_archetype_table(df: pd.DataFrame) -> list[dict]:
    """Stats agregadas por arquetipo para la tabla del PDF."""
    n_total = len(df)
    rows = []
    grouped = df.groupby("archetype_name", dropna=False)

    for archetype_name, group in grouped:
        if pd.isna(archetype_name):
            archetype_name = "Sin clasificar"

        n_users = len(group)
        pct = round(100 * n_users / n_total, 1) if n_total else 0.0

        # 14d es el horizonte de referencia del TFG.
        if "churn_14d_final" in group.columns:
            mean_prob = group["churn_14d_final"].mean()
            pct_high = round(
                100 * (group["churn_14d_final"] >= HIGH_RISK_THRESHOLD).mean(), 1
            )
        else:
            mean_prob = float("nan")
            pct_high = 0.0

        risk_class = (
            "risk-high"
            if pct_high >= 50
            else "risk-medium"
            if pct_high >= 25
            else "risk-low"
        )

        rows.append({
            "name": archetype_name,
            "n_users": f"{n_users:,}",
            "pct": pct,
            "mean_prob": f"{mean_prob:.3f}" if pd.notna(mean_prob) else "n/a",
            "pct_high_risk": pct_high,
            "risk_class": risk_class,
        })

    rows.sort(key=lambda r: float(r["pct"]), reverse=True)
    return rows


def _build_countermeasure_table(df: pd.DataFrame) -> list[dict]:
    """Distribución de la contramedida primaria sobre los jugadores."""
    if "contramedida_primaria_cod" not in df.columns:
        return []

    n_total = len(df)
    label_col = (
        "contramedida_primaria_label"
        if "contramedida_primaria_label" in df.columns
        else "contramedida_primaria_cod"
    )
    grouped = (
        df.groupby(["contramedida_primaria_cod", label_col], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )

    rows = []
    for _, r in grouped.iterrows():
        cod = r["contramedida_primaria_cod"]
        if pd.isna(cod):
            continue
        rows.append({
            "cod": cod,
            "label": r[label_col] if pd.notna(r[label_col]) else "",
            "n_users": f"{int(r['n']):,}",
            "pct": round(100 * int(r["n"]) / n_total, 1) if n_total else 0.0,
        })
    return rows


def generate_pdf_report(
    predictions_path: Path,
    summary_path: Optional[Path],
    output_path: Path,
    model_version: str = "v2_rf_L22_2026-05-19",
) -> Path:
    """
    Genera un informe PDF ejecutivo.

    Args:
        predictions_path: ruta al predictions.csv del pipeline.
        summary_path: ruta al summary.json (opcional, no usado actualmente).
        output_path: dónde escribir el PDF.
        model_version: versión del modelo para el header.

    Returns:
        Path al PDF generado.
    """
    logger.info("Generando PDF: %s", output_path)

    df = pd.read_csv(predictions_path)
    n_users = len(df)

    # KPI central: 14d es el horizonte de referencia del TFG.
    n_high_risk = (
        int((df["churn_14d_final"] >= HIGH_RISK_THRESHOLD).sum())
        if "churn_14d_final" in df.columns
        else 0
    )
    pct_high_risk = round(100 * n_high_risk / n_users, 1) if n_users else 0.0
    n_archetypes = (
        int(df["archetype_n1"].nunique(dropna=True))
        if "archetype_n1" in df.columns
        else 0
    )

    # Alto riesgo por horizonte (coherente con la UI: >= threshold).
    high_risk_rows = []
    for horizon in ("7d", "14d", "30d"):
        col = f"churn_{horizon}_final"
        if col in df.columns:
            n_hr = int((df[col] >= HIGH_RISK_THRESHOLD).sum())
            high_risk_rows.append({
                "horizon": horizon,
                "n_users": f"{n_hr:,}",
                "pct": round(100 * n_hr / n_users, 1) if n_users else 0.0,
            })

    chart_b64 = _create_archetype_chart_b64(df)
    archetype_table = _build_archetype_table(df)
    countermeasure_table = _build_countermeasure_table(df)
    cm_coverage = (
        round(100 * (df["contramedida_primaria_cod"].notna()).mean(), 1)
        if "contramedida_primaria_cod" in df.columns
        else 0.0
    )

    template = Template(PDF_TEMPLATE)
    html = template.render(
        run_date=datetime.now().strftime("%d/%m/%Y"),
        model_version=model_version,
        n_users_formatted=f"{n_users:,}",
        n_high_risk_formatted=f"{n_high_risk:,}",
        pct_high_risk=pct_high_risk,
        n_archetypes=n_archetypes,
        high_risk_threshold=HIGH_RISK_THRESHOLD,
        chart_archetypes_b64=chart_b64,
        archetype_table=archetype_table,
        high_risk_rows=high_risk_rows,
        countermeasure_table=countermeasure_table,
        cm_coverage=cm_coverage,
    )

    HTML(string=html).write_pdf(output_path)
    logger.info("PDF generado: %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)
    return output_path
