"""
Cleanup pipeline dinámico parametrizable.

A diferencia del cleanup hardcodeado del TFG original (02zz_master_cleanup.ipynb),
este módulo calcula DINÁMICAMENTE las listas de columnas a eliminar sobre cada
master, aplicando umbrales explícitos.

Resuelve los problemas detectados en `validation_study/informes/cleanup_audit.md`:
- Cleanup hardcoded → ahora dinámico
- Umbrales declarativos no aplicados → ahora aplicados estrictamente
- 14/49 cols listadas ausentes → no es problema (las listas se calculan en runtime)
- Sin tie-break documentado → tie-break por menor %NaN, alfabético en empate
- 10 cuasi-constantes y 3 pares corr>=0.95 sobrevivientes → ahora se detectan

Uso (en deployment, sin target_cols porque NO hay target):
    from pipeline import cleanup_pipeline as cp

    drops = cp.compute_dynamic_cleanup(master, target_cols=())
    masters = cp.apply_cleanups(master, drops)
    # masters = {'v1_conservative': df, 'v2_intermediate': df, 'v3_aggressive': df}
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Defaults locales (sustituyen a config.py del validation_study)
# ============================================================

# Umbrales por defecto. Inyectables por parámetro en compute_dynamic_cleanup()
# desde config/pipeline_config.yaml en el caller.
CLEANUP_THRESHOLDS = {
    'high_nan_threshold': 0.95,
    'quasi_constant_threshold': 0.95,
    'corr_threshold_v2': 0.99,
    'corr_threshold_v3': 0.95,
}

# Target leakage = decisión schema-level (no sample-level).
# Heredado del TFG: cols que son snapshot al REFERENCE_DATE (no al CUTOFF).
# Estas SIEMPRE se eliminan en v1/v2/v3 — el modelo del TFG NO las vio en
# training, así que no pueden ir al predict.
TARGET_LEAKAGE_COLS = [
    # Derivados directos del target (last_login_dt)
    'user_last_login_date',          # unix timestamp = ground truth
    'user_last_login_days_ago',      # (CUTOFF - last_login).days
    'user_days_since_last_login',    # idem clip(0)
    # Snapshot post-cutoff (estado del user en REFERENCE, no en CUTOFF)
    'user_game_version',
    'user_gold',
    'user_gems',
    'user_dark_steel',
    'user_runes',
    'user_current_character',
    'user_current_session',
    'user_num_logins',
    'user_last_completed_tutorial_block',
    'user_updated_at',
]


# ============================================================
# Detectores individuales
# ============================================================

def compute_high_nan_cols(
    master: pd.DataFrame,
    exclude_cols: Sequence[str],
    threshold: float = 0.95,
) -> List[str]:
    """
    Detecta columnas con %NaN > threshold.

    Args:
        master: dataframe a auditar
        exclude_cols: columnas a no considerar (user_id, targets)
        threshold: umbral de NaN (default 0.95 = 95%)

    Returns:
        Lista alfabética de columnas con high NaN.
    """
    candidates = [c for c in master.columns if c not in exclude_cols]
    if not candidates:
        return []
    nan_ratio = master[candidates].isna().mean()
    high_nan = nan_ratio[nan_ratio > threshold].index.tolist()
    return sorted(high_nan)


def compute_quasi_constant_cols(
    master: pd.DataFrame,
    exclude_cols: Sequence[str],
    threshold: float = 0.95,
) -> List[str]:
    """
    Detecta columnas donde un único valor cubre >threshold del sample.

    Aplicable a numéricas, booleanas y categóricas.

    Args:
        master: dataframe
        exclude_cols: cols a no auditar
        threshold: umbral de concentración del top value

    Returns:
        Lista alfabética de columnas cuasi-constantes.
    """
    candidates = [c for c in master.columns if c not in exclude_cols]
    quasi_constants = []
    for col in candidates:
        s = master[col].dropna()
        if s.empty:
            continue
        vc = s.value_counts(normalize=True)
        if len(vc) == 0:
            continue
        top_freq = float(vc.iloc[0])
        if top_freq > threshold:
            quasi_constants.append(col)
    return sorted(quasi_constants)


def compute_correlation_drops(
    master: pd.DataFrame,
    exclude_cols: Sequence[str],
    threshold: float,
    already_dropped: Optional[Set[str]] = None,
) -> Tuple[List[str], List[Tuple[str, str, float]]]:
    """
    Detecta pares de columnas numéricas con |corr| >= threshold y decide cuál eliminar.

    Tie-break: mantiene la columna de MENOR %NaN. En empate, la primera alfabéticamente
    se mantiene y la segunda se elimina.

    Args:
        master: dataframe
        exclude_cols: cols a no considerar
        threshold: umbral de |corr| (0.99 o 0.95)
        already_dropped: cols ya marcadas para eliminar en pasos previos

    Returns:
        (cols_to_drop, pairs_detected): lista de cols a eliminar y lista de
        tuplas (col_a, col_b, |corr|) para trazabilidad.
    """
    if already_dropped is None:
        already_dropped = set()

    # Solo numéricas, excluir cols ya marcadas
    candidates = [
        c for c in master.columns
        if c not in exclude_cols
        and c not in already_dropped
        and pd.api.types.is_numeric_dtype(master[c])
    ]

    if len(candidates) < 2:
        return [], []

    # Orden alfabético para reproducibilidad
    candidates = sorted(candidates)

    # Matriz de correlación absoluta
    corr_matrix = master[candidates].corr(method='pearson').abs()
    nan_ratio = master[candidates].isna().mean()

    to_drop: Set[str] = set()
    pairs_detected: List[Tuple[str, str, float]] = []

    for i, col_a in enumerate(candidates):
        if col_a in to_drop:
            continue
        for col_b in candidates[i + 1:]:
            if col_b in to_drop:
                continue
            corr_val = corr_matrix.loc[col_a, col_b]
            if pd.isna(corr_val):
                continue
            if corr_val >= threshold:
                nan_a = nan_ratio[col_a]
                nan_b = nan_ratio[col_b]
                if nan_a < nan_b:
                    drop = col_b
                elif nan_b < nan_a:
                    drop = col_a
                else:
                    # Empate: eliminar la segunda alfabéticamente (col_b por construcción del bucle)
                    drop = col_b
                to_drop.add(drop)
                pairs_detected.append((col_a, col_b, float(corr_val)))

    return sorted(to_drop), pairs_detected


# ============================================================
# Pipeline completo
# ============================================================

def compute_dynamic_cleanup(
    master: pd.DataFrame,
    # DEPLOYMENT: deployment NO tiene target. En training del TFG era
    # target_cols=('churn_14d', 'churn_30d'). Aquí default vacío.
    target_cols: Sequence[str] = (),
    user_id_col: str = 'user_id',
    high_nan_threshold: Optional[float] = None,
    quasi_constant_threshold: Optional[float] = None,
    corr_threshold_v2: Optional[float] = None,
    corr_threshold_v3: Optional[float] = None,
    target_leakage_cols: Optional[Sequence[str]] = None,
) -> Dict:
    """
    Calcula DINÁMICAMENTE las 4 listas de columnas a eliminar.

    Args:
        master: dataframe
        target_cols: nombres de cols target a proteger
        user_id_col: nombre de la col user_id (también protegida)
        high_nan_threshold: default 0.95 (de config.CLEANUP_THRESHOLDS)
        quasi_constant_threshold: default 0.95
        corr_threshold_v2: default 0.99
        corr_threshold_v3: default 0.95
        target_leakage_cols: lista fija schema-level (default config.TARGET_LEAKAGE_COLS)

    Returns:
        dict con keys:
        - 'high_missing': list
        - 'quasi_constant': list
        - 'target_leakage': list (intersección con master.columns)
        - 'corr_99_drop': list
        - 'corr_95_drop': list (incremental sobre corr_99)
        - 'corr_99_pairs': list of (col_a, col_b, corr) detectados
        - 'corr_95_pairs': list of (col_a, col_b, corr)
    """
    # Defaults
    thresholds = CLEANUP_THRESHOLDS
    if high_nan_threshold is None:
        high_nan_threshold = thresholds['high_nan_threshold']
    if quasi_constant_threshold is None:
        quasi_constant_threshold = thresholds['quasi_constant_threshold']
    if corr_threshold_v2 is None:
        corr_threshold_v2 = thresholds['corr_threshold_v2']
    if corr_threshold_v3 is None:
        corr_threshold_v3 = thresholds['corr_threshold_v3']
    if target_leakage_cols is None:
        target_leakage_cols = TARGET_LEAKAGE_COLS

    # Cols a proteger de todos los análisis
    exclude_cols = [user_id_col] + list(target_cols)
    exclude_cols = [c for c in exclude_cols if c in master.columns]

    # 1. HIGH NaN
    high_missing = compute_high_nan_cols(master, exclude_cols, high_nan_threshold)

    # 2. CUASI-CONSTANTES
    quasi_constant = compute_quasi_constant_cols(
        master, exclude_cols, quasi_constant_threshold
    )

    # 3. TARGET LEAKAGE (lista fija intersectada con cols presentes)
    target_leakage = sorted([c for c in target_leakage_cols if c in master.columns])

    # 4. CORR_99 (excluir ya marcadas)
    already_dropped_v1 = (
        set(high_missing) | set(quasi_constant) | set(target_leakage)
    )
    corr_99_drop, corr_99_pairs = compute_correlation_drops(
        master, exclude_cols, corr_threshold_v2, already_dropped_v1
    )

    # 5. CORR_95 incremental (excluir las dropeadas en corr_99)
    already_dropped_v2 = already_dropped_v1 | set(corr_99_drop)
    corr_95_drop, corr_95_pairs = compute_correlation_drops(
        master, exclude_cols, corr_threshold_v3, already_dropped_v2
    )

    return {
        'high_missing': high_missing,
        'quasi_constant': quasi_constant,
        'target_leakage': target_leakage,
        'corr_99_drop': corr_99_drop,
        'corr_95_drop': corr_95_drop,
        'corr_99_pairs': corr_99_pairs,
        'corr_95_pairs': corr_95_pairs,
    }


def apply_cleanups(
    master: pd.DataFrame,
    drops: Dict,
) -> Dict[str, pd.DataFrame]:
    """
    Aplica los 3 cleanups (v1, v2, v3) usando los drops calculados dinámicamente.

    v1_conservative = drop high_missing ∪ quasi_constant ∪ target_leakage
    v2_intermediate = v1 ∪ corr_99_drop
    v3_aggressive   = v2 ∪ corr_95_drop
    """
    to_drop_v1 = set(drops['high_missing']) | set(drops['quasi_constant']) | set(drops['target_leakage'])
    to_drop_v2 = to_drop_v1 | set(drops['corr_99_drop'])
    to_drop_v3 = to_drop_v2 | set(drops['corr_95_drop'])

    # Intersectar con cols realmente presentes (por seguridad)
    cols = set(master.columns)
    return {
        'v1_conservative': master.drop(columns=list(to_drop_v1 & cols)),
        'v2_intermediate': master.drop(columns=list(to_drop_v2 & cols)),
        'v3_aggressive':   master.drop(columns=list(to_drop_v3 & cols)),
    }


def generate_drops_report(drops: Dict, output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Genera CSV con (feature, motivo, threshold_value) para trazabilidad.
    """
    rows = []
    for col in drops['high_missing']:
        rows.append({'feature': col, 'motivo': 'high_nan',
                     'threshold': CLEANUP_THRESHOLDS['high_nan_threshold']})
    for col in drops['quasi_constant']:
        rows.append({'feature': col, 'motivo': 'quasi_constant',
                     'threshold': CLEANUP_THRESHOLDS['quasi_constant_threshold']})
    for col in drops['target_leakage']:
        rows.append({'feature': col, 'motivo': 'target_leakage', 'threshold': None})
    for col in drops['corr_99_drop']:
        rows.append({'feature': col, 'motivo': 'corr_99',
                     'threshold': CLEANUP_THRESHOLDS['corr_threshold_v2']})
    for col in drops['corr_95_drop']:
        rows.append({'feature': col, 'motivo': 'corr_95',
                     'threshold': CLEANUP_THRESHOLDS['corr_threshold_v3']})

    df = pd.DataFrame(rows, columns=['feature', 'motivo', 'threshold'])
    if output_path is not None:
        df.to_csv(output_path, index=False)
    return df
