"""
_stats.py — helpers estadísticos compartidos entre feature pipelines.

Implementaciones fieles a las del TFG (notebook 04c, cell c01):
- Shannon entropy (base 2)
- Gini coefficient sobre distribución de counts
- Simpson diversity (1 - Σ p_i²)
- Binge index (max/median de daily counts)

Las funciones aceptan iterables de números (lista, values de dict, Series).
Tratan listas vacías / sumas cero devolviendo NaN (mismo comportamiento
que el TFG).
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


def shannon(values: Iterable[float]) -> float:
    """
    Entropía de Shannon en base 2 sobre una distribución de counts.

    Replica EXACTA de `04c_features_diversidad.ipynb` (c01):
        s = arr.sum()
        if s == 0: return NaN
        p = arr / s
        p = p[p > 0]
        return -np.sum(p * np.log2(p))
    """
    arr = np.asarray(list(values), dtype=float)
    s = arr.sum()
    if s == 0:
        return float("nan")
    p = arr / s
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def gini(values: Iterable[float]) -> float:
    """
    Coeficiente de Gini sobre la distribución (ordenada ascendentemente).

    Replica EXACTA de `04c_features_diversidad.ipynb` (c01):
        sorted = sort(arr)
        return (2·Σ(i·x_i) - (n+1)·s) / (n·s)
    """
    arr = np.sort(np.asarray(list(values), dtype=float))
    n = len(arr)
    s = arr.sum()
    if n == 0 or s == 0:
        return float("nan")
    return float((2 * np.sum(np.arange(1, n + 1) * arr) - (n + 1) * s) / (n * s))


def simpson(values: Iterable[float]) -> float:
    """
    Diversidad de Simpson: 1 - Σ p_i².

    Replica EXACTA de `04c_features_diversidad.ipynb` (c01).
    """
    arr = np.asarray(list(values), dtype=float)
    s = arr.sum()
    if s == 0:
        return float("nan")
    p = arr / s
    return float(1 - np.sum(p ** 2))


def binge_index(daily_counts: Mapping) -> float:
    """
    Índice de "binge": ratio max(daily) / median(daily).

    Replica del TFG (04d, función `binge` inline):
        if no counts: NaN
        m = median(values)
        return max(values)/m if m > 0 else NaN
    """
    if not daily_counts:
        return float("nan")
    vals = list(daily_counts.values())
    m = np.median(vals)
    if m <= 0:
        return float("nan")
    return float(max(vals) / m)
