"""
pipeline_context.py — contexto de ejecución del pipeline.

Sustituye al `config.py` del validation_study. En deployment, el contexto NO
viene de constantes globales — viene de los parámetros que el CLI / la UI le
pasan (paths de CSVs subidos por el cliente, fecha de referencia derivada del
last_login_date máximo, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path


@dataclass
class PipelineContext:
    """Contexto de una ejecución concreta del pipeline."""

    raw_csvs_dir: Path                # Carpeta donde están los CSVs del cliente
    reference_date: date              # Fecha de "ahora" derivada de max(last_login_date)
    cutoff_days: int = 90             # L22 (RF L22 v1, modelo final del TFG)
    spike_days: int = 7
    min_logins: int = 2               # L22 (antes L32 usaba 5)

    @property
    def cutoff_date(self) -> date:
        return self.reference_date - timedelta(days=self.cutoff_days)
