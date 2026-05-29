"""
perfilado.py — perfilado de gustos (7 ejes) + asignación de contramedida primaria.

Port del sistema del TFG (scripts/gustos/perfilado_gustos.py, archetypes.yaml
v1.1). REEMPLAZA al antiguo Nivel 2 (HDBSCAN de sub-arquetipos, no operacional).

Para cada jugador calcula:
  - 5 ejes universales (100% de jugadores): clase_favorita, estilo_build,
    monetizacion, perfil_enhance, perfil_coleccion.
  - 2 ejes Tier 2 (solo has_tier2=1): pvp_perfil, perfil_oro.
  - has_tier2: flag derivado de la presencia de datos transaccionales
    (fights OR currency en la ventana de 30d).
  - contramedida_primaria_cod / _label: una de las 12 del catálogo, asignada
    por orden de prelación (primer match gana) + fallback CM04 → cobertura 100%.

Arquitectura estable-vs-intercambiable:
  - El catálogo (12 CM, labels, orden de prelación, fallback) y los umbrales
    viven en config/archetypes.yaml + config/_tercile_thresholds.json.
  - La lógica ejecutable de los disparadores vive aquí (como en el TFG). El
    orden de evaluación y el fallback se LEEN del YAML, no se hardcodean.

Diferencia clave con el TFG: en inferencia los terciles/percentiles NO se
recalculan sobre el sample (sería incorrecto con pocos usuarios). Se usan los
umbrales persistidos del training (config/_tercile_thresholds.json).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# Features que el perfilado espera en el master de gustos.
UNIVERSAL_FEATURES = [
    "char_class_main",
    "items_attack_defense_ratio",
    "iap_trial_only",
    "iap_is_payer",
    "reward_has_ad",
    "items_max_enhance_level",
    "pct_items_high_enhance",
    "items_redundancy_ratio",
    "coll_total_items",
]
TIER2_FEATURES = [
    "fights_pct_pvp",
    "fights_pct_won",
    "currency_pct_inflow",
    "currency_pct_outflow",
]

EJES = [
    "clase_favorita",
    "estilo_build",
    "monetizacion",
    "perfil_enhance",
    "perfil_coleccion",
    "pvp_perfil",
    "perfil_oro",
]


class TasteConfig:
    """Config del perfilado leída de archetypes.yaml + _tercile_thresholds.json."""

    def __init__(self, archetypes_yaml: Path, thresholds_json: Path) -> None:
        cfg = yaml.safe_load(archetypes_yaml.read_text())
        th = json.loads(thresholds_json.read_text())

        # arquetipo_n1 code -> name
        self.arch_names: dict[int, str] = {
            int(k): v["name"] for k, v in cfg["archetypes_n1"].items()
        }
        # char_class_main code -> clase
        self.clase_map: dict[int, str] = {
            int(k): v for k, v in cfg["clase_mapping"].items()
        }
        # catálogo: code -> label
        self.cm_labels: dict[str, str] = {
            cm["id"]: cm["label"] for cm in cfg["contramedidas"]
        }
        # orden de prelación + fallback (intercambiables vía YAML)
        self.orden: list[str] = list(cfg["prioridad"]["orden"])
        self.fallback_cod: str = cfg["prioridad"]["fallback_final"]["contramedida"]

        # umbrales persistidos del training
        self.t33: float = float(th["estilo_build"]["t33"])
        self.t67: float = float(th["estilo_build"]["t67"])
        self.p75_coll: float = float(th["perfil_coleccion"]["p75"])

    # Conjuntos de arquetipos usados por los disparadores (resueltos por nombre)
    @property
    def arch_activos(self) -> set[str]:
        return {"Jugador Establecido Activo", "Hardcore End-Game"}

    @property
    def arch_establecido_o_veterano(self) -> set[str]:
        return {
            "Jugador Establecido Activo",
            "Veterano Especializado",
            "Veterano Inversor",
        }

    @property
    def arch_veteranos(self) -> set[str]:
        return {"Veterano Especializado", "Veterano Inversor"}


def load_taste_config(repo_root: Path) -> TasteConfig:
    return TasteConfig(
        archetypes_yaml=repo_root / "config" / "archetypes.yaml",
        thresholds_json=repo_root / "config" / "_tercile_thresholds.json",
    )


def _derive_has_tier2(master: pd.DataFrame) -> pd.Series:
    """has_tier2 = tiene datos transaccionales (fights OR currency) en ventana 30d.

    Réplica fiel del TFG: has_tier2 == (fights notna OR currency notna), 100%
    de coincidencia con el flag del two_stage_assignments del training.
    """
    present = pd.Series(False, index=master.index)
    for col in ["fights_pct_pvp", "fights_pct_won",
                "currency_pct_inflow", "currency_pct_outflow"]:
        if col in master.columns:
            present = present | master[col].notna()
    return present.astype(int)


def build_profile(
    master_gustos: pd.DataFrame,
    archetypes_df: pd.DataFrame,
    config: TasteConfig,
) -> pd.DataFrame:
    """Calcula los 7 ejes de gusto + has_tier2 para cada jugador.

    Args:
        master_gustos: master de gustos (user_id + features). Debe contener las
            9 features universales; las 4 Tier 2 son opcionales (NaN si faltan).
        archetypes_df: user_id + archetype_n1 (salida del clustering N1).
        config: TasteConfig con mappings, umbrales y catálogo.

    Returns:
        DataFrame: user_id, arquetipo_n1 (nombre), has_tier2, + 7 ejes.
    """
    df = master_gustos.merge(
        archetypes_df[["user_id", "archetype_n1"]], on="user_id", how="left"
    )

    out = pd.DataFrame({"user_id": df["user_id"]})
    out["arquetipo_n1"] = df["archetype_n1"].map(config.arch_names)
    out["has_tier2"] = _derive_has_tier2(df)

    # === EJE 1: clase_favorita ===
    out["clase_favorita"] = (
        df["char_class_main"].fillna(0).astype(int).map(config.clase_map).fillna("Caballero")
    )

    # === EJE 2: estilo_build (terciles persistidos del training) ===
    ratio = df["items_attack_defense_ratio"]
    out["estilo_build"] = np.where(
        ratio >= config.t67, "agresivo",
        np.where(ratio <= config.t33, "tanque", "equilibrado"),
    )

    # === EJE 3: monetizacion (trial PRIMERO) ===
    out["monetizacion"] = np.where(
        df["iap_trial_only"] == 1, "trial_no_convertido",
        np.where(
            df["iap_is_payer"] == 1, "pagador",
            np.where(df["reward_has_ad"] == 1, "no_pagador_ads", "no_pagador"),
        ),
    )

    # === EJE 4: perfil_enhance (mediana) ===
    out["perfil_enhance"] = np.where(
        (df["items_max_enhance_level"] > 2) | (df["pct_items_high_enhance"] > 0),
        "inversor", "no_inversor",
    )

    # === EJE 5: perfil_coleccion (p75 persistido) ===
    out["perfil_coleccion"] = np.where(
        (df["items_redundancy_ratio"] > 1.5) | (df["coll_total_items"] > config.p75_coll),
        "coleccionista", "funcional",
    )

    t2 = out["has_tier2"] == 1

    # === EJE 6: pvp_perfil (Tier 2) ===
    pvp = df["fights_pct_pvp"] if "fights_pct_pvp" in df.columns else pd.Series(np.nan, index=df.index)
    won = df["fights_pct_won"] if "fights_pct_won" in df.columns else pd.Series(np.nan, index=df.index)
    pvp_frustrado = t2 & (pvp.fillna(0) > 0.3) & (won.fillna(0) < 0.3)
    out["pvp_perfil"] = "null"
    out.loc[t2, "pvp_perfil"] = "pve_focus"
    out.loc[pvp_frustrado, "pvp_perfil"] = "pvp_frustrado"

    # === EJE 7: perfil_oro (Tier 2) ===
    inflow = df["currency_pct_inflow"] if "currency_pct_inflow" in df.columns else pd.Series(np.nan, index=df.index)
    outflow = df["currency_pct_outflow"] if "currency_pct_outflow" in df.columns else pd.Series(np.nan, index=df.index)
    out["perfil_oro"] = "null"
    out.loc[t2 & (inflow > 0.85), "perfil_oro"] = "acumulador"
    out.loc[t2 & (out["perfil_oro"] == "null") & (outflow > 0.25), "perfil_oro"] = "gastador"
    out.loc[t2 & (out["perfil_oro"] == "null"), "perfil_oro"] = "neutro"

    return out


def _build_predicates(profile: pd.DataFrame, config: TasteConfig) -> dict[str, pd.Series]:
    """Disparador booleano por código de contramedida (réplica del TFG).

    Devuelve un dict cod -> mask. El orden de aplicación lo decide el YAML.
    """
    a = profile["arquetipo_n1"]
    mon = profile["monetizacion"]
    has_t2 = profile["has_tier2"] == 1

    return {
        "CM03": mon == "trial_no_convertido",
        "CM01": (a == "Hardcore End-Game") & (mon == "pagador"),
        "CM02": (mon == "pagador") & (a != "Hardcore End-Game"),
        "CM10": has_t2 & (profile["perfil_oro"] == "acumulador"),
        "CM11": has_t2 & (profile["pvp_perfil"] == "pvp_frustrado"),
        "CM12": a == "Recién Llegado Explorador",
        "CM04": a.isin(config.arch_veteranos) & (mon != "pagador"),
        "CM06": (profile["perfil_coleccion"] == "coleccionista")
                & a.isin(config.arch_establecido_o_veterano),
        "CM05": profile["perfil_enhance"] == "inversor",
        "CM07": profile["estilo_build"].isin(["agresivo", "tanque"])
                & a.isin(config.arch_activos),
        "CM08": mon == "no_pagador_ads",
        "CM09": (a == "Casual Dormido") & (mon == "no_pagador"),
    }


def assign_countermeasures(profile: pd.DataFrame, config: TasteConfig) -> pd.DataFrame:
    """Asigna la contramedida primaria por el orden de prelación del YAML.

    Primer match gana. Los jugadores sin match reciben el fallback (CM04).
    Garantiza cobertura 100% (0 jugadores sin asignar).
    """
    n = len(profile)
    code = pd.Series(["CM00"] * n, index=profile.index, dtype="object")

    predicates = _build_predicates(profile, config)

    for cm_code in config.orden:
        mask = predicates.get(cm_code)
        if mask is None:
            logger.warning("Código %s en orden de prelación sin disparador definido", cm_code)
            continue
        m = mask & (code == "CM00")
        code.loc[m] = cm_code

    # Fallback final: cobertura 100%
    code.loc[code == "CM00"] = config.fallback_cod

    out = profile.copy()
    out["contramedida_primaria_cod"] = code
    out["contramedida_primaria_label"] = code.map(config.cm_labels).fillna("desconocida")
    return out


def profile_and_assign(
    master_gustos: pd.DataFrame,
    archetypes_df: pd.DataFrame,
    repo_root: Path,
    config: Optional[TasteConfig] = None,
) -> pd.DataFrame:
    """Pipeline completo: build_profile + assign_countermeasures.

    Returns:
        DataFrame: user_id, arquetipo_n1, has_tier2, 7 ejes,
        contramedida_primaria_cod, contramedida_primaria_label.
    """
    if config is None:
        config = load_taste_config(repo_root)
    profile = build_profile(master_gustos, archetypes_df, config)
    return assign_countermeasures(profile, config)
