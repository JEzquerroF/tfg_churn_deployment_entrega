"""Tests del perfilado de gustos (7 ejes + 12 contramedidas + prelación).

Verifica con datos sintéticos que:
  - los 7 ejes se calculan (5 universales + 2 Tier 2);
  - la prelación asigna la contramedida correcta (primer match gana);
  - el fallback CM04 cierra el gap (0 jugadores sin asignar);
  - la cobertura es del 100%;
  - has_tier2 habilita los ejes pvp_perfil / perfil_oro y las CM10/CM11.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline import perfilado  # noqa: E402


# Arquetipos (code → name) para construir archetypes_df.
ARCH = {
    0: "Recién Llegado Explorador",
    1: "Jugador Establecido Activo",
    2: "Hardcore End-Game",
    3: "Veterano Especializado",
    4: "Casual Dormido",
    5: "Veterano Inversor",
}


def _base_row() -> dict:
    """Fila 'neutra': no_pagador, build equilibrado, no inversor, funcional, sin Tier 2."""
    return {
        "char_class_main": 0,            # Caballero
        "items_attack_defense_ratio": 1.10,  # equilibrado (entre t33=1.037 y t67=1.170)
        "iap_trial_only": 0,
        "iap_is_payer": 0,
        "reward_has_ad": 0,
        "items_max_enhance_level": 1,    # <= 2 → no inversor
        "pct_items_high_enhance": 0.0,
        "items_redundancy_ratio": 1.0,   # <= 1.5
        "coll_total_items": 10,          # <= 42 → funcional
        "fights_pct_pvp": np.nan,        # sin Tier 2
        "fights_pct_won": np.nan,
        "currency_pct_inflow": np.nan,
        "currency_pct_outflow": np.nan,
    }


@pytest.fixture
def scenario() -> tuple[pd.DataFrame, pd.DataFrame, perfilado.TasteConfig]:
    """Construye un master sintético con un caso por contramedida esperada."""
    rows = []
    arche = []

    def add(user_id, arch_code, **overrides):
        r = _base_row()
        r.update(overrides)
        r["user_id"] = user_id
        rows.append(r)
        arche.append({"user_id": user_id, "archetype_n1": arch_code})

    # CM03 — trial_no_convertido (gana sobre todo lo demás)
    add("u_cm03", 2, iap_trial_only=1, iap_is_payer=1)
    # CM01 — Hardcore + pagador
    add("u_cm01", 2, iap_is_payer=1)
    # CM02 — pagador no-Hardcore
    add("u_cm02", 1, iap_is_payer=1)
    # CM10 — Tier 2 acumulador (currency inflow > 0.85), no pagador
    add("u_cm10", 4, currency_pct_inflow=0.92, currency_pct_outflow=0.05)
    # CM11 — Tier 2 pvp_frustrado (pvp>0.3, won<0.3), no acumulador
    add("u_cm11", 1, fights_pct_pvp=0.5, fights_pct_won=0.1)
    # CM12 — Recién Llegado (onboarding, prima sobre gustos)
    add("u_cm12", 0, items_max_enhance_level=5)  # aunque sería inversor, gana CM12
    # CM04 (regla) — Veterano Inversor no pagador
    add("u_cm04", 5)
    # CM06 — coleccionista en arquetipo Establecido
    add("u_cm06", 1, coll_total_items=100)
    # CM05 — inversor (enhance) en arquetipo NO veterano (si fuera veterano,
    # CM04 prioridad 7 lo capturaría antes que CM05 prioridad 9).
    add("u_cm05", 4, items_max_enhance_level=5)
    # CM07 — build agresivo en arquetipo activo
    add("u_cm07", 1, items_attack_defense_ratio=1.30)
    # CM08 — no_pagador_ads
    add("u_cm08", 1, reward_has_ad=1)
    # CM09 — Casual Dormido no_pagador
    add("u_cm09", 4)
    # FALLBACK CM04 — Establecido no_pagador sin ningún rasgo (no match en 12 reglas)
    add("u_fallback", 1)

    master = pd.DataFrame(rows)
    archetypes_df = pd.DataFrame(arche)
    config = perfilado.load_taste_config(REPO_ROOT)
    return master, archetypes_df, config


def test_seven_axes_computed(scenario) -> None:
    """build_profile produce las 7 columnas de ejes + has_tier2."""
    master, archetypes_df, config = scenario
    profile = perfilado.build_profile(master, archetypes_df, config)
    for eje in perfilado.EJES:
        assert eje in profile.columns, f"falta eje {eje}"
    assert "has_tier2" in profile.columns
    # Los 5 ejes universales no tienen NaN
    for eje in ("clase_favorita", "estilo_build", "monetizacion",
                "perfil_enhance", "perfil_coleccion"):
        assert profile[eje].notna().all(), f"eje universal {eje} con NaN"


def test_has_tier2_derivation(scenario) -> None:
    """has_tier2=1 solo para usuarios con datos transaccionales (fights o currency)."""
    master, archetypes_df, config = scenario
    profile = perfilado.build_profile(master, archetypes_df, config)
    by_user = dict(zip(profile["user_id"], profile["has_tier2"]))
    assert by_user["u_cm10"] == 1, "u_cm10 tiene currency → Tier 2"
    assert by_user["u_cm11"] == 1, "u_cm11 tiene fights → Tier 2"
    assert by_user["u_cm09"] == 0, "u_cm09 sin transaccional → no Tier 2"


def test_tier2_axes_only_for_tier2(scenario) -> None:
    """pvp_perfil / perfil_oro son 'null' para no-Tier2, con valor para Tier 2."""
    master, archetypes_df, config = scenario
    profile = perfilado.build_profile(master, archetypes_df, config).set_index("user_id")
    # Tier 2 acumulador
    assert profile.loc["u_cm10", "perfil_oro"] == "acumulador"
    # Tier 2 pvp frustrado
    assert profile.loc["u_cm11", "pvp_perfil"] == "pvp_frustrado"
    # No Tier 2 → null
    assert profile.loc["u_cm09", "pvp_perfil"] == "null"
    assert profile.loc["u_cm09", "perfil_oro"] == "null"


def test_prelacion_assigns_expected_countermeasures(scenario) -> None:
    """Cada usuario sintético recibe la contramedida esperada por la prelación."""
    master, archetypes_df, config = scenario
    profile = perfilado.build_profile(master, archetypes_df, config)
    result = perfilado.assign_countermeasures(profile, config)
    got = dict(zip(result["user_id"], result["contramedida_primaria_cod"]))

    expected = {
        "u_cm03": "CM03",
        "u_cm01": "CM01",
        "u_cm02": "CM02",
        "u_cm10": "CM10",
        "u_cm11": "CM11",
        "u_cm12": "CM12",
        "u_cm04": "CM04",
        "u_cm06": "CM06",
        "u_cm05": "CM05",
        "u_cm07": "CM07",
        "u_cm08": "CM08",
        "u_cm09": "CM09",
        "u_fallback": "CM04",
    }
    for user_id, exp in expected.items():
        assert got[user_id] == exp, f"{user_id}: esperaba {exp}, got {got[user_id]}"


def test_fallback_closes_gap_100pct_coverage(scenario) -> None:
    """Tras la prelación + fallback, 0 jugadores en CM00 (cobertura 100%)."""
    master, archetypes_df, config = scenario
    profile = perfilado.build_profile(master, archetypes_df, config)
    result = perfilado.assign_countermeasures(profile, config)

    assert (result["contramedida_primaria_cod"] == "CM00").sum() == 0
    valid = {f"CM{i:02d}" for i in range(1, 13)}
    assert set(result["contramedida_primaria_cod"].unique()).issubset(valid)
    # labels poblados para todos
    assert result["contramedida_primaria_label"].notna().all()
    assert (result["contramedida_primaria_label"] == "desconocida").sum() == 0


def test_trial_beats_payer_prelacion() -> None:
    """Prelación: trial (CM03) gana sobre pagador (CM01/CM02) aunque ambos apliquen."""
    config = perfilado.load_taste_config(REPO_ROOT)
    row = _base_row()
    row.update({"user_id": "u", "iap_trial_only": 1, "iap_is_payer": 1})
    master = pd.DataFrame([row])
    archetypes_df = pd.DataFrame([{"user_id": "u", "archetype_n1": 2}])  # Hardcore
    result = perfilado.profile_and_assign(master, archetypes_df, REPO_ROOT, config)
    assert result.iloc[0]["contramedida_primaria_cod"] == "CM03"
