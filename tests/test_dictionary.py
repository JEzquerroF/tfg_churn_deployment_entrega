"""Tests del diccionario.json + limpieza de emojis en outputs (Fase 1.6)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_dictionary import build_dictionary  # noqa: E402


# Regex que detecta cualquier emoji o pictograma (rangos Unicode comunes).
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"   # símbolos & pictogramas
    "\U0001F600-\U0001F64F"   # emoticonos
    "\U0001F680-\U0001F6FF"   # transporte & símbolos
    "\U0001F700-\U0001F77F"   # alquímia
    "\U0001F780-\U0001F7FF"   # geométricos
    "\U0001F800-\U0001F8FF"   # suplementarios
    "\U0001F900-\U0001F9FF"   # suplementarios símbolos & pictogramas
    "\U0001FA00-\U0001FA6F"   # símbolos extendidos
    "\U0001FA70-\U0001FAFF"   # símbolos & pictogramas extendidos
    "☀-⛿"           # símbolos misceláneos (incluye ⚔)
    "✀-➿"           # dingbats (incluye ✂, ✈, etc.)
    "]+"
)


def _contains_emoji(s: str) -> bool:
    return bool(EMOJI_PATTERN.search(s))


def test_archetypes_yaml_names_have_no_emoji():
    """Los `name` de arquetipo NO deben tener emojis."""
    with open(REPO_ROOT / "config" / "archetypes.yaml") as f:
        cfg = yaml.safe_load(f)

    for cluster_id, archetype in cfg["archetypes_n1"].items():
        name = archetype["name"]
        assert not _contains_emoji(name), (
            f"Arquetipo {cluster_id} ({name!r}) contiene emoji en `name`."
        )


def test_catalog_has_12_countermeasures_with_stable_ids():
    """El catálogo `contramedidas` tiene 12 entradas con id CM0X único + label + mecánica."""
    with open(REPO_ROOT / "config" / "archetypes.yaml") as f:
        cfg = yaml.safe_load(f)

    contramedidas = cfg.get("contramedidas", [])
    assert len(contramedidas) == 12, f"esperaba 12 contramedidas, hay {len(contramedidas)}"

    ids_seen: set[str] = set()
    prioridades_seen: set[int] = set()
    for cm in contramedidas:
        for field in ("id", "label", "prioridad", "mecanica", "disparador"):
            assert field in cm, f"contramedida {cm.get('id', '?')}: falta '{field}'"
        assert re.match(r"^CM\d{2}$", cm["id"]), f"id {cm['id']!r} no es CM0X"
        assert cm["id"] not in ids_seen, f"ID {cm['id']} duplicado"
        ids_seen.add(cm["id"])
        prioridades_seen.add(cm["prioridad"])

    assert ids_seen == {f"CM{i:02d}" for i in range(1, 13)}, "ids deben ser CM01..CM12"
    assert prioridades_seen == set(range(1, 13)), "prioridades deben ser 1..12 únicas"


def test_prelacion_order_and_fallback():
    """El orden de prelación tiene las 12 CM y el fallback es CM04."""
    with open(REPO_ROOT / "config" / "archetypes.yaml") as f:
        cfg = yaml.safe_load(f)

    orden = cfg["prioridad"]["orden"]
    assert len(orden) == 12
    assert set(orden) == {f"CM{i:02d}" for i in range(1, 13)}
    assert cfg["prioridad"]["fallback_final"]["contramedida"] == "CM04"


def test_seven_taste_axes_defined():
    """Los 7 ejes de gusto están definidos; 5 universales + 2 Tier 2."""
    with open(REPO_ROOT / "config" / "archetypes.yaml") as f:
        cfg = yaml.safe_load(f)

    ejes = cfg["ejes_gusto"]
    assert set(ejes.keys()) == {
        "clase_favorita", "estilo_build", "monetizacion", "perfil_enhance",
        "perfil_coleccion", "pvp_perfil", "perfil_oro",
    }
    tier2 = [e for e, c in ejes.items() if c.get("requiere_tier2")]
    assert set(tier2) == {"pvp_perfil", "perfil_oro"}


def test_dictionary_builds_without_emojis():
    """build_dictionary() no debe producir emojis en ningún campo del JSON serializado."""
    diccionario = build_dictionary(REPO_ROOT)
    serialized = json.dumps(diccionario, ensure_ascii=False)
    assert not _contains_emoji(serialized), (
        "diccionario.json contiene emojis. No deben estar en outputs."
    )


def test_dictionary_has_expected_structure():
    """Estructura del diccionario v2.0: arquetipos + 12 CM + ejes, sin archetype_n2."""
    diccionario = build_dictionary(REPO_ROOT)

    for key in (
        "version", "generated_at", "model_version",
        "schema_description", "archetype_n1", "ejes_gusto",
        "countermeasures", "source_codes",
    ):
        assert key in diccionario, f"falta '{key}' en diccionario"

    # Nivel 2 eliminado: no debe haber archetype_n2
    assert "archetype_n2" not in diccionario, "archetype_n2 no debe existir (Nivel 2 eliminado)"

    # 6 arquetipos N1 (sin la antigua sub-lista de countermeasures)
    assert len(diccionario["archetype_n1"]) == 6
    for k, arch in diccionario["archetype_n1"].items():
        assert isinstance(arch.get("code"), int)
        assert arch["code"] == int(k), f"code != clave en arquetipo {k}"
        assert "name" in arch
        assert "countermeasures" not in arch, "archetype_n1 ya no lleva countermeasures inline"

    # 12 contramedidas con id CM01..CM12
    cms = diccionario["countermeasures"]
    assert set(cms.keys()) == {f"CM{i:02d}" for i in range(1, 13)}
    for cod, cm in cms.items():
        assert cm["cod"] == cod
        assert "label" in cm and "mecanica" in cm and "disparador" in cm

    # 7 ejes de gusto
    assert set(diccionario["ejes_gusto"].keys()) == {
        "clase_favorita", "estilo_build", "monetizacion", "perfil_enhance",
        "perfil_coleccion", "pvp_perfil", "perfil_oro",
    }

    # 3 source codes
    assert set(diccionario["source_codes"].keys()) == {"oof_stable", "live_drift", "live_new"}
