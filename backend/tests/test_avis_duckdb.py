"""Parité `avis.avis_list` (Python) ↔ `avis.avis_list_duckdb` (SQL) — même contrat.

Le hot path `/avis_list` gagne un moteur de LECTURE DuckDB (`backend.bake_duckdb`) MAIS
doit rendre EXACTEMENT le même résultat que le fallback Python (mêmes `total`, mêmes ids
d'items dans le même ordre, même shape). On bake une mini-base en mémoire depuis une
`avis.json` synthétique + `claim_stance` et on compare les DEUX chemins sur un panel de
requêtes (sans filtre, thème macro/feuille, recherche accent/casse, stance, combiné,
pagination). `duckdb` absent → skip propre (extra `collect`/`serve`).
"""

from __future__ import annotations

import pytest

from backend.avis import avis_list, avis_list_duckdb

duckdb = pytest.importorskip("duckdb")
from backend.bake_duckdb import build_db   # noqa: E402  (après le skip conditionnel)

# Hiérarchie : macro `m1` {feuilles `c1`,`c2`}, macro `m2` isolé (mêmes champs que /analysis).
THEMES = [
    {"id": "m1", "parent_id": None, "title": "Macro 1", "color": "#111"},
    {"id": "c1", "parent_id": "m1", "title": "Feuille 1", "color": "#f00"},
    {"id": "c2", "parent_id": "m1", "title": "Feuille 2", "color": "#0f0"},
    {"id": "m2", "parent_id": None, "title": "Macro 2", "color": "#00f"},
]

# Provenance : `leaf_id` = feuille réelle, `cluster_id` = macro (couleur), comme en prod.
AVIS = {
    "a1": {"id": "a1", "text": "Réglementer l'accès des mineurs aux réseaux.", "lang": "fr",
           "claims": [{"id": "a1#0", "cluster_id": "m1", "leaf_id": "c1",
                       "theme_title": "F1", "color": "#f00", "spans": [{"start": 0, "end": 5}]}]},
    "a2": {"id": "a2", "text": "Protéger les données personnelles des citoyens.", "lang": "fr",
           "claims": [{"id": "a2#0", "cluster_id": "m1", "leaf_id": "c2",
                       "theme_title": "F2", "color": "#0f0", "spans": [{"start": 0, "end": 8}]},
                      {"id": "a2#1", "cluster_id": "m1", "leaf_id": "c2",
                       "theme_title": "F2", "color": "#0f0", "spans": [{"start": 9, "end": 12}]}]},
    "a3": {"id": "a3", "text": "Liberté d'expression avant tout, sans réglementation.",
           "lang": "fr",
           "claims": [{"id": "a3#0", "cluster_id": "m2", "leaf_id": "m2",
                       "theme_title": "M2", "color": "#00f", "spans": [{"start": 0, "end": 7}]}]},
    "a4": {"id": "a4", "text": "Aucun claim ici (avis sans idée extraite).", "lang": "fr",
           "claims": []},
}

# Stance par claim (baké au build ; joint par id de claim `avis_id#idx`).
CLAIM_STANCE = {
    "a1#0": {"stance": "favorable"},
    "a2#0": {"stance": "defavorable"},
    "a2#1": {"stance": "favorable"},
    "a3#0": {"stance": "favorable"},
}

# Panel de requêtes : couvre chaque prédicat + combinaisons + pagination.
QUERIES = [
    {},                                                  # tout
    {"theme_id": "m1"},                                  # macro → sous-arbre {c1,c2}
    {"theme_id": "c1"},                                  # feuille exacte
    {"theme_id": "m2"},                                  # autre macro
    {"q": "REGLEMENTER"},                                # casse + accents
    {"q": "réglement"},                                  # sous-chaîne (matche a1 ET a3)
    {"q": "zzznope"},                                    # aucun match
    {"stance": "favorable"},                             # stance seule
    {"stance": "defavorable"},
    {"theme_id": "m1", "stance": "favorable"},           # thème ∧ stance
    {"theme_id": "c2", "stance": "defavorable"},         # feuille ∧ stance (même claim)
    {"theme_id": "m1", "q": "donnees"},                  # thème ∧ recherche
    {"limit": 1, "offset": 1},                           # pagination
    {"limit": 2, "offset": 2},
]


@pytest.fixture()
def con():
    c = duckdb.connect(":memory:")
    build_db(c, AVIS, CLAIM_STANCE, THEMES)
    try:
        yield c
    finally:
        c.close()


@pytest.mark.parametrize("params", QUERIES, ids=lambda p: str(p) or "all")
def test_parity_fallback_vs_duckdb(con, params):
    """Même `total`, mêmes ids/ordre, même shape d'item entre les deux chemins."""
    py = avis_list(AVIS, THEMES, claim_stance=CLAIM_STANCE, **params)
    db = avis_list_duckdb(con, THEMES, claim_stance=CLAIM_STANCE, **params)
    assert db["total"] == py["total"], params
    assert [it["avis_id"] for it in db["items"]] == [it["avis_id"] for it in py["items"]], params
    # Parité de shape ET de contenu, item par item (payload verbatim ⇒ byte-identique).
    assert db["items"] == py["items"], params


def test_full_item_shape(con):
    """Un item DuckDB porte le contrat complet `/avis_list` (avis entier + aperçu + thèmes)."""
    db = avis_list_duckdb(con, THEMES, theme_id="c2")
    assert db["total"] == 1
    item = db["items"][0]
    assert set(item) == {"avis_id", "excerpt", "themes", "text", "text_fr", "lang", "claims"}
    assert item["avis_id"] == "a2"
    assert item["claims"] == AVIS["a2"]["claims"]      # claims verbatim préservés
    assert item["themes"] == [{"id": "m1", "title": "F2", "color": "#0f0"}]
