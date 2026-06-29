"""Tests `/avis_list` — liste/recherche/filtre cluster paginée (page d'exploration).

Deux niveaux, sans réseau ni build LLM :
  * **Unitaire (toujours actif)** : le helper pur `backend.avis.avis_list` sur une mini
    `avis.json` synthétique — filtre par sous-arbre, recherche insensible casse/accents,
    pagination, thèmes uniques. Prouve le contrat hors-cache (cf. `test_avis_pii`).
  * **Shape serveur (gardé)** : `/avis_list?dataset=tiktok` sur le VRAI cache si l'analyse
    est précalculée (sinon skip propre via `require_ready`).
"""

from __future__ import annotations

from backend.avis import avis_list
from ._helpers import require_ready

# Hiérarchie de thèmes minimale : un macro `m1` avec deux feuilles `c1`,`c2` ; un macro
# `m2` isolé. (mêmes champs que `/analysis`.themes : id + parent_id.)
THEMES = [
    {"id": "m1", "parent_id": None},
    {"id": "c1", "parent_id": "m1"},
    {"id": "c2", "parent_id": "m1"},
    {"id": "m2", "parent_id": None},
]

# Mini provenance `{avis_id: {id, text, claims:[{cluster_id, theme_title, color}]}}`.
AVIS = {
    "a1": {"id": "a1", "text": "Réglementer l'accès des mineurs aux réseaux.",
           "claims": [{"cluster_id": "c1", "theme_title": "Mineurs", "color": "#f00"}]},
    "a2": {"id": "a2", "text": "Protéger les données personnelles.",
           "claims": [{"cluster_id": "c2", "theme_title": "Données", "color": "#0f0"},
                      {"cluster_id": "c2", "theme_title": "Données", "color": "#0f0"}]},
    "a3": {"id": "a3", "text": "Liberté d'expression avant tout.",
           "claims": [{"cluster_id": "m2", "theme_title": "Liberté", "color": "#00f"}]},
}


def test_filter_macro_includes_subtree():
    """Filtrer le macro `m1` ramène ses deux feuilles (`a1`,`a2`), pas `m2` (`a3`)."""
    out = avis_list(AVIS, THEMES, theme_id="m1")
    ids = {it["avis_id"] for it in out["items"]}
    assert ids == {"a1", "a2"}
    assert out["total"] == 2


def test_filter_leaf_is_exact():
    """Filtrer une feuille `c1` ne ramène que ses avis."""
    out = avis_list(AVIS, THEMES, theme_id="c1")
    assert [it["avis_id"] for it in out["items"]] == ["a1"]


def test_search_is_case_and_accent_insensitive():
    """`q='REGLEMENTER'` matche « Réglementer » (casse + accents ignorés)."""
    out = avis_list(AVIS, THEMES, q="REGLEMENTER")
    assert [it["avis_id"] for it in out["items"]] == ["a1"]
    # Aucune correspondance → total 0, items vides.
    assert avis_list(AVIS, THEMES, q="zzzznope")["total"] == 0


def test_search_and_filter_combine():
    out = avis_list(AVIS, THEMES, theme_id="m1", q="donnees")
    assert [it["avis_id"] for it in out["items"]] == ["a2"]


def test_pagination_slices_filtered_total():
    out = avis_list(AVIS, THEMES, limit=1, offset=1)
    assert out["total"] == 3          # total = AVANT pagination
    assert len(out["items"]) == 1


def test_item_shape_unique_themes_and_excerpt():
    out = avis_list(AVIS, THEMES, theme_id="c2")
    item = out["items"][0]
    # Item enrichi : aperçu + thèmes + avis ENTIER (text/text_fr/lang/claims) pour le
    # rendu inline côté front (plus d'appel `/avis/{id}` par carte).
    assert set(item) == {"avis_id", "excerpt", "themes",
                         "text", "text_fr", "lang", "claims"}
    # a2 porte 2 claims du même cluster → un seul thème listé.
    assert item["themes"] == [{"id": "c2", "title": "Données", "color": "#0f0"}]
    assert isinstance(item["excerpt"], str) and item["excerpt"]
    # Avis complet servi tel quel depuis `avis.json` (claims + défauts FR).
    assert item["text"] == AVIS["a2"]["text"]
    assert item["claims"] == AVIS["a2"]["claims"]
    assert item["text_fr"] is None and item["lang"] == "fr"


def test_avis_list_server_shape(client):
    """Sur le VRAI cache tiktok (si prêt) : enveloppe `{total, items:[…]}` cohérente."""
    require_ready(client, "tiktok")
    r = client.get("/avis_list", params={"dataset": "tiktok", "limit": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["total"], int)
    assert isinstance(body["items"], list) and len(body["items"]) <= 5
    for it in body["items"]:
        assert set(it) >= {"avis_id", "excerpt", "themes",
                           "text", "lang", "claims"}
        assert isinstance(it["text"], str) and isinstance(it["claims"], list)
        for th in it["themes"]:
            assert set(th) == {"id", "title", "color"}
