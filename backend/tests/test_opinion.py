"""Opinion — agrégation (pur, sans LLM) + endpoint `/opinion` (sert l'artefact à part).

L'agrégation est testée DIRECTEMENT (logique de répartition + garde-fous de pureté,
zéro appel réseau). L'endpoint est testé en monkeypatchant `analysis_store.read_opinion`
pour ne dépendre d'AUCUN cache disque : on fige la forme servie et la dégradation
gracieuse (liste vide quand l'opinion n'a pas été bakée).
"""

from __future__ import annotations

from collections import Counter

from backend import analysis_store
from backend.build_opinion import aggregate
from backend.recluster import DEFAULT_DATASET


# --------------------------------------------------------------------------- #
# Agrégation — profils clivant / consensuel / impur
# --------------------------------------------------------------------------- #
def test_aggregate_clivant():
    """Opposition réelle (fav et def tous deux substantiels) → 'clivant'."""
    o = aggregate("n1", "instaurer le RIC", Counter(favorable=10, defavorable=5, nuance=5), 20)
    assert o["profil"] == "clivant"
    assert o["fav"] == 10 and o["def"] == 5 and o["nuance"] == 5 and o["n"] == 20
    assert o["pct_favorable"] == round(10 / 15, 3)
    assert o["opposition"] == round(5 / 15, 3)
    assert o["engagement"] == round(15 / 20, 3)
    assert o["theme_id"] == "n1" and o["proposition"] == "instaurer le RIC"


def test_aggregate_consensuel():
    """Large adhésion, opposition < seuil → 'consensuel' (minorité de sceptiques)."""
    o = aggregate("n2", "rapprocher élus et citoyens", Counter(favorable=18, defavorable=1, nuance=1), 20)
    assert o["profil"] == "consensuel"
    assert o["opposition"] < 0.15
    assert o["pct_favorable"] == round(18 / 19, 3)


def test_aggregate_impur_low_engagement():
    """Trop de nuance (engagement < seuil) → 'impur' : pas de répartition affichée."""
    o = aggregate("n3", "objet diffus", Counter(favorable=2, defavorable=1, nuance=17), 20)
    assert o["profil"] == "impur"
    assert o["engagement"] < 0.35


def test_aggregate_impur_few_claims():
    """Sous le plancher de claims, signal trop faible → 'impur' même très engagé."""
    o = aggregate("n4", "petite feuille", Counter(favorable=3, defavorable=2), 5)
    assert o["profil"] == "impur"
    assert o["n"] == 5


def test_aggregate_empty():
    """Aucun claim : ne lève pas (divisions gardées) et sort 'impur'."""
    o = aggregate("n5", "vide", Counter(), 0)
    assert o["profil"] == "impur"
    assert o["engagement"] == 0.0 and o["opposition"] == 0.0 and o["pct_favorable"] == 0.0


# --------------------------------------------------------------------------- #
# Endpoint /opinion
# --------------------------------------------------------------------------- #
def test_opinion_absent_is_graceful(client, monkeypatch):
    """Opinion non bakée → 200 + liste vide (le front dégrade sans bloquer la synthèse)."""
    monkeypatch.setattr(analysis_store, "read_opinion", lambda _ds: None)
    r = client.get("/opinion", params={"dataset": DEFAULT_DATASET})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dataset"] == DEFAULT_DATASET
    assert body["themes"] == [] and body["status"] == "absent"


def test_opinion_served_shape(client, monkeypatch):
    """Opinion présente → l'endpoint sert le payload tel quel, par theme_id."""
    fixture = {
        "dataset": DEFAULT_DATASET,
        "model": "mistral-small-latest",
        "themes": [
            {"theme_id": "n1", "proposition": "instaurer le RIC", "fav": 10, "def": 5,
             "nuance": 5, "n": 20, "engagement": 0.75, "opposition": 0.333,
             "pct_favorable": 0.667, "profil": "clivant"},
        ],
    }
    monkeypatch.setattr(analysis_store, "read_opinion", lambda _ds: fixture)
    r = client.get("/opinion", params={"dataset": DEFAULT_DATASET})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dataset"] == DEFAULT_DATASET
    themes = body["themes"]
    assert isinstance(themes, list) and len(themes) == 1
    t = themes[0]
    assert set(t) >= {"theme_id", "proposition", "fav", "def", "nuance",
                      "pct_favorable", "opposition", "profil"}
    assert t["profil"] in ("clivant", "consensuel", "impur")


def test_opinion_unknown_dataset(client):
    """Id non whitelisté → 404 (garde path-traversal partagée avec les autres endpoints)."""
    r = client.get("/opinion", params={"dataset": "../etc"})
    assert r.status_code == 404, r.text
