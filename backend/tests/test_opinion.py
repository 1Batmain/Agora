"""Opinion — agrégation (pur, sans LLM) + endpoint `/opinion` (sert l'artefact à part).

L'agrégation est testée DIRECTEMENT (logique de répartition + garde-fous de pureté,
zéro appel réseau). L'endpoint est testé en monkeypatchant `analysis_store.read_opinion`
pour ne dépendre d'AUCUN cache disque : on fige la forme servie et la dégradation
gracieuse (liste vide quand l'opinion n'a pas été bakée).
"""

from __future__ import annotations

from collections import Counter

from backend import analysis_store
from backend.avis import join_claim_stance
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


# --------------------------------------------------------------------------- #
# Stance PAR CLAIM — join gracieux dans /avis (transparence par claim)
# --------------------------------------------------------------------------- #
STANCE_MAP = {
    "a1#0": {"stance": "favorable", "justif": "soutient la mesure",
             "proposition": "instaurer le RIC", "theme_id": "n1"},
    "a1#2": {"stance": "defavorable", "justif": "rejette la mesure",
             "proposition": "instaurer le RIC", "theme_id": "n1"},
}


def test_join_claim_stance_enriches_matching_claims():
    """Les claims dont l'id est dans la map reçoivent stance/proposition/justif."""
    claims = [{"id": "a1#0", "spans": [{"start": 0, "end": 5}], "target": None},
              {"id": "a1#1", "spans": [{"start": 6, "end": 9}], "target": None},
              {"id": "a1#2", "spans": [{"start": 10, "end": 15}], "target": None}]
    out = join_claim_stance(claims, STANCE_MAP)
    assert out[0]["stance"] == "favorable"
    assert out[0]["proposition"] == "instaurer le RIC"
    assert out[0]["stance_justif"] == "soutient la mesure"
    assert out[2]["stance"] == "defavorable"
    # Claim sans entrée : inchangé, pas de clés de stance ajoutées (gracieux).
    assert "stance" not in out[1]
    # L'ancrage verbatim (spans/target) est intact sur tous les claims.
    assert all(c["spans"] == claims[i]["spans"] for i, c in enumerate(out))


def test_join_claim_stance_graceful_without_map():
    """Map absente (opinion non bakée) → claims renvoyés tels quels."""
    claims = [{"id": "a1#0", "spans": []}]
    assert join_claim_stance(claims, None) is claims
    assert join_claim_stance(claims, {}) is claims


def test_claim_stance_store_round_trip(tmp_path, monkeypatch):
    """write_claim_stance → read_claim_stance restitue la map (et le cache s'invalide)."""
    monkeypatch.setattr(analysis_store, "claim_stance_path",
                        lambda _ds: tmp_path / "claim_stance.json")
    analysis_store._CLAIM_STANCE_CACHE.clear()
    assert analysis_store.read_claim_stance("ds") is None      # absent → None gracieux
    analysis_store.write_claim_stance("ds", STANCE_MAP)
    got = analysis_store.read_claim_stance("ds")
    assert got == STANCE_MAP


def test_avis_endpoint_joins_stance(client, monkeypatch):
    """`/avis` joint la stance par claim quand l'artefact existe (gracieux sinon)."""
    avis_fixture = {
        "id": "a1", "text": "Texte de l'avis.", "text_fr": None, "lang": "fr",
        "claims": [{"id": "a1#0", "cluster_id": "n1", "color": "#f00",
                    "spans": [{"start": 0, "end": 5}], "target": None,
                    "theme_title": "Démocratie"}],
    }
    monkeypatch.setattr(analysis_store, "state", lambda _ds: analysis_store.READY)
    monkeypatch.setattr(analysis_store, "read_avis", lambda _ds, _id: avis_fixture)
    monkeypatch.setattr(analysis_store, "read_claim_stance", lambda _ds: STANCE_MAP)
    r = client.get("/avis/a1", params={"dataset": DEFAULT_DATASET})
    assert r.status_code == 200, r.text
    claim = r.json()["claims"][0]
    assert claim["stance"] == "favorable"
    assert claim["proposition"] == "instaurer le RIC"
    assert claim["spans"] == [{"start": 0, "end": 5}]   # ancrage verbatim intact
