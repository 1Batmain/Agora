"""Démographie — jointure CSV enrichi ↔ avis (par ligne), agrégats par thème + rollup.

Zéro LLM, zéro torch : pure jointure sur artefacts servis (avis.json, analysis.json).
Artefact À PART et OPTIONNEL (`demographics.json`) — même contrat de rétro-compat que
l'opinion et les arguments : absent → rien ne change nulle part.
"""

from __future__ import annotations

import json

from backend import analysis_store
from backend.recluster import DEFAULT_DATASET

CSV = "\n".join([
    '"ID";"Contribution";"sexe";"age"',
    '"a";"texte 0";"Homme";"18 à 29 ans"',
    '"b";"texte 1";"Femme";"30 à 49 ans"',
    '"c";"texte 2";"Femme";""',
    # Ligne SANS avis analysé (ex. doublon regroupé) : compte dans le GLOBAL
    # (le panel = toutes les contributions), pas dans les thèmes.
    '"d";"texte 3 doublon";"Homme";"30 à 49 ans"',
])

AVIS = {
    "src:file.csv:0:1": {"claims": []},
    "src:file.csv:1:1": {"claims": []},
    "src:file.csv:2:1": {"claims": []},
    "src:pas-de-ligne": {"claims": []},      # id non parsable → ignoré
    "src:file.csv:99:1": {"claims": []},      # ligne hors CSV → ignoré
}

ANALYSIS = {"themes": [
    {"id": "n0", "parent_id": None},
    {"id": "n1", "parent_id": "n0"},
    {"id": "n2", "parent_id": "n0"},
]}

# Les fichiers citations/<theme>.json listent TOUS les claims d'un nœud (avec avis_id),
# parents inclus — c'est la source d'appartenance avis↔thème.
CITATIONS = {
    "n0": [{"avis_id": "src:file.csv:0:1"}, {"avis_id": "src:file.csv:1:1"},
           {"avis_id": "src:file.csv:2:1"}, {"avis_id": "src:pas-de-ligne"}],
    "n1": [{"avis_id": "src:file.csv:0:1"}, {"avis_id": "src:file.csv:1:1"},
           {"avis_id": "src:file.csv:2:1"},
           {"avis_id": "src:file.csv:1:1"}],  # 2 claims du même avis → compté UNE fois
    "n2": [{"avis_id": "src:file.csv:1:1"}, {"avis_id": "src:file.csv:2:1"},
           {"avis_id": "src:file.csv:99:1"}],
}


def _built(tmp_path, monkeypatch):
    from backend.build_demographics import build_demographics

    csv_path = tmp_path / "brute.csv"
    csv_path.write_text(CSV, encoding="utf-8")
    monkeypatch.setattr(analysis_store, "avis_path", lambda _ds: tmp_path / "avis.json")
    monkeypatch.setattr(analysis_store, "analysis_path", lambda _ds: tmp_path / "analysis.json")
    monkeypatch.setattr(analysis_store, "demographics_path",
                        lambda _ds: tmp_path / "demographics.json")
    monkeypatch.setattr(analysis_store, "read_citations",
                        lambda _ds, tid: CITATIONS.get(tid))
    analysis_store._AVIS_CACHE.clear()
    (tmp_path / "avis.json").write_text(json.dumps(AVIS), encoding="utf-8")
    (tmp_path / "analysis.json").write_text(json.dumps(ANALYSIS), encoding="utf-8")
    return build_demographics("ds", csv_path)


def test_global_counts_over_all_contributions(tmp_path, monkeypatch):
    """Le GLOBAL (note « Panel ») couvre TOUTES les lignes du CSV — pas seulement
    les avis analysés (les doublons regroupés restent des contributions)."""
    payload = _built(tmp_path, monkeypatch)
    assert payload["axes"] == ["sexe", "age"]
    assert payload["global"]["sexe"] == {"Femme": 2, "Homme": 2}
    assert payload["global"]["age"] == {"30 à 49 ans": 2, "18 à 29 ans": 1}
    assert payload["n_contributions"] == 4
    assert payload["n_avis_matched"] == 3   # les 2 ids non joignables sont écartés


def test_theme_majority_and_parent_rollup(tmp_path, monkeypatch):
    payload = _built(tmp_path, monkeypatch)
    themes = {t["theme_id"]: t for t in payload["themes"]}
    # n1 : avis 0(H), 1(F), 2(F) → majorité Femme 2/3.
    n1 = themes["n1"]["majority"]["sexe"]
    assert n1["label"] == "Femme" and n1["n"] == 2
    assert n1["share"] == round(2 / 3, 3)
    # n2 : avis 1(F), 2(F) → Femme 2/2 (l'avis hors CSV est écarté).
    assert themes["n2"]["majority"]["sexe"]["share"] == 1.0
    assert themes["n2"]["n_avis"] == 2
    # Parent n0 : ses citations couvrent les avis 0,1,2 → Femme 2/3 (id non parsable écarté).
    assert themes["n0"]["majority"]["sexe"]["label"] == "Femme"
    assert themes["n0"]["n_avis"] == 3
    # Axe age sur n2 : avis 1 (30-49) seul renseigné → majorité 30-49, share 1.0 sur 1 réponse.
    n2_age = themes["n2"]["majority"]["age"]
    assert n2_age["label"] == "30 à 49 ans" and n2_age["n"] == 1


def test_majority_tie_is_deterministic(tmp_path, monkeypatch):
    payload = _built(tmp_path, monkeypatch)
    # n1 âge : 18-29 (1) vs 30-49 (1) — égalité → plus grand compte puis ordre alphabétique.
    n1_age = {t["theme_id"]: t for t in payload["themes"]}["n1"]["majority"]["age"]
    assert n1_age["label"] == "18 à 29 ans"


def test_demographics_store_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis_store, "demographics_path",
                        lambda _ds: tmp_path / "demographics.json")
    assert analysis_store.read_demographics("ds") is None
    analysis_store.write_demographics("ds", {"dataset": "ds", "themes": []})
    assert analysis_store.read_demographics("ds") == {"dataset": "ds", "themes": []}


def test_demographics_endpoint_graceful(client, monkeypatch):
    monkeypatch.setattr(analysis_store, "read_demographics", lambda _ds: None)
    r = client.get("/demographics", params={"dataset": DEFAULT_DATASET})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["themes"] == [] and body["status"] == "absent"


def test_demographics_endpoint_served(client, monkeypatch):
    fixture = {"dataset": DEFAULT_DATASET, "axes": ["sexe"],
               "global": {"sexe": {"Femme": 2}},
               "themes": [{"theme_id": "n1", "n_avis": 3,
                           "majority": {"sexe": {"label": "Femme", "share": 0.67, "n": 2}}}]}
    monkeypatch.setattr(analysis_store, "read_demographics", lambda _ds: fixture)
    r = client.get("/demographics", params={"dataset": DEFAULT_DATASET})
    assert r.status_code == 200
    assert r.json()["themes"][0]["majority"]["sexe"]["label"] == "Femme"


def test_demographics_unknown_dataset(client):
    r = client.get("/demographics", params={"dataset": "../etc"})
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# /datasets : flag `sampled` — la dédup exacte N'EST PAS un échantillonnage
# --------------------------------------------------------------------------- #
def _descriptor_with_meta(tmp_path, monkeypatch, meta: dict):
    from backend import recluster

    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    monkeypatch.setattr(recluster, "cache_paths",
                        lambda _ds: (tmp_path / "e.npy", tmp_path / "i.jsonl", meta_path))
    return recluster.dataset_descriptor("ds")


_BASE_META = {"label": "Démo", "n_nodes": 198, "n_loaded": 1933, "n_responses": 1933,
              "languages": ["fr"], "lang_counts": {"fr": 198}, "source": "ds"}


def test_dataset_not_sampled_when_only_deduped(tmp_path, monkeypatch):
    """cap/balance absents → couverture complète (198 textes uniques = 1933 voix),
    le front ne doit PAS afficher la note « Échantillon »."""
    meta = {**_BASE_META, "built_with": {"min_chars": 1, "dedup_exact": True,
                                         "balance": None, "cap": None}}
    out = _descriptor_with_meta(tmp_path, monkeypatch, meta)
    assert out["sampled"] is False


def test_dataset_sampled_when_capped(tmp_path, monkeypatch):
    meta = {**_BASE_META, "built_with": {"min_chars": 1, "dedup_exact": True,
                                         "balance": "lang", "cap": 3000}}
    out = _descriptor_with_meta(tmp_path, monkeypatch, meta)
    assert out["sampled"] is True


def test_dataset_sampled_absent_without_built_with(tmp_path, monkeypatch):
    """Vieux meta.json sans built_with → pas de flag (le front garde son heuristique)."""
    out = _descriptor_with_meta(tmp_path, monkeypatch, dict(_BASE_META))
    assert "sampled" not in out
