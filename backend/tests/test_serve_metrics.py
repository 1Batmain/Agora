"""Enrichissement SERVE-TIME des indices — couverture + fidélité verbatim.

Teste la LOGIQUE PURE de `backend.serve_metrics` (dérivée de l'arbre caché + de
`claims.json`), SANS aucun build : payloads synthétiques + `claims.json` temporaire.
Garantit la forme M5 `{key, value, detail}`, l'idempotence, et le comptage verbatim.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from backend import serve_metrics
from backend.claims_endpoint import CLAIMS_NAME


def _payload(themes, indices=None):
    return {
        "themes": themes,
        "dataset_stats": {"totals": {}, "indices": list(indices or [])},
    }


def test_coverage_excludes_noise_and_uses_total_ideas():
    """couverture = avis des thèmes RÉELS / TOUTES les contributions ingérées."""
    themes = [
        {"id": "n0", "parent_id": None, "n_avis": 60},
        {"id": "n1", "parent_id": None, "n_avis": 20},
        {"id": "__noise__", "parent_id": None, "n_avis": 5},
        {"id": "n0a", "parent_id": "n0", "n_avis": 30},  # enfant : ignoré (pas racine)
    ]
    cov = serve_metrics.coverage_index(_payload(themes), total_ideas=100)
    assert cov["key"] == "couverture"
    assert set(cov) == {"key", "value", "detail"}
    # 80 avis classés (hors __noise__) / 100 contributions = 0.8.
    assert cov["value"] == 0.8
    assert cov["detail"] == {"classes": 80, "noise": 20, "total": 100}


def test_coverage_fallback_without_total():
    """Sans `total_ideas`, dénominateur = themed + noise des thèmes racine."""
    themes = [
        {"id": "n0", "parent_id": None, "n_avis": 8},
        {"id": "__noise__", "parent_id": None, "n_avis": 2},
    ]
    cov = serve_metrics.coverage_index(_payload(themes), total_ideas=0)
    assert cov["value"] == 0.8
    assert cov["detail"]["total"] == 10


def test_fidelity_counts_exact_substrings(tmp_path, monkeypatch):
    """fidélité = part des claims dont les spans (+ cible) sont sous-chaînes exactes."""
    monkeypatch.setattr(serve_metrics, "CACHE_DIR", tmp_path)
    text = "J'aime les chats et les chiens du quartier."
    ideas = [SimpleNamespace(id="a1", text_clean=text, text=text, weight=1.0)]
    claims = {
        "model": "test",
        "claims": {
            "a1": [
                {"text": "J'aime les chats", "spans": [[0, 16]], "target": None},  # verbatim
                {"text": "blah", "spans": [[0, 4]], "target": None},               # NON
            ]
        },
    }
    ddir = tmp_path / "demo"
    ddir.mkdir()
    (ddir / CLAIMS_NAME).write_text(json.dumps(claims, ensure_ascii=False), encoding="utf-8")

    fid = serve_metrics.fidelity_index("demo", ideas)
    assert fid["key"] == "fidelite_verbatim"
    assert set(fid) == {"key", "value", "detail"}
    assert fid["value"] == 0.5
    assert fid["detail"] == {"n_claims": 2, "n_verbatim": 1}


def test_fidelity_absent_when_no_claims_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(serve_metrics, "CACHE_DIR", tmp_path)
    assert serve_metrics.fidelity_index("missing", []) is None


def test_enrich_indices_appends_and_is_idempotent(tmp_path, monkeypatch):
    """enrich ajoute couverture (+ fidélité si dispo) et remplace les doublons de clé."""
    monkeypatch.setattr(serve_metrics, "CACHE_DIR", tmp_path)  # pas de claims.json → couverture seule
    themes = [{"id": "n0", "parent_id": None, "n_avis": 9}]
    base = [{"key": "effusion", "value": 0.5, "detail": {}}]
    ideas = [SimpleNamespace(id=str(i), text_clean="x" * 20, weight=1.0) for i in range(10)]
    payload = _payload(themes, base)

    out = serve_metrics.enrich_indices(payload, "demo", ideas)
    keys = [ix["key"] for ix in out["dataset_stats"]["indices"]]
    assert "effusion" in keys and "couverture" in keys
    cov = next(ix for ix in out["dataset_stats"]["indices"] if ix["key"] == "couverture")
    assert cov["value"] == 0.9  # 9 classés / 10 idées

    # Idempotent : un 2e passage ne duplique pas la clé couverture.
    out2 = serve_metrics.enrich_indices(out, "demo", ideas)
    keys2 = [ix["key"] for ix in out2["dataset_stats"]["indices"]]
    assert keys2.count("couverture") == 1


def test_enrich_noop_on_malformed_payload():
    assert serve_metrics.enrich_indices({}, "demo", []) == {}
    assert serve_metrics.enrich_indices({"dataset_stats": 3}, "demo", []) == {"dataset_stats": 3}
