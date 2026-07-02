"""Groupe — MODE PUBLIC (AGORA_PUBLIC) : fail-CLOSED + compute/build désactivés.

En mode public, la posture s'inverse : sans `AGORA_API_TOKEN` configuré, les endpoints
protégés sont REFUSÉS (403) au lieu d'être ouverts ; les endpoints de COMPUTE/BUILD
(`/recluster`, `/density`, `/build`) sont désactivés (403) ; et un read sur un dataset
NON prêt renvoie 404 SANS jamais déclencher de build (zéro extraction LLM).

`auth.PUBLIC_MODE` est lu à chaque requête (dépendances `forbid_in_public` /
`require_token`, et `_not_ready_response`) → on le MONKEYPATCH pour simuler le mode public
sans réimporter l'app.
"""

from __future__ import annotations

import pytest

from backend import analysis_store, auth, build_manager


@pytest.fixture
def public(monkeypatch):
    """Active le mode public, token absent (fail-CLOSED), build neutralisé (jamais lancé)."""
    monkeypatch.setattr(auth, "PUBLIC_MODE", True)
    monkeypatch.setattr(auth, "API_TOKEN", None)

    def _boom(*a, **k):  # un build NE doit jamais être déclenché en mode public
        raise AssertionError("ensure_build ne doit pas être appelé en mode public")

    monkeypatch.setattr(build_manager, "ensure_build", _boom)


def test_protected_endpoints_fail_closed(client, public):
    """Sans token, en public, les endpoints PROTÉGÉS renvoient 403 (et non 200/ouvert)."""
    assert client.post("/flag", json={"dataset": "tiktok", "target_type": "avis",
                                      "target_id": "a1", "text": "x"}).status_code == 403
    # /submit est OUVERT en public (participation citoyenne, mode COLLECTE sans embedding) :
    # consultation inconnue → 404 (validation), PAS 403. Voir test_integration (parcours).
    assert client.post("/submit", json={"consultation_id": "inconnue",
                                        "text": "bonjour le monde"}).status_code == 404


def test_compute_endpoints_disabled(client, public):
    """Les endpoints de COMPUTE/BUILD sont désactivés (403) en mode public."""
    assert client.post("/recluster", json={"dataset": "tiktok"}).status_code == 403
    assert client.get("/density", params={"dataset": "tiktok"}).status_code == 403
    assert client.post("/build", json={"dataset": "tiktok"}).status_code == 403


def test_build_disabled_even_with_token(client, public, monkeypatch):
    """`/build` reste désactivé en public MÊME avec un token valide (forbid_in_public)."""
    monkeypatch.setattr(auth, "API_TOKEN", "tok")
    r = client.post("/build", json={"dataset": "tiktok"}, headers={"X-API-Token": "tok"})
    assert r.status_code == 403, r.text


def test_unbuilt_dataset_returns_404_without_build(client, public, monkeypatch):
    """Un read sur un dataset NON prêt → 404, SANS déclencher de build (ensure_build boom)."""
    monkeypatch.setattr(analysis_store, "state", lambda ds: analysis_store.ABSENT)
    assert client.post("/analysis", json={"dataset": "tiktok"}).status_code == 404
    assert client.get("/insights", params={"dataset": "tiktok"}).status_code == 404
    assert client.get("/citations", params={"dataset": "tiktok",
                                            "theme_id": "n0"}).status_code == 404
    assert client.get("/avis/whatever", params={"dataset": "tiktok"}).status_code == 404


def test_cache_reads_stay_open(client, public):
    """Les lectures de méta (cache) restent servies en public (sans token)."""
    assert client.get("/health").status_code == 200
    assert client.get("/datasets").status_code == 200


def test_dev_mode_still_open_without_public(client):
    """Hors public + sans token (défaut des tests), les protégés ne sont PAS 403."""
    r = client.post("/flag", json={"dataset": "tiktok", "target_type": "avis",
                                   "target_id": "a1", "text": "x"})
    assert r.status_code != 403
