"""Groupe 4 — AUTH (P1) : les endpoints coûteux exigent le token quand il est défini.

`backend.auth.API_TOKEN` est lu À L'IMPORT (env `AGORA_API_TOKEN`). En test, on le
MONKEYPATCH directement (`require_token` lit le global à chaque appel) pour simuler le
mode prod : sans header → 401, avec `X-API-Token`/`Authorization: Bearer` correct → 200.

On NEUTRALISE `build_manager.ensure_build` (le cas 200 sur `/build` ne doit pas spawner
un vrai sous-process de build) — on teste la GRILLE d'auth, pas le pipeline.
"""

from __future__ import annotations

import pytest

from backend import analysis_store, auth, build_manager

TOKEN = "s3cr3t-test-token"


@pytest.fixture
def with_token(monkeypatch):
    """Pose un token API (mode prod) et neutralise le build de fond."""
    monkeypatch.setattr(auth, "API_TOKEN", TOKEN)
    # /build appelle ensure_build : on le rend inerte (READY) pour éviter tout subprocess.
    monkeypatch.setattr(build_manager, "ensure_build", lambda ds, **kw: analysis_store.READY)
    return TOKEN


def test_build_requires_token(client, with_token):
    # Sans header → 401.
    assert client.post("/build", json={"dataset": "tiktok"}).status_code == 401
    # Mauvais token → 401.
    r_bad = client.post("/build", json={"dataset": "tiktok"},
                        headers={"X-API-Token": "wrong"})
    assert r_bad.status_code == 401
    # Bon token (header dédié) → passe l'auth (ensure_build neutralisé → 200).
    r_ok = client.post("/build", json={"dataset": "tiktok"},
                       headers={"X-API-Token": with_token})
    assert r_ok.status_code == 200, r_ok.text


def test_build_accepts_bearer(client, with_token):
    r = client.post("/build", json={"dataset": "tiktok"},
                    headers={"Authorization": f"Bearer {with_token}"})
    assert r.status_code == 200, r.text


def test_flag_requires_token(client, with_token):
    """`POST /flag` est protégé : sans token → 401 (avant tout upsert)."""
    assert client.post("/flag", json={"dataset": "tiktok",
                                       "target_type": "avis",
                                       "target_id": "a1",
                                       "text": "x"}).status_code == 401


def test_open_endpoints_never_require_token(client, with_token):
    """Les lectures restent OUVERTES même token posé (cache only, faible risque)."""
    assert client.get("/health").status_code == 200
    assert client.get("/datasets").status_code == 200
    assert client.get("/build_status", params={"dataset": "tiktok"}).status_code == 200


def test_dev_mode_open_without_token(client):
    """Sans `AGORA_API_TOKEN` (mode dev, défaut des tests), `POST /flag` n'est PAS 401."""
    # API_TOKEN est None par défaut ici → require_token laisse passer. Le statut dépend
    # ensuite de la validation du corps (200 ou 422), mais JAMAIS 401.
    r = client.post("/flag", json={"dataset": "tiktok",
                                   "target_type": "avis",
                                   "target_id": "a1",
                                   "text": "x"})
    assert r.status_code != 401
