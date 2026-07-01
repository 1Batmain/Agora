"""Groupe 3 — WHITELIST sécurité : un id de dataset hors-whitelist ⇒ 404.

`_resolve` garde une whitelist O(1) (`_IDSET`) découverte au démarrage : tout id absent
lève 404 AVANT toute construction de `_Dataset` (aucun accès disque, aucun build). Bloque
notamment le path-traversal (`dataset="../etc"`). Toujours actif (aucun précalcul requis).
"""

from __future__ import annotations

import pytest

# NB : `dataset=""`/`None` est LÉGITIME (→ dataset par défaut, cf. `dataset or DEFAULT`),
# ce n'est pas une tentative de traversal — on ne le teste donc pas ici.
BOGUS = ["../etc", "../../etc/passwd", "..", "tiktok/../secret", "does-not-exist"]


@pytest.mark.parametrize("bogus", BOGUS)
def test_analysis_rejects_unknown_dataset(client, bogus):
    r = client.post("/analysis", json={"dataset": bogus})
    assert r.status_code == 404, f"{bogus!r} aurait dû être rejeté (404), eu {r.status_code}"


@pytest.mark.parametrize("bogus", BOGUS)
def test_read_endpoints_reject_unknown_dataset(client, bogus):
    """La whitelist garde aussi les autres lectures par dataset (cohérence)."""
    assert client.get("/build_status", params={"dataset": bogus}).status_code == 404
    assert client.get("/citations", params={"dataset": bogus, "theme_id": "n0"}).status_code == 404
    assert client.get("/avis/whatever", params={"dataset": bogus}).status_code == 404


def test_path_traversal_does_not_leak(client):
    """Le path-traversal ne doit jamais renvoyer 200 ni un payload servable."""
    r = client.post("/analysis", json={"dataset": "../etc"})
    assert r.status_code == 404
    assert "themes" not in r.json()
