"""Groupe — DURCISSEMENTS : limite de taille de corps (413) + en-têtes de sécurité.

Audit input #2 (limite de corps) et privacy #6 (en-têtes). Ces gardes sont actives en
permanence (indépendantes de tout précalcul ou du mode public).
"""

from __future__ import annotations

from backend.server import MAX_BODY_BYTES


def test_oversize_body_rejected_413(client):
    """Un corps POST au-delà de 64 Ko est rejeté (413) AVANT tout parsing/handler."""
    big = "x" * (MAX_BODY_BYTES + 1024)
    r = client.post("/analysis", json={"dataset": "tiktok", "model": big})
    assert r.status_code == 413, r.status_code
    assert "volumineux" in r.json()["detail"].lower()


def test_normal_body_passes(client):
    """Un corps de taille raisonnable n'est PAS bloqué par la limite."""
    r = client.post("/analysis", json={"dataset": "tiktok"})
    assert r.status_code != 413, r.status_code


def test_security_headers_present(client):
    """Chaque réponse porte les en-têtes de sécurité de base."""
    h = client.get("/health").headers
    assert h.get("x-content-type-options") == "nosniff"
    assert h.get("x-frame-options") == "DENY"
    assert h.get("referrer-policy") == "no-referrer"
    assert "frame-ancestors" in h.get("content-security-policy", "")
