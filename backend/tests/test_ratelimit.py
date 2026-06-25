"""Groupe 5 — RATE-LIMIT : au-delà du quota, un endpoint protégé renvoie 429.

`rate_limit` (fenêtre glissante par IP) lit le global `_RATE` à chaque appel → on
l'ABAISSE par monkeypatch pour ne pas avoir à envoyer des milliers de requêtes. On
frappe `/explain` SANS params : la dépendance `rate_limit` s'exécute AVANT le handler,
donc sous le quota on obtient 422 (params manquants) et au-delà 429 — sans aucun
travail lourd. La fenêtre est vidée entre tests par l'autouse `_reset_rate_limit`.
"""

from __future__ import annotations

from backend import auth

PROTECTED_GET = "/explain"
PARAMS = {"dataset": "tiktok"}          # pas de cluster/pair → 422 tant qu'on passe le quota


def test_rate_limit_trips_429(client, monkeypatch):
    monkeypatch.setattr(auth, "_RATE", 3)
    auth._hits.clear()
    codes = [client.get(PROTECTED_GET, params=PARAMS).status_code for _ in range(4)]
    # Les 3 premières passent le quota (handler → 422 params manquants), la 4ᵉ est bloquée.
    assert codes[:3] == [422, 422, 422], codes
    assert codes[3] == 429, codes
    # Le corps 429 porte bien le message anti-abus.
    blocked = client.get(PROTECTED_GET, params=PARAMS)
    assert blocked.status_code == 429
    assert "requêtes" in blocked.json()["detail"].lower()


def test_rate_limit_resets_after_clear(client, monkeypatch):
    """Vider la fenêtre (comme le ferait l'écoulement du temps) ré-ouvre l'accès."""
    monkeypatch.setattr(auth, "_RATE", 1)
    auth._hits.clear()
    assert client.get(PROTECTED_GET, params=PARAMS).status_code == 422   # 1ʳᵉ passe
    assert client.get(PROTECTED_GET, params=PARAMS).status_code == 429   # 2ᵉ bloquée
    auth._hits.clear()                                                    # « le temps passe »
    assert client.get(PROTECTED_GET, params=PARAMS).status_code == 422   # ré-ouvert


def test_open_endpoint_not_rate_limited(client, monkeypatch):
    """Les lectures ouvertes (sans dépendance `rate_limit`) ne sont jamais 429."""
    monkeypatch.setattr(auth, "_RATE", 1)
    auth._hits.clear()
    codes = {client.get("/health").status_code for _ in range(5)}
    assert codes == {200}
