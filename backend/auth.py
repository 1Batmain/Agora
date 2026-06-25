"""Sécurité d'exposition du backend (audit prod P1 / SEC1).

Deux dépendances FastAPI à poser sur les endpoints MUTATIFS / COÛTEUX (`/build`,
`/flag`) :

  - ``require_token`` : exige le header ``X-API-Token`` (ou ``Authorization: Bearer``)
    égal à ``AGORA_API_TOKEN``. Si la variable n'est PAS définie → mode DEV ouvert,
    avec un avertissement unique au démarrage. **En prod, définir AGORA_API_TOKEN.**
  - ``rate_limit`` : fenêtre glissante par IP (``AGORA_RATE_LIMIT`` requêtes par
    ``AGORA_RATE_WINDOW`` secondes) → 429 au-delà. Anti-DoS / anti-abus de facture LLM.

Lecture seule (``/analysis``, ``/avis``, ``/citations``, ``/insights``, ``/datasets``)
reste ouverte (cache only, faible risque) — à placer derrière le reverse-proxy.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request

API_TOKEN = os.environ.get("AGORA_API_TOKEN") or None
_warned = False


def require_token(
    x_api_token: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    """Refuse (401) si AGORA_API_TOKEN est défini et le header ne correspond pas."""
    global _warned
    if API_TOKEN is None:
        if not _warned:
            print(
                "[auth] AGORA_API_TOKEN non defini -> endpoints couteux NON proteges "
                "(mode dev). Definir AGORA_API_TOKEN en production."
            )
            _warned = True
        return
    token = x_api_token
    if not token and authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if not token or token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Token API invalide ou manquant.")


_RATE = int(os.environ.get("AGORA_RATE_LIMIT", "30"))        # requêtes
_WINDOW = float(os.environ.get("AGORA_RATE_WINDOW", "60"))   # secondes
_hits: dict[str, deque] = defaultdict(deque)


def rate_limit(request: Request) -> None:
    """Fenêtre glissante par IP : 429 au-delà de _RATE requêtes / _WINDOW s."""
    ip = request.client.host if request.client else "?"
    now = time.monotonic()
    dq = _hits[ip]
    while dq and dq[0] < now - _WINDOW:
        dq.popleft()
    if len(dq) >= _RATE:
        raise HTTPException(
            status_code=429,
            detail="Trop de requêtes — réessayez dans un instant.",
        )
    dq.append(now)
