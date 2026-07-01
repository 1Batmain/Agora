"""Sécurité d'exposition du backend (audit prod P1 / SEC1).

Dépendances FastAPI à poser sur les endpoints MUTATIFS / COÛTEUX :

  - ``require_token`` : exige le header ``X-API-Token`` (ou ``Authorization: Bearer``)
    égal à ``AGORA_API_TOKEN``. Comparaison à temps constant (``hmac.compare_digest``).
  - ``rate_limit`` : fenêtre glissante par IP (``AGORA_RATE_LIMIT`` requêtes par
    ``AGORA_RATE_WINDOW`` secondes) → 429 au-delà. Anti-DoS / anti-abus de facture LLM.
  - ``forbid_in_public`` : refuse (403) un endpoint de COMPUTE/BUILD en mode public.

MODE PUBLIC (``AGORA_PUBLIC=1``) — durcissement avant exposition Internet :
  - ``require_token`` devient **FAIL-CLOSED** : sans ``AGORA_API_TOKEN`` configuré, les
    endpoints protégés sont REFUSÉS (403), au lieu de l'ancien « fail-open » (token
    absent → tout ouvert). Corrige CRIT-1 des audits.
  - les endpoints de COMPUTE/BUILD (``/recluster``, ``/density``, ``/build``) sont
    désactivés via ``forbid_in_public`` ; seules les lectures de cache restent servies.

Lecture seule (``/analysis``, ``/avis``, ``/citations``, ``/insights``, ``/datasets``)
reste ouverte (cache only, faible risque) — à placer derrière le reverse-proxy.
"""
from __future__ import annotations

import hmac
import os
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request

API_TOKEN = os.environ.get("AGORA_API_TOKEN") or None
# Mode public : posture fail-CLOSED (cf. docstring). Lu à l'import, surchargeable en test.
PUBLIC_MODE = os.environ.get("AGORA_PUBLIC", "").strip().lower() in ("1", "true", "yes", "on")
_warned = False


def require_token(
    x_api_token: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    """Exige un token valide. Fail-CLOSED en mode public, fail-open en dev seulement.

    - ``AGORA_API_TOKEN`` défini → header obligatoire et correct, sinon 401.
    - token NON défini + mode public → 403 (fail-CLOSED : refus total).
    - token NON défini + mode dev   → laisse passer, avec un avertissement unique.
    """
    global _warned
    if API_TOKEN is None:
        if PUBLIC_MODE:
            raise HTTPException(
                status_code=403,
                detail="Endpoint désactivé (mode public : accès restreint).",
            )
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
    if not token or not hmac.compare_digest(token, API_TOKEN):
        raise HTTPException(status_code=401, detail="Token API invalide ou manquant.")


def forbid_in_public() -> None:
    """Refuse (403) un endpoint de COMPUTE/BUILD quand ``AGORA_PUBLIC`` est actif.

    En mode public, un nœud ne sert QUE des analyses pré-construites : tout calcul lourd
    à la requête (UMAP, Leiden) ou tout déclenchement de build est désactivé.
    """
    if PUBLIC_MODE:
        raise HTTPException(
            status_code=403,
            detail="Endpoint désactivé en mode public.",
        )


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
