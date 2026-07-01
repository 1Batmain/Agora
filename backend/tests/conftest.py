"""Fixtures partagées de la suite de régression API (TestClient in-process).

Principe : on lit les VRAIS caches des 4 datasets via `fastapi.testclient.TestClient`,
sans jamais déclencher de calcul lourd.

  * **Aucun build LLM** : `AGORA_AUTOBUILD=0` est posé AVANT l'import de l'app (sinon
    l'event `startup` lancerait un sous-process de build). Les endpoints SERVE ne
    LISENT que le cache persisté.
  * **Skip propre, jamais d'échec parasite** : les tests qui exigent une analyse PRÊTE
    (`/analysis`, `/avis`, `/citations`, `/insights`) sont SKIPPÉS quand le cache
    `analysis/` du dataset n'est pas construit — au lieu d'échouer OU de déclencher un
    build (cf. `require_ready`). Ils s'activent automatiquement dès que le précalcul
    existe.
  * **Tests toujours actifs** : lecture/shape de base (`/health`, `/datasets`,
    `/build_status`, `/flags`), whitelist sécurité (404 path-traversal), auth (401/200)
    et rate-limit (429) ne dépendent d'AUCUN précalcul et tournent inconditionnellement.

Lancer :  uv run --extra embed-contender --extra faiss --with fastapi --with pytest pytest -q
"""

from __future__ import annotations

import os

# DOIVENT précéder tout import de `backend.server`/`backend.auth` (lus à l'import) :
#  - autobuild OFF : pas de build de fond au démarrage du TestClient ;
#  - token absent : endpoints protégés ouverts par défaut (le test d'auth le pose lui-même) ;
#  - quota large : on ne veut pas de 429 parasite (le test de rate-limit l'abaisse lui-même).
os.environ["AGORA_AUTOBUILD"] = "0"
os.environ.pop("AGORA_API_TOKEN", None)
os.environ["AGORA_RATE_LIMIT"] = "100000"

import pytest
from fastapi.testclient import TestClient

from backend import auth
from backend.server import app


@pytest.fixture(scope="session")
def client():
    """TestClient in-process partagé (déclenche startup/shutdown ; autobuild OFF)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_rate_limit():
    """Vide la fenêtre glissante par IP entre chaque test.

    Tous les tests partagent l'IP « testclient » ; sans ce reset, les appels aux
    endpoints PROTÉGÉS (comptés par `rate_limit`) s'accumuleraient et finiraient par
    déclencher des 429 parasites dans des tests qui n'ont rien à voir avec le quota.
    """
    auth._hits.clear()
    yield
    auth._hits.clear()
