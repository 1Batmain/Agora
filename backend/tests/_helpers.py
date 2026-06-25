"""Helpers de gating partagés par la suite (lecture du cache RÉEL, sans build).

`require_ready` est la garde centrale : elle SKIP un test (sans jamais déclencher de
build) quand le dataset n'est pas dans la whitelist ou que son analyse n'est pas
précalculée. `/build_status` LIT le status persisté — il n'appelle PAS `ensure_build`,
contrairement à `/analysis`/`/avis`/`/citations` sur un dataset froid.
"""

from __future__ import annotations

import pytest

from backend.recluster import CACHE_DIR, list_datasets

# Les 4 datasets attendus dans le cache (ordre du brief). On teste l'INTERSECTION avec
# ce qui est réellement construit : un dataset absent du cache → skip ciblé.
EXPECTED_DATASETS = ("tiktok", "granddebat", "xstance", "republique-numerique")


def available_datasets() -> set[str]:
    """Ids réellement présents dans le cache (`embeddings.npy` + `ideas.jsonl`)."""
    return set(list_datasets())


def analysis_ready(client, dataset: str) -> bool:
    """True si l'analyse du dataset est PRÊTE — via `/build_status` (ne build jamais)."""
    r = client.get("/build_status", params={"dataset": dataset})
    return r.status_code == 200 and r.json().get("status") == "ready"


def require_ready(client, dataset: str) -> None:
    """Skip si le dataset est absent du cache ou si son analyse n'est pas construite."""
    if dataset not in available_datasets():
        pytest.skip(f"dataset {dataset!r} absent du cache backend/cache/")
    if not analysis_ready(client, dataset):
        pytest.skip(
            f"analyse {dataset!r} non précalculée (cache analysis/ absent) — "
            f"construis-la (`POST /build`) pour activer ce test"
        )


def has_claims(dataset: str) -> bool:
    """True si les claims extraits sont cachés (prérequis de `/sandbox`/`/explain`)."""
    return (CACHE_DIR / dataset / "claims.json").exists()
