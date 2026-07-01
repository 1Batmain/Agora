"""Tests d'INTÉGRATION bout-à-bout — l'app FastAPI vue du dehors (TestClient in-process).

Contrairement aux tests de shape (qui figent un endpoint à la fois), on suit ici des
PARCOURS complets, tels qu'un client réel les enchaîne, SERVIS UNIQUEMENT DEPUIS LE
CACHE (aucun build LLM, `AGORA_AUTOBUILD=0` posé par `conftest`) :

  1. **Landing** — `/datasets` peuple le sélecteur (≥5 consultations, hiérarchie
     mère→enfants pour x-stance).
  2. **Exploration d'une consultation PRÊTE** — `/analysis` → on récolte un thème réel,
     puis on frappe `/insights`, `/citations`, `/opinion`, `/avis_list` AVEC CET id :
     c'est le vrai chaînage front (le thème vient de l'analyse, pas d'une constante).
  3. **Posture MODE PUBLIC (fail-CLOSED)** — `AGORA_PUBLIC=1` sans token : les endpoints
     protégés/coûteux renvoient 403 et AUCUN build n'est jamais déclenché (souveraineté
     + zéro extraction LLM en prod publique).

Skip PROPRE, jamais d'échec parasite : le parcours (2) est gardé par `require_ready`
(intersection avec les caches réellement construits) — un dataset non précalculé le
SKIP au lieu d'échouer ou de déclencher un build. Les parcours (1) et (3) ne dépendent
d'aucun précalcul et tournent inconditionnellement.

Lancer :  uv run --extra embed-contender --extra faiss --with fastapi --with pytest pytest -q
"""

from __future__ import annotations

import pytest

from backend import analysis_store, auth, build_manager
from backend.recluster import DEFAULT_DATASET
from ._helpers import (
    EXPECTED_DATASETS,
    analysis_ready,
    available_datasets,
    require_ready,
)


# --------------------------------------------------------------------------- #
# Sélection d'un dataset PRÊT pour le parcours d'exploration bout-à-bout.
# --------------------------------------------------------------------------- #
def _first_ready_dataset(client) -> str:
    """Premier dataset attendu dont l'analyse est précalculée, sinon SKIP propre.

    On privilégie le défaut serveur (`tiktok`) puis l'ordre du brief. Aucun build
    n'est déclenché : `analysis_ready` LIT `/build_status`.
    """
    order = [DEFAULT_DATASET, *[d for d in EXPECTED_DATASETS if d != DEFAULT_DATASET]]
    present = available_datasets()
    for ds in order:
        if ds in present and analysis_ready(client, ds):
            return ds
    pytest.skip(
        "aucun dataset attendu n'a d'analyse précalculée (cache analysis/ absent) — "
        "construis-en une (`POST /build`) pour activer le parcours d'exploration"
    )


# ============================ 1. Landing (toujours actif) ==================== #
def test_landing_datasets(client):
    """`/datasets` peuple le sélecteur (≥5, 4 fermées attendues, x-stance a des enfants)
    — SANS aucun précalcul d'analyse."""
    r = client.get("/datasets")
    assert r.status_code == 200, r.text
    items = r.json()
    assert isinstance(items, list) and len(items) >= 5, f"≥5 consultations attendues, {len(items)}"

    by_id = {d["id"]: d for d in items}
    # Les 4 consultations fermées de référence sont servies depuis leur cache.
    assert set(EXPECTED_DATASETS) <= set(by_id), \
        f"datasets manquants: {set(EXPECTED_DATASETS) - set(by_id)}"

    # Hiérarchie mère→enfants : x-stance expose ses enfants par-topic (build_children).
    xstance = by_id["xstance"]
    assert isinstance(xstance.get("children"), list) and xstance["children"], \
        "x-stance doit exposer une liste `children` non vide"
    # Un enfant n'apparaît PAS dans la liste top-level (servi par id ailleurs).
    assert not (set(xstance["children"]) & set(by_id)), \
        "les enfants de x-stance ne doivent pas remonter en top-level"


# =============== 2. Parcours d'exploration d'une consultation PRÊTE ========== #
def test_explore_ready_consultation_end_to_end(client):
    """Chaîne réelle : `/analysis` → thème réel → `/insights`+`/citations`+`/avis_list`+`/opinion`.

    Le `theme_id` n'est PAS une constante : il est récolté sur la carte des thèmes servie
    par `/analysis`, exactement comme le front l'enchaîne. Skip propre si rien n'est prêt.
    """
    dataset = _first_ready_dataset(client)

    # -- /analysis : la carte des thèmes précalculée --------------------------- #
    r = client.post("/analysis", json={"dataset": dataset})
    assert r.status_code == 200, r.text
    analysis = r.json()
    assert analysis["status"] == "ready"
    themes = analysis["themes"]
    assert isinstance(themes, list) and themes, "carte de thèmes non vide"
    root = next((t for t in themes if t.get("parent_id") is None), None)
    assert root is not None, "au moins un thème racine"
    theme_id = root["id"]

    # -- /insights : synthèse Markdown précalculée (global + ce thème) ---------- #
    r = client.get("/insights", params={"dataset": dataset, "level": "global"})
    assert r.status_code == 200, r.text
    assert isinstance(r.json(), dict) and r.json(), "synthèse globale non vide"

    r = client.get("/insights", params={"dataset": dataset, "level": "theme", "id": theme_id})
    # 200 si la synthèse du thème a été bakée, 404 sinon (thème sans synthèse propre) —
    # jamais 202 (l'analyse EST prête) ni 5xx.
    assert r.status_code in (200, 404), r.text

    # -- /citations : claims du thème, triées par proximité au centroïde -------- #
    r = client.get("/citations", params={"dataset": dataset, "theme_id": theme_id})
    assert r.status_code == 200, r.text
    citations = r.json()
    assert isinstance(citations, list), "citations = liste de claims"
    for c in citations[:5]:
        assert isinstance(c.get("text"), str) and c["text"], "claim verbatim non vide"

    # -- /avis_list : page d'exploration filtrée par ce thème ------------------- #
    r = client.get("/avis_list", params={"dataset": dataset, "theme_id": theme_id, "limit": 5})
    assert r.status_code == 200, r.text
    avis = r.json()
    assert isinstance(avis["total"], int)
    assert isinstance(avis["items"], list) and len(avis["items"]) <= 5
    # Un thème racine a forcément des avis dans son sous-arbre → page non vide.
    assert avis["items"], "le thème racine doit ramener au moins un avis"
    for it in avis["items"]:
        assert {"avis_id", "excerpt", "themes", "text", "claims"} <= set(it)
        assert isinstance(it["text"], str) and isinstance(it["claims"], list)

    # -- /opinion : répartition d'opinion par feuille (artefact À PART) --------- #
    r = client.get("/opinion", params={"dataset": dataset})
    assert r.status_code == 200, r.text
    opinion = r.json()
    assert opinion["dataset"] == dataset
    assert isinstance(opinion["themes"], list)  # vide si non baké → dégrade sans bloquer
    for t in opinion["themes"][:5]:
        assert {"theme_id", "proposition", "pct_favorable"} <= set(t)


def test_avis_detail_matches_list(client):
    """`/avis_list` puis `/avis/{id}` : le détail d'un avis listé est servi et cohérent."""
    dataset = _first_ready_dataset(client)
    r = client.get("/avis_list", params={"dataset": dataset, "limit": 1})
    assert r.status_code == 200, r.text
    items = r.json()["items"]
    if not items:
        pytest.skip(f"aucun avis dans le cache {dataset!r}")
    avis_id = items[0]["avis_id"]

    r = client.get(f"/avis/{avis_id}", params={"dataset": dataset})
    assert r.status_code == 200, r.text
    detail = r.json()
    assert detail["id"] == avis_id
    assert isinstance(detail["text"], str) and detail["text"]
    assert isinstance(detail["claims"], list)


# ================== 3. Mode PUBLIC — fail-CLOSED bout-à-bout ================= #
@pytest.fixture
def public(monkeypatch):
    """`AGORA_PUBLIC=1`, token absent (fail-CLOSED), build NEUTRALISÉ (jamais lancé).

    On monkeypatch les drapeaux lus à chaque requête (au lieu de réimporter l'app) et on
    piège `ensure_build` : tout parcours public qui tenterait un build ferait échouer le
    test — c'est l'invariant « zéro extraction LLM en prod publique ».
    """
    monkeypatch.setattr(auth, "PUBLIC_MODE", True)
    monkeypatch.setattr(auth, "API_TOKEN", None)

    def _no_build(*a, **k):
        raise AssertionError("ensure_build ne doit JAMAIS être appelé en mode public")

    monkeypatch.setattr(build_manager, "ensure_build", _no_build)


def test_public_mode_locks_down_end_to_end(client, public, monkeypatch):
    """Sous `AGORA_PUBLIC=1` sans token : lectures méta OUVERTES, mutations/compute 403,
    read d'un dataset non prêt = 404 SANS build."""
    # Lectures de méta : restent servies (la landing publique fonctionne).
    assert client.get("/health").status_code == 200
    assert client.get("/datasets").status_code == 200

    # Endpoints PROTÉGÉS (mutations citoyennes) : fail-CLOSED → 403.
    assert client.post("/flag", json={"dataset": "tiktok", "target_type": "avis",
                                      "target_id": "a1", "text": "x"}).status_code == 403
    # /submit reste OUVERT en public (participation citoyenne aux consultations OUVERTES) :
    # rate-limité + collecte seule (pas d'embedding/clé). Consultation inconnue → 404, PAS 403.
    assert client.post("/submit", json={"consultation_id": "inconnue",
                                        "text": "bonjour le monde"}).status_code == 404

    # Endpoints de COMPUTE/BUILD : désactivés → 403 (aucun calcul lourd exposé).
    assert client.post("/recluster", json={"dataset": "tiktok"}).status_code == 403
    assert client.get("/density", params={"dataset": "tiktok"}).status_code == 403
    assert client.post("/build", json={"dataset": "tiktok"}).status_code == 403

    # Read d'un dataset NON prêt : 404 propre, et `ensure_build` (piégé) n'est PAS appelé.
    monkeypatch.setattr(analysis_store, "state", lambda ds: analysis_store.ABSENT)
    assert client.post("/analysis", json={"dataset": "tiktok"}).status_code == 404
    assert client.get("/insights", params={"dataset": "tiktok"}).status_code == 404
    assert client.get("/citations", params={"dataset": "tiktok",
                                            "theme_id": "n0"}).status_code == 404


def test_dev_mode_is_open_without_public(client):
    """Contrôle NÉGATIF : hors mode public + sans token (défaut des tests), un protégé
    n'est PAS refusé (403) — la posture ne s'inverse qu'en public."""
    r = client.post("/flag", json={"dataset": "tiktok", "target_type": "avis",
                                   "target_id": "a1", "text": "x"})
    assert r.status_code != 403
