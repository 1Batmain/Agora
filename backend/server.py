"""Serveur FastAPI :8010 — carte spatiale PRÉCALCULÉE, MULTI-DATASET (SERVE-only).

Léger : au démarrage, DÉCOUVRE tous les caches `backend/cache/<dataset>/` et les
charge (vecteurs `.npy` + `ideas.jsonl`), PAS le modèle torch. Le pipeline lourd
(claims→embed→cluster→UMAP→hiérarchie→insights) est PRÉCALCULÉ et PERSISTÉ par
`backend.build_analysis` (en tâche de fond, cf. `build_manager`) ; les endpoints
ne font que LIRE le cache persisté — AUCUN calcul lourd à la requête.

GÉNÉRIQUE : aucun nom de corpus en dur. Rétro-compat : sans `dataset`, tout se
comporte comme sur le défaut (`tiktok`).

Endpoints :
  - GET  /health        → {ok, datasets, default_dataset}
  - GET  /datasets      → [{id, label, n_nodes, languages, source, namings}]
  - POST /analysis      → carte spatiale précalculée (UMAP + arbre adaptatif + edges)
  - GET  /insights      → synthèse Markdown LLM précalculée (global | theme)
  - GET  /citations     → claims d'un thème, triées par proximité au centroïde
  - GET  /avis/{id}     → un avis entier + ses portions verbatim surlignables
  - POST /build         → (re)déclenche le précalcul d'un dataset (non bloquant)
  - GET  /build_status  → état du build d'un dataset (polling front)

Lancer :
    uv run --extra contender uvicorn backend.server:app --host 0.0.0.0 --port 8010
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.recluster import (
    DEFAULT_DATASET,
    DEFAULT_NAMING_METHOD,
    MODEL_ID,
    NAMINGS,
    dataset_descriptor,
    list_datasets,
    load_cache,
)
from backend import analysis_store, build_manager


class _Dataset:
    """Un dataset chargé en mémoire : cache aligné (avis + vecteurs) + descripteur."""

    def __init__(self, dataset_id: str) -> None:
        self.id = dataset_id
        self.ideas, self.vecs, self.weights = load_cache(dataset_id)
        self.descriptor = dataset_descriptor(dataset_id, self.ideas)


# Registre MULTI-DATASET chargé une fois au démarrage (process léger, pas de torch).
_ids = list_datasets()
if not _ids:
    raise RuntimeError(
        "Aucun cache de dataset trouvé sous backend/cache/<dataset>/.\n"
        "Construis-en un : uv run --extra embed-contender "
        "python -m backend.build_cache --dataset tiktok"
    )
DATASETS: dict[str, _Dataset] = {ds: _Dataset(ds) for ds in _ids}
# Défaut rétro-compat : "tiktok" s'il existe, sinon le premier découvert.
DEFAULT = DEFAULT_DATASET if DEFAULT_DATASET in DATASETS else _ids[0]


def _resolve(dataset: str | None) -> _Dataset:
    ds = dataset or DEFAULT
    if ds not in DATASETS:
        raise HTTPException(
            status_code=404,
            detail=f"dataset inconnu: {ds!r} (disponibles: {list(DATASETS)})",
        )
    return DATASETS[ds]


app = FastAPI(title="Agora — carte spatiale précalculée (multi-dataset)", version="2.0")

# CORS permissif en dev (le front passe par un proxy vite mais on couvre l'accès
# direct depuis localhost/forge au cas où).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "model_id": MODEL_ID,
        "default_dataset": DEFAULT,
        "datasets": {
            ds.id: {"n_cached": len(ds.ideas), "dim": int(ds.vecs.shape[1])}
            for ds in DATASETS.values()
        },
    }


@app.get("/datasets")
def datasets() -> list[dict]:
    """Datasets disponibles (caches construits) → de quoi peupler le sélecteur."""
    return [
        {**DATASETS[ds].descriptor,
         "namings": list(NAMINGS), "default_naming": DEFAULT_NAMING_METHOD}
        for ds in DATASETS
    ]


# ===================== Refonte « carte spatiale » (B1–B4) ===================== #
# SÉPARATION BUILD / SERVE. Le pipeline lourd (claims→embed→cluster→UMAP→hiérarchie→
# insights) est PRÉCALCULÉ et PERSISTÉ par `backend.build_analysis` (en tâche de fond,
# cf. `build_manager`). Les trois endpoints du CONTRAT figé (queue/front-redesign.md)
# ne font ici que LIRE le cache persisté — AUCUN calcul lourd à la requête. Si l'analyse
# n'est pas prête, ils déclenchent/poursuivent le build de fond et renvoient un état
# clair `{status: building|absent|error}` (HTTP 202 en cours, 503 en échec).

# Build au démarrage : pour chaque dataset SANS analyse prête, on lance un build de
# fond (non bloquant). Désactivable (tests/dev) via AGORA_AUTOBUILD=0 ; restreignable à
# une liste via AGORA_AUTOBUILD_DATASETS="a,b" (sinon tous).
_AUTOBUILD = (os.environ.get("AGORA_AUTOBUILD", "1").strip().lower()
              not in ("0", "false", "no", ""))
_AUTOBUILD_ONLY = {s.strip() for s in os.environ.get("AGORA_AUTOBUILD_DATASETS", "").split(",")
                   if s.strip()}


@app.on_event("startup")
def _startup_autobuild() -> None:
    if not _AUTOBUILD:
        return
    targets = [ds for ds in DATASETS.values()
               if not _AUTOBUILD_ONLY or ds.id in _AUTOBUILD_ONLY]
    build_manager.ensure_all(targets)


def _not_ready_response(ds, response: Response) -> dict:
    """Réponse SERVE quand l'analyse n'est pas prête : (re)lance le build, renvoie l'état.

    Ne calcule JAMAIS à la requête — délègue au build de fond (`ensure_build`) et renvoie
    la progression. 202 si ça construit/va construire, 503 si le dernier build a échoué.
    """
    build_manager.ensure_build(ds)
    prog = analysis_store.progress(ds.id)
    prog["status"] = analysis_store.state(ds.id)
    prog["dataset"] = ds.id
    response.status_code = 503 if prog["status"] == analysis_store.ERROR else 202
    return prog


class AnalysisBody(BaseModel):
    """Corps de `/analysis` — lecture de la carte spatiale PRÉCALCULÉE d'un dataset.

    Contrat : `{dataset, backend?(api|mac|auto)}`. SERVE-only : `backend`/`model`/
    `embedder`/`resolution` sont acceptés pour compat mais n'influent PAS sur la lecture
    (l'analyse canonique est précalculée côté backend) ; utilise `POST /build` pour
    (re)construire.
    """
    dataset: str | None = None
    backend: str | None = None          # api (défaut) | mac | auto
    model: str | None = None
    embedder: str | None = None
    resolution: float = Field(1.0, gt=0.0)


@app.post("/analysis")
def do_analysis(body: AnalysisBody, response: Response) -> dict:
    """SERVE-only : sert la carte spatiale PRÉCALCULÉE (UMAP + arbre adaptatif + edges).

    Lit `backend/cache/<dataset>/analysis/analysis.json` (instantané). Si l'analyse
    n'est pas prête, déclenche un build de fond et renvoie `{status: building|absent|
    error}` (202/503) — JAMAIS de calcul lourd ici. Le front affiche « Analyse en cours… »
    puis re-sonde jusqu'au résultat réel.
    """
    ds = _resolve(body.dataset)
    if analysis_store.state(ds.id) == analysis_store.READY:
        payload = analysis_store.read_analysis(ds.id)
        if payload is not None:
            payload.setdefault("status", "ready")
            return payload
    return _not_ready_response(ds, response)


@app.get("/insights")
def get_insights(
    response: Response,
    dataset: str | None = Query(None),
    level: str = Query("global"),
    id: str | None = Query(None),
) -> dict:
    """SERVE-only : synthèse Markdown LLM PRÉCALCULÉE, liée au niveau (global | theme).

    Lit `analysis/insights/<…>.json` (instantané). `level=global` → toute la consultation ;
    `level=theme&id=<theme_id>` → un thème. Si l'analyse n'est pas prête → 202 `building`.
    404 si le niveau/thème demandé n'existe pas dans une analyse pourtant prête.
    """
    ds = _resolve(dataset)
    level = (level or "global").strip().lower()
    if level not in ("global", "theme"):
        raise HTTPException(status_code=422, detail=f"level inconnu: {level!r} (global|theme).")
    if level == "theme" and not id:
        raise HTTPException(status_code=422, detail="level='theme' exige un `id` de thème.")
    if analysis_store.state(ds.id) != analysis_store.READY:
        return _not_ready_response(ds, response)
    data = analysis_store.read_insights(ds.id, level, id)
    if data is None:
        raise HTTPException(status_code=404,
                            detail=f"synthèse absente: level={level} id={id!r}.")
    return data


@app.get("/citations")
def get_citations(
    response: Response,
    dataset: str | None = Query(None),
    theme_id: str = Query(...),
) -> list[dict] | dict:
    """SERVE-only : claims d'un thème PRÉCALCULÉES, triées par proximité au centroïde.

    Lit `analysis/citations/<theme_id>.json` (instantané) → `[{text, dist_to_centroid,
    weight}]` (+ `avis_id`/`rank` bonus). Si l'analyse n'est pas prête → 202 `building`.
    404 si le thème est inconnu dans une analyse prête.
    """
    ds = _resolve(dataset)
    if analysis_store.state(ds.id) != analysis_store.READY:
        return _not_ready_response(ds, response)
    data = analysis_store.read_citations(ds.id, theme_id)
    if data is None:
        raise HTTPException(status_code=404,
                            detail=f"thème inconnu: {theme_id!r} (dataset {ds.id!r}).")
    return data


@app.get("/avis/{avis_id}")
def get_avis(
    avis_id: str,
    response: Response,
    dataset: str | None = Query(None),
) -> dict:
    """SERVE-only : un avis EN ENTIER + ses portions verbatim surlignables.

    Lit `analysis/avis.json` (précalculé, instantané) → `{id, text, spans}` où chaque
    span `{start, end, cluster_id, color, theme_label}` est une portion extractive
    (sous-chaîne exacte) colorée à la couleur de son macro-thème (= couleur des bulles).
    Si l'analyse n'est pas prête → 202 `building` ; 404 si l'avis est inconnu.
    """
    ds = _resolve(dataset)
    if analysis_store.state(ds.id) != analysis_store.READY:
        return _not_ready_response(ds, response)
    data = analysis_store.read_avis(ds.id, avis_id)
    if data is None:
        raise HTTPException(status_code=404,
                            detail=f"avis inconnu: {avis_id!r} (dataset {ds.id!r}).")
    return data


class BuildBody(BaseModel):
    """Corps de `POST /build` — (re)déclenche le précalcul d'un dataset.

    `force=true` efface l'analyse persistée avant de reconstruire (sinon no-op si déjà
    prête). Le build tourne EN TÂCHE DE FOND : la réponse est immédiate (202).
    """
    dataset: str | None = None
    force: bool = False


@app.post("/build")
def do_build(body: BuildBody, response: Response) -> dict:
    """Déclenche/relance le build de fond de l'analyse d'un dataset (non bloquant)."""
    ds = _resolve(body.dataset)
    if body.force:
        analysis_store.clear(ds.id)
    state = build_manager.ensure_build(ds)
    response.status_code = 200 if state == analysis_store.READY else 202
    prog = analysis_store.progress(ds.id)
    prog["status"] = analysis_store.state(ds.id)
    prog["dataset"] = ds.id
    return prog


@app.get("/build_status")
def build_status(dataset: str | None = Query(None)) -> dict:
    """État du build d'un dataset (pour le polling front) : status + progression."""
    ds = _resolve(dataset)
    prog = analysis_store.progress(ds.id)
    prog["status"] = analysis_store.state(ds.id)
    prog["dataset"] = ds.id
    prog["building"] = build_manager.is_building(ds.id)
    return prog
