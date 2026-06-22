"""Serveur FastAPI :8010 — re-clustering LIVE, MULTI-DATASET (embeddings cachés).

Léger : au démarrage, DÉCOUVRE tous les caches `backend/cache/<dataset>/` et les
charge (vecteurs `.npy` + `ideas.jsonl`), PAS le modèle torch. Le front (`:5180`,
proxy vite `/api`) choisit un dataset + bouge les knobs → `POST /recluster` →
GraphPayload hiérarchique recalculé en ~1–3 s.

GÉNÉRIQUE : aucun nom de corpus en dur. Les défauts des knobs sont DÉRIVÉS des
données de CHAQUE dataset (pas des magic-numbers TikTok). Rétro-compat : sans
`dataset`, tout se comporte comme avant sur le défaut (`tiktok`).

Endpoints :
  - GET  /health    → {ok, datasets, default_dataset}
  - GET  /datasets  → [{id, label, n_nodes, languages, source}]
  - GET  /params    → table des knobs (?dataset=…, défaut tiktok)
  - POST /recluster → GraphPayload hiérarchique + meta.stats (body.dataset)
  - POST /synthesize→ rapport Markdown (synthèse + pertinence) via Mistral

Lancer :
    uv run --extra contender uvicorn backend.server:app --host 0.0.0.0 --port 8010
"""

from __future__ import annotations

import os

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.recluster import (
    DEFAULT_DATASET,
    DEFAULT_METHOD,
    DEFAULT_NAMING_METHOD,
    METHODS,
    MODEL_ID,
    NAMINGS,
    SEED,
    dataset_descriptor,
    list_datasets,
    load_cache,
    recluster,
)
from backend.claims_endpoint import (
    DEFAULT_MIN_CHARS as CLAIMS_MIN_CHARS,
    OllamaUnavailable,
    claims_payload,
)
from backend import analysis_store, build_manager
from backend.synthesize import synthesize
from pipeline.claims.pipeline import DEFAULT_EMBEDDER
from pipeline.cluster.naming_methods import MISTRAL_MODEL
from pipeline.cluster.adaptive import EDGE_SIGMA, derive_defaults
from pipeline.cluster.dedup import dedup_near
from pipeline.cluster.hdbscan_contender import N_COMPONENTS, derive_hdbscan_defaults

# Filtres par défaut (mêmes que recluster) pour DÉRIVER les défauts data-driven.
_DEFAULT_DEDUP = 0.95
_DEFAULT_MIN_CHARS = 12

# Options de NOMMAGE switchable (orthogonales à method/dataset). Le front bâtit
# son sélecteur depuis cette liste — aucune valeur de corpus en dur.
NAMING_OPTIONS = [
    {"name": "ctfidf", "label": "c-TF-IDF",
     "help": "mots-clés distinctifs dérivés du corpus (défaut, déterministe)"},
    {"name": "centroid", "label": "Centroïde",
     "help": "verbatim citoyen le plus représentatif du cluster"},
    {"name": "llm", "label": "LLM (Mistral)",
     "help": f"titre court généré via l'API Mistral ({MISTRAL_MODEL}) ; repli c-TF-IDF si indisponible"},
]


def _derive_startup_defaults(ideas, vecs, weights):
    """Dérive k/seuil/min_sub/dup sur un cache APRÈS les filtres par défaut.

    Les défauts des sliders ne sont donc PAS des magic-numbers : ils sont la
    valeur que produit la dérivation sur CE cache (corpus/modèle). Chaque dataset
    a SES propres défauts, calculés ici (audit #6/#7/#9).
    """
    keep = [i for i, idea in enumerate(ideas)
            if len((idea.text_clean or idea.text).strip()) >= _DEFAULT_MIN_CHARS]
    v = np.ascontiguousarray(vecs[keep])
    w = weights[keep]
    dd = dedup_near(v, w, threshold=_DEFAULT_DEDUP)
    v = np.ascontiguousarray(v[dd.keep])
    return derive_defaults(v)


def _build_hdbscan_knobs(hd) -> list[dict]:
    """Knobs de la méthode HDBSCAN — défauts DÉRIVÉS de N (zéro magic-number).

    `n_components=5` est FIXE (contrat) donc N'EST PAS un knob. `min_cluster_size`
    et `min_samples` ∝ N (cf. min_sub_size) ; `umap_n_neighbors` ∝ log N (cf. k).
    Bornes = suggestions de slider, pas une validation dure.
    """
    return [
        {"name": "dedup", "label": "Dédup (cosine)", "default": _DEFAULT_DEDUP,
         "min": 0.80, "max": 1.0, "step": 0.01, "help": "fusion near-dups",
         "derived": False},
        {"name": "min_chars", "label": "Min. caractères", "default": _DEFAULT_MIN_CHARS,
         "min": 0, "max": 200, "step": 1, "help": "filtre avis courts",
         "derived": False},
        {"name": "min_cluster_size", "label": "Taille mini cluster", "default": hd.min_cluster_size,
         "min": 2, "max": 200, "step": 1, "help": "plus grand → moins de clusters (défaut ∝ N)",
         "derived": True},
        {"name": "min_samples", "label": "min_samples", "default": hd.min_samples,
         "min": 1, "max": 200, "step": 1, "help": "plus grand → plus de bruit (défaut = min_cluster_size)",
         "derived": True},
        {"name": "umap_n_neighbors", "label": "UMAP voisins", "default": hd.umap_n_neighbors,
         "min": 2, "max": 100, "step": 1, "help": "voisinage UMAP (défaut ∝ log N)",
         "derived": True},
    ]


def _build_knobs(derived) -> list[dict]:
    """Table des knobs pour un dataset (défauts DÉRIVÉS ou réglages d'usage).

    Bornes ÉLARGIES aux limites physiques pour ne JAMAIS rejeter (422) une valeur
    légitime sur un autre modèle/corpus (audit #8) ; `min/max` = suggestions de
    slider, pas une validation dure.
    """
    return [
        {"name": "dedup", "label": "Dédup (cosine)", "default": _DEFAULT_DEDUP,
         "min": 0.80, "max": 1.0, "step": 0.01, "help": "fusion near-dups",
         "derived": False},
        {"name": "min_chars", "label": "Min. caractères", "default": _DEFAULT_MIN_CHARS,
         "min": 0, "max": 200, "step": 1, "help": "filtre avis courts",
         "derived": False},
        {"name": "k", "label": "k voisins", "default": derived.k,
         "min": 2, "max": 100, "step": 1, "help": "densité k-NN (défaut ∝ log N)",
         "derived": True},
        {"name": "threshold", "label": "Seuil arêtes (cosine)", "default": round(derived.threshold, 4),
         "min": 0.0, "max": 0.999, "step": 0.01, "help": "coupe les arêtes (défaut μ−σ·k)",
         "derived": True},
        {"name": "resolution_macro", "label": "Résolution macros", "default": 1.0,
         "min": 0.05, "max": 10.0, "step": 0.1, "help": "granularité macros",
         "derived": False},
        {"name": "resolution_sub", "label": "Résolution sous-thèmes", "default": 1.5,
         "min": 0.05, "max": 10.0, "step": 0.1, "help": "granularité sous-thèmes",
         "derived": False},
        {"name": "min_sub_size", "label": "Taille mini sous-thème", "default": derived.min_sub_size,
         "min": 1, "max": 1000, "step": 1, "help": "fusion des miettes (défaut frac·N)",
         "derived": True},
    ]


class _Dataset:
    """Un dataset chargé en mémoire : cache aligné + défauts dérivés + knobs."""

    def __init__(self, dataset_id: str) -> None:
        self.id = dataset_id
        self.ideas, self.vecs, self.weights = load_cache(dataset_id)
        self.derived = _derive_startup_defaults(self.ideas, self.vecs, self.weights)
        self.hdbscan_derived = derive_hdbscan_defaults(self.derived.n)
        # Knobs PAR MÉTHODE : le front affiche les bons selon `method`.
        self.knobs_by_method = {
            "leiden": _build_knobs(self.derived),
            "hdbscan": _build_hdbscan_knobs(self.hdbscan_derived),
        }
        self.defaults_by_method = {
            m: {k["name"]: k["default"] for k in knobs}
            for m, knobs in self.knobs_by_method.items()
        }
        # Rétro-compat : `knobs`/`defaults` = méthode par défaut (leiden).
        self.knobs = self.knobs_by_method[DEFAULT_METHOD]
        self.defaults = self.defaults_by_method[DEFAULT_METHOD]
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


class ReclusterBody(BaseModel):
    """Corps de /recluster — tous les knobs sont optionnels.

    `dataset` (défaut `"tiktok"`, rétro-compat) sélectionne le cache. Bornes
    (`ge`/`le`) = limites PHYSIQUES seulement (audit #8). `threshold`/`k`/
    `min_sub_size`/`dup_threshold` à ``None`` ⇒ **dérivés** des données.
    """
    dataset: str | None = None
    method: str | None = None
    naming: str | None = None
    dedup: float | None = Field(_DEFAULT_DEDUP, ge=0.0, le=1.0)
    min_chars: int = Field(_DEFAULT_MIN_CHARS, ge=0)
    # Leiden
    k: int | None = Field(None, ge=2)
    threshold: float | None = Field(None, ge=0.0, le=1.0)
    resolution_macro: float = Field(1.0, gt=0.0)
    resolution_sub: float = Field(1.5, gt=0.0)
    min_sub_size: int | None = Field(None, ge=1)
    dup_threshold: float | None = Field(None, ge=0.0, le=1.0)
    # HDBSCAN
    min_cluster_size: int | None = Field(None, ge=2)
    min_samples: int | None = Field(None, ge=1)
    umap_n_neighbors: int | None = Field(None, ge=2)


app = FastAPI(title="Agora — recluster live (multi-dataset)", version="2.0")

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


@app.get("/params")
def params(dataset: str | None = Query(None),
           method: str | None = Query(None)) -> dict:
    # `derived` expose la PROVENANCE des défauts data-driven du DATASET demandé.
    # `method` (défaut leiden) choisit la table de knobs renvoyée dans `knobs`.
    ds = _resolve(dataset)
    meth = (method or DEFAULT_METHOD).lower()
    if meth not in METHODS:
        raise HTTPException(
            status_code=404,
            detail=f"méthode inconnue: {meth!r} (disponibles: {list(METHODS)})",
        )
    d = ds.derived
    hd = ds.hdbscan_derived
    return {
        "dataset": ds.id,
        "method": meth,
        "methods": list(METHODS),
        "default_method": DEFAULT_METHOD,
        "namings": NAMING_OPTIONS,
        "naming_methods": list(NAMINGS),
        "default_naming": DEFAULT_NAMING_METHOD,
        "llm_model": MISTRAL_MODEL,
        "knobs": ds.knobs_by_method[meth],
        "defaults": ds.defaults_by_method[meth],
        "knobs_by_method": ds.knobs_by_method,
        "seed": SEED,
        "hdbscan_derived": {
            "min_cluster_size": hd.min_cluster_size,
            "min_samples": hd.min_samples,
            "umap_n_neighbors": hd.umap_n_neighbors,
            "n_components": N_COMPONENTS,
            "n": hd.n,
        },
        "derived": {
            "k": d.k,
            "threshold": round(d.threshold, 4),
            "min_sub_size": d.min_sub_size,
            "dup_threshold": round(d.dup_threshold, 4),
            "knn_sim_mean": d.pool_mean,
            "knn_sim_std": d.pool_std,
            "edge_sigma": EDGE_SIGMA,
            "note": "défauts dérivés sur le cache après dedup/min_chars par défaut",
        },
    }


@app.post("/recluster")
def do_recluster(body: ReclusterBody) -> dict:
    ds = _resolve(body.dataset)
    meth = (body.method or DEFAULT_METHOD).lower()
    if meth not in METHODS:
        raise HTTPException(
            status_code=404,
            detail=f"méthode inconnue: {meth!r} (disponibles: {list(METHODS)})",
        )
    naming = (body.naming or DEFAULT_NAMING_METHOD).lower()
    if naming not in NAMINGS:
        raise HTTPException(
            status_code=404,
            detail=f"nommage inconnu: {naming!r} (disponibles: {list(NAMINGS)})",
        )
    return recluster(
        ds.ideas, ds.vecs, ds.weights,
        method=meth,
        naming=naming,
        dedup=body.dedup,
        min_chars=body.min_chars,
        k=body.k,
        threshold=body.threshold,
        resolution_macro=body.resolution_macro,
        resolution_sub=body.resolution_sub,
        min_sub_size=body.min_sub_size,
        dup_threshold=body.dup_threshold,
        min_cluster_size=body.min_cluster_size,
        min_samples=body.min_samples,
        umap_n_neighbors=body.umap_n_neighbors,
        dataset=ds.id,
    )


class ClaimsBody(BaseModel):
    """Corps de `/claims` — thèmes ÉMERGENTS (pipeline ouvert avis→claims→cluster).

    `dataset` sélectionne le cache (défaut rétro-compat). `resolution` règle la
    granularité Leiden (rejouable sans ré-extraire). `model`/`embedder` sont
    optionnels (défauts souverains) ; changer de `model` invalide le cache claims.
    """
    dataset: str | None = None
    resolution: float = Field(1.0, gt=0.0)
    backend: str | None = None      # api (défaut) | mac | auto ; sinon AGORA_CLAIMS_BACKEND
    model: str | None = None
    embedder: str | None = None
    min_chars: int = Field(CLAIMS_MIN_CHARS, ge=0)


@app.post("/claims")
def do_claims(body: ClaimsBody) -> dict:
    """Carte des thèmes émergents : extraction LLM cachée → embed caché → clustering.

    1er run : extrait les claims via le backend (`api` API Mistral par DÉFAUT, `mac`
    Ollama souverain, `auto` Mac→repli API) puis embed — lent. Runs suivants (même
    dataset/modèle), y compris autre `resolution` : rejoue le clustering depuis le cache,
    SANS ré-extraire. La réponse expose `meta.backend`/`meta.sovereign`. 503 si une
    extraction est nécessaire mais le backend est inutilisable ; 500 sinon.
    """
    ds = _resolve(body.dataset)
    try:
        return claims_payload(
            ds,
            resolution=body.resolution,
            backend=body.backend,            # None → AGORA_CLAIMS_BACKEND (défaut api)
            model=body.model,                # None → modèle par défaut du backend
            embedder=body.embedder or DEFAULT_EMBEDDER,
            min_chars=body.min_chars,
        )
    except OllamaUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


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


class SynthesizeBody(BaseModel):
    """Corps de /synthesize — sélectionne le dataset + la vue à synthétiser.

    `method`/`naming` (mêmes valeurs que /recluster) déterminent le découpage et
    les titres résumés. Le reste du clustering utilise les défauts dérivés.
    """
    dataset: str | None = None
    method: str | None = None
    naming: str | None = None


@app.post("/synthesize")
def do_synthesize(body: SynthesizeBody) -> dict:
    """Rapport Markdown (synthèse + pertinence des clusters) via Mistral.

    Repli gracieux côté `synthesize` : sans clé Mistral, renvoie un rapport
    « indisponible » avec `meta.fallback=True` — pas une erreur HTTP.
    """
    ds = _resolve(body.dataset)
    meth = (body.method or DEFAULT_METHOD).lower()
    if meth not in METHODS:
        raise HTTPException(
            status_code=404,
            detail=f"méthode inconnue: {meth!r} (disponibles: {list(METHODS)})",
        )
    naming = (body.naming or DEFAULT_NAMING_METHOD).lower()
    if naming not in NAMINGS:
        raise HTTPException(
            status_code=404,
            detail=f"nommage inconnu: {naming!r} (disponibles: {list(NAMINGS)})",
        )
    return synthesize(
        ds.ideas, ds.vecs, ds.weights,
        dataset=ds.id, method=meth, naming=naming,
        languages=ds.descriptor.get("languages"),
    )
