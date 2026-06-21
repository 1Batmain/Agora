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

import numpy as np
from fastapi import FastAPI, HTTPException, Query
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
from backend.synthesize import synthesize
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
