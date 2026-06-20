"""Serveur FastAPI :8010 — re-clustering LIVE des avis (embeddings en cache).

Léger : charge `backend/cache/embeddings.npy` (+ `ideas.jsonl`) au démarrage,
PAS le modèle torch. Le front (`:5180`, via proxy vite `/api`) bouge les knobs →
`POST /recluster` → GraphPayload hiérarchique recalculé en ~1–3 s.

Endpoints :
  - GET  /health    → {ok, n_cached, model_id, dim}
  - GET  /params    → table des knobs (défaut + min/max + step)
  - POST /recluster → GraphPayload hiérarchique + meta.stats

Lancer :
    uv run --extra contender uvicorn backend.server:app --host 0.0.0.0 --port 8010
"""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.recluster import MODEL_ID, SEED, load_cache, recluster
from pipeline.cluster.adaptive import EDGE_SIGMA, derive_defaults
from pipeline.cluster.dedup import dedup_near

# Cache chargé une fois au démarrage (process léger, pas de torch).
IDEAS, VECS, WEIGHTS = load_cache()

# Filtres par défaut (mêmes que recluster) pour DÉRIVER les défauts data-driven.
_DEFAULT_DEDUP = 0.95
_DEFAULT_MIN_CHARS = 12


def _derive_startup_defaults():
    """Dérive k/seuil/min_sub/dup sur le cache APRÈS les filtres par défaut.

    Les défauts des sliders ne sont donc PAS des magic-numbers TikTok mais la
    valeur que produit la dérivation sur CE corpus/modèle (audit #6/#7/#9). Sur
    un autre cache, /params renverrait d'autres défauts, automatiquement.
    """
    keep = [i for i, idea in enumerate(IDEAS)
            if len((idea.text_clean or idea.text).strip()) >= _DEFAULT_MIN_CHARS]
    v = np.ascontiguousarray(VECS[keep])
    w = WEIGHTS[keep]
    dd = dedup_near(v, w, threshold=_DEFAULT_DEDUP)
    v = np.ascontiguousarray(v[dd.keep])
    return derive_defaults(v)


DERIVED = _derive_startup_defaults()

# Knobs : défauts DÉRIVÉS (k/seuil/min_sub) ou réglages d'usage (dedup/min_chars/
# résolutions). Bornes ÉLARGIES aux limites physiques pour ne JAMAIS rejeter (422)
# une valeur légitime sur un autre modèle/corpus (audit #8) ; les `min/max` ne sont
# plus que des suggestions de slider, pas une validation dure.
KNOBS = [
    {"name": "dedup", "label": "Dédup (cosine)", "default": _DEFAULT_DEDUP,
     "min": 0.80, "max": 1.0, "step": 0.01, "help": "fusion near-dups",
     "derived": False},
    {"name": "min_chars", "label": "Min. caractères", "default": _DEFAULT_MIN_CHARS,
     "min": 0, "max": 200, "step": 1, "help": "filtre avis courts",
     "derived": False},
    {"name": "k", "label": "k voisins", "default": DERIVED.k,
     "min": 2, "max": 100, "step": 1, "help": "densité k-NN (défaut ∝ log N)",
     "derived": True},
    {"name": "threshold", "label": "Seuil arêtes (cosine)", "default": round(DERIVED.threshold, 4),
     "min": 0.0, "max": 0.999, "step": 0.01, "help": "coupe les arêtes (défaut μ−σ·k)",
     "derived": True},
    {"name": "resolution_macro", "label": "Résolution macros", "default": 1.0,
     "min": 0.05, "max": 10.0, "step": 0.1, "help": "granularité macros",
     "derived": False},
    {"name": "resolution_sub", "label": "Résolution sous-thèmes", "default": 1.5,
     "min": 0.05, "max": 10.0, "step": 0.1, "help": "granularité sous-thèmes",
     "derived": False},
    {"name": "min_sub_size", "label": "Taille mini sous-thème", "default": DERIVED.min_sub_size,
     "min": 1, "max": 1000, "step": 1, "help": "fusion des miettes (défaut frac·N)",
     "derived": True},
]
DEFAULTS = {k["name"]: k["default"] for k in KNOBS}


class ReclusterBody(BaseModel):
    """Corps de /recluster — tous les knobs sont optionnels.

    Bornes (`ge`/`le`) = limites PHYSIQUES seulement (audit #8) : on ne rejette
    plus une valeur légitime parce qu'elle sort d'un intervalle calé sur TikTok.
    `threshold`/`k`/`min_sub_size`/`dup_threshold` à ``None`` ⇒ **dérivés** des
    données (le défaut data-driven, pas un magic-number).
    """
    dedup: float | None = Field(DEFAULTS["dedup"], ge=0.0, le=1.0)
    min_chars: int = Field(DEFAULTS["min_chars"], ge=0)
    k: int | None = Field(None, ge=2)
    threshold: float | None = Field(None, ge=0.0, le=1.0)
    resolution_macro: float = Field(DEFAULTS["resolution_macro"], gt=0.0)
    resolution_sub: float = Field(DEFAULTS["resolution_sub"], gt=0.0)
    min_sub_size: int | None = Field(None, ge=1)
    dup_threshold: float | None = Field(None, ge=0.0, le=1.0)


app = FastAPI(title="Agora — recluster live", version="1.0")

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
        "n_cached": len(IDEAS),
        "model_id": MODEL_ID,
        "dim": int(VECS.shape[1]),
    }


@app.get("/params")
def params() -> dict:
    # `derived` expose la PROVENANCE des défauts data-driven : valeurs + la
    # distribution des cosinus k-NN dont sort le seuil (audit #6/#7/#9).
    return {
        "knobs": KNOBS,
        "defaults": DEFAULTS,
        "seed": SEED,
        "derived": {
            "k": DERIVED.k,
            "threshold": round(DERIVED.threshold, 4),
            "min_sub_size": DERIVED.min_sub_size,
            "dup_threshold": round(DERIVED.dup_threshold, 4),
            "knn_sim_mean": DERIVED.pool_mean,
            "knn_sim_std": DERIVED.pool_std,
            "edge_sigma": EDGE_SIGMA,
            "note": "défauts dérivés sur le cache après dedup/min_chars par défaut",
        },
    }


@app.post("/recluster")
def do_recluster(body: ReclusterBody) -> dict:
    return recluster(
        IDEAS, VECS, WEIGHTS,
        dedup=body.dedup,
        min_chars=body.min_chars,
        k=body.k,
        threshold=body.threshold,
        resolution_macro=body.resolution_macro,
        resolution_sub=body.resolution_sub,
        min_sub_size=body.min_sub_size,
        dup_threshold=body.dup_threshold,
    )
