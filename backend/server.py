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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.recluster import MODEL_ID, SEED, load_cache, recluster

# Contrat FROZEN (queue/cross-lane.md § « Console live ») : défauts ← WINNER nomic-v2.
KNOBS = [
    {"name": "dedup", "label": "Dédup (cosine)", "default": 0.95,
     "min": 0.90, "max": 0.99, "step": 0.01, "help": "fusion near-dups"},
    {"name": "min_chars", "label": "Min. caractères", "default": 12,
     "min": 0, "max": 40, "step": 1, "help": "filtre avis courts"},
    {"name": "k", "label": "k voisins", "default": 12,
     "min": 5, "max": 30, "step": 1, "help": "densité k-NN"},
    {"name": "threshold", "label": "Seuil arêtes (cosine)", "default": 0.60,
     "min": 0.40, "max": 0.85, "step": 0.01, "help": "coupe les arêtes"},
    {"name": "resolution_macro", "label": "Résolution macros", "default": 1.0,
     "min": 0.3, "max": 3.0, "step": 0.1, "help": "granularité macros"},
    {"name": "resolution_sub", "label": "Résolution sous-thèmes", "default": 1.5,
     "min": 0.5, "max": 4.0, "step": 0.1, "help": "granularité sous-thèmes"},
    {"name": "min_sub_size", "label": "Taille mini sous-thème", "default": 18,
     "min": 5, "max": 40, "step": 1, "help": "fusion des miettes"},
]
DEFAULTS = {k["name"]: k["default"] for k in KNOBS}


class ReclusterBody(BaseModel):
    """Corps de /recluster — tous les knobs sont optionnels (repli sur défaut)."""
    dedup: float = Field(DEFAULTS["dedup"], ge=0.90, le=0.99)
    min_chars: int = Field(DEFAULTS["min_chars"], ge=0, le=40)
    k: int = Field(DEFAULTS["k"], ge=5, le=30)
    threshold: float = Field(DEFAULTS["threshold"], ge=0.40, le=0.85)
    resolution_macro: float = Field(DEFAULTS["resolution_macro"], ge=0.3, le=3.0)
    resolution_sub: float = Field(DEFAULTS["resolution_sub"], ge=0.5, le=4.0)
    min_sub_size: int = Field(DEFAULTS["min_sub_size"], ge=5, le=40)


app = FastAPI(title="Agora — recluster live", version="1.0")

# CORS permissif en dev (le front passe par un proxy vite mais on couvre l'accès
# direct depuis localhost/forge au cas où).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache chargé une fois au démarrage (process léger, pas de torch).
IDEAS, VECS, WEIGHTS = load_cache()


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
    return {"knobs": KNOBS, "defaults": DEFAULTS, "seed": SEED}


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
    )
