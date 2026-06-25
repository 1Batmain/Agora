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
  - POST /analysis      → carte précalculée (arbre incrémental + co-occurrence, d3-pack)
  - GET  /stream        → rejoue le build EN INCRÉMENTAL (SSE, claims cachés, zéro LLM)
  - GET  /insights      → synthèse Markdown LLM précalculée (global | theme)
  - GET  /citations     → claims d'un thème, triées par proximité au centroïde
  - GET  /avis/{id}     → un avis entier + ses portions verbatim surlignables
  - POST /build         → (re)déclenche le précalcul d'un dataset (non bloquant)
  - GET  /build_status  → état du build d'un dataset (polling front)
  - GET  /flags         → flags de feedback d'un dataset (réafficher l'état)
  - POST /flag          → upsert le flag d'un avis (commentaire libre, horodaté)
  - DELETE /flag/{id}   → retire le flag d'un avis

Lancer :
    uv run --extra contender uvicorn backend.server:app --host 0.0.0.0 --port 8010
"""

from __future__ import annotations

import json
import os

from fastapi import Depends, FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.auth import rate_limit, require_token

# Dépendances posées sur les endpoints MUTATIFS / COÛTEUX (audit prod SEC1).
PROTECTED = [Depends(require_token), Depends(rate_limit)]
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
from backend import analysis_store, build_manager, flags_store


class _Dataset:
    """Un dataset chargé en mémoire : cache aligné (avis + vecteurs) + descripteur."""

    def __init__(self, dataset_id: str) -> None:
        self.id = dataset_id
        self.ideas, self.vecs, self.weights = load_cache(dataset_id)
        self.descriptor = dataset_descriptor(dataset_id, self.ideas)


# Registre MULTI-DATASET : la WHITELIST des ids est découverte au démarrage (process
# léger), mais les `_Dataset` (vecteurs + ideas, lourds en RAM) sont LAZY-LOADÉS au
# 1ᵉʳ `_resolve` et mémoïsés. Charger tout au boot ferait ×N la RAM dès l'amorçage.
_ids = list_datasets()
if not _ids:
    raise RuntimeError(
        "Aucun cache de dataset trouvé sous backend/cache/<dataset>/.\n"
        "Construis-en un : uv run --extra embed-contender "
        "python -m backend.build_cache --dataset tiktok"
    )
_IDSET = set(_ids)                       # whitelist O(1) (sécurité path-traversal)
_LOADED: dict[str, _Dataset] = {}        # cache des datasets effectivement chargés
# Défaut rétro-compat : "tiktok" s'il existe, sinon le premier découvert.
DEFAULT = DEFAULT_DATASET if DEFAULT_DATASET in _IDSET else _ids[0]


def _resolve(dataset: str | None) -> _Dataset:
    """Résout un id de dataset → `_Dataset` (lazy-load + mémoïsation).

    La whitelist `_IDSET` est la garde de sécurité (path-traversal) : un id absent →
    404, JAMAIS de construction de `_Dataset` (qui toucherait le disque). L'objet n'est
    construit (vecteurs + ideas) qu'au 1ᵉʳ accès d'un id whitelisté, puis caché.
    """
    ds = dataset or DEFAULT
    if ds not in _IDSET:
        raise HTTPException(
            status_code=404,
            detail=f"dataset inconnu: {ds!r} (disponibles: {_ids})",
        )
    obj = _LOADED.get(ds)
    if obj is None:
        obj = _LOADED[ds] = _Dataset(ds)
    return obj


app = FastAPI(title="Agora — carte spatiale précalculée (multi-dataset)", version="2.0")

# CORS restreint (audit prod SEC1) : origines connues seulement, JAMAIS "*".
# Surchargeable par AGORA_ALLOWED_ORIGINS (liste séparée par des virgules).
_origins_env = os.environ.get(
    "AGORA_ALLOWED_ORIGINS",
    "http://localhost:5180,http://127.0.0.1:5180",
)
ALLOWED_ORIGINS = [o.strip() for o in _origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    # Lazy-load : NE force PAS le chargement des vecteurs. `n_cached` vient du
    # descripteur léger (meta.json / ideas, sans vecteurs) ; `loaded` indique si
    # l'objet `_Dataset` (vecteurs en RAM) est effectivement chargé.
    return {
        "ok": True,
        "model_id": MODEL_ID,
        "default_dataset": DEFAULT,
        "datasets": {
            ds_id: {"n_cached": dataset_descriptor(ds_id).get("n_nodes", 0),
                    "loaded": ds_id in _LOADED}
            for ds_id in _ids
        },
    }


@app.get("/datasets")
def datasets() -> list[dict]:
    """Datasets disponibles (caches construits) → de quoi peupler le sélecteur.

    Lazy-load : le descripteur est lu depuis `meta.json`/`ideas` (léger), SANS forcer
    le chargement des vecteurs en RAM (qui n'a lieu qu'au 1ᵉʳ `_resolve` du dataset).
    """
    return [
        {**dataset_descriptor(ds_id),
         "namings": list(NAMINGS), "default_naming": DEFAULT_NAMING_METHOD}
        for ds_id in _ids
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
    # Lazy-load : ne charge en RAM (`_resolve`) QUE les datasets sans analyse prête (ceux
    # qui doivent réellement builder) ; les datasets déjà prêts restent déchargés au boot.
    for ds_id in _ids:
        if _AUTOBUILD_ONLY and ds_id not in _AUTOBUILD_ONLY:
            continue
        if analysis_store.state(ds_id) == analysis_store.READY:
            continue
        build_manager.ensure_build(_resolve(ds_id))


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
    """SERVE-only : un avis EN ENTIER + ses CLAIMS verbatim surlignables.

    Lit `analysis/avis.json` (précalculé, instantané) → `{id, text, claims}` où chaque
    claim `{id, cluster_id, color, spans:[{start,end}], target:{start,end}|null,
    theme_title}` regroupe ses portions extractives (sous-chaînes exactes, 1..N spans
    non-contigus) + sa cible verbatim, colorées à la couleur de son macro-thème (=
    couleur des bulles). Si l'analyse n'est pas prête → 202 `building` ; 404 si inconnu.
    """
    ds = _resolve(dataset)
    if analysis_store.state(ds.id) != analysis_store.READY:
        return _not_ready_response(ds, response)
    data = analysis_store.read_avis(ds.id, avis_id)
    if data is None:
        raise HTTPException(status_code=404,
                            detail=f"avis inconnu: {avis_id!r} (dataset {ds.id!r}).")
    return data


# ===================== Bac à sable « console de mixage » ===================== #
# RECLUSTER LIVE sans LLM (~1 s) : knobs α/k/resolution/coarsen_mult/tau_mult sur les
# embeddings CACHÉS (claims + cibles). + decision-trace chiffrée. Cf. /tmp/contract-sandbox.md.

class SandboxBody(BaseModel):
    """Corps de `POST /sandbox` — recluster live. Tous les knobs optionnels (défauts dérivés)."""
    dataset: str | None = None
    alpha: float | None = Field(None, ge=0.0, le=1.0)   # poids cible dans le blend
    k: int | None = Field(None, ge=2)                   # voisins k-NN
    resolution: float | None = Field(None, gt=0.0)      # résolution Leiden
    coarsen_mult: float | None = Field(None, gt=0.0)    # × seuil μ+σ de fusion des racines
    tau_mult: float | None = Field(None, gt=0.0)        # × seuil τ de subdivision


@app.post("/sandbox", dependencies=PROTECTED)
def do_sandbox(body: SandboxBody, response: Response) -> dict:
    """RECLUSTER LIVE (aucun LLM) selon les knobs → clusters + decision-trace + ms.

    Lit les claims + embeddings (claims & cibles) CACHÉS. Si les claims ne sont pas
    encore extraits (cache absent), renvoie 503 avec un indice (lancer POST /build).
    """
    from backend.sandbox import recluster_payload
    from backend.claims_endpoint import OllamaUnavailable
    from pipeline.claims.backend import BackendUnavailable

    ds = _resolve(body.dataset)
    try:
        return recluster_payload(
            ds, alpha=body.alpha, k=body.k, resolution=body.resolution,
            coarsen_mult=body.coarsen_mult, tau_mult=body.tau_mult,
        )
    except (OllamaUnavailable, BackendUnavailable) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"claims non extraits ({exc}). Lance POST /build d'abord.",
        ) from exc


@app.get("/explain", dependencies=PROTECTED)
def get_explain(
    dataset: str | None = Query(None),
    cluster: str | None = Query(None),
    pair: str | None = Query(None),
) -> dict:
    """Decision-trace d'un nœud (`cluster=nX`) ou d'une paire (`pair=nX,nY`).

    S'appuie sur le DERNIER `/sandbox` du dataset (mémoïsé) ; si aucun, en relance un
    aux knobs neutres. Renvoie voisinage (k plus proches centroïdes) + critères de
    fusion/subdivision chiffrés (cf. contrat).
    """
    from backend.sandbox import explain_cluster, explain_pair

    ds = _resolve(dataset)
    if pair:
        parts = [p.strip() for p in pair.split(",") if p.strip()]
        if len(parts) != 2:
            raise HTTPException(status_code=422, detail="pair attend 'nX,nY'.")
        out = explain_pair(ds, parts[0], parts[1])
    elif cluster:
        out = explain_cluster(ds, cluster.strip())
    else:
        raise HTTPException(status_code=422, detail="fournir cluster=nX ou pair=nX,nY.")
    if "error" in out:
        raise HTTPException(status_code=404, detail=out["error"])
    return out


class BuildBody(BaseModel):
    """Corps de `POST /build` — (re)déclenche le précalcul d'un dataset.

    `force=true` efface l'analyse persistée avant de reconstruire (sinon no-op si déjà
    prête). Le build tourne EN TÂCHE DE FOND : la réponse est immédiate (202).
    """
    dataset: str | None = None
    force: bool = False


@app.post("/build", dependencies=PROTECTED)
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


# ============================ Stream LIVE (SSE) ============================ #
# Rejoue le build d'un dataset EN INCRÉMENTAL via `AnalysisState` : rattachement
# plus-proche + maj O(1) + split local sur divergence, à partir des claims +
# embeddings DÉJÀ CACHÉS (AUCUN LLM, pas de recompute global → rapide). Émet les
# events du contrat figé (`/tmp/contract-live.md`) en Server-Sent Events.

def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.get("/stream", dependencies=PROTECTED)
def stream(dataset: str | None = Query(None),
           backend: str | None = Query(None)) -> StreamingResponse:
    """SSE : rejoue les claims CACHÉS d'un dataset via une `AnalysisState` fraîche.

    Émet `snapshot` (état initial) → `claim_added`/`theme_split` (au fil de l'eau) →
    `done`. Réutilise `claims.json` + `claims_emb.npz` : zéro appel LLM. Si les claims
    ne sont pas encore extraits (cache absent), émet un event `error` puis se ferme.
    """
    from backend.analysis import _dataset_context
    from backend.claims_endpoint import OllamaUnavailable
    from backend.state import AnalysisState
    from pipeline.claims.backend import BackendUnavailable

    ds = _resolve(dataset)

    def gen():
        try:
            state = AnalysisState.from_dataset(ds, backend=backend)
        except (OllamaUnavailable, BackendUnavailable) as exc:
            yield _sse({"type": "error", "detail": str(exc),
                        "hint": "claims non extraits : lance POST /build d'abord."})
            return
        except Exception as exc:  # noqa: BLE001 — on remonte l'erreur au client SSE
            yield _sse({"type": "error", "detail": str(exc)})
            return
        for event in state.stream_events(dataset_context=_dataset_context(ds.id)):
            yield _sse(event)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/build_status")
def build_status(dataset: str | None = Query(None)) -> dict:
    """État du build d'un dataset (pour le polling front) : status + progression."""
    ds = _resolve(dataset)
    prog = analysis_store.progress(ds.id)
    prog["status"] = analysis_store.state(ds.id)
    prog["dataset"] = ds.id
    prog["building"] = build_manager.is_building(ds.id)
    return prog


# ============================ Flags de feedback ============================ #
# Bob signale un avis mal découpé / mal ciblé / mal extrait avec un commentaire
# libre, pour affiner ensuite. Artefact LÉGER, persistant et éditable, indépendant
# de l'analyse précalculée : upsert par avis_id dans `backend/cache/<dataset>/flags.json`.
# AUCUN calcul lourd ici.

class FlagBody(BaseModel):
    """Corps de `POST /flag` — feedback libre sur l'extraction d'un avis (upsert)."""
    dataset: str | None = None
    avis_id: str
    text: str = ""


@app.get("/flags")
def get_flags(dataset: str | None = Query(None)) -> dict:
    """Tous les flags d'un dataset (pour réafficher l'état au chargement)."""
    ds = _resolve(dataset)
    return {"dataset": ds.id, "flags": flags_store.list_flags(ds.id)}


@app.post("/flag", dependencies=PROTECTED)
def post_flag(body: FlagBody) -> dict:
    """UPSERT du flag d'un avis (crée ou met à jour, horodaté) → {ok, flag}."""
    ds = _resolve(body.dataset)
    avis_id = (body.avis_id or "").strip()
    if not avis_id:
        raise HTTPException(status_code=422, detail="avis_id requis.")
    flag = flags_store.upsert_flag(ds.id, avis_id, body.text)
    return {"ok": True, "flag": flag}


@app.delete("/flag/{avis_id}", dependencies=PROTECTED)
def remove_flag(avis_id: str, dataset: str | None = Query(None)) -> dict:
    """Retire le flag d'un avis → {ok, removed}."""
    ds = _resolve(dataset)
    removed = flags_store.delete_flag(ds.id, avis_id)
    return {"ok": True, "removed": removed}
