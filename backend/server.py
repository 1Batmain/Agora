"""Serveur FastAPI :8010 — carte spatiale PRÉCALCULÉE, MULTI-DATASET (SERVE-only).

Léger : au démarrage, DÉCOUVRE tous les caches `backend/cache/<dataset>/` et les
charge (vecteurs `.npy` + `ideas.jsonl`), PAS le modèle torch. Le pipeline lourd
(claims→embed→cluster→hiérarchie→insights) est PRÉCALCULÉ et PERSISTÉ par
`backend.build_analysis` (en tâche de fond, cf. `build_manager`) ; les endpoints
ne font que LIRE le cache persisté — AUCUN calcul lourd à la requête.

GÉNÉRIQUE : aucun nom de corpus en dur. Rétro-compat : sans `dataset`, tout se
comporte comme sur le défaut (`tiktok`).

Endpoints :
  - GET  /health        → {ok, datasets, default_dataset}
  - GET  /datasets      → [{id, label, status, n_nodes, languages, source, namings}]
  - POST /analysis      → carte précalculée (arbre incrémental + co-occurrence, d3-pack)
  - GET  /insights      → synthèse Markdown LLM précalculée (global | theme)
  - GET  /citations     → claims d'un thème, triées par proximité au centroïde
  - GET  /avis/{id}     → un avis entier + ses portions verbatim surlignables
  - POST /build         → (re)déclenche le précalcul d'un dataset (non bloquant)
  - GET  /build_status  → état du build d'un dataset (polling front)
  - GET  /flags         → flags de feedback d'un dataset (réafficher l'état)
  - POST /flag          → upsert le flag d'un avis OU d'une synthèse de thème (horodaté)
  - DELETE /flag/{id}   → retire le flag d'une cible (target_type, défaut "avis")

Lancer :
    uv run --extra contender uvicorn backend.server:app --host 0.0.0.0 --port 8010
"""

from __future__ import annotations

import os

from fastapi import Depends, FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware

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
    list_open_consultations,
    load_cache,
    open_consultation_descriptor,
)
from backend import analysis_store, build_manager, density, flags_store, live_cluster


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

    Inclut aussi les consultations OUVERTES (descripteurs `status:"open"`), même
    SANS cache d'analyse : elles n'ont pas de dossier cache/ mais doivent apparaître
    sur la landing (carte « Ouvert » → vue Participer).
    """
    # Capacités serveur (méthodes de nommage) ajoutées à CHAQUE descripteur
    # `Consultation` — orthogonales au schéma, identiques pour tous les items.
    def _with_capabilities(c: dict) -> dict:
        return {**c, "namings": list(NAMINGS), "default_naming": DEFAULT_NAMING_METHOD}

    closed = [_with_capabilities(dataset_descriptor(ds_id)) for ds_id in _ids]
    open_ = [_with_capabilities(open_consultation_descriptor(name))
             for name in list_open_consultations()]
    # Ouvertes en tête : ce sont celles où l'on peut encore agir.
    return open_ + closed


# ===================== Participation (consultations OUVERTES) ===================== #
# Une contribution citoyenne sur une consultation OUVERTE est embeddée nomic LOCAL
# (aucun LLM/clé) et corrélée AU MOMENT MÊME aux contributions déjà reçues : on
# renvoie combien de personnes ont déjà évoqué un sujet proche + l'extrait le plus
# proche, puis on stocke la contribution. Voir backend/submissions.py.

# Bornes de saisie (anti-abus léger) : un avis court mais réel ↔ pas un pavé.
SUBMIT_MIN_CHARS = 3
SUBMIT_MAX_CHARS = 5000


class SubmitBody(BaseModel):
    """Corps de `POST /submit` — une contribution sur une consultation ouverte."""
    consultation_id: str
    text: str


@app.post("/submit", dependencies=PROTECTED)
def submit(body: SubmitBody) -> dict:
    """Reçoit une contribution citoyenne, la corrèle aux retours déjà reçus, la stocke.

    1. valide la consultation (doit être OUVERTE) et le texte,
    2. embedde le texte (nomic local),
    3. corrèle au cosinus aux contributions existantes → `n_similar` + extrait proche,
    4. stocke {text, vec, ts},
    5. renvoie `{ok, n_similar, nearest_excerpt, message}`.
    """
    from datetime import datetime, timezone

    from backend import submissions

    if body.consultation_id not in set(list_open_consultations()):
        raise HTTPException(
            status_code=404,
            detail=f"consultation ouverte inconnue: {body.consultation_id!r}",
        )
    text = (body.text or "").strip()
    if len(text) < SUBMIT_MIN_CHARS:
        raise HTTPException(status_code=422, detail="Contribution trop courte.")
    if len(text) > SUBMIT_MAX_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"Contribution trop longue (max {SUBMIT_MAX_CHARS} caractères).",
        )

    vec = submissions.embed_text(text)
    existing = submissions.load_submissions(body.consultation_id)
    corr = submissions.correlate(vec, existing)
    ts = datetime.now(timezone.utc).isoformat()
    submissions.append_submission(body.consultation_id, text, vec, ts)

    n = corr["n_similar"]
    if n > 0:
        message = (
            f"{n} personne{'s' if n > 1 else ''} ont déjà évoqué un sujet proche."
            if n > 1
            else "1 personne a déjà évoqué un sujet proche."
        )
    else:
        message = "Vous êtes parmi les premiers à soulever ce point !"
    return {
        "ok": True,
        "n_similar": n,
        "nearest_excerpt": corr["nearest_excerpt"] if n > 0 else None,
        "message": message,
    }


# ===================== Refonte « carte spatiale » (B1–B4) ===================== #
# SÉPARATION BUILD / SERVE. Le pipeline lourd (claims→embed→cluster→hiérarchie→
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
    """SERVE-only : sert la carte des thèmes PRÉCALCULÉE (arbre variance-adaptatif + edges de co-occurrence ; sans positions x,y — front en d3-pack).

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


@app.get("/density")
def get_density(
    dataset: str | None = Query(None),
) -> dict:
    """Paysage de densité 3D : UMAP 2D des embeddings PRÉ-clustering + KDE sur grille.

    Renvoie `{nx, nz, x_range, z_range, heights[nz][nx], zmax}` — la hauteur de chaque
    sommet d'une grille 96×96 est la densité locale (front : surface rotatable, hauteur
    normalisable par `zmax`). Calcul PARESSEUX au 1ᵉʳ appel puis CACHE disque
    (`umap2d.npy` + `density.json`), INDÉPENDANT des caches d'analyse. La whitelist
    `_resolve` garde le path-traversal. 503 si la projection UMAP est indisponible
    (umap-learn absent ET pas de cache).
    """
    ds = _resolve(dataset)
    try:
        return density.density_payload(ds.id)
    except density.DensityUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


class ReclusterBody(BaseModel):
    """Corps de `POST /recluster` — re-clustering LIVE piloté par le seuil k-NN.

    `knn_threshold=None` → défaut DÉRIVÉ du dataset (la Console démarre comme `/analysis`).
    SERVE/COMPUTE léger (zéro LLM, < ~2 s) : lit les vecteurs cachés et la projection
    UMAP cachée — NE touche AUCUN cache d'analyse.
    """
    dataset: str | None = None
    knn_threshold: float | None = Field(None, ge=0.0, le=1.0)
    # `k` (nombre de voisins du graphe k-NN) = LEVIER de la Console (≥2).
    k: int | None = Field(None, ge=2, le=200)
    resolution: float = Field(1.0, gt=0.0)


@app.post("/recluster")
def do_recluster(body: ReclusterBody) -> dict:
    """Re-clustering LIVE au seuil k-NN donné → `{themes, points, indices, meta}`.

    Reconstruit la carte des thèmes À LA VOLÉE (Leiden hiérarchique + variance-adaptatif
    + coarsening, nommage c-TF-IDF, indices M5, points UMAP 2D) en faisant varier le seuil
    d'arête k-NN, SANS aucun appel LLM. La whitelist `_resolve` garde le path-traversal.
    """
    ds = _resolve(body.dataset)
    return live_cluster.recluster_payload(
        ds.id, body.knn_threshold, k=body.k, resolution=body.resolution
    )


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
    """Corps de `POST /flag` — feedback libre sur un AVIS ou une SYNTHÈSE de thème.

    Modèle généralisé : `target_type` ("avis"|"theme") + `target_id` (+ `layer`,
    `category` pour les thèmes). RÉTRO-COMPAT : si l'ancien `avis_id` est fourni
    sans `target_type`, on le mappe en `target_type="avis"`, `target_id=avis_id`.
    """
    dataset: str | None = None
    target_type: str | None = None
    target_id: str | None = None
    avis_id: str | None = None  # rétro-compat (ancien front avis)
    layer: int | None = None
    category: str | None = None
    text: str = ""


# Cibles flaguables connues — garde-fou contre un type arbitraire.
_FLAG_TARGETS = {"avis", "theme"}


def _flag_target(target_type: str | None, target_id: str | None, avis_id: str | None):
    """Résout (target_type, target_id) en honorant la rétro-compat `avis_id`."""
    ttype = (target_type or "").strip().lower()
    tid = (target_id or "").strip()
    if not ttype and avis_id:  # ancien contrat : avis_id seul → avis
        ttype, tid = "avis", str(avis_id).strip()
    if ttype not in _FLAG_TARGETS:
        raise HTTPException(status_code=422, detail="target_type doit valoir avis ou theme.")
    if not tid:
        raise HTTPException(status_code=422, detail="target_id requis.")
    return ttype, tid


@app.get("/flags")
def get_flags(dataset: str | None = Query(None)) -> dict:
    """Tous les flags d'un dataset, tous types (pour réafficher l'état au chargement)."""
    ds = _resolve(dataset)
    return {"dataset": ds.id, "flags": flags_store.list_flags(ds.id)}


@app.post("/flag", dependencies=PROTECTED)
def post_flag(body: FlagBody) -> dict:
    """UPSERT du flag d'une cible (avis ou thème), horodaté → {ok, flag}."""
    ds = _resolve(body.dataset)
    ttype, tid = _flag_target(body.target_type, body.target_id, body.avis_id)
    flag = flags_store.upsert_flag(
        ds.id, ttype, tid, body.text, layer=body.layer, category=body.category
    )
    return {"ok": True, "flag": flag}


@app.delete("/flag/{target_id}", dependencies=PROTECTED)
def remove_flag(
    target_id: str,
    dataset: str | None = Query(None),
    target_type: str = Query("avis"),
) -> dict:
    """Retire le flag d'une cible → {ok, removed}. `target_type` défaut "avis" (rétro-compat)."""
    ds = _resolve(dataset)
    ttype, tid = _flag_target(target_type, target_id, None)
    removed = flags_store.delete_flag(ds.id, ttype, tid)
    return {"ok": True, "removed": removed}
