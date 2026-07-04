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
  - GET  /avis_list     → liste paginée/filtrée des avis (cluster + recherche)
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

from backend import auth
from backend.auth import forbid_in_public, rate_limit, require_token

# Dépendances posées sur les endpoints MUTATIFS / COÛTEUX (audit prod SEC1).
PROTECTED = [Depends(require_token), Depends(rate_limit)]
# Endpoints de COMPUTE/BUILD : désactivés en mode public + rate-limités hors public.
COMPUTE = [Depends(forbid_in_public), Depends(rate_limit)]
from pydantic import BaseModel, Field

# Léger (numpy only, aucun torch) : la résolution Leiden par défaut, source unique.
from pipeline.cluster.leiden_cluster import DEFAULT_RESOLUTION

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
from backend import (
    analysis_store,
    avis,
    build_manager,
    density,
    flags_store,
    live_cluster,
    serve_metrics,
)


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


# /docs, /redoc & /openapi.json DÉSACTIVÉS en mode public (pas de divulgation du schéma
# d'API à un anonyme — audit privacy #9). Conservés en dev pour le confort.
_DOCS = dict(docs_url=None, redoc_url=None, openapi_url=None) if auth.PUBLIC_MODE else {}
app = FastAPI(title="Agora — carte spatiale précalculée (multi-dataset)", version="2.0", **_DOCS)


# Limite de taille de corps : 413 au-delà de 64 Ko (anti-DoS mémoire — audit input #2).
# Le check Content-Length rejette AVANT de bufferiser le corps. Le reverse-proxy
# (nginx `client_max_body_size`) reste une défense en profondeur recommandée.
MAX_BODY_BYTES = 64 * 1024


class _BodySizeLimitMiddleware:
    """Rejette (413) toute requête HTTP dont le corps annoncé dépasse `max_bytes`."""

    def __init__(self, app, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            for name, value in scope.get("headers") or []:
                if name == b"content-length":
                    try:
                        too_big = int(value) > self.max_bytes
                    except ValueError:
                        too_big = False
                    if too_big:
                        await send({
                            "type": "http.response.start",
                            "status": 413,
                            "headers": [(b"content-type", b"application/json; charset=utf-8")],
                        })
                        await send({
                            "type": "http.response.body",
                            "body": b'{"detail":"Corps de requete trop volumineux (max 64 Ko)."}',
                        })
                        return
                    break
        await self.app(scope, receive, send)


@app.middleware("http")
async def _security_headers(request, call_next):
    """En-têtes de sécurité de base sur chaque réponse (audit privacy #6)."""
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    # Anti-clickjacking sans casser /docs (ne restreint pas le chargement de ressources).
    response.headers.setdefault("Content-Security-Policy", "frame-ancestors 'none'")
    response.headers.setdefault(
        "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
    )
    return response


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
# Outermost : la borne de taille s'applique AVANT tout parsing (anti-DoS mémoire).
app.add_middleware(_BodySizeLimitMiddleware, max_bytes=MAX_BODY_BYTES)


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

    # Hiérarchie mère→enfants : la liste top-level montre les MÈRES + les
    # consultations SIMPLES, PAS les enfants (qui restent servis par id sur tous
    # les autres endpoints via `_resolve`). Un enfant = descripteur avec `parent_id`.
    closed = [_with_capabilities(d) for ds_id in _ids
              if not (d := dataset_descriptor(ds_id)).get("parent_id")]
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


@app.post("/submit", dependencies=[Depends(rate_limit)])
def submit(body: SubmitBody) -> dict:
    """Reçoit une contribution citoyenne, la corrèle aux retours déjà reçus, la stocke.

    Vie privée (SEC3, audit privacy #1) : le texte est MASQUÉ (`clean_text` : PII +
    normalisation) AVANT tout embedding/stockage, et la réponse ne renvoie JAMAIS le
    verbatim d'un autre citoyen — seulement un AGRÉGAT non-PII.

    1. valide la consultation (doit être OUVERTE) et le texte (longueur brute),
    2. masque le texte (PII) puis l'embedde (nomic local),
    3. corrèle au cosinus aux contributions existantes → `n_similar`,
    4. stocke {text_clean, vec, ts},
    5. renvoie `{ok, n_similar, pct_panel, message}` (zéro verbatim d'autrui).
    """
    from datetime import datetime, timezone

    from backend import submissions
    from pipeline.ingest import normalize

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

    # MASQUAGE PII avant tout traitement : rien de brut n'est embeddé ni persisté.
    clean = normalize.clean_text(text)
    if not clean:
        raise HTTPException(status_code=422, detail="Contribution vide après nettoyage.")

    ts = datetime.now(timezone.utc).isoformat()
    existing = submissions.load_submissions(body.consultation_id)

    # MODE PUBLIC (prod serve-only, SANS torch/clé) : on COLLECTE seulement — le texte
    # (PII masquée) est stocké SANS embedding ni corrélation live. L'analyse est construite
    # plus tard en DEV quand la consultation a réuni assez de contributions, puis promue.
    # Prod reste léger (aucun modèle lourd chargé) et n'a besoin d'AUCUNE clé.
    if auth.PUBLIC_MODE:
        submissions.append_submission(body.consultation_id, clean, None, ts)
        return {
            "ok": True,
            "n_similar": 0,
            "pct_panel": 0,
            "message": (
                "Merci, votre contribution est enregistrée. "
                f"{len(existing) + 1} contributions collectées jusqu'ici — l'analyse sera "
                "publiée quand la consultation en aura réuni assez."
            ),
        }

    # DEV (embedder local disponible) : embedding + corrélation → feedback riche live.
    vec = submissions.embed_text(clean)
    corr = submissions.correlate(vec, existing)
    submissions.append_submission(body.consultation_id, clean, vec, ts)

    n = corr["n_similar"]
    # `pct_panel` : part du panel (contributions déjà reçues) ayant évoqué un sujet proche.
    panel = len(existing)
    pct_panel = round(100 * n / panel) if panel else 0
    if n > 0:
        message = (
            f"{n} personnes ont déjà évoqué un sujet proche."
            if n > 1
            else "1 personne a déjà évoqué un sujet proche."
        )
    else:
        message = "Vous êtes parmi les premiers à soulever ce point !"
    return {
        "ok": True,
        "n_similar": n,
        "pct_panel": pct_panel,
        "message": message,
    }


# ===================== Refonte « carte spatiale » (B1–B4) ===================== #
# SÉPARATION BUILD / SERVE. Le pipeline lourd (claims→embed→cluster→hiérarchie→
# insights) est PRÉCALCULÉ et PERSISTÉ par `backend.build_analysis` (en tâche de fond,
# cf. `build_manager`). Les trois endpoints du CONTRAT figé (`.agent/queue/front-redesign.md`)
# ne font ici que LIRE le cache persisté — AUCUN calcul lourd à la requête. Si l'analyse
# n'est pas prête, ils déclenchent/poursuivent le build de fond et renvoient un état
# clair `{status: building|absent|error}` (HTTP 202 en cours, 503 en échec).

# Build au démarrage : pour chaque dataset SANS analyse prête, on lance un build de fond.
# OFF PAR DÉFAUT (serve-only) : un clone frais SANS clé Mistral ne doit JAMAIS partir en
# extraction en boucle. On l'OPT-IN explicitement (AGORA_AUTOBUILD=1) là où c'est voulu ;
# restreignable à une liste via AGORA_AUTOBUILD_DATASETS="a,b" (sinon tous).
_AUTOBUILD = (os.environ.get("AGORA_AUTOBUILD", "0").strip().lower()
              in ("1", "true", "yes", "on"))
_AUTOBUILD_ONLY = {s.strip() for s in os.environ.get("AGORA_AUTOBUILD_DATASETS", "").split(",")
                   if s.strip()}


@app.on_event("startup")
def _startup_autobuild() -> None:
    # Mode public : un nœud exposé ne build JAMAIS (extraction LLM faite hors-ligne par
    # l'opérateur, cf. AGORA_PUBLIC). Aucune extraction mistral-large sur un nœud public.
    if auth.PUBLIC_MODE or not _AUTOBUILD:
        return
    # Lazy-load : ne charge en RAM (`_resolve`) QUE les datasets sans analyse prête (ceux
    # qui doivent réellement builder) ; les datasets déjà prêts restent déchargés au boot.
    for ds_id in _ids:
        if _AUTOBUILD_ONLY and ds_id not in _AUTOBUILD_ONLY:
            continue
        if analysis_store.state(ds_id) == analysis_store.READY:
            continue
        build_manager.ensure_build(_resolve(ds_id))


def _sanitize_progress(prog: dict) -> dict:
    """Masque les détails d'exception internes (str(exc)) avant envoi au client.

    Le détail complet reste persisté dans `status.json` (côté serveur, pour l'opérateur) ;
    le client ne reçoit qu'un message générique — pas de chemin disque / trace / détail API.
    """
    if prog.get("error"):
        prog = {**prog, "error": "échec du build (voir logs serveur)"}
    return prog


def _not_ready_response(ds, response: Response) -> dict:
    """Réponse SERVE quand l'analyse n'est pas prête.

    MODE PUBLIC **ou AUTOBUILD OFF (défaut)** : ne déclenche AUCUN build — dataset non
    pré-construit → 404. C'est ce qui empêche un clone SANS clé Mistral de partir en boucle
    d'extraction pilotée par les lectures/polling du front (bug corrigé).

    AUTOBUILD ON (opt-in explicite `AGORA_AUTOBUILD=1`, hors public) : (re)lance le build de
    fond et renvoie la progression — 202 si ça construit, 503 si le dernier build a échoué.
    """
    if auth.PUBLIC_MODE or not _AUTOBUILD:
        raise HTTPException(
            status_code=404,
            detail="Analyse non disponible pour ce dataset (cache absent). Récupère les "
                   "caches via scripts/setup.sh, ou construis avec AGORA_AUTOBUILD=1 + clé Mistral.",
        )
    build_manager.ensure_build(ds)
    prog = analysis_store.progress(ds.id)
    prog["status"] = analysis_store.state(ds.id)
    prog["dataset"] = ds.id
    response.status_code = 503 if prog["status"] == analysis_store.ERROR else 202
    return _sanitize_progress(prog)


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
    resolution: float = Field(DEFAULT_RESOLUTION, gt=0.0)


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
            # Enrichissement SERVE-TIME (couverture + fidélité verbatim) : dérivé de
            # l'arbre caché + claims.json, muté EN MÉMOIRE sur ce payload fraîchement
            # parsé — ZÉRO écriture au cache, AUCUN rebuild.
            return serve_metrics.enrich_indices(payload, ds.id, ds.ideas)
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


@app.get("/cost")
def get_cost(dataset: str | None = Query(None)) -> dict:
    """SERVE-only : coût LLM PRÉCALCULÉ du build (`analysis/cost.json`) — tokens + $ estimé.

    Transparence des coûts d'analyse (tout le trafic Mistral est compté à `chat()`).
    404 si non mesuré (dataset construit avant l'instrumentation)."""
    ds = _resolve(dataset)
    from backend import cost as _cost
    data = _cost.read_cost(ds.id)
    if data is None:
        raise HTTPException(status_code=404, detail="coût non mesuré pour ce dataset.")
    # DURÉES au serve-time (jamais bakées dans cost.json) : build d'analyse (status.json)
    # + phase opinion (opinion.json) — pour l'affichage « coût · tokens · durée » du front.
    durations: dict = {}
    prog = analysis_store.progress(ds.id)
    if isinstance(prog, dict) and prog.get("took_seconds"):
        durations["analysis_seconds"] = prog["took_seconds"]
    op = analysis_store.read_opinion(ds.id)
    if isinstance(op, dict) and op.get("took_seconds"):
        durations["opinion_seconds"] = op["took_seconds"]
    args_art = analysis_store.read_arguments(ds.id)
    if isinstance(args_art, dict) and args_art.get("took_seconds"):
        durations["arguments_seconds"] = args_art["took_seconds"]
    if durations:
        data = {**data, "durations": durations}
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


@app.get("/opinion")
def get_opinion(dataset: str | None = Query(None)) -> dict:
    """SERVE-only : répartition d'opinion PRÉCALCULÉE par thème feuille.

    Lit `analysis/opinion.json` (baké par `backend.build_opinion`, instantané) →
    `{dataset, model, themes:[{theme_id, proposition, fav, def, nuance, pct_favorable,
    opposition, profil}]}`. Artefact À PART, INDÉPENDANT du build d'analyse : s'il n'a
    pas (encore) été baké, on renvoie une liste VIDE (200) — le front dégrade
    gracieusement (pas de barre d'opinion) sans jamais bloquer la synthèse.
    """
    ds = _resolve(dataset)
    data = analysis_store.read_opinion(ds.id)
    if data is None:
        return {"dataset": ds.id, "themes": [], "status": "absent"}
    return data


@app.get("/arguments")
def get_arguments(dataset: str | None = Query(None)) -> dict:
    """SERVE-only : arguments minés PRÉCALCULÉS par thème (synthèses sourcées).

    Lit `analysis/arguments.json` (baké par `backend.build_arguments`, OPTIONNEL) →
    `{dataset, model, themes:[{theme_id, mode, proposition, arguments:[{argument,
    stance, n_support, sources:[{avis_id, claim_id, text, similarity}]}]}]}`.
    Artefact À PART : les datasets déjà analysés n'en ont pas → liste VIDE (200),
    le front ne rend simplement pas le panneau (contrat de rétro-compat).
    """
    ds = _resolve(dataset)
    data = analysis_store.read_arguments(ds.id)
    if data is None:
        return {"dataset": ds.id, "themes": [], "status": "absent"}
    return data


@app.get("/demographics")
def get_demographics(dataset: str | None = Query(None)) -> dict:
    """SERVE-only : profil démographique PRÉCALCULÉ (global + majorités par thème).

    Lit `analysis/demographics.json` (baké par `backend.build_demographics`,
    OPTIONNEL — pure jointure, zéro LLM). Absent → liste vide (200), le front ne
    rend ni la description du panel ni les groupes majoritaires.
    """
    ds = _resolve(dataset)
    data = analysis_store.read_demographics(ds.id)
    if data is None:
        return {"dataset": ds.id, "themes": [], "status": "absent"}
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
    # Join gracieux de la stance par claim (artefact à part, absent → claims inchangés).
    stance_map = analysis_store.read_claim_stance(ds.id)
    if stance_map:
        data = {**data, "claims": avis.join_claim_stance(data.get("claims", []), stance_map)}
    return data


@app.get("/avis_list")
def get_avis_list(
    response: Response,
    dataset: str | None = Query(None),
    theme_id: str | None = Query(None),
    q: str | None = Query(None),
    stance: str | None = Query(None),  # "favorable"|"defavorable" → n'garde que les avis de ce sentiment
    limit: int = Query(15, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict:
    """SERVE-only : liste paginée/filtrée de TOUS les avis (page d'exploration).

    Lit `analysis/avis.json` (précalculé) → `{total, items:[…]}` où chaque item porte
    l'avis ENTIER (`text`, `text_fr`, `lang`, `claims:[{spans,target,color,theme_title,
    cluster_id}]`) en plus de `excerpt` + `themes:[{id,title,color}]` : l'explorateur
    rend chaque avis INLINE (texte + surlignages) sans `/avis/{id}` par carte. Items
    lourds → `limit` par défaut bas (15). `theme_id` filtre les avis ayant ≥1 claim dans
    le sous-arbre du thème (un macro filtre ses sous-thèmes, hiérarchie de `/analysis`) ;
    `q` est une recherche sous-chaîne insensible casse/accents sur le texte ;
    `limit`/`offset` paginent. Si l'analyse n'est pas prête → 202 `building`.
    """
    ds = _resolve(dataset)
    if analysis_store.state(ds.id) != analysis_store.READY:
        return _not_ready_response(ds, response)
    payload = analysis_store.read_analysis(ds.id)
    themes = (payload or {}).get("themes", [])
    # Stance lue AVANT le filtrage : `avis.avis_list` peut ne garder que les avis dont ≥1 claim
    # (dans le thème) porte le sentiment demandé (`stance`), pour les cartes cliquables.
    stance_map = analysis_store.read_claim_stance(ds.id)
    # Hot path RECHERCHE : le coût flagué par l'audit #1 est le scan O(N) + fold Unicode
    # par requête, qui n'intervient QUE quand `q` est présent. On délègue alors le filtrage +
    # la pagination à l'index DuckDB (`analysis.duckdb`, ×10–30 mesuré, cf. research/
    # bench_duckdb_avis.py). SANS `q`, le fallback RAM (itération dict + court-circuit, aucun
    # fold) est déjà optimal et bat le coût fixe d'une requête SQL → on le garde. Index
    # absent/périmé/duckdb non installé → curseur None → fallback inchangé (parité prouvée).
    con = analysis_store.avis_duckdb_con(ds.id) if (q and q.strip()) else None
    if con is not None:
        result = avis.avis_list_duckdb(con, themes, theme_id=theme_id, q=q,
                                       stance=stance, claim_stance=stance_map,
                                       limit=limit, offset=offset)
    else:
        avis_data = analysis_store.read_avis_all(ds.id)
        if avis_data is None:
            return _not_ready_response(ds, response)
        result = avis.avis_list(avis_data, themes, theme_id=theme_id, q=q,
                                stance=stance, claim_stance=stance_map, limit=limit, offset=offset)
    # Join gracieux de la stance par claim sur les avis de la page (absent → inchangé).
    if stance_map:
        for item in result.get("items", []):
            item["claims"] = avis.join_claim_stance(item.get("claims", []), stance_map)
    return result


@app.get("/density", dependencies=COMPUTE)
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
        # Message générique au client (pas de détail interne) ; cause loggée côté serveur.
        raise HTTPException(status_code=503, detail="Projection de densité indisponible.") from exc


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
    # Bornée HAUT (audit) : une `resolution` énorme alourdit Leiden gratuitement.
    resolution: float = Field(DEFAULT_RESOLUTION, gt=0.0, le=10.0)


@app.post("/recluster", dependencies=COMPUTE)
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


@app.post("/build", dependencies=[Depends(forbid_in_public), *PROTECTED])
def do_build(body: BuildBody, response: Response) -> dict:
    """Déclenche/relance le build de fond de l'analyse d'un dataset (non bloquant).

    DÉSACTIVÉ en mode public (`forbid_in_public`) : aucun build pilotable à distance —
    l'opérateur pré-construit hors-ligne (CLI) puis sert le cache.
    """
    ds = _resolve(body.dataset)
    if body.force:
        analysis_store.clear(ds.id)
    state = build_manager.ensure_build(ds)
    response.status_code = 200 if state == analysis_store.READY else 202
    prog = analysis_store.progress(ds.id)
    prog["status"] = analysis_store.state(ds.id)
    prog["dataset"] = ds.id
    return _sanitize_progress(prog)


@app.get("/build_status")
def build_status(dataset: str | None = Query(None)) -> dict:
    """État du build d'un dataset (pour le polling front) : status + progression."""
    ds = _resolve(dataset)
    prog = analysis_store.progress(ds.id)
    prog["status"] = analysis_store.state(ds.id)
    prog["dataset"] = ds.id
    prog["building"] = build_manager.is_building(ds.id)
    return _sanitize_progress(prog)


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
