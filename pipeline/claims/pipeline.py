"""Pipeline CLAIMS → thèmes ÉMERGENTS (réutilisable, sans gold ni taxonomie).

    avis citoyens bruts
      → extraction des CLAIMS atomiques (ministral local, vocab libre)   [extract.py]
      → embed nomic-v2 (espace PROD)                                       [pipeline.embed]
      → graphe k-NN + Leiden, défauts DÉRIVÉS des données                  [pipeline.cluster]
      → noms c-TF-IDF (mots-vides corpus-dérivés)                          [pipeline.cluster.naming]
      → carte : poids social, consensus, claims représentatives, CO-OCCURRENCE

Les thèmes ne sont PAS imposés : un cluster = un thème découvert. Tout dérive des
données (généricité — marche sur des centaines de consultations originales).

API publique :
  - `run_claims(avis, *, resolution, ollama_url, model, ...) -> dict`  (bout en bout)
  - `extract_claims(...)`        (étape lente, à cacher par l'appelant)
  - `cluster_claims(...)`        (étape rapide : rejouable à résolution variable)

La CO-OCCURRENCE relie deux thèmes quand un MÊME avis porte des claims tombant
dans les deux : `count` = nombre d'avis qui « pontent » les deux thèmes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from pipeline.claims.backend import resolve_backend
from pipeline.claims.extract import extract_claims
from pipeline.claims.ollama import OllamaStats
from pipeline.claims.span import Claim, as_claim
from pipeline.cluster.adaptive import derive_defaults
from pipeline.cluster.palette import color_for
from pipeline.cluster.knn import build_knn_graph
from pipeline.cluster.leiden_cluster import run_leiden
from pipeline.cluster.naming import derive_corpus_stopwords, name_clusters
from pipeline.cluster.scoring import score_cluster

DEFAULT_MODEL = "ministral-3:latest"
DEFAULT_EMBEDDER = "nomic-v2"
DEFAULT_SEED = 42
N_REPRESENTATIVE = 4


@dataclass
class Avis:
    """Un avis citoyen en entrée du pipeline (id stable + texte + poids social)."""
    id: str
    text: str
    weight: float = 1.0


@dataclass
class Theme:
    """Un thème ÉMERGENT (cluster de claims)."""
    cluster_id: int
    name: str                       # label c-TF-IDF
    keywords: list[str]
    n_claims: int
    n_avis: int
    weight: float
    consensus: float
    diversity: float
    representative_claims: list[str]
    color: str                      # couleur cluster (source unique : palette.py)

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "name": self.name,
            "keywords": self.keywords,
            "n_claims": self.n_claims,
            "n_avis": self.n_avis,
            "weight": self.weight,
            "consensus": self.consensus,
            "diversity": self.diversity,
            "representative_claims": self.representative_claims,
            "color": self.color,
        }


# --------------------------------------------------------------------------- #
# Normalisation des entrées (accepte Avis / objets .id/.text/.weight / dicts)
# --------------------------------------------------------------------------- #
def as_avis(items: list) -> list[Avis]:
    out: list[Avis] = []
    for it in items:
        if isinstance(it, Avis):
            out.append(it)
        elif isinstance(it, dict):
            text = (it.get("text") or it.get("text_clean") or "").strip()
            out.append(Avis(id=str(it.get("id") or f"idea-{len(out)}"),
                            text=text, weight=float(it.get("weight", 1.0) or 1.0)))
        else:  # objet à attributs (ex. pipeline.cluster.io.Idea)
            text = (getattr(it, "text", "") or getattr(it, "text_clean", "") or "").strip()
            out.append(Avis(id=str(getattr(it, "id", f"idea-{len(out)}")),
                            text=text, weight=float(getattr(it, "weight", 1.0) or 1.0)))
    return out


def _normalize_rows(M: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(M, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return M / nrm


def embed_claim_texts(texts: list[str], *, embedder: str = DEFAULT_EMBEDDER) -> np.ndarray:
    """Vecteurs L2-normalisés des claims (espace PROD nomic-v2, `search_document:`)."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float64)
    from pipeline.embed.embedder import Embedder
    from pipeline.embed.registry import resolve_model_id

    vecs = Embedder(model_id=resolve_model_id(embedder)).embed(texts)
    return _normalize_rows(np.asarray(vecs, dtype=np.float64))


# --------------------------------------------------------------------------- #
# Aplatissement claims (alignés à l'ordre des avis)
# --------------------------------------------------------------------------- #
def _flatten(avis: list[Avis], claims_by_id: dict[str, list]
             ) -> tuple[list[str], list[int], np.ndarray,
                        list[list[tuple[int, int]]], list[tuple[int, int] | None]]:
    """→ (claim_texts, claim_owner[idx d'avis], claim_weight, claim_spans, claim_target).

    `claims_by_id` porte des `Claim` (ou des dicts du cache / str legacy, normalisés
    via `as_claim`). `claim_texts[i]` = texte JOINT des portions (sert à l'embedding).
    `claim_spans[i]` = liste des spans verbatim `(start, end)` du claim (1..N portions ;
    `[(-1,-1)]` si non ancré). `claim_target[i]` = span de la cible verbatim, ou `None`.
    """
    texts: list[str] = []
    owner: list[int] = []
    weight: list[float] = []
    spans: list[list[tuple[int, int]]] = []
    targets: list[tuple[int, int] | None] = []
    for ai, a in enumerate(avis):
        for raw in claims_by_id.get(a.id, []):
            c = as_claim(raw, avis_text=a.text)
            texts.append(c.text)
            owner.append(ai)
            weight.append(a.weight)
            spans.append(list(c.spans))
            targets.append(c.target)
    return texts, owner, np.asarray(weight, dtype=np.float64), spans, targets


# --------------------------------------------------------------------------- #
# Étape clustering (rapide, rejouable à résolution variable)
# --------------------------------------------------------------------------- #
def _build_themes(membership: list[int], claim_vecs: np.ndarray, claim_texts: list[str],
                  claim_weight: np.ndarray, claim_owner: list[int], dup_threshold: float,
                  names: dict) -> list[Theme]:
    by_cluster: dict[int, list[int]] = {}
    for i, cid in enumerate(membership):
        by_cluster.setdefault(cid, []).append(i)
    n_clusters = len(by_cluster)

    themes: list[Theme] = []
    for cid, idx in by_cluster.items():
        sc = score_cluster(idx, claim_vecs, claim_weight, dup_threshold=dup_threshold)
        cent = np.asarray(sc.centroid, dtype=np.float64)
        sims = claim_vecs[idx] @ cent
        order = np.argsort(-sims)
        # claims représentatives = les plus centrales, sans quasi-doublon littéral.
        reps: list[str] = []
        for j in order:
            t = claim_texts[idx[j]]
            if any(t.lower() == e.lower() for e in reps):
                continue
            reps.append(t)
            if len(reps) >= N_REPRESENTATIVE:
                break
        themes.append(Theme(
            cluster_id=cid,
            name=names.get(cid, {}).get("label", f"thème {cid}"),
            keywords=names.get(cid, {}).get("keywords", []),
            n_claims=len(idx),
            n_avis=len({claim_owner[i] for i in idx}),
            weight=round(sc.weight_sum, 1),
            consensus=round(sc.consensus, 3),
            diversity=round(sc.diversity, 3),
            representative_claims=reps,
            color=color_for(cid, n_clusters),
        ))
    # tri : poids social × consensus (préoccupations partagées ET cohérentes d'abord).
    themes.sort(key=lambda t: -(t.weight * max(t.consensus, 0.0)))
    return themes


def _cooccurrence(membership: list[int], claim_owner: list[int]) -> list[dict]:
    """Liens entre thèmes : un avis dont les claims tombent dans ≥2 thèmes les ponte.

    `count` = nombre d'avis liant la paire de thèmes (force du lien). Symétrique,
    a < b, trié par count décroissant.
    """
    themes_by_avis: dict[int, set[int]] = {}
    for ci, ai in enumerate(claim_owner):
        themes_by_avis.setdefault(ai, set()).add(membership[ci])
    counts: dict[tuple[int, int], int] = {}
    for cids in themes_by_avis.values():
        for a, b in combinations(sorted(cids), 2):
            counts[(a, b)] = counts.get((a, b), 0) + 1
    out = [{"a": a, "b": b, "count": c} for (a, b), c in counts.items()]
    out.sort(key=lambda e: -e["count"])
    return out


def cluster_claims(
    avis: list,
    claims_by_id: dict[str, list],
    *,
    resolution: float = 1.0,
    seed: int = DEFAULT_SEED,
    embedder: str = DEFAULT_EMBEDDER,
    claim_vecs: np.ndarray | None = None,
) -> dict:
    """Embed (si besoin) → k-NN+Leiden → naming → thèmes + co-occurrence.

    Étape RAPIDE et déterministe : aucune extraction LLM. `claim_vecs` permet de
    réutiliser des embeddings cachés (rejouer une autre résolution sans ré-embed).
    Renvoie le dict de sortie du pipeline (themes / cooccurrence / params).
    """
    avis = as_avis(avis)
    claim_texts, claim_owner, claim_weight, _spans, _targets = _flatten(avis, claims_by_id)
    n_claims = len(claim_texts)
    n_avis = len(avis)

    if n_claims == 0:
        return {"themes": [], "cooccurrence": [],
                "params": {"resolution": resolution, "seed": seed, "embedder": embedder,
                           "n_avis": n_avis, "n_claims": 0}}

    if claim_vecs is None:
        claim_vecs = embed_claim_texts(claim_texts, embedder=embedder)
    else:
        claim_vecs = _normalize_rows(np.asarray(claim_vecs, dtype=np.float64))
    if claim_vecs.shape[0] != n_claims:
        raise ValueError(
            f"claim_vecs désaligné : {claim_vecs.shape[0]} vecteurs vs {n_claims} claims."
        )

    defaults = derive_defaults(claim_vecs.astype(np.float32))
    graph = build_knn_graph(claim_vecs, k=defaults.k, threshold=defaults.threshold)
    res = run_leiden(graph, resolution=resolution, seed=seed)
    membership = res.membership

    # naming c-TF-IDF (mots-vides dérivés du corpus de claims).
    by_cluster: dict[int, list[int]] = {}
    for i, cid in enumerate(membership):
        by_cluster.setdefault(cid, []).append(i)
    corpus_stop, _ = derive_corpus_stopwords(claim_texts)
    cluster_docs = {cid: [claim_texts[i] for i in idx] for cid, idx in by_cluster.items()}
    names = name_clusters(cluster_docs, corpus_stopwords=corpus_stop)

    themes = _build_themes(membership, claim_vecs, claim_texts, claim_weight,
                           claim_owner, defaults.dup_threshold, names)
    cooccurrence = _cooccurrence(membership, claim_owner)

    return {
        "themes": [t.to_dict() for t in themes],
        "cooccurrence": cooccurrence,
        "params": {
            "resolution": resolution,
            "seed": seed,
            "embedder": embedder,
            "n_avis": n_avis,
            "n_claims": n_claims,
            "claims_per_avis": round(n_claims / n_avis, 3) if n_avis else 0.0,
            "n_themes": res.n_clusters,
            "modularity": res.modularity,
            "derived": {
                "k": defaults.k,
                "threshold": round(defaults.threshold, 4),
                "dup_threshold": round(defaults.dup_threshold, 4),
            },
        },
    }


# --------------------------------------------------------------------------- #
# Bout en bout
# --------------------------------------------------------------------------- #
def run_claims(
    avis: list,
    *,
    resolution: float = 1.0,
    backend: str | None = None,
    ollama_url: str | None = None,
    model: str | None = None,
    embedder: str = DEFAULT_EMBEDDER,
    seed: int = DEFAULT_SEED,
    claims_by_id: dict[str, list[str]] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Pipeline complet : extraction (backend claims) → clustering émergent → carte.

    `backend` choisit le moteur d'extraction (``api`` par défaut via Mistral, ``mac``
    Ollama souverain, ``auto`` Mac→repli API ; sinon `AGORA_CLAIMS_BACKEND`). `model`
    surcharge le modèle du backend. `claims_by_id` permet de RÉUTILISER une extraction
    cachée (rejeu de résolution sans ré-extraire). Si le backend est inutilisable, lève
    `BackendUnavailable` (l'appelant renvoie une erreur HTTP claire). La sortie inclut
    `params.backend`, `params.model` et `params.cost`.
    """
    avis = as_avis(avis)
    stats = OllamaStats()

    used_backend = (backend or "?")
    used_model = model or DEFAULT_MODEL
    sovereign = None
    if claims_by_id is None:
        be = resolve_backend(backend, ollama_url=ollama_url, model=model)
        used_backend, used_model, sovereign = be.name, be.model, be.sovereign
        claims_by_id = extract_claims(avis, backend=be, stats=stats, progress=progress)

    result = cluster_claims(avis, claims_by_id, resolution=resolution, seed=seed,
                            embedder=embedder)
    result["params"]["backend"] = used_backend
    result["params"]["sovereign"] = sovereign
    result["params"]["model"] = used_model
    result["params"]["cost"] = {
        "calls": stats.calls,
        "cache_hits": stats.cache_hits,
        "errors": stats.errors,
        "cold_seconds": round(stats.cold_seconds, 2),
        "eval_tokens": stats.eval_tokens,
    }
    return result
