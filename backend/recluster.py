"""Re-clustering LIVE sur les embeddings CACHÉS (jamais ré-embeddés).

Cœur du serveur :8010. À partir des vecteurs nomic-v2 en cache, applique la
chaîne du contrat — `min_chars` → `dedup` → k-NN(`k`,`threshold`) → Leiden
**hiérarchique** (macro/sub) → scoring → naming TF-IDF → **GraphPayload
hiérarchique** — en RÉUTILISANT `pipeline.cluster.*`. Aucun appel au modèle torch.

Le payload a la même shape que `data/graph.json` (`meta, nodes, links, themes`),
augmenté de `meta.stats { n_macros, n_subs, n_nodes, modularity, took_ms }`.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from pipeline.cluster.build import _build_hierarchical
from pipeline.cluster.dedup import dedup_near
from pipeline.cluster.io import Idea
from pipeline.cluster.knn import build_knn_graph

SEED = 42

CACHE_DIR = Path(__file__).resolve().parent / "cache"
EMB_PATH = CACHE_DIR / "embeddings.npy"
IDEAS_PATH = CACHE_DIR / "ideas.jsonl"

MODEL_ID = "nomic-ai/nomic-embed-text-v2-moe"


def load_cache() -> tuple[list[Idea], np.ndarray, np.ndarray]:
    """Charge le superset caché (vecteurs + ideas alignés). Aucun torch."""
    if not EMB_PATH.exists() or not IDEAS_PATH.exists():
        raise RuntimeError(
            f"Cache absent ({EMB_PATH}). Lance d'abord :\n"
            "  uv run --extra embed-contender python -m backend.build_cache"
        )
    vecs = np.load(EMB_PATH).astype(np.float32)
    ideas: list[Idea] = []
    with open(IDEAS_PATH, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if line:
                ideas.append(Idea.from_row(json.loads(line), i))
    if len(ideas) != vecs.shape[0]:
        raise RuntimeError(
            f"Cache désaligné : {len(ideas)} ideas vs {vecs.shape[0]} vecteurs."
        )
    weights = np.array([idea.weight for idea in ideas], dtype=np.float32)
    return ideas, vecs, weights


def recluster(
    ideas: list,
    vecs: np.ndarray,
    weights: np.ndarray,
    *,
    dedup: float | None = 0.95,
    min_chars: int = 12,
    k: int = 12,
    threshold: float = 0.60,
    resolution_macro: float = 1.0,
    resolution_sub: float = 1.5,
    min_sub_size: int = 18,
    seed: int = SEED,
) -> dict:
    """Re-clusterise le superset caché et renvoie un GraphPayload hiérarchique.

    `ideas`/`vecs`/`weights` sont le superset CACHÉ aligné. Les filtres
    `min_chars`/`dedup` réduisent ce set (sans ré-embedder).
    """
    t0 = perf_counter()
    n_cached = len(ideas)

    # 1) Filtre des avis trop courts (sur le set caché).
    if min_chars:
        keep = [
            i for i, idea in enumerate(ideas)
            if len((idea.text_clean or idea.text).strip()) >= min_chars
        ]
        ideas = [ideas[i] for i in keep]
        vecs = np.ascontiguousarray(vecs[keep])
        weights = weights[keep]
    n_after_minlen = len(ideas)

    if len(ideas) == 0:
        raise ValueError("Aucun avis après filtrage (min_chars trop élevé ?).")

    # 2) Déduplication near-dup (cumule le poids, ne perd aucune voix).
    dedup_meta = None
    if dedup is not None:
        dd = dedup_near(vecs, weights, threshold=dedup)
        ideas = [ideas[i] for i in dd.keep]
        vecs = np.ascontiguousarray(vecs[dd.keep])
        weights = dd.weights
        dedup_meta = {
            "threshold": dedup,
            "n_in": dd.n_in,
            "n_out": dd.n_out,
            "n_collapsed": dd.n_collapsed,
        }

    # 3) Graphe k-NN cosine.
    knn = build_knn_graph(vecs, k=k, threshold=threshold)

    # 4-6) Leiden hiérarchique + scoring + naming + nœuds (réutilisé du pipeline).
    nodes, themes, clustering_meta = _build_hierarchical(
        ideas, vecs, weights, knn,
        resolution_macro=resolution_macro,
        resolution_sub=resolution_sub,
        min_sub_size=min_sub_size,
        seed=seed,
    )

    id_by_idx = [idea.id for idea in ideas]
    links = [
        {
            "source": id_by_idx[i],
            "target": id_by_idx[j],
            "type": "knn",
            "props": {"weight": round(float(w), 4)},
        }
        for (i, j, w) in knn.edges
    ]

    took_ms = round((perf_counter() - t0) * 1000)
    lh = clustering_meta["leiden_hierarchy"]
    stats = {
        "n_macros": lh["n_macros"],
        "n_subs": lh["n_leaves"],
        "n_nodes": len(nodes),
        "modularity": lh["macro_modularity"],
        "took_ms": took_ms,
    }

    return {
        "meta": {
            "model_id": MODEL_ID,
            "embedding_dim": int(vecs.shape[1]),
            "n_nodes": len(nodes),
            "n_links": len(links),
            "n_themes": len(themes),
            "subset": {
                "n_cached": n_cached,
                "min_chars": min_chars,
                "n_after_minlen": n_after_minlen,
            },
            "dedup": dedup_meta,
            "params": {
                "dedup": dedup,
                "min_chars": min_chars,
                "k": k,
                "threshold": threshold,
                "resolution_macro": resolution_macro,
                "resolution_sub": resolution_sub,
                "min_sub_size": min_sub_size,
                "seed": seed,
                "knn_backend": knn.backend,
                "avg_degree": round(knn.avg_degree, 3),
            },
            "clustering": clustering_meta,
            "stats": stats,
        },
        "nodes": nodes,
        "links": links,
        "themes": themes,
    }
