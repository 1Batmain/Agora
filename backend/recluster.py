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
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from pipeline.cluster.adaptive import EDGE_SIGMA, derive_defaults
from pipeline.cluster.build import _build_hierarchical
from pipeline.cluster.dedup import dedup_near
from pipeline.cluster.io import Idea
from pipeline.cluster.knn import build_knn_graph

SEED = 42

# Cache MULTI-DATASET : `backend/cache/<dataset>/{embeddings.npy, ideas.jsonl,
# meta.json}`. Un dataset = un sous-dossier (aucun nom de corpus codé en dur ;
# les datasets sont DÉCOUVERTS en scannant le dossier). Défaut rétro-compat =
# "tiktok".
CACHE_DIR = Path(__file__).resolve().parent / "cache"
DEFAULT_DATASET = "tiktok"

MODEL_ID = "nomic-ai/nomic-embed-text-v2-moe"

EMB_NAME = "embeddings.npy"
IDEAS_NAME = "ideas.jsonl"
META_NAME = "meta.json"


def dataset_dir(dataset: str) -> Path:
    return CACHE_DIR / dataset


def cache_paths(dataset: str) -> tuple[Path, Path, Path]:
    d = dataset_dir(dataset)
    return d / EMB_NAME, d / IDEAS_NAME, d / META_NAME


def list_datasets() -> list[str]:
    """Datasets disponibles = sous-dossiers de cache/ avec un cache complet.

    Découverte pure (zéro littéral de corpus) : on liste les dossiers qui
    contiennent à la fois `embeddings.npy` et `ideas.jsonl`. Triés avec le défaut
    (`tiktok`) en tête pour la rétro-compat de l'UI.
    """
    if not CACHE_DIR.exists():
        return []
    found = [
        p.name for p in CACHE_DIR.iterdir()
        if p.is_dir() and (p / EMB_NAME).exists() and (p / IDEAS_NAME).exists()
    ]
    found.sort(key=lambda n: (n != DEFAULT_DATASET, n))
    return found


def load_cache(dataset: str = DEFAULT_DATASET) -> tuple[list[Idea], np.ndarray, np.ndarray]:
    """Charge le cache d'UN dataset (vecteurs + ideas alignés). Aucun torch."""
    emb_path, ideas_path, _ = cache_paths(dataset)
    if not emb_path.exists() or not ideas_path.exists():
        raise RuntimeError(
            f"Cache absent ({emb_path}). Construis-le d'abord :\n"
            f"  uv run --extra embed-contender python -m backend.build_cache --dataset {dataset}"
        )
    vecs = np.load(emb_path).astype(np.float32)
    ideas: list[Idea] = []
    with open(ideas_path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if line:
                ideas.append(Idea.from_row(json.loads(line), i))
    if len(ideas) != vecs.shape[0]:
        raise RuntimeError(
            f"Cache désaligné ({dataset}) : {len(ideas)} ideas vs {vecs.shape[0]} vecteurs."
        )
    weights = np.array([idea.weight for idea in ideas], dtype=np.float32)
    return ideas, vecs, weights


def dataset_descriptor(dataset: str, ideas: list[Idea] | None = None) -> dict:
    """Métadonnées d'un dataset pour `GET /datasets`.

    Lit `meta.json` s'il existe (écrit par build_cache), sinon DÉRIVE tout des
    `ideas` cachés (langues, n, source). Générique : aucune valeur en dur.
    """
    _, ideas_path, meta_path = cache_paths(dataset)
    meta: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            meta = {}

    if ideas is None and (not meta or "languages" not in meta or "n_nodes" not in meta):
        ideas = []
        if ideas_path.exists():
            with open(ideas_path, "r", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    line = line.strip()
                    if line:
                        ideas.append(Idea.from_row(json.loads(line), i))

    if ideas is not None:
        lang_counts = Counter(idea.lang for idea in ideas if idea.lang)
        src_counts = Counter(idea.source for idea in ideas if idea.source)
        derived = {
            "n_nodes": len(ideas),
            "languages": [lg for lg, _ in lang_counts.most_common()],
            "lang_counts": dict(lang_counts.most_common()),
            "source": src_counts.most_common(1)[0][0] if src_counts else dataset,
        }
    else:
        derived = {}

    return {
        "id": dataset,
        "label": meta.get("label", dataset),
        "n_nodes": meta.get("n_nodes", derived.get("n_nodes", 0)),
        "languages": meta.get("languages", derived.get("languages", [])),
        "lang_counts": meta.get("lang_counts", derived.get("lang_counts", {})),
        "source": meta.get("source", derived.get("source", dataset)),
    }


def recluster(
    ideas: list,
    vecs: np.ndarray,
    weights: np.ndarray,
    *,
    dedup: float | None = 0.95,
    min_chars: int = 12,
    k: int | None = None,
    threshold: float | None = None,
    resolution_macro: float = 1.0,
    resolution_sub: float = 1.5,
    min_sub_size: int | None = None,
    dup_threshold: float | None = None,
    seed: int = SEED,
    dataset: str = DEFAULT_DATASET,
) -> dict:
    """Re-clusterise le superset caché et renvoie un GraphPayload hiérarchique.

    `ideas`/`vecs`/`weights` sont le superset CACHÉ aligné. Les filtres
    `min_chars`/`dedup` réduisent ce set (sans ré-embedder).

    `k`, `threshold`, `min_sub_size`, `dup_threshold` valant ``None`` sont
    **dérivés des données** (audit #6/#7/#9) sur le set filtré/dédupliqué — aucun
    ré-embed : on réutilise les vecteurs cachés. Une valeur explicite la force.
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

    # 2.5) Défauts DÉRIVÉS des données (audit #6/#7/#9) sur le set dédupliqué.
    #      Aucun ré-embed : la distribution des cosinus k-NN sort des vecteurs
    #      cachés. Un knob explicite (non-None) est toujours respecté.
    derived = derive_defaults(vecs, k=k)
    k = k if k is not None else derived.k
    if threshold is None:
        threshold = derived.threshold
    if min_sub_size is None:
        min_sub_size = derived.min_sub_size
    # near-dup (diversity) DÉRIVÉ de la distribution (p98) — volontairement SOUS
    # le seuil `dedup` : dedup a déjà collapsé les paires > dedup, donc diversity
    # mesure les quasi-doublons RÉSIDUELS juste en dessous (sinon diversity≈1).
    if dup_threshold is None:
        dup_threshold = derived.dup_threshold

    # 3) Graphe k-NN cosine.
    knn = build_knn_graph(vecs, k=k, threshold=threshold)

    # 4-6) Leiden hiérarchique + scoring + naming + nœuds (réutilisé du pipeline).
    nodes, themes, clustering_meta = _build_hierarchical(
        ideas, vecs, weights, knn,
        resolution_macro=resolution_macro,
        resolution_sub=resolution_sub,
        min_sub_size=min_sub_size,
        seed=seed,
        dup_threshold=dup_threshold,
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
            "dataset": dataset,
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
                "threshold": round(float(threshold), 4),
                "resolution_macro": resolution_macro,
                "resolution_sub": resolution_sub,
                "min_sub_size": min_sub_size,
                "dup_threshold": round(float(dup_threshold), 4),
                "seed": seed,
                "knn_backend": knn.backend,
                "avg_degree": round(knn.avg_degree, 3),
            },
            # Défauts DÉRIVÉS effectivement calculés (traçabilité audit #6/#7/#9).
            "derived": {
                "k": derived.k,
                "threshold": round(derived.threshold, 4),
                "min_sub_size": derived.min_sub_size,
                "dup_threshold": round(derived.dup_threshold, 4),
                "knn_sim_mean": derived.pool_mean,
                "knn_sim_std": derived.pool_std,
                "edge_sigma": EDGE_SIGMA,
            },
            "clustering": clustering_meta,
            "stats": stats,
        },
        "nodes": nodes,
        "links": links,
        "themes": themes,
    }
