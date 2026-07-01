"""R&D — EXPRESSIVITÉ DE LA NORME des embeddings (vecteur BRUT vs L2-normalisé).

Concern (Bob) : le cache stocke des vecteurs L2-normalisés (norme≡1, vérifié) → on
JETTE la magnitude native de l'embedding. Perte d'expressivité ? Cette étude tranche
DEUX questions, sur les datasets à GOLD (repnum → axes du projet de loi ; xstance →
topics + stance), SANS toucher granddebat (rebuild en cours) :

  (1) La NORME native porte-t-elle un signal, ou est-ce un artefact (longueur du texte) ?
      → corrélations norme↔longueur, variance intra/inter-cluster (η²), norme↔stance.

  (2) Le CLUSTERING gagne-t-il à GARDER la norme ?
      → kNN+Leiden avec 3 métriques : COSINUS (normalisé = prod) vs PRODUIT SCALAIRE
        (brut, IP) vs EUCLIDIEN (brut, L2). Mesure : modularité + alignement gold (ARI/NMI/V).

Méthode : on ré-embedde les claims en BRUT (`embedder.embed(..., normalize=False)`) avec
le modèle de PROD (nomic-v2), une seule passe ; la version normalisée s'en DÉRIVE
(V_norm = V_raw/‖V_raw‖) — garantit que la comparaison ne porte QUE sur la magnitude.
On RÉUTILISE les briques de prod (knn/adaptive/leiden) ; le gold repnum vient de
`research/k_sweep.repnum_gold_axes`. R&D pur : aucun fichier produit modifié.

Lancer :
    uv run --extra contender --extra embed-contender --extra faiss --with fastapi \
        python research/embed_norm.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.recluster import load_cache
from pipeline.cluster.adaptive import derive_defaults, derive_k
from pipeline.cluster.knn import KnnNeighbors, build_knn_graph
from pipeline.cluster.leiden_cluster import DEFAULT_SEED, run_leiden
from pipeline.embed.embedder import DEFAULT_MODEL_ID

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
)
from scipy.stats import pearsonr, spearmanr

# repnum gold (axes officiels du projet de loi) — réutilise la jointure CSV de k_sweep.
from research.k_sweep import repnum_gold_axes

SEED = DEFAULT_SEED
SCRATCH = Path("/tmp/claude-1000/-home-bat-agora-worktrees-embed-norm/raw_embeds")
SCRATCH.mkdir(parents=True, exist_ok=True)

DATASETS = ["xstance", "republique-numerique"]


# --------------------------------------------------------------------------- #
# Ré-embed BRUT = pooled AVANT normalisation (la VRAIE magnitude native).
#
# PIÈGE CONFIRMÉ : nomic-v2 a un module `Normalize` (idx 2) DANS son pipeline
# sentence-transformers → `encode(normalize_embeddings=False)` rend QUAND MÊME des
# vecteurs unitaires (norme≡1, cv=0). Le flag d'encode est redondant avec ce module.
# La magnitude native que Bob questionne vit dans le POOLED (modules [Transformer,
# Pooling]) AVANT le Normalize. On tronque le modèle (`[:2]`) pour la récupérer.
# --------------------------------------------------------------------------- #
DOC_PREFIX = "search_document: "


def raw_embeddings(dataset: str, ideas) -> np.ndarray:
    """Pooled pré-normalisation (magnitude native), même préfixe doc que la prod."""
    cache = SCRATCH / f"{dataset}_pooled.npy"
    if cache.exists():
        v = np.load(cache).astype(np.float32)
        if v.shape[0] == len(ideas):
            print(f"  [pooled] cache hit {cache} {v.shape}", flush=True)
            return v
    print(f"  [pooled] ré-embed {len(ideas)} claims (PRÉ-normalisation, {DEFAULT_MODEL_ID}) …",
          flush=True)
    from sentence_transformers import SentenceTransformer

    full = SentenceTransformer(DEFAULT_MODEL_ID, device="cpu", trust_remote_code=True)
    # Tronque le Normalize : ne garde que [Transformer, Pooling] → pooled brut.
    trunc = SentenceTransformer(modules=[full[0], full[1]], device="cpu")
    texts = [f"{DOC_PREFIX}{idea.text_clean or idea.text}" for idea in ideas]
    v = trunc.encode(texts, batch_size=32, convert_to_numpy=True,
                     normalize_embeddings=False, show_progress_bar=False).astype(np.float32)
    np.save(cache, v)
    print(f"  [pooled] sauvé → {cache} {v.shape}", flush=True)
    return v


# --------------------------------------------------------------------------- #
# η² (rapport de corrélation) : part de variance de `x` expliquée par les groupes.
# --------------------------------------------------------------------------- #
def eta_squared(x: np.ndarray, labels: list) -> dict:
    x = np.asarray(x, dtype=np.float64)
    groups: dict = {}
    for v, lab in zip(x, labels):
        if lab is None:
            continue
        groups.setdefault(lab, []).append(v)
    vals = np.concatenate([np.array(g) for g in groups.values()]) if groups else x
    if vals.size < 2:
        return {"eta2": None}
    grand = vals.mean()
    ss_total = float(((vals - grand) ** 2).sum())
    ss_between = float(sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups.values()))
    eta2 = ss_between / ss_total if ss_total > 0 else 0.0
    return {
        "eta2": round(eta2, 4),
        "n_groups": len(groups),
        "n": int(vals.size),
        "intra_std_mean": round(float(np.mean([np.std(g) for g in groups.values()])), 4),
        "inter_std": round(float(np.std([np.mean(g) for g in groups.values()])), 4),
    }


# --------------------------------------------------------------------------- #
# kNN par MÉTRIQUE → affinité (haut = proche), réutilisable par build_knn_graph.
# --------------------------------------------------------------------------- #
def metric_neighbors(V: np.ndarray, k: int, metric: str) -> KnnNeighbors:
    """Voisinage k-NN exact (faiss) par métrique, exprimé en AFFINITÉ (haut=proche).

      - cosine    : V normalisé, IndexFlatIP → produit scalaire == cosinus ∈ [-1,1].
      - ip        : V BRUT,      IndexFlatIP → produit scalaire brut (norme-pondéré).
      - euclidean : V BRUT,      IndexFlatL2 → distance L2 ; affinité = 1/(1+d).

    L'affinité sert À LA FOIS de pool pour le seuil dérivé (μ−3.2σ) et de poids
    d'arête Leiden — exactement comme le cosinus en prod. Comparaison apples-to-apples :
    même k, même chaîne (derive_defaults → build_knn_graph → run_leiden) pour les 3.
    """
    import faiss

    V = np.ascontiguousarray(V, dtype=np.float32)
    d = V.shape[1]
    kk = min(k + 1, V.shape[0])
    if metric in ("cosine", "ip"):
        index = faiss.IndexFlatIP(d)
        index.add(V)
        sims, idx = index.search(V, kk)
        affinity = sims.astype(np.float32)
    elif metric == "euclidean":
        index = faiss.IndexFlatL2(d)
        index.add(V)
        d2, idx = index.search(V, kk)
        dist = np.sqrt(np.maximum(d2, 0.0))
        affinity = (1.0 / (1.0 + dist)).astype(np.float32)
    else:
        raise ValueError(metric)
    return KnnNeighbors(sims=affinity, idx=idx.astype(np.int64), backend="faiss")


def cluster_with_metric(V: np.ndarray, k: int, metric: str):
    """Chaîne de prod (seuil dérivé → graphe → Leiden racine) pour une métrique."""
    neigh = metric_neighbors(V, k, metric)
    derived = derive_defaults(V, k=k, neighbors=neigh)
    graph = build_knn_graph(
        np.ascontiguousarray(V, dtype=np.float64), k=k,
        threshold=float(derived.threshold), neighbors=neigh,
    )
    res = run_leiden(graph, resolution=1.0, seed=SEED)
    return {
        "metric": metric,
        "threshold": round(float(derived.threshold), 4),
        "avg_degree": round(graph.avg_degree, 2),
        "n_clusters": res.n_clusters,
        "modularity": round(res.modularity, 4),
        "membership": res.membership,
    }


def alignment(pred: list, gold: list) -> dict | None:
    pairs = [(p, g) for p, g in zip(pred, gold) if p is not None and g is not None]
    if len(pairs) < 2:
        return None
    pr = [p for p, _ in pairs]
    gl = [g for _, g in pairs]
    return {
        "n": len(pairs),
        "ari": round(adjusted_rand_score(gl, pr), 4),
        "nmi": round(normalized_mutual_info_score(gl, pr), 4),
        "v": round(v_measure_score(gl, pr), 4),
    }


# --------------------------------------------------------------------------- #
def study(dataset: str) -> dict:
    print(f"\n=== {dataset} ===", flush=True)
    ideas, vecs_cached, _ = load_cache(dataset)
    n = len(ideas)
    k = derive_k(n)
    print(f"  N={n}  derive_k={k}", flush=True)

    V_raw = raw_embeddings(dataset, ideas)
    norms = np.linalg.norm(V_raw, axis=1)
    V_norm = V_raw / np.where(norms[:, None] > 0, norms[:, None], 1.0)

    # Sanity : le cache normalisé == notre normalisé dérivé du brut ? (même modèle/préfixe)
    cos_to_cache = float(np.mean(np.sum(V_norm * vecs_cached, axis=1)))
    print(f"  norme brute  min={norms.min():.3f} med={np.median(norms):.3f} "
          f"max={norms.max():.3f} cv={norms.std()/norms.mean():.3f}", flush=True)
    print(f"  cos(V_norm, cache)  μ={cos_to_cache:.4f}  (≈1 ⇒ cache = V_raw L2-normalisé)", flush=True)

    # ---- Gold (clustering target) ----
    if dataset == "xstance":
        # Idea.from_row ne remonte pas props.topic/label → relire le jsonl.
        gold, stance = _xstance_labels(dataset, n)
    else:
        gold = repnum_gold_axes(ideas)
        stance = None
    cov = sum(g is not None for g in gold)
    print(f"  gold couverture : {cov}/{n}", flush=True)

    # ===================================================================== #
    # (1) La NORME porte-t-elle un signal ?
    # ===================================================================== #
    char_len = np.array([len((i.text_clean or i.text)) for i in ideas], dtype=np.float64)
    tok_len = np.array([len((i.text_clean or i.text).split()) for i in ideas], dtype=np.float64)

    # Cluster d'appartenance = partition COSINUS racine (proxy prod) pour l'η² norme/cluster.
    cos_clusters = cluster_with_metric(V_norm, k, "cosine")
    clu = cos_clusters["membership"]

    signal = {
        "norm_stats": {
            "min": round(float(norms.min()), 4), "max": round(float(norms.max()), 4),
            "mean": round(float(norms.mean()), 4), "std": round(float(norms.std()), 4),
            "cv": round(float(norms.std() / norms.mean()), 4),
        },
        "corr_norm_charlen": {
            "pearson": round(float(pearsonr(norms, char_len)[0]), 4),
            "spearman": round(float(spearmanr(norms, char_len)[0]), 4),
        },
        "corr_norm_toklen": {
            "pearson": round(float(pearsonr(norms, tok_len)[0]), 4),
            "spearman": round(float(spearmanr(norms, tok_len)[0]), 4),
        },
        "eta2_norm_by_cluster": eta_squared(norms, clu),
        "eta2_norm_by_gold": eta_squared(norms, gold),
    }
    if stance is not None:
        signal["eta2_norm_by_stance"] = eta_squared(norms, stance)
        # moyennes de norme par stance (FAVOR vs AGAINST)
        by = {}
        for v, s in zip(norms, stance):
            if s:
                by.setdefault(s, []).append(v)
        signal["norm_by_stance_mean"] = {k2: round(float(np.mean(v)), 4) for k2, v in by.items()}

    print(f"  [signal] corr(norm,charlen) pearson={signal['corr_norm_charlen']['pearson']} "
          f"spearman={signal['corr_norm_charlen']['spearman']}", flush=True)
    print(f"  [signal] η²(norm|cluster)={signal['eta2_norm_by_cluster']['eta2']} "
          f"η²(norm|gold)={signal['eta2_norm_by_gold']['eta2']}", flush=True)
    if stance is not None:
        print(f"  [signal] η²(norm|stance)={signal['eta2_norm_by_stance']['eta2']} "
              f"means={signal['norm_by_stance_mean']}", flush=True)

    # ===================================================================== #
    # (2) Le CLUSTERING gagne-t-il à garder la norme ? (3 métriques)
    # ===================================================================== #
    metric_rows = []
    for metric, V in (("cosine", V_norm), ("ip", V_raw), ("euclidean", V_raw)):
        r = cluster_with_metric(V, k, metric)
        al = alignment(r["membership"], gold)
        row = {kk: r[kk] for kk in ("metric", "threshold", "avg_degree", "n_clusters", "modularity")}
        row["align"] = al
        metric_rows.append(row)
        ex = (f"  ARI={al['ari']:.3f} NMI={al['nmi']:.3f} V={al['v']:.3f}" if al else "")
        print(f"  [{metric:9}] thr={row['threshold']:.4f} deg={row['avg_degree']:5.1f} "
              f"clusters={row['n_clusters']:>3} Q={row['modularity']:.4f}{ex}", flush=True)

    return {
        "dataset": dataset, "n": n, "k": k,
        "cos_to_cache": round(cos_to_cache, 4),
        "signal": signal,
        "metrics": metric_rows,
    }


def _xstance_labels(dataset: str, n: int):
    """(topic, stance) par idée alignés à l'ordre du cache (relit le jsonl props)."""
    from backend.recluster import cache_paths
    _, ideas_path, _ = cache_paths(dataset)
    topics, stances = [], []
    with open(ideas_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line).get("props", {})
            topics.append(p.get("topic"))
            stances.append(p.get("label"))
    return topics, stances


def main():
    results = [study(ds) for ds in DATASETS]
    out = Path(__file__).resolve().parent / "embed_norm_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRésultats → {out}", flush=True)

    print("\n=== RÉSUMÉ (métrique gagnante par gold ARI) ===", flush=True)
    for r in results:
        best = max(r["metrics"], key=lambda m: (m["align"]["ari"] if m["align"] else -1))
        cos = next(m for m in r["metrics"] if m["metric"] == "cosine")
        print(f"  {r['dataset']:>22}: cosine ARI={cos['align']['ari']:.3f} "
              f"Q={cos['modularity']:.3f} | best={best['metric']} "
              f"ARI={best['align']['ari']:.3f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
