"""Métriques d'arbitrage : clustering prédit vs labels FAVOR/AGAINST.

- **NMI** / **ARI** : accord entre les clusters prédits et la vérité terrain
  (invariants au renommage des clusters). NMI∈[0,1], ARI∈[-1,1] (0 = hasard).
- **pureté** : fraction de points bien classés si on étiquette chaque cluster
  par sa classe majoritaire. ∈[0,1], monte trivialement quand il y a beaucoup
  de clusters → à lire AVEC le nombre de clusters.
- **silhouette** : qualité INTERNE (séparation) des clusters prédits dans
  l'espace d'embedding (cosine). Indépendante des labels. None si < 2 clusters.

Le bruit HDBSCAN (label -1) est traité comme un cluster à part entière pour
NMI/ARI/pureté (honnête : c'est une décision du clustering), et exclu de la
silhouette (les points -1 ne forment pas un groupe cohérent).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def purity(pred: list[int], truth: list[int]) -> float:
    """Pureté = Σ_cluster max_classe(|cluster ∩ classe|) / N."""
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    n = len(truth)
    if n == 0:
        return 0.0
    total = 0
    for c in set(pred.tolist()):
        mask = pred == c
        classes, counts = np.unique(truth[mask], return_counts=True)
        total += int(counts.max())
    return total / n


def silhouette(vecs: np.ndarray, pred: list[int], exclude_noise: bool = True):
    """Silhouette cosine des clusters prédits ; None si non calculable.

    Nécessite ≥ 2 clusters distincts et ≥ 2 points par configuration valide.
    """
    pred = np.asarray(pred)
    X = vecs
    if exclude_noise:
        keep = pred != -1
        if keep.sum() < 2:
            return None
        X = vecs[keep]
        pred = pred[keep]
    labels = set(pred.tolist())
    if len(labels) < 2 or len(pred) <= len(labels):
        return None
    try:
        return float(silhouette_score(X, pred, metric="cosine"))
    except ValueError:
        return None


def score_against_labels(
    pred: list[int], truth: list[int], vecs: np.ndarray
) -> dict:
    """Toutes les métriques d'une exécution de clustering vs labels."""
    n_clusters = len(set(c for c in pred if c != -1))
    n_noise = sum(1 for c in pred if c == -1)
    return {
        "nmi": float(normalized_mutual_info_score(truth, pred)),
        "ari": float(adjusted_rand_score(truth, pred)),
        "purity": purity(pred, truth),
        "silhouette": silhouette(vecs, pred),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
    }


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    """Moyenne ± écart-type en ignorant les None ; (None, None) si vide."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    arr = np.asarray(vals, dtype=float)
    return float(arr.mean()), float(arr.std())
