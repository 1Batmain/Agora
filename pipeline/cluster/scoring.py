"""T-N5 · Scoring des thèmes.

Pour chaque communauté :
  - `weight_sum`  : somme des poids sociaux des avis du thème.
  - `diversity`   : 1 − densité de duplicats. Densité dup = fraction de paires
                    de membres quasi-identiques (cosine > `dup_threshold`).
                    1.0 = aucune redondance littérale ; 0.0 = tout est dupliqué.
  - `consensus`   : cohérence sémantique intra-thème = cosinus moyen des paires.
                    Haut = même intention. Couplé à `diversity` haut → "même
                    intention, formulations variées".
  - `centroid`    : barycentre L2-normalisé (pour assignation live ultérieure).

Tout est déterministe (pas d'aléatoire) → reproductible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DUP_THRESHOLD = 0.93


@dataclass
class ClusterScore:
    weight_sum: float
    diversity: float
    consensus: float
    centroid: list[float]
    size: int


def _pairwise_cos(sub: np.ndarray) -> np.ndarray:
    """Matrice de cosinus (vecteurs supposés L2-normalisés)."""
    return sub @ sub.T


def score_cluster(
    member_idx: list[int],
    vecs: np.ndarray,
    weights: np.ndarray,
    dup_threshold: float = DUP_THRESHOLD,
) -> ClusterScore:
    size = len(member_idx)
    sub = vecs[member_idx]
    w = weights[member_idx]
    weight_sum = float(w.sum())

    centroid = sub.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm

    if size <= 1:
        return ClusterScore(
            weight_sum=round(weight_sum, 4),
            diversity=1.0,
            consensus=1.0 if size == 1 else 0.0,
            centroid=[round(float(x), 6) for x in centroid],
            size=size,
        )

    sim = _pairwise_cos(sub)
    iu = np.triu_indices(size, k=1)
    pair_sims = sim[iu]
    n_pairs = pair_sims.size

    dup_density = float((pair_sims > dup_threshold).sum()) / n_pairs
    diversity = 1.0 - dup_density
    consensus = float(pair_sims.mean())

    return ClusterScore(
        weight_sum=round(weight_sum, 4),
        diversity=round(diversity, 4),
        consensus=round(consensus, 4),
        centroid=[round(float(x), 6) for x in centroid],
        size=size,
    )


def rank_clusters(scores: dict[int, ClusterScore]) -> list[int]:
    """Classement lisible : une idée minoritaire mais forte > bruit majoritaire.

    Score d'intérêt = weight_sum pondéré par la qualité (consensus × diversity),
    de sorte qu'un thème cohérent et non redondant remonte même s'il est petit.
    """
    def key(cid: int) -> float:
        s = scores[cid]
        quality = max(0.0, s.consensus) * max(0.0, s.diversity)
        return s.weight_sum * (0.5 + quality)

    return sorted(scores.keys(), key=key, reverse=True)
