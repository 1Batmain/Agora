"""Défauts DÉRIVÉS des données — zéro magic-number corpus-spécifique.

Les défauts du graphe sémantique (`k`, seuil d'arête, `min_sub_size`, seuil
near-dup) ne sont PAS des constantes calées sur un corpus : ils se **dérivent**
de la distribution observée (similarités k-NN) ou de la taille `N`. Ne subsistent
que des hyper-paramètres de **FORME** — sans unité corpus-spécifique :

  - `K_LOG_COEF`     : densité du voisinage,   k ∝ log10(N) (borné).
  - `EDGE_SIGMA`     : seuil d'arête = μ − σ·EDGE_SIGMA de la distribution des
                       cosinus k-NN (plancher d'aberrations gaussien).
  - `MIN_SUB_FRAC`   : `min_sub_size` relatif à N (plancher absolu petit).
  - `DUP_PERCENTILE` : near-dup = haut percentile des cosinus k-NN.

Ces formes sont langue / corpus / modèle-agnostiques ; la VALEUR obtenue s'adapte
au **modèle** (e5 « chaud » → similarités hautes → seuil haut ; nomic plus froid →
seuil bas) et à la **densité** du corpus (clusters serrés → σ petit → seuil proche
de la moyenne). C'est exactement ce que le seuil cosine fixe (0.84/0.80/0.60,
incohérent entre modules) ne pouvait pas faire.

Calibration de référence — corpus TikTok FR, nomic-v2 (cache backend), après les
filtres live (`min_chars=12`, `dedup=0.95`, N=1597) :

    derive_k(1597)              = 12
    μ=0.766  σ=0.052
    derive_threshold            = 0.602   (μ − 3.2·σ)
    derive_min_sub_size(1597)   = 18
    derive_dup_threshold        ≈ 0.869   (p98)

→ 8 macros / 47 sous-thèmes, modularité 0.60 : IDENTIQUE au réglage manuel gelé
que ces défauts remplacent (non-régression prouvée, cf. ADAPTIVE_NOTE.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --- Hyper-paramètres de FORME (sans unité corpus-spécifique) -------------- #
K_LOG_COEF = 3.8          # k ≈ round(K_LOG_COEF · log10(N)), borné [K_MIN, K_MAX]
K_MIN, K_MAX = 8, 30
EDGE_SIGMA = 3.2          # seuil = μ − EDGE_SIGMA · σ des cosinus k-NN
MIN_SUB_FRAC = 0.011      # min_sub_size = max(MIN_SUB_FLOOR, round(frac · N))
MIN_SUB_FLOOR = 5
DUP_PERCENTILE = 98.0     # near-dup (diversity) = ce percentile des cosinus k-NN
DUP_MIN, DUP_MAX = 0.80, 0.995


@dataclass
class DerivedDefaults:
    """Défauts dérivés d'un set de vecteurs (L2-normalisés) + traçabilité."""
    n: int
    k: int
    threshold: float
    dup_threshold: float
    min_sub_size: int
    pool_mean: float | None
    pool_std: float | None


def derive_k(n: int) -> int:
    """`k` (voisins k-NN) ∝ log10(N), borné. Réconcilie build/knn/backend.

    Petit corpus → peu de voisins (sinon on relie tout) ; gros corpus → plus de
    voisins, mais plafonné (coût quadratique du tri). Sur le corpus de référence
    (N=1597) retombe sur **12**, le défaut gelé du contrat.
    """
    if n <= 2:
        return max(1, n - 1)
    return int(min(K_MAX, max(K_MIN, round(K_LOG_COEF * math.log10(n)))))


def knn_sim_pool(vecs: np.ndarray, k: int, block: int = 512) -> np.ndarray:
    """Pool des cosinus des `k` plus proches voisins de chaque nœud (self exclu).

    Calcul par blocs de lignes (sobre en RAM). Vecteurs supposés L2-normalisés
    (cosine = produit scalaire). C'est la distribution dont on dérive le seuil —
    aucun ré-embed : on réutilise les vecteurs déjà calculés (cache backend).
    """
    n = vecs.shape[0]
    if n <= 1:
        return np.empty(0, dtype=np.float32)
    kk = min(k, n - 1)
    out: list[np.ndarray] = []
    for s in range(0, n, block):
        e = min(s + block, n)
        sim = vecs[s:e] @ vecs.T
        for r in range(e - s):
            i = s + r
            row = sim[r].copy()
            row[i] = -2.0  # exclut le self-match (cosine 1.0)
            idx = np.argpartition(row, -kk)[-kk:]
            out.append(row[idx])
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


def pool_from_neighbors(sims: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Pool des cosinus k-NN (self exclu) À PARTIR d'un voisinage déjà calculé.

    `sims`/`idx` = sortie de `knn_search` (top-(k+1), self inclus). On retire, par
    ligne, l'unique entrée `idx == ligne` (self) → restent les `k` cosinus voisins,
    MÊME multiset que `knn_sim_pool` (les `k` plus grands non-self). Dérive le seuil
    SANS 2ᵉ passe O(n²) — voisinage déjà calculé pour le graphe."""
    n = sims.shape[0]
    self_mask = idx == np.arange(n)[:, None]
    return sims[~self_mask].astype(np.float32, copy=False)


def derive_threshold(vecs: np.ndarray, k: int,
                     pool: np.ndarray | None = None) -> float:
    """Seuil d'arête = μ − EDGE_SIGMA·σ de la distribution des cosinus k-NN.

    Plancher d'aberrations : on coupe les arêtes dont le cosinus est anormalement
    bas par rapport au voisinage typique. S'adapte au modèle (μ haut → seuil haut)
    et à la densité (σ petit → seuil proche de μ). Borné dans [0, 1).
    """
    if pool is None:
        pool = knn_sim_pool(vecs, k)
    if pool.size == 0:
        return 0.0
    mean = float(pool.mean())
    std = float(pool.std())
    return float(min(0.999, max(0.0, mean - EDGE_SIGMA * std)))


def derive_min_sub_size(n: int) -> int:
    """Taille mini d'un sous-thème, RELATIVE à N (sinon écrase le petit corpus,
    laisse de la poussière sur le gros). `max(floor, round(frac·N))`."""
    return int(max(MIN_SUB_FLOOR, round(MIN_SUB_FRAC * n)))


def derive_dup_threshold(vecs: np.ndarray, k: int,
                         pool: np.ndarray | None = None) -> float:
    """Seuil near-dup (pour `diversity`) = haut percentile des cosinus k-NN.

    « Quasi-doublon » = similarité dans le haut de la distribution observée —
    notion relative au modèle, pas un 0.93 figé. Borné [DUP_MIN, DUP_MAX].
    """
    if pool is None:
        pool = knn_sim_pool(vecs, k)
    if pool.size == 0:
        return DUP_MAX
    return float(min(DUP_MAX, max(DUP_MIN, float(np.percentile(pool, DUP_PERCENTILE)))))


def derive_defaults(vecs: np.ndarray, *, k: int | None = None,
                    neighbors=None) -> DerivedDefaults:
    """Dérive en UNE passe (un seul pool de cosinus) tous les défauts du graphe.

    Si `k` est fourni, on le respecte (et on dérive le reste cohéremment) ; sinon
    `k` lui-même est dérivé de N. C'est le point d'entrée central — build et
    backend l'appellent pour rester cohérents.

    `neighbors` (optionnel, `KnnNeighbors` de `knn_search`) = voisinage k-NN DÉJÀ
    calculé : le pool des cosinus en est extrait (`pool_from_neighbors`) au lieu d'une
    2ᵉ passe dense O(n²). À fournir UNIQUEMENT si calculé avec le même `k_eff`.
    """
    n = vecs.shape[0]
    k_eff = derive_k(n) if k is None else int(k)
    if neighbors is not None:
        pool = pool_from_neighbors(neighbors.sims, neighbors.idx)
    else:
        pool = knn_sim_pool(vecs, k_eff)
    return DerivedDefaults(
        n=n,
        k=k_eff,
        threshold=derive_threshold(vecs, k_eff, pool=pool),
        dup_threshold=derive_dup_threshold(vecs, k_eff, pool=pool),
        min_sub_size=derive_min_sub_size(n),
        pool_mean=round(float(pool.mean()), 4) if pool.size else None,
        pool_std=round(float(pool.std()), 4) if pool.size else None,
    )
