"""T-N2 · Graphe k-NN sémantique.

À partir de vecteurs (L2-normalisés), construit les arêtes cosine > seuil.
Backend : `faiss-cpu` si dispo (rapide), sinon `sklearn.NearestNeighbors`
(fallback robuste — pas de dépendance native pénible). Arêtes non dirigées,
dédupliquées (i<j), poids = cosine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KnnGraph:
    n: int
    edges: list[tuple[int, int, float]]  # (i, j, cosine) avec i < j
    k: int
    threshold: float
    backend: str

    @property
    def avg_degree(self) -> float:
        if self.n == 0:
            return 0.0
        return 2.0 * len(self.edges) / self.n


def _knn_faiss(vecs: np.ndarray, k: int):
    import faiss  # type: ignore

    d = vecs.shape[1]
    index = faiss.IndexFlatIP(d)  # vecteurs normalisés -> IP == cosine
    index.add(vecs)
    sims, idx = index.search(vecs, min(k + 1, vecs.shape[0]))
    return sims, idx, "faiss"


def _knn_sklearn(vecs: np.ndarray, k: int):
    from sklearn.neighbors import NearestNeighbors

    k_eff = min(k + 1, vecs.shape[0])
    nn = NearestNeighbors(n_neighbors=k_eff, metric="cosine")
    nn.fit(vecs)
    dist, idx = nn.kneighbors(vecs)
    sims = 1.0 - dist  # cosine distance -> similarité
    return sims, idx, "sklearn"


@dataclass
class KnnNeighbors:
    """Voisinage k-NN BRUT (self inclus) — réutilisable pour le graphe ET les stats.

    `sims`/`idx` ont la forme `(n, k+1)` : pour chaque nœud, ses `k+1` plus proches
    voisins par cosinus (self typiquement en tête, sim≈1). Calculé UNE fois (faiss
    si dispo), il alimente à la fois `build_knn_graph` (arêtes) et `derive_defaults`
    (pool des cosinus pour le seuil) — au lieu de deux passes O(n²) redondantes.
    """
    sims: np.ndarray            # (n, k+1) float32, cosinus décroissants
    idx: np.ndarray             # (n, k+1) int, indices des voisins
    backend: str


def knn_search(vecs: np.ndarray, k: int, prefer_faiss: bool = True) -> KnnNeighbors:
    """k-NN brut (top-(k+1), self inclus) — faiss exact si dispo, sinon sklearn.

    EXACT (`IndexFlatIP` = produit scalaire == cosine sur vecteurs normalisés) :
    pas d'approximation, résultats identiques au dense. Sert de source unique de
    vérité aux deux consommateurs (graphe + seuil dérivé)."""
    vecs = np.ascontiguousarray(vecs, dtype=np.float32)
    sims = idx = None
    backend = ""
    if prefer_faiss:
        try:
            sims, idx, backend = _knn_faiss(vecs, k)
        except Exception:
            sims = None
    if sims is None:
        sims, idx, backend = _knn_sklearn(vecs, k)
    return KnnNeighbors(sims=sims, idx=idx, backend=backend)


def slice_neighbors(neighbors: "KnnNeighbors", k: int) -> "KnnNeighbors":
    """Restreint un voisinage à son top-(k+1) — un PRÉFIXE exact (sims/idx triés décroissants).

    `knn_search` calcule tous les cosinus (O(n²d)) et n'en garde que les k+1 meilleurs, triés.
    Le top-(k'+1) d'un k' plus petit est donc le PRÉFIXE du top-(k+1). Slicer revient exactement
    à `knn_search(vecs, k')` — au bit près — sans recalculer les distances. Permet UN seul
    `knn_search` au k max puis tous les paliers plus fins par slice (cf. `cluster.layers`)."""
    c = k + 1
    return KnnNeighbors(sims=neighbors.sims[:, :c], idx=neighbors.idx[:, :c],
                        backend=neighbors.backend)


def build_knn_graph(
    vecs: np.ndarray,
    k: int = 10,
    threshold: float = 0.80,
    prefer_faiss: bool = True,
    neighbors: "KnnNeighbors | None" = None,
) -> KnnGraph:
    """Construit le graphe k-NN cosine (arêtes > seuil).

    Les défauts `k`/`threshold` ne sont que des REPLIS neutres : en production,
    build et le backend passent des valeurs DÉRIVÉES des données (k ∝ log N,
    seuil = μ−σ·k des cosinus k-NN — cf. `pipeline.cluster.adaptive`, audit #6).
    Ce module ne porte donc plus de magic-number corpus-spécifique « actif ».

    `neighbors` (optionnel) = voisinage k-NN DÉJÀ calculé (`knn_search`) : on évite
    une 2ᵉ passe O(n²) quand l'appelant l'a déjà pour dériver le seuil (cf. sandbox).
    La construction des arêtes est VECTORISÉE (numpy) — résultat identique à l'ancien
    double-boucle : arêtes non dirigées i<j, poids = cosine (symétrique en exact, donc
    `max` sur doublon == la même valeur).
    """
    n = vecs.shape[0]
    if n <= 1:
        return KnnGraph(n=n, edges=[], k=k, threshold=threshold, backend="none")

    if neighbors is None:
        neighbors = knn_search(vecs, k, prefer_faiss=prefer_faiss)
    sims, idx, backend = neighbors.sims, neighbors.idx, neighbors.backend

    # Aplatissement ROW-MAJOR (i croissant, col croissant) == ordre de scan d'origine.
    rows = np.repeat(np.arange(n), idx.shape[1])
    cols = idx.reshape(-1).astype(np.int64)
    w = sims.reshape(-1).astype(np.float64)
    keep = (cols != rows) & (w >= threshold)
    rows, cols, w = rows[keep], cols[keep], w[keep]
    if rows.size == 0:
        return KnnGraph(n=n, edges=[], k=k, threshold=threshold, backend=backend)

    a = np.minimum(rows, cols)
    b = np.maximum(rows, cols)
    key = a * n + b
    # Dédup en conservant l'ORDRE DE PREMIÈRE APPARITION (comme l'ancien dict) et le
    # poids MAX par paire (en exact, symétrique → max == la même valeur).
    uniq, first_idx, inv = np.unique(key, return_index=True, return_inverse=True)
    maxw = np.zeros(uniq.shape[0], dtype=np.float64)
    np.maximum.at(maxw, inv, w)
    o = np.argsort(first_idx, kind="stable")        # ordre de 1ʳᵉ apparition
    fi = first_idx[o]
    edges = list(zip(a[fi].tolist(), b[fi].tolist(), maxw[o].tolist()))
    return KnnGraph(n=n, edges=edges, k=k, threshold=threshold, backend=backend)
