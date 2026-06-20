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


def build_knn_graph(
    vecs: np.ndarray,
    k: int = 10,
    threshold: float = 0.80,
    prefer_faiss: bool = True,
) -> KnnGraph:
    """Construit le graphe k-NN cosine (arêtes > seuil).

    Les défauts `k`/`threshold` ne sont que des REPLIS neutres : en production,
    build et le backend passent des valeurs DÉRIVÉES des données (k ∝ log N,
    seuil = μ−σ·k des cosinus k-NN — cf. `pipeline.cluster.adaptive`, audit #6).
    Ce module ne porte donc plus de magic-number corpus-spécifique « actif ».
    """
    n = vecs.shape[0]
    if n <= 1:
        return KnnGraph(n=n, edges=[], k=k, threshold=threshold, backend="none")

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

    seen: dict[tuple[int, int], float] = {}
    for i in range(n):
        for col in range(idx.shape[1]):
            j = int(idx[i, col])
            if j == i:
                continue
            s = float(sims[i, col])
            if s < threshold:
                continue
            a, b = (i, j) if i < j else (j, i)
            prev = seen.get((a, b))
            if prev is None or s > prev:
                seen[(a, b)] = s

    edges = [(a, b, w) for (a, b), w in seen.items()]
    return KnnGraph(n=n, edges=edges, k=k, threshold=threshold, backend=backend)
