"""T-N3 · Clustering Leiden (approche primaire).

Leiden (`igraph` + `leidenalg`) sur le graphe k-NN sémantique pondéré.
Seed fixé → reproductible. Retourne l'appartenance (membership) par nœud.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.cluster.knn import KnnGraph

DEFAULT_SEED = 42
DEFAULT_RESOLUTION = 1.0


@dataclass
class LeidenResult:
    membership: list[int]  # cluster_id par index de nœud
    n_clusters: int
    modularity: float
    resolution: float
    seed: int


def run_leiden(
    graph: KnnGraph,
    resolution: float = DEFAULT_RESOLUTION,
    seed: int = DEFAULT_SEED,
    n_iterations: int = -1,
) -> LeidenResult:
    import igraph as ig
    import leidenalg as la

    n = graph.n
    if n == 0:
        return LeidenResult([], 0, 0.0, resolution, seed)

    g = ig.Graph(n=n)
    if graph.edges:
        g.add_edges([(i, j) for (i, j, _) in graph.edges])
        g.es["weight"] = [w for (_, _, w) in graph.edges]

    # RBConfiguration = modularité avec paramètre de résolution réglable.
    part = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights="weight" if graph.edges else None,
        resolution_parameter=resolution,
        n_iterations=n_iterations,
        seed=seed,
    )
    membership = list(part.membership)
    n_clusters = len(set(membership))
    try:
        modularity = float(g.modularity(membership, weights="weight" if graph.edges else None))
    except Exception:
        modularity = 0.0
    return LeidenResult(
        membership=membership,
        n_clusters=n_clusters,
        modularity=round(modularity, 4),
        resolution=resolution,
        seed=seed,
    )
