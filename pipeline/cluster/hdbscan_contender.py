"""T-N3 (contender) · UMAP + HDBSCAN, pour le banc d'éval.

Approche alternative à Leiden. Dépendances optionnelles (`umap-learn`,
`hdbscan`) — importées paresseusement. Si non installées, `available()`
renvoie False et `run_hdbscan` lève un ImportError explicite. Seed fixé.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_SEED = 42


@dataclass
class HdbscanResult:
    membership: list[int]  # cluster_id par nœud ; -1 = bruit
    n_clusters: int
    n_noise: int
    params: dict


def available() -> bool:
    try:
        import hdbscan  # noqa: F401
        import umap  # noqa: F401

        return True
    except Exception:
        return False


def run_hdbscan(
    vecs: np.ndarray,
    n_neighbors: int = 15,
    n_components: int = 5,
    min_cluster_size: int = 3,
    seed: int = DEFAULT_SEED,
    use_umap: bool = True,
) -> HdbscanResult:
    import hdbscan as _hdbscan

    X = vecs
    if use_umap:
        import umap as _umap

        reducer = _umap.UMAP(
            n_neighbors=min(n_neighbors, max(2, vecs.shape[0] - 1)),
            n_components=min(n_components, max(2, vecs.shape[1])),
            metric="cosine",
            random_state=seed,
        )
        X = reducer.fit_transform(vecs)

    clusterer = _hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(X)
    membership = [int(x) for x in labels]
    uniq = set(membership) - {-1}
    return HdbscanResult(
        membership=membership,
        n_clusters=len(uniq),
        n_noise=sum(1 for m in membership if m == -1),
        params={
            "n_neighbors": n_neighbors,
            "n_components": n_components,
            "min_cluster_size": min_cluster_size,
            "seed": seed,
            "use_umap": use_umap,
        },
    )
