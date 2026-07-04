"""T-N3 (contender) · UMAP + HDBSCAN — banc d'éval ET méthode switchable console.

Approche alternative à Leiden. Dépendances optionnelles (`umap-learn`,
`hdbscan`) — importées paresseusement. Si non installées, `available()`
renvoie False et `run_hdbscan` lève un ImportError explicite. Seed fixé.

Pipeline : **UMAP(n_components=5)** sur les vecteurs (cosine) → **HDBSCAN** →
clusters PLATS + bruit (`cluster_id=-1`). En option, une seconde **UMAP-2D**
fournit des coords `(x,y)` par nœud pour un affichage 2D.

GÉNÉRICITÉ : aucun magic-number corpus-spécifique. Les défauts (`min_cluster_size`,
`min_samples`, `umap_n_neighbors`) sont **dérivés de N** via les mêmes formes que
le reste du pipeline (`pipeline.cluster.adaptive`) :
  - `min_cluster_size` = `derive_min_sub_size(N)` (relatif à N, cf. `min_sub_size`).
  - `min_samples`       = `MIN_SAMPLES_FLOOR` (=1) : sensibilité MAXIMALE à la
    structure de densité (plancher absolu, langue/corpus-agnostique). Le monter
    lisse la densité → moins de clusters, plus de bruit (knob). Empiriquement, la
    valeur HDBSCAN par défaut (=min_cluster_size) écrase tout en 2 macro-blobs
    après l'UMAP : le plancher révèle la vraie structure fine.
  - `umap_n_neighbors`  = `derive_k(N)` (∝ log10 N, même voisinage que le k-NN).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.cluster.adaptive import derive_k, derive_min_sub_size
from pipeline.cluster.leiden_cluster import DEFAULT_SEED  # source unique du seed de clustering

# n_components de l'UMAP de clustering : FIXE (contrat). 5D = compromis usuel
# (assez pour séparer, assez bas pour densifier avant HDBSCAN). PAS un knob.
N_COMPONENTS = 5
# n_components de l'UMAP d'affichage : 2D (x, y).
N_COMPONENTS_2D = 2
# Plancher de min_samples : 1 = sensibilité max à la densité (cf. docstring).
MIN_SAMPLES_FLOOR = 1


@dataclass
class HdbscanDefaults:
    """Défauts HDBSCAN dérivés de N (zéro magic-number corpus-spécifique)."""
    n: int
    min_cluster_size: int
    min_samples: int
    umap_n_neighbors: int


def derive_hdbscan_defaults(n: int) -> HdbscanDefaults:
    """Dérive les défauts HDBSCAN de la taille du set filtré `n`.

    Mêmes FORMES que le reste du pipeline (langue/corpus/modèle-agnostiques) :
    la valeur s'adapte à N, rien n'est calé sur un corpus.
    """
    mcs = derive_min_sub_size(n)
    return HdbscanDefaults(
        n=n,
        min_cluster_size=mcs,
        # min_samples = plancher (1) : sensibilité max à la densité. Le monter
        # lisse → moins de clusters, plus de bruit. Exposé en knob (cf. docstring).
        min_samples=MIN_SAMPLES_FLOOR,
        umap_n_neighbors=derive_k(n),
    )


@dataclass
class HdbscanResult:
    membership: list[int]  # cluster_id par nœud ; -1 = bruit
    n_clusters: int
    n_noise: int
    coords_2d: list[list[float]] | None  # [x, y] par nœud (UMAP-2D), si demandé
    params: dict


def available() -> bool:
    try:
        import hdbscan  # noqa: F401
        import umap  # noqa: F401

        return True
    except Exception:
        return False


def _umap_embed(vecs: np.ndarray, n_neighbors: int, n_components: int,
                seed: int) -> np.ndarray:
    import umap as _umap

    reducer = _umap.UMAP(
        n_neighbors=min(n_neighbors, max(2, vecs.shape[0] - 1)),
        n_components=min(n_components, max(2, vecs.shape[1])),
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(vecs)


def run_hdbscan(
    vecs: np.ndarray,
    n_neighbors: int = 15,
    n_components: int = N_COMPONENTS,
    min_cluster_size: int = 3,
    min_samples: int | None = None,
    seed: int = DEFAULT_SEED,
    use_umap: bool = True,
    compute_2d: bool = False,
) -> HdbscanResult:
    """UMAP(n_components)→HDBSCAN. `compute_2d` ajoute une UMAP-2D (coords x,y).

    Rétro-compatible avec le banc d'éval (`n_neighbors`, `n_components`,
    `min_cluster_size`, `seed`). `min_samples=None` ⇒ HDBSCAN utilise
    `min_cluster_size`.
    """
    import hdbscan as _hdbscan

    X = _umap_embed(vecs, n_neighbors, n_components, seed) if use_umap else vecs

    clusterer = _hdbscan.HDBSCAN(
        min_cluster_size=max(2, int(min_cluster_size)),
        min_samples=(int(min_samples) if min_samples is not None else None),
        metric="euclidean",
    )
    labels = clusterer.fit_predict(X)
    membership = [int(x) for x in labels]
    uniq = set(membership) - {-1}

    coords_2d = None
    if compute_2d:
        xy = _umap_embed(vecs, n_neighbors, N_COMPONENTS_2D, seed)
        coords_2d = [[round(float(a), 4), round(float(b), 4)] for a, b in xy]

    return HdbscanResult(
        membership=membership,
        n_clusters=len(uniq),
        n_noise=sum(1 for m in membership if m == -1),
        coords_2d=coords_2d,
        params={
            "n_neighbors": n_neighbors,
            "n_components": n_components,
            "min_cluster_size": int(min_cluster_size),
            "min_samples": (int(min_samples) if min_samples is not None
                            else int(min_cluster_size)),
            "seed": seed,
            "use_umap": use_umap,
        },
    )
