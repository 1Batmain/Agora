"""Paysage de densité 3D — UMAP 2D des embeddings PRÉ-clustering + KDE 2D.

Pas un nuage de points : une SURFACE. Le plan (x, z) est la projection UMAP 2D des
vecteurs nomic en cache (`cache/<dataset>/embeddings.npy`, juste avant le clustering),
la hauteur y est la DENSITÉ locale estimée par KDE gaussien évaluée sur une grille
96×96. Pics = zones denses (futurs thèmes), vallées = clairsemé.

Calcul PARESSEUX + CACHE disque (UMAP est coûteux ; KDE l'est moins mais on cache le
payload servi) :
  - `cache/<dataset>/umap2d.npy`   : projection 2D (recalcul SEULEMENT si absent).
  - `cache/<dataset>/density.json` : payload servi (recalcul si absent / grille changée).

Ces deux fichiers sont DÉRIVÉS (gitignorés) ; ce module NE touche JAMAIS aux caches
d'analyse (`analysis/`, `claims*`, …). `umap-learn` est importé PARESSEUSEMENT : sans
cache `umap2d.npy` ni `umap-learn` installé, `density_payload` lève `DensityUnavailable`
(l'endpoint la traduit en 503 propre, jamais un 500 opaque).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from backend.recluster import cache_paths, dataset_dir

# Grille de la surface (nx × nz). 96 = compromis lisse/léger (9 216 évaluations KDE).
GRID = 96
# Graine UMAP fixe → projection REPRODUCTIBLE (même cache d'un run à l'autre).
UMAP_SEED = 42
# Marge ajoutée autour de l'étendue (x, z) : la surface retombe vers ~0 sur les bords
# (paysage avec littoral) au lieu d'être tranchée net au point extrême.
RANGE_MARGIN = 0.05

UMAP2D_NAME = "umap2d.npy"
DENSITY_NAME = "density.json"


class DensityUnavailable(RuntimeError):
    """Densité non calculable (ex. `umap-learn` absent et pas de cache `umap2d.npy`)."""


def _umap2d_path(dataset: str) -> Path:
    return dataset_dir(dataset) / UMAP2D_NAME


def _density_path(dataset: str) -> Path:
    return dataset_dir(dataset) / DENSITY_NAME


def _load_embeddings(dataset: str) -> np.ndarray:
    """Charge `embeddings.npy` (vecteurs PRÉ-clustering) sans toucher au reste du cache."""
    emb_path, _, _ = cache_paths(dataset)
    if not emb_path.exists():
        raise DensityUnavailable(f"embeddings absents pour {dataset!r} ({emb_path}).")
    return np.load(emb_path).astype(np.float32)


def compute_umap2d(dataset: str) -> np.ndarray:
    """Projection UMAP 2D des embeddings, CACHÉE sur disque (recalcul si absent).

    Import PARESSEUX de `umap-learn` : on ne paie la dépendance que lors d'un VRAI
    recalcul. Si le cache existe, aucune importation lourde.
    """
    path = _umap2d_path(dataset)
    if path.exists():
        return np.load(path).astype(np.float32)

    try:
        import umap  # noqa: PLC0415 — import paresseux (dépendance lourde optionnelle).
    except ImportError as exc:  # pragma: no cover - dépend de l'env d'exécution.
        raise DensityUnavailable(
            "umap-learn non installé et pas de cache umap2d.npy "
            f"({dataset}). Installe l'extra `contender` pour (re)calculer."
        ) from exc

    vecs = _load_embeddings(dataset)
    # n_neighbors borné par la taille de l'échantillon (petits datasets) ; reste des
    # défauts UMAP (générique, aucun réglage corpus-spécifique).
    n_neighbors = min(15, max(2, vecs.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        random_state=UMAP_SEED,
        metric="cosine",
    )
    coords = reducer.fit_transform(vecs).astype(np.float32)
    np.save(path, coords)
    return coords


def _ranged(values: np.ndarray) -> tuple[float, float]:
    """Étendue [min, max] élargie de `RANGE_MARGIN` (bords en pente douce)."""
    lo, hi = float(values.min()), float(values.max())
    span = hi - lo or 1.0
    pad = span * RANGE_MARGIN
    return lo - pad, hi + pad


def _compute_density(dataset: str) -> dict:
    """Calcule le payload de densité : UMAP 2D → KDE gaussien sur grille GRID×GRID."""
    from scipy.stats import gaussian_kde  # noqa: PLC0415 — local, scipy via sklearn.

    coords = compute_umap2d(dataset)          # (N, 2) : colonnes x, z.
    xs_pts, zs_pts = coords[:, 0], coords[:, 1]
    x_range = _ranged(xs_pts)
    z_range = _ranged(zs_pts)

    kde = gaussian_kde(np.vstack([xs_pts, zs_pts]))
    xs = np.linspace(x_range[0], x_range[1], GRID)
    zs = np.linspace(z_range[0], z_range[1], GRID)
    grid_x, grid_z = np.meshgrid(xs, zs)      # (GRID, GRID), indexé [iz][ix].
    flat = np.vstack([grid_x.ravel(), grid_z.ravel()])
    heights = kde(flat).reshape(GRID, GRID)   # heights[iz][ix].

    zmax = float(heights.max())
    return {
        "nx": GRID,
        "nz": GRID,
        "x_range": [x_range[0], x_range[1]],
        "z_range": [z_range[0], z_range[1]],
        "heights": heights.tolist(),
        "zmax": zmax,
    }


def density_payload(dataset: str) -> dict:
    """Payload servi par `GET /density` : caché sur disque, calculé paresseusement.

    Lit `density.json` s'il est présent ET cohérent (même grille GRID) ; sinon
    (re)calcule, écrit le cache, et renvoie. Lève `DensityUnavailable` si la projection
    UMAP ne peut être ni lue ni recalculée.
    """
    path = _density_path(dataset)
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("nx") == GRID and cached.get("nz") == GRID:
                return cached
        except (json.JSONDecodeError, OSError):
            pass  # cache illisible → on recalcule.

    payload = _compute_density(dataset)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload
