"""Endpoint `/density` — paysage 3D (UMAP 2D + KDE).

Vérifie le CONTRAT de forme du payload (grille 96×96, ranges, zmax) sur le dataset
par défaut. Le calcul UMAP est coûteux et `umap-learn` est OPTIONNEL (extra
`contender`) : si la projection ne peut être ni lue (cache `umap2d.npy`) ni recalculée
(umap absent), l'endpoint renvoie 503 et le test se SKIPPE proprement — jamais d'échec
parasite, à l'image des autres tests gardés (`require_ready`).

La whitelist de sécurité (404 path-traversal) est testée inconditionnellement : elle
ne dépend d'aucun calcul lourd.
"""

from __future__ import annotations

import pytest

from backend.density import GRID
from backend.recluster import DEFAULT_DATASET


def test_density_unknown_dataset_404(client):
    """Path-traversal / id hors whitelist → 404, AUCUN calcul (garde `_resolve`)."""
    r = client.get("/density", params={"dataset": "../etc"})
    assert r.status_code == 404, r.text


def test_density_shape(client):
    """`/density` sur le défaut → 200 + grille GRID×GRID cohérente (ou skip si UMAP absent)."""
    r = client.get("/density", params={"dataset": DEFAULT_DATASET})
    if r.status_code == 503:
        pytest.skip("UMAP indisponible (umap-learn absent et pas de cache umap2d.npy).")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["nx"] == GRID and body["nz"] == GRID
    assert len(body["x_range"]) == 2 and body["x_range"][0] < body["x_range"][1]
    assert len(body["z_range"]) == 2 and body["z_range"][0] < body["z_range"][1]

    heights = body["heights"]
    assert len(heights) == GRID                      # nz lignes…
    assert all(len(row) == GRID for row in heights)  # …de nx colonnes.

    zmax = body["zmax"]
    assert zmax > 0
    flat = [v for row in heights for v in row]
    assert max(flat) == pytest.approx(zmax)
    assert min(flat) >= 0  # densité KDE ≥ 0.
