"""Endpoint `/citations` — claims d'un thème TRIÉES par proximité au centroïde (B4).

Au niveau le plus fin du canvas (une feuille), le front liste les verbatims citoyens
du thème, **les plus représentatifs en tête** : on trie les claims du thème par leur
proximité au centroïde du thème (distance cosinus croissante), pondérée par le poids
social. Réutilise l'arbre variance-adaptatif (`backend.analysis`) — aucun recalcul.

    GET /citations {dataset, theme_id} -> [{text, dist_to_centroid, weight}]

Fonctionne sur n'importe quel nœud (feuille ou non) : on renvoie toutes les claims
du sous-arbre porté par le nœud (ses `members`). Champs hors-contrat (`avis_id`,
`rank`) ajoutés en bonus — le front peut les ignorer.
"""

from __future__ import annotations

import numpy as np

from backend.analysis import DEFAULT_SEED, ThemeTree, get_or_build_tree
from pipeline.claims.pipeline import DEFAULT_EMBEDDER


def citations_for_theme(tree: ThemeTree, theme_id: str) -> list[dict]:
    """Claims du thème, triées par distance croissante au centroïde (plus proche d'abord)."""
    node = tree.get(theme_id)
    if node is None:
        raise ValueError(f"thème inconnu: {theme_id!r} (dataset {tree.dataset!r}).")
    if not node.members:
        return []

    prep = tree.prepared
    members = node.members
    sims = prep.claim_vecs[members] @ node.centroid     # cos au centroïde
    out: list[dict] = []
    for local, ci in enumerate(members):
        avis_idx = prep.claim_owner[ci]
        out.append({
            "text": prep.claim_texts[ci],
            "dist_to_centroid": round(float(1.0 - sims[local]), 4),
            "weight": round(float(prep.claim_weight[ci]), 4),
            # bonus hors-contrat :
            "avis_id": prep.avis[avis_idx].id,
        })
    # Tri principal : proximité au centroïde ; départage : poids social décroissant.
    out.sort(key=lambda c: (c["dist_to_centroid"], -c["weight"]))
    for rank, c in enumerate(out):
        c["rank"] = rank
    return out


def citations_payload(
    ds,
    *,
    theme_id: str,
    backend: str | None = None,
    model: str | None = None,
    embedder: str = DEFAULT_EMBEDDER,
    resolution: float = 1.0,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    """Construit (ou réutilise) l'arbre puis renvoie les citations triées du thème."""
    tree = get_or_build_tree(
        ds, backend=backend, model=model, embedder=embedder,
        resolution=resolution, seed=seed,
    )
    return citations_for_theme(tree, theme_id)
