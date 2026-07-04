"""Endpoint `/citations` — claims d'un thème triées par CENTRALITÉ × DÉVELOPPEMENT (B4+D1).

Au niveau le plus fin du canvas (une feuille), le front liste les verbatims citoyens
du thème, **les plus représentatifs en tête**. D1 : on ne surface plus le claim court
et générique (le plus proche du centroïde) mais l'argument ÉTOFFÉ et on-topic — score
`centralité(garde-fou) × développement` (`backend.develop`). La centralité reste en
garde-fou anti hors-sujet. Réutilise l'arbre variance-adaptatif — aucun recalcul.

    GET /citations {dataset, theme_id} -> [{text, dist_to_centroid, weight, development}]

Fonctionne sur n'importe quel nœud (feuille ou non) : on renvoie toutes les claims
du sous-arbre porté par le nœud (ses `members`). Champs hors-contrat (`avis_id`,
`rank`, `development`) ajoutés en bonus — le front peut les ignorer.
"""

from __future__ import annotations

import numpy as np

from backend.analysis import DEFAULT_RESOLUTION, DEFAULT_SEED, ThemeTree, get_or_build_tree
from backend.develop import corpus_idf, development_scores, guard_gate
from pipeline.claims.pipeline import DEFAULT_EMBEDDER


def citations_for_theme(tree: ThemeTree, theme_id: str) -> list[dict]:
    """Claims du thème, triées par `centralité(garde-fou) × développement` décroissant (D1)."""
    node = tree.get(theme_id)
    if node is None:
        raise ValueError(f"thème inconnu: {theme_id!r} (dataset {tree.dataset!r}).")
    if not node.members:
        return []

    prep = tree.prepared
    members = node.members
    texts = [prep.claim_texts[ci] for ci in members]
    sims = prep.claim_vecs[members] @ node.centroid     # cos au centroïde (centralité)
    idf = tree.claim_idf if tree.claim_idf is not None else corpus_idf(texts)
    dev = development_scores(texts, idf)                 # développement ∈ [0,1]
    score = guard_gate(sims) * dev                       # garde-fou × développement
    out: list[dict] = []
    for local, ci in enumerate(members):
        avis_idx = prep.claim_owner[ci]
        out.append({
            "text": texts[local],
            "dist_to_centroid": round(float(1.0 - sims[local]), 4),
            "weight": round(float(prep.claim_weight[ci]), 4),
            # bonus hors-contrat :
            "development": round(float(dev[local]), 4),
            "_score": float(score[local]),
            "_sim": float(sims[local]),
            "avis_id": prep.avis[avis_idx].id,
        })
    # Tri principal : score centralité×développement décroissant ; départage : centralité.
    out.sort(key=lambda c: (-c["_score"], -c["_sim"]))
    for rank, c in enumerate(out):
        c["rank"] = rank
        del c["_score"], c["_sim"]
    return out


def citations_payload(
    ds,
    *,
    theme_id: str,
    backend: str | None = None,
    model: str | None = None,
    embedder: str = DEFAULT_EMBEDDER,
    resolution: float = DEFAULT_RESOLUTION,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    """Construit (ou réutilise) l'arbre puis renvoie les citations triées du thème."""
    tree = get_or_build_tree(
        ds, backend=backend, model=model, embedder=embedder,
        resolution=resolution, seed=seed,
    )
    return citations_for_theme(tree, theme_id)
