"""Re-clustering LIVE piloté par le seuil k-NN — fondation de la page Console.

À partir des embeddings CACHÉS (jamais ré-embeddés) d'un dataset, reconstruit la
carte des thèmes en faisant VARIER le seuil d'arête k-NN (`knn_threshold`). C'est la
brique « lean » qui remplace l'ancien `recluster.recluster()` (QW1, mort) : on
réutilise telles quelles les briques d'`analysis.py` (Leiden hiérarchique +
subdivision variance-adaptative + coarsening racine, nommage c-TF-IDF, indices M5,
couleurs) mais SANS extraction LLM de claims — l'unité clusterisée est l'IDÉE elle-même.

Chaîne (zéro LLM, < ~2 s) :
  1. `load_cache(dataset)` → ideas, vecs (nomic-v2 cachés), poids.
  2. graphe k-NN au **seuil donné** (`build_knn_graph`), k dérivé de N.
  3. Leiden → clusters fins, **coarsening** racine, **subdivision** variance-adaptative.
  4. nommage **c-TF-IDF** (PAS de LLM), couleurs par macro, convergence intra.
  5. indices globaux M5 (`_dataset_stats`, forme `{key, value, detail}`).
  6. points **UMAP 2D** (réutilise `density.compute_umap2d`, cache `umap2d.npy`) — un
     point `(x, z)` par idée, `cluster_id` aligné EXACTEMENT sur l'ordre des ideas.

Réponse : `{themes, points, indices, meta}`. Le `knn_threshold` par défaut est celui
DÉRIVÉ du dataset (`derive_defaults`) — la Console démarre alors comme `/analysis`.
"""

from __future__ import annotations

from time import perf_counter
from types import SimpleNamespace

import numpy as np

# Briques de construction d'arbre réutilisées telles quelles (réutilisation EXPLICITE,
# cf. brief) — l'unité « membre » est ici une idée et non un claim, mais toute la
# logique d'indices/hiérarchie est agnostique au type de membre.
from backend.analysis import (
    _assign_colors,
    _assign_convergence,
    _build_subtree,
    _coarsen_roots,
    _dataset_stats,
    _derive_tau,
    _name_nodes,
    _node_stats,
    theme_dict,
)
from backend import density
from backend.recluster import load_cache
from pipeline.cluster.adaptive import derive_defaults, derive_k
from pipeline.cluster.knn import build_knn_graph, knn_search
from pipeline.cluster.leiden_cluster import DEFAULT_SEED, run_leiden


def default_threshold(vecs: np.ndarray) -> float:
    """Seuil k-NN par défaut = celui DÉRIVÉ du dataset (μ − σ·k des cosinus k-NN).

    Donne à la Console le MÊME point de départ que `/analysis` : tant que l'usager
    n'a pas bougé le curseur, le re-clustering live reproduit le défaut adaptatif.
    """
    return float(derive_defaults(np.ascontiguousarray(vecs, dtype=np.float32)).threshold)


def build_live_tree(
    ideas,
    vecs: np.ndarray,
    weights: np.ndarray,
    knn_threshold: float | None = None,
    *,
    k: int | None = None,
    resolution: float = 1.0,
    seed: int = DEFAULT_SEED,
) -> SimpleNamespace:
    """Construit l'arbre de thèmes LIVE au seuil `knn_threshold` (idées = membres).

    Reproduit le corps de `analysis.build_theme_tree`, mais : (a) part des idées + vecs
    cachés (aucun re-embed, aucun LLM) ; (b) le graphe racine utilise le seuil DONNÉ au
    lieu du seuil dérivé (`None` → seuil dérivé, comme `/analysis`) ; (c) `owner[i] = i`
    (chaque idée est son propre avis). Les sous-thèmes re-dérivent leur graphe local
    normalement (variance-adaptatif).

    Renvoie un namespace léger duck-typé `nodes/order/macros/...` — suffisant pour les
    helpers d'indices/couleurs et la sérialisation, sans porter de `PreparedClaims`.
    """
    n = len(ideas)
    # L2-normalisation défensive (les embeddings cachés le sont déjà ; idempotent) →
    # garantit l'exactitude des stats de nœud (dispersion = 1 − ‖Σv‖/n).
    v32 = np.ascontiguousarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(v32, axis=1, keepdims=True)
    v32 = v32 / np.where(norms > 0, norms, 1.0)
    vecs64 = v32.astype(np.float64)
    weights_arr = np.asarray(weights, dtype=np.float64)
    texts = [(getattr(idea, "text_clean", None) or idea.text) for idea in ideas]
    owner = list(range(n))                       # une idée = un avis

    nodes: dict = {}
    order: list[str] = []
    macros: list[str] = []
    tau = float("inf")
    derived = None
    merge_thr = float("nan")
    root_modularity = None        # Q Leiden de la partition RACINE (None si n==0)
    # Seuil EFFECTIF : donné, sinon dérivé (rempli ci-dessous quand n>0).
    thr = float(knn_threshold) if knn_threshold is not None else None

    if n:
        # `k` (nombre de voisins) = LEVIER de la Console. Donné → borné [2, n−1] ;
        # sinon dérivé de N (démarre comme `/analysis`). Le seuil suit (dérivé du k).
        k = max(2, min(int(k), n - 1)) if k is not None else derive_k(n)
        neighbors = knn_search(v32, k)           # 1 passe k-NN → seuil dérivé + graphe
        derived = derive_defaults(v32, k=k, neighbors=neighbors)
        if thr is None:                          # défaut = seuil DÉRIVÉ (démarre comme /analysis)
            thr = float(derived.threshold)
        # Graphe RACINE au seuil EFFECTIF (le levier de la Console) ; k & voisinage réutilisés.
        graph = build_knn_graph(vecs64, k=k, threshold=thr, neighbors=neighbors)
        leiden = run_leiden(graph, resolution=resolution, seed=seed)
        membership = leiden.membership
        # Modularité Q de la partition RACINE = qualité de la coupe AU SEUIL/k courant
        # (pédagogique : CHUTE quand k monte — cohérent avec le verdict k-sweep).
        root_modularity = float(leiden.modularity)
        by_cluster: dict[int, list[int]] = {}
        for i, c in enumerate(membership):
            by_cluster.setdefault(c, []).append(i)
        fine_groups = list(by_cluster.values())

        # Coarsening de l'ENTRÉE (fusion des racines trop proches) + seuil de dispersion
        # DÉRIVÉ sur les clusters réellement subdivisibles — identique à `/analysis`.
        super_groups, merge_thr = _coarsen_roots(fine_groups, vecs64)
        floor = derived.min_sub_size
        macro_disp = [_node_stats(g, vecs64, weights_arr)[1]
                      for g in fine_groups if len(g) >= floor]
        tau = _derive_tau(macro_disp)

        counter = [0]

        def _w(members: list[int]) -> float:
            return _node_stats(members, vecs64, weights_arr)[3]

        merged = [sorted((fine_groups[i] for i in sg), key=lambda g: -_w(g))
                  for sg in super_groups]
        merged.sort(key=lambda fine: -_w([m for g in fine for m in g]))
        for fine in merged:
            union = [m for g in fine for m in g]
            forced = fine if len(fine) >= 2 else None
            mid = _build_subtree(union, None, 0, counter, nodes, order, vecs64,
                                 weights_arr, owner, tau, resolution, seed,
                                 forced_children=forced)
            macros.append(mid)

        _name_nodes(nodes, texts)               # c-TF-IDF (zéro LLM)
        _assign_colors(nodes, macros)
        _assign_convergence(nodes, macros)

    return SimpleNamespace(
        nodes=nodes, order=order, macros=macros, dataset=None,
        tau=tau, base_resolution=resolution, seed=seed,
        derived_global=derived, knn_threshold=thr, root_modularity=root_modularity,
        root_coarsen={
            "n_fine": len(by_cluster) if n else 0,
            "n_macros": len(macros),
            "merge_threshold": (None if merge_thr != merge_thr else round(merge_thr, 4)),
        },
    )


def _leaf_of_idea(tree: SimpleNamespace, n: int) -> list[str | None]:
    """Index d'idée → id de la FEUILLE (nœud sans enfant) qui la contient.

    Les feuilles partitionnent les idées (membres propagés vers le haut), donc chaque
    idée tombe dans exactement une feuille → un `cluster_id` le plus spécifique.
    """
    leaf_of: list[str | None] = [None] * n
    for nid, node in tree.nodes.items():
        if not node.children:
            for i in node.members:
                leaf_of[i] = nid
    return leaf_of


def _points(tree: SimpleNamespace, dataset: str, n: int) -> list[dict]:
    """Un point UMAP 2D par idée, aligné à l'ordre des ideas, coloré par macro.

    Réutilise la projection cachée `umap2d.npy` (`density.compute_umap2d`). Si UMAP est
    indisponible (umap-learn absent ET pas de cache), renvoie `[]` — le re-clustering
    (themes + indices) reste servi ; le front peut afficher la carte sans le paysage.
    """
    try:
        coords = density.compute_umap2d(dataset)
    except density.DensityUnavailable:
        return []
    if coords.shape[0] != n:                     # cache désaligné → on s'abstient (pas de 500)
        return []
    leaf_of = _leaf_of_idea(tree, n)
    out = []
    for i in range(n):
        nid = leaf_of[i]
        color = tree.nodes[nid].color if nid is not None else ""
        out.append({
            "x": round(float(coords[i, 0]), 4),
            "z": round(float(coords[i, 1]), 4),
            "cluster_id": nid,
            "color": color,
        })
    return out


def recluster_payload(
    dataset: str,
    knn_threshold: float | None = None,
    *,
    k: int | None = None,
    resolution: float = 1.0,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Payload `POST /recluster` : `{themes, points, indices, meta}`. Rapide, zéro LLM.

    `knn_threshold=None` → défaut DÉRIVÉ du dataset (`default_threshold`). Le payload
    NE touche AUCUN cache d'analyse : seuls les vecteurs cachés et `umap2d.npy` (cache
    de densité, dérivé) sont lus.
    """
    t0 = perf_counter()
    ideas, vecs, weights = load_cache(dataset)
    n = len(ideas)

    tree = build_live_tree(ideas, vecs, weights, knn_threshold,
                           k=k, resolution=resolution, seed=seed)
    themes = [theme_dict(tree.nodes[i]) for i in tree.order]
    stats = _dataset_stats(SimpleNamespace(nodes=tree.nodes, macros=tree.macros))
    points = _points(tree, dataset, n)

    took_ms = round((perf_counter() - t0) * 1000)
    dg = tree.derived_global
    # Seuil dérivé (défaut) = celui que la Console reproduit curseur au repos.
    default_thr = round(float(dg.threshold), 4) if dg is not None else None
    return {
        "themes": themes,
        "points": points,
        "indices": stats["indices"],
        "meta": {
            "dataset": dataset,
            "knn_threshold": None if tree.knn_threshold is None else round(tree.knn_threshold, 4),
            "knn_threshold_default": default_thr,
            # Modularité Q de la partition RACINE (Console : chute pédagogique quand k↑).
            "modularity": tree.root_modularity,
            "k": dg.k if dg is not None else None,
            "k_default": derive_k(n) if n else None,
            "n_themes": len(themes),
            "n_macros": len(tree.macros),
            "n_ideas": n,
            "n_points": len(points),
            "resolution": resolution,
            "seed": seed,
            "took_ms": took_ms,
            "totals": stats["totals"],
            "derived": None if dg is None else {"k": dg.k, "threshold": round(dg.threshold, 4)},
            "root_coarsen": tree.root_coarsen,
        },
    }
