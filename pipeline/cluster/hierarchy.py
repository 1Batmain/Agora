"""T-N4b · Thèmes HIÉRARCHIQUES (macro → sous-thèmes) via Leiden 2 niveaux.

Idée : un député lit d'abord quelques **grandes communautés** (macro, `level=0`),
puis ouvre chacune pour voir ses **sous-thèmes** (`level=1`). Ex.
`harcèlement` → `cyberharcèlement` / `comparaison physique` / `haine LGBT-racisme`.

Procédé :
  1. **Niveau 0 (macro)** : Leiden basse résolution sur le graphe k-NN complet
     → quelques grandes communautés.
  2. **Niveau 1 (sous-thèmes)** : pour chaque macro, on extrait le SOUS-GRAPHE
     induit (arêtes dont les deux extrémités sont dans le macro) et on relance
     Leiden à plus haute résolution. Les miettes (< `min_sub_size`) sont fusionnées
     dans le plus gros sous-thème du macro → pas de poussière de singletons.

Identifiants (espaces disjoints, pour que l'intégrité macro/feuille soit triviale) :
  - macro `cluster_id` ∈ [0, M)
  - feuille (sous-thème) `cluster_id` ∈ [M, M+L)
Chaque nœud appartient à exactement une feuille ; chaque feuille à exactement un
macro. L'arbre est donc bien formé par construction.

Le naming (TF-IDF, décision Bob) reste INCHANGÉ et s'applique à chaque niveau :
  - macro : TF-IDF inter-macros (chaque macro = un document) ;
  - sous-thème : TF-IDF CONTRASTÉ dans le macro (sous-thèmes du macro entre eux).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from pipeline.cluster.knn import KnnGraph
from pipeline.cluster.leiden_cluster import run_leiden

DEFAULT_RESOLUTION_MACRO = 0.6
DEFAULT_RESOLUTION_SUB = 2.0
DEFAULT_MIN_SUB_SIZE = 5


@dataclass
class HierarchyResult:
    macro_membership: list[int]            # id macro par index de nœud
    leaf_membership: list[int]             # id feuille (sous-thème) par index de nœud
    macro_children: dict[int, list[int]]   # macro_id -> [leaf_id, ...] (trié)
    leaf_parent: dict[int, int]            # leaf_id -> macro_id
    n_macros: int
    n_leaves: int
    resolution_macro: float
    resolution_sub: float
    min_sub_size: int
    seed: int
    macro_modularity: float = 0.0
    leaf_sizes: dict[int, int] = field(default_factory=dict)


def _induced_subgraph(
    edges: list[tuple[int, int, float]],
    members: list[int],
) -> tuple[KnnGraph, list[int]]:
    """Sous-graphe induit par `members`. Retourne (KnnGraph local, local→global)."""
    member_set = set(members)
    local_of = {g: l for l, g in enumerate(members)}
    sub_edges: list[tuple[int, int, float]] = []
    for (i, j, w) in edges:
        if i in member_set and j in member_set:
            sub_edges.append((local_of[i], local_of[j], w))
    g = KnnGraph(
        n=len(members),
        edges=sub_edges,
        k=0,
        threshold=0.0,
        backend="induced",
    )
    return g, members


def _merge_crumbs(local_membership: list[int], min_sub_size: int) -> list[int]:
    """Fusionne les sous-clusters < min_sub_size dans le plus gros (déterministe).

    Si tout est trop petit (macro minuscule), on retombe sur un seul sous-thème.
    """
    sizes: dict[int, int] = defaultdict(int)
    for c in local_membership:
        sizes[c] += 1
    # Plus gros sous-cluster (tie-break : plus petit id → déterministe).
    biggest = max(sizes, key=lambda c: (sizes[c], -c))
    big_enough = {c for c, n in sizes.items() if n >= min_sub_size}
    if not big_enough:
        # Aucun sous-cluster viable → le macro reste indivis.
        return [0 for _ in local_membership]
    # Réassigne les membres des sous-clusters trop petits au plus gros viable.
    target_for: dict[int, int] = {}
    for c in sizes:
        target_for[c] = c if c in big_enough else biggest
    remapped = [target_for[c] for c in local_membership]
    # Compacte les ids en 0..k-1 (ordre stable par 1re apparition).
    seen: dict[int, int] = {}
    out: list[int] = []
    for c in remapped:
        if c not in seen:
            seen[c] = len(seen)
        out.append(seen[c])
    return out


def run_hierarchical(
    graph: KnnGraph,
    resolution_macro: float = DEFAULT_RESOLUTION_MACRO,
    resolution_sub: float = DEFAULT_RESOLUTION_SUB,
    min_sub_size: int = DEFAULT_MIN_SUB_SIZE,
    seed: int = 42,
) -> HierarchyResult:
    n = graph.n
    if n == 0:
        return HierarchyResult([], [], {}, {}, 0, 0,
                               resolution_macro, resolution_sub, min_sub_size, seed)

    # 1) Macro -------------------------------------------------------------
    macro = run_leiden(graph, resolution=resolution_macro, seed=seed)
    macro_membership = list(macro.membership)

    macro_members: dict[int, list[int]] = defaultdict(list)
    for idx, m in enumerate(macro_membership):
        macro_members[m].append(idx)
    macro_ids = sorted(macro_members)
    n_macros = len(macro_ids)

    # 2) Sous-thèmes par macro (sous-graphe induit, Leiden plus fine) -------
    leaf_membership = [0] * n
    macro_children: dict[int, list[int]] = {}
    leaf_parent: dict[int, int] = {}
    leaf_sizes: dict[int, int] = {}
    next_leaf_id = n_macros  # espace d'ids disjoint des macros

    for m in macro_ids:
        members = macro_members[m]
        if len(members) <= 1:
            local = [0] * len(members)
        else:
            subg, _ = _induced_subgraph(graph.edges, members)
            sub = run_leiden(subg, resolution=resolution_sub, seed=seed)
            local = _merge_crumbs(list(sub.membership), min_sub_size)

        # Mappe les sous-clusters locaux vers des ids feuilles globaux.
        local_to_leaf: dict[int, int] = {}
        children: list[int] = []
        for pos, lc in enumerate(local):
            if lc not in local_to_leaf:
                lid = next_leaf_id
                next_leaf_id += 1
                local_to_leaf[lc] = lid
                children.append(lid)
                leaf_parent[lid] = m
                leaf_sizes[lid] = 0
            lid = local_to_leaf[lc]
            leaf_membership[members[pos]] = lid
            leaf_sizes[lid] += 1
        macro_children[m] = sorted(children)

    n_leaves = next_leaf_id - n_macros
    return HierarchyResult(
        macro_membership=macro_membership,
        leaf_membership=leaf_membership,
        macro_children=macro_children,
        leaf_parent=leaf_parent,
        n_macros=n_macros,
        n_leaves=n_leaves,
        resolution_macro=resolution_macro,
        resolution_sub=resolution_sub,
        min_sub_size=min_sub_size,
        seed=seed,
        macro_modularity=macro.modularity,
        leaf_sizes=leaf_sizes,
    )
