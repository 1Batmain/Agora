"""T-N4b · Thèmes HIÉRARCHIQUES (macro → sous-thèmes) via Leiden 2 niveaux.

⚠️ LEGACY — HORS chemin servi. Le pipeline de prod construit sa hiérarchie via le moteur
d'ABSTRACTION (profil de thème ré-embeddé, `backend/analysis.py` + `pipeline/cluster/
abstraction.py`), PAS par ce Leiden 2-niveaux. Ce module n'est plus importé que par
`cluster/build.py` (CLI batch legacy) et `research/`. Conservé comme référence historique.


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

import numpy as np

from pipeline.cluster.knn import KnnGraph
from pipeline.cluster.leiden_cluster import DEFAULT_SEED, run_leiden

# Résolutions = granularité, exposées en knobs (défauts d'usage raisonnables).
# `DEFAULT_MIN_SUB_SIZE` n'est plus qu'un REPLI : le vrai défaut est DÉRIVÉ,
# relatif à N (`pipeline.cluster.adaptive.derive_min_sub_size`, audit #7) ; build
# et le backend passent la valeur dérivée. Cette constante ne sert que si un
# appelant direct n'en fournit aucune.
DEFAULT_RESOLUTION_MACRO = 1.0
# Réconcilié avec le backend live (était 3.0 dans build, 1.5 dans le backend —
# l'incohérence inter-modules signalée par l'audit #6). Valeur unique = celle du
# contrat FROZEN (`.agent/queue/cross-lane.md`), pour que build et le backend produisent
# la MÊME structure sur les mêmes données.
DEFAULT_RESOLUTION_SUB = 1.5
DEFAULT_MIN_SUB_SIZE = 15


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


def _merge_crumbs(
    local_membership: list[int],
    sub_vecs: np.ndarray,
    min_sub_size: int,
) -> list[int]:
    """Fusionne les sous-clusters < min_sub_size dans le plus PROCHE (cosine).

    Plutôt que de tout déverser dans le plus gros (qui gonflerait en fourre-tout),
    chaque miette rejoint le sous-thème viable dont le centroïde lui ressemble le
    plus. Si aucun sous-cluster n'atteint la taille mini (macro minuscule), le
    macro reste indivis (un seul sous-thème). Déterministe.
    """
    sizes: dict[int, int] = defaultdict(int)
    for c in local_membership:
        sizes[c] += 1
    big_enough = sorted(c for c, n in sizes.items() if n >= min_sub_size)
    if not big_enough:
        return [0 for _ in local_membership]
    if len(big_enough) == len(sizes):
        target_for = {c: c for c in sizes}
    else:
        # Centroïdes (L2-normalisés) des sous-clusters viables et des miettes.
        members_of: dict[int, list[int]] = defaultdict(list)
        for pos, c in enumerate(local_membership):
            members_of[c].append(pos)

        def centroid(idxs: list[int]) -> np.ndarray:
            v = sub_vecs[idxs].mean(axis=0)
            nrm = np.linalg.norm(v)
            return v / nrm if nrm > 0 else v

        viable_cent = {c: centroid(members_of[c]) for c in big_enough}
        target_for = {}
        for c in sizes:
            if c in viable_cent:
                target_for[c] = c
                continue
            cc = centroid(members_of[c])
            # tie-break déterministe : plus petit id viable
            target_for[c] = max(big_enough, key=lambda b: (float(viable_cent[b] @ cc), -b))
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
    vecs: np.ndarray,
    resolution_macro: float = DEFAULT_RESOLUTION_MACRO,
    resolution_sub: float = DEFAULT_RESOLUTION_SUB,
    min_sub_size: int = DEFAULT_MIN_SUB_SIZE,
    seed: int = DEFAULT_SEED,
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
            local = _merge_crumbs(list(sub.membership), vecs[members], min_sub_size)

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


def check_integrity(payload: dict) -> list[str]:
    """Vérifie l'arbre macro→sous d'un GraphPayload hiérarchique.

    Retourne la liste des erreurs (vide = arbre cohérent). Règles :
      - chaque thème a un `level` ∈ {0,1} ;
      - `children` d'un macro = EXACTEMENT les feuilles dont `parent_id` = ce macro ;
      - chaque feuille (level=1) pointe un `parent_id` macro valide ;
      - chaque nœud référence une feuille existante (`cluster_id`) et un macro
        (`macro_id`) qui CONCORDENT (la feuille appartient bien à ce macro).
    """
    errors: list[str] = []
    themes = payload.get("themes", [])
    macros = {t["cluster_id"]: t for t in themes if t.get("level") == 0}
    leaves = {t["cluster_id"]: t for t in themes if t.get("level") == 1}

    for t in themes:
        if t.get("level") not in (0, 1):
            errors.append(f"thème {t.get('cluster_id')} : level invalide {t.get('level')}")

    # macros : parent_id null, children = feuilles dont parent_id == macro
    leaf_parent = {lid: lt.get("parent_id") for lid, lt in leaves.items()}
    for mid, mt in macros.items():
        if mt.get("parent_id") is not None:
            errors.append(f"macro {mid} : parent_id devrait être null")
        declared = set(mt.get("children", []))
        actual = {lid for lid, p in leaf_parent.items() if p == mid}
        if declared != actual:
            errors.append(
                f"macro {mid} : children {sorted(declared)} ≠ feuilles réelles {sorted(actual)}")

    # feuilles : parent macro valide, children vide
    for lid, lt in leaves.items():
        p = lt.get("parent_id")
        if p not in macros:
            errors.append(f"feuille {lid} : parent_id {p} n'est pas un macro")
        if lt.get("children"):
            errors.append(f"feuille {lid} : ne devrait pas avoir d'enfants")

    # nœuds : cluster_id = feuille valide ; macro_id concorde avec le parent
    for nd in payload.get("nodes", []):
        leaf = nd.get("cluster_id")
        macro = nd.get("macro_id")
        if leaf not in leaves:
            errors.append(f"nœud {nd.get('id')} : cluster_id {leaf} n'est pas une feuille")
            continue
        if macro not in macros:
            errors.append(f"nœud {nd.get('id')} : macro_id {macro} n'est pas un macro")
            continue
        if leaf_parent.get(leaf) != macro:
            errors.append(
                f"nœud {nd.get('id')} : feuille {leaf} appartient au macro "
                f"{leaf_parent.get(leaf)}, pas {macro}")
    return errors
