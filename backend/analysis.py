"""Endpoint `/analysis` — carte spatiale des thèmes émergents (B1 + B2 du contrat).

Part des CLAIMS émergents (pipeline ouvert avis→claims→cluster, cf. `claims_endpoint`)
et construit l'objet que le canvas du front affiche :

  - **B1 — projection + relations** : positions **UMAP 2D** des centroïdes de thèmes
    (seed → stables) + arêtes de **co-occurrence** (deux thèmes liés quand un même
    avis porte des claims tombant dans les deux).
  - **B2 — hiérarchie VARIANCE-ADAPTATIVE** : pour chaque thème on mesure la
    **dispersion interne** (distance cosinus moyenne au centroïde). Un thème ne se
    subdivise que si sa dispersion dépasse un **seuil DÉRIVÉ** des données (séparation
    par plus grand écart sur la distribution des dispersions macro — ZÉRO magic-number)
    ET que la re-clusterisation de ses propres claims dégage ≥2 sous-thèmes viables.
    Profondeur variable : thèmes homogènes = feuilles, thèmes hétérogènes = subdivisés.

Tout dérive des données (généricité) : aucune liste de thèmes, aucun seuil de corpus
codé en dur. La sortie suit le contrat figé `queue/front-redesign.md` :

    POST /analysis {dataset, backend?} -> {themes, edges, params, backend_used}
    themes[i] = {id, label, x, y, n_avis, n_claims, weight, consensus, dispersion,
                 parent_id|null, has_children}
    edges[j]  = {a, b, weight}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from time import perf_counter

import numpy as np

from backend.claims_endpoint import PreparedClaims, prepare_claims
from pipeline.claims.pipeline import DEFAULT_EMBEDDER, DEFAULT_SEED, N_REPRESENTATIVE
from pipeline.cluster.adaptive import derive_defaults
from pipeline.cluster.knn import build_knn_graph
from pipeline.cluster.leiden_cluster import run_leiden
from pipeline.cluster.naming import derive_corpus_stopwords, name_clusters
from pipeline.cluster.palette import color_for

# --- Hyper-paramètres de FORME (sans unité corpus-spécifique) -------------- #
# Échelle de résolutions ESSAYÉES pour subdiviser un thème hétérogène : on part de
# la résolution de base et on la monte tant que Leiden ne dégage pas ≥2 sous-thèmes
# viables. Ce sont des PAS DE RECHERCHE (multiplicateurs), pas des seuils de corpus.
RES_LADDER = (1.0, 1.5, 2.0, 3.0)
MAX_DEPTH = 4          # garde-fou de profondeur (l'arbre s'arrête bien avant en pratique)


@dataclass
class ThemeNode:
    """Un nœud de l'arbre de thèmes (macro, sous-thème, ou feuille)."""
    id: str
    parent_id: str | None
    depth: int
    members: list[int]              # indices GLOBAUX des claims du nœud
    centroid: np.ndarray            # barycentre L2-normalisé (espace claims)
    dispersion: float               # 1 − cos moyen au centroïde (0 = très serré)
    consensus: float                # cos moyen entre paires de claims
    weight: float                   # somme des poids sociaux des claims
    n_claims: int
    n_avis: int
    label: str = ""
    title: str = ""                 # titre court LLM (3-7 mots) précalculé au build
    keywords: list[str] = field(default_factory=list)
    representative_claims: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    color: str = ""                 # couleur cluster (source unique : palette.py)
    x: float = 0.0
    y: float = 0.0

    @property
    def has_children(self) -> bool:
        return bool(self.children)


@dataclass
class ThemeTree:
    """Arbre de thèmes complet + contexte (claims, défauts dérivés, traçabilité).

    Partagé entre `/analysis` (ajoute x,y + edges), `/insights` (synthèse par niveau)
    et `/citations` (claims triées) — d'où le cache en mémoire côté serveur.
    """
    nodes: dict[str, ThemeNode]
    order: list[str]                # ids en parcours préfixe (ordre d'affichage)
    macros: list[str]              # ids de niveau 0
    dataset: str
    prepared: PreparedClaims
    tau: float                      # seuil de dispersion DÉRIVÉ (subdivision si >)
    base_resolution: float
    seed: int
    derived_global: object          # DerivedDefaults sur tout le corpus de claims

    def get(self, node_id: str) -> ThemeNode | None:
        return self.nodes.get(node_id)


# --------------------------------------------------------------------------- #
# Statistiques d'un nœud — O(n·d), sans matrice de paires (vecs L2-normalisés)
# --------------------------------------------------------------------------- #
def _node_stats(members: list[int], vecs: np.ndarray, weights: np.ndarray
                ) -> tuple[np.ndarray, float, float, float]:
    """→ (centroïde unitaire, dispersion, consensus, poids).

    Pour des vecteurs L2-normalisés, avec S = Σ vᵢ et n = |membres| :
      - cos moyen au centroïde = ‖S‖ / n  → dispersion = 1 − ‖S‖/n ;
      - cos moyen entre paires  = (‖S‖² − n) / (n²−n).
    Identités exactes : aucune matrice n×n n'est matérialisée.
    """
    sub = vecs[members]
    n = len(members)
    s = sub.sum(axis=0)
    norm_s = float(np.linalg.norm(s))
    centroid = s / norm_s if norm_s > 0 else s
    mean_to_centroid = norm_s / n if n else 0.0
    dispersion = max(0.0, 1.0 - mean_to_centroid)
    if n > 1:
        consensus = (norm_s * norm_s - n) / (n * (n - 1))
    else:
        consensus = 1.0
    weight = float(weights[members].sum())
    return centroid, dispersion, float(consensus), weight


# --------------------------------------------------------------------------- #
# Seuil de dispersion DÉRIVÉ — plus grand écart sur la distribution macro
# --------------------------------------------------------------------------- #
def _derive_tau(dispersions: list[float]) -> float:
    """Seuil de subdivision = milieu du PLUS GRAND ÉCART des dispersions triées.

    On sépare les thèmes « serrés » des « étalés » là où la distribution observée
    présente sa plus forte discontinuité (gap). Entièrement dérivé des données
    (aucune constante de corpus) ; si <2 thèmes ou distribution dégénérée, renvoie
    +inf (personne ne subdivise). La FAISABILITÉ (Leiden doit dégager ≥2 sous-thèmes)
    reste le garde-fou contre les coupes spurious.
    """
    vals = sorted(d for d in dispersions)
    if len(vals) < 2:
        return float("inf")
    gaps = [(vals[i + 1] - vals[i], i) for i in range(len(vals) - 1)]
    max_gap, idx = max(gaps, key=lambda g: g[0])
    if max_gap <= 0.0:
        return float("inf")
    return (vals[idx] + vals[idx + 1]) / 2.0


# --------------------------------------------------------------------------- #
# Subdivision d'un nœud — re-graphe ses propres claims, Leiden résolution montante
# --------------------------------------------------------------------------- #
def _subdivide(members: list[int], vecs: np.ndarray, base_resolution: float,
               seed: int) -> list[list[int]] | None:
    """Partitionne `members` en ≥2 sous-thèmes viables, ou None si homogène.

    Re-dérive les défauts du graphe SUR LE SOUS-ENSEMBLE (k, seuil, min_sub_size à
    l'échelle locale), construit le graphe k-NN, et monte la résolution Leiden tant
    qu'il ne dégage pas ≥2 communautés ≥ `min_sub_size`. Les miettes (communautés
    sous le seuil) sont fusionnées dans le sous-thème viable le plus proche → la
    partition couvre tout le nœud parent. Renvoie des indices GLOBAUX.
    """
    if len(members) < 2:
        return None
    subvecs = np.ascontiguousarray(vecs[members], dtype=np.float64)
    dd = derive_defaults(subvecs.astype(np.float32))
    graph = build_knn_graph(subvecs, k=dd.k, threshold=dd.threshold)

    chosen: list[list[int]] | None = None     # groupes d'indices LOCAUX
    for mult in RES_LADDER:
        membership = run_leiden(graph, resolution=base_resolution * mult, seed=seed).membership
        by_cluster: dict[int, list[int]] = {}
        for i, c in enumerate(membership):
            by_cluster.setdefault(c, []).append(i)
        groups = list(by_cluster.values())
        big = [g for g in groups if len(g) >= dd.min_sub_size]
        if len(big) >= 2:
            chosen = groups
            break
    if chosen is None:
        return None

    big = [g for g in chosen if len(g) >= dd.min_sub_size]
    small = [g for g in chosen if len(g) < dd.min_sub_size]
    # Centroïdes des sous-thèmes viables (pour absorber les miettes).
    big_centroids = []
    for g in big:
        s = subvecs[g].sum(axis=0)
        nrm = np.linalg.norm(s)
        big_centroids.append(s / nrm if nrm > 0 else s)
    big_centroids = np.asarray(big_centroids)
    for g in small:
        s = subvecs[g].sum(axis=0)
        nrm = np.linalg.norm(s)
        gc = s / nrm if nrm > 0 else s
        nearest = int(np.argmax(big_centroids @ gc))
        big[nearest] = big[nearest] + g
    # Indices LOCAUX → GLOBAUX.
    return [[members[i] for i in g] for g in big]


# --------------------------------------------------------------------------- #
# Construction de l'arbre (récursive, variance-adaptative)
# --------------------------------------------------------------------------- #
def _build_subtree(members: list[int], parent_id: str | None, depth: int,
                   counter: list[int], nodes: dict[str, ThemeNode], order: list[str],
                   vecs: np.ndarray, weights: np.ndarray, owner: list[int],
                   tau: float, base_resolution: float, seed: int) -> str:
    """Crée le nœud de `members`, le subdivise si hétérogène (dispersion > τ), récursif."""
    node_id = f"n{counter[0]}"
    counter[0] += 1
    centroid, dispersion, consensus, weight = _node_stats(members, vecs, weights)
    node = ThemeNode(
        id=node_id, parent_id=parent_id, depth=depth, members=members,
        centroid=centroid, dispersion=round(dispersion, 4), consensus=round(consensus, 4),
        weight=round(weight, 1), n_claims=len(members),
        n_avis=len({owner[i] for i in members}),
    )
    nodes[node_id] = node
    order.append(node_id)

    if depth < MAX_DEPTH and dispersion > tau:
        child_groups = _subdivide(members, vecs, base_resolution, seed)
        if child_groups:
            for grp in child_groups:
                cid = _build_subtree(grp, node_id, depth + 1, counter, nodes, order,
                                     vecs, weights, owner, tau, base_resolution, seed)
                node.children.append(cid)
    return node_id


def _name_nodes(nodes: dict[str, ThemeNode], claim_texts: list[str]) -> None:
    """Nomme TOUS les nœuds via c-TF-IDF (mots-clés distinctifs sur tout l'arbre).

    Naming partagé : les mots-vides sont dérivés du corpus de claims une fois, et le
    c-TF-IDF est calculé sur l'ensemble des nœuds → les labels macro et sous-thèmes
    sont distinctifs les uns vis-à-vis des autres.
    """
    ids = list(nodes.keys())
    idx_of = {nid: i for i, nid in enumerate(ids)}
    cluster_docs = {idx_of[nid]: [claim_texts[i] for i in nodes[nid].members] for nid in ids}
    corpus_stop, _ = derive_corpus_stopwords(claim_texts)
    names = name_clusters(cluster_docs, corpus_stopwords=corpus_stop)
    for nid in ids:
        info = names.get(idx_of[nid], {})
        node = nodes[nid]
        node.label = info.get("label", f"thème {nid}")
        node.keywords = info.get("keywords", [])


def macro_of(node: ThemeNode, nodes: dict[str, ThemeNode]) -> ThemeNode:
    """Remonte au macro-thème (ancêtre de profondeur 0) qui contient `node`."""
    cur = node
    while cur.parent_id is not None:
        cur = nodes[cur.parent_id]
    return cur


def _assign_colors(nodes: dict[str, ThemeNode], macros: list[str]) -> None:
    """Couleur cluster par MACRO (source unique : palette.py), héritée par les enfants.

    Tous les nœuds d'un même macro partagent sa couleur → les bulles, les thèmes de
    `/analysis` et les surlignages d'avis (provenance) restent COHÉRENTS, sans aucune
    couleur dupliquée côté front. Teintes équiréparties sur le nombre de macros.
    """
    n = len(macros)
    rank = {mid: i for i, mid in enumerate(macros)}
    for node in nodes.values():
        node.color = color_for(rank.get(macro_of(node, nodes).id, 0), n)


def _representatives(node: ThemeNode, vecs: np.ndarray, claim_texts: list[str],
                     k: int = N_REPRESENTATIVE) -> list[str]:
    """Claims les plus proches du centroïde (médoïdes), sans quasi-doublon littéral."""
    if not node.members:
        return []
    sims = vecs[node.members] @ node.centroid
    order = np.argsort(-sims)
    reps: list[str] = []
    for j in order:
        t = claim_texts[node.members[j]]
        if any(t.lower() == e.lower() for e in reps):
            continue
        reps.append(t)
        if len(reps) >= k:
            break
    return reps


# --------------------------------------------------------------------------- #
# Projection 2D (UMAP des centroïdes ; repli déterministe pour très peu de nœuds)
# --------------------------------------------------------------------------- #
def _project_2d(centroids: np.ndarray, seed: int) -> np.ndarray:
    """Positions 2D STABLES des centroïdes de thèmes (distance = proximité sémantique).

    UMAP (cosine, seed fixe) dès qu'il y a assez de nœuds ; sinon repli déterministe
    (cercle régulier) pour ne jamais planter sur un arbre minuscule. Sortie centrée.
    """
    n = centroids.shape[0]
    if n == 0:
        return np.zeros((0, 2))
    if n == 1:
        return np.zeros((1, 2))
    if n <= 3:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])
    try:
        import umap

        n_neighbors = int(min(15, n - 1))
        reducer = umap.UMAP(
            n_components=2, n_neighbors=n_neighbors, min_dist=0.1,
            metric="cosine", random_state=seed,
        )
        emb = reducer.fit_transform(centroids)
    except Exception:
        # Repli linéaire déterministe : 2 premières composantes principales.
        c = centroids - centroids.mean(axis=0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        emb = c @ vt[:2].T
    emb = np.asarray(emb, dtype=np.float64)
    emb = emb - emb.mean(axis=0)
    scale = float(np.abs(emb).max()) or 1.0
    return emb / scale


# --------------------------------------------------------------------------- #
# Co-occurrence — entre frères de chaque groupe (macros, puis enfants de chaque nœud)
# --------------------------------------------------------------------------- #
def _cooccurrence(tree: ThemeTree) -> list[dict]:
    """Arêtes de co-occurrence entre thèmes FRÈRES (même niveau, même parent).

    Pour chaque groupe de frères (les macros sous la racine, puis les enfants de
    chaque nœud interne), un avis dont les claims tombent dans ≥2 frères les ponte.
    `weight` = nombre d'avis pontant la paire. On ne relie jamais parent↔enfant
    (trivial). Symétrique, a<b, trié par poids décroissant.
    """
    owner = tree.prepared.claim_owner
    groups: list[list[str]] = [tree.macros]
    for node in tree.nodes.values():
        if node.children:
            groups.append(node.children)

    counts: dict[tuple[str, str], int] = {}
    for sibling_ids in groups:
        if len(sibling_ids) < 2:
            continue
        claim_to_node: dict[int, str] = {}
        for nid in sibling_ids:
            for ci in tree.nodes[nid].members:
                claim_to_node[ci] = nid
        themes_by_avis: dict[int, set[str]] = {}
        for ci, nid in claim_to_node.items():
            themes_by_avis.setdefault(owner[ci], set()).add(nid)
        for nids in themes_by_avis.values():
            for a, b in combinations(sorted(nids), 2):
                counts[(a, b)] = counts.get((a, b), 0) + 1

    edges = [{"a": a, "b": b, "weight": c} for (a, b), c in counts.items()]
    edges.sort(key=lambda e: -e["weight"])
    return edges


# --------------------------------------------------------------------------- #
# Point d'entrée : construit l'arbre (sans x,y) — réutilisé par insights/citations
# --------------------------------------------------------------------------- #
def build_theme_tree(
    ds,
    *,
    backend: str | None = None,
    model: str | None = None,
    embedder: str = DEFAULT_EMBEDDER,
    min_chars: int | None = None,
    resolution: float = 1.0,
    seed: int = DEFAULT_SEED,
    prepared: PreparedClaims | None = None,
) -> ThemeTree:
    """Extrait/charge les claims puis construit l'arbre variance-adaptatif.

    `prepared` permet de réutiliser une extraction+embed déjà faite (insights/citations
    rejouent l'arbre sans re-préparer). Ne calcule PAS les positions UMAP (c'est le
    rôle de `analysis_payload`) → reste léger pour les endpoints texte.
    """
    if prepared is None:
        kw = {} if min_chars is None else {"min_chars": min_chars}
        prepared = prepare_claims(ds, backend=backend, model=model, embedder=embedder, **kw)

    vecs = prepared.claim_vecs
    weights = prepared.claim_weight
    owner = prepared.claim_owner
    n_claims = len(prepared.claim_texts)

    nodes: dict[str, ThemeNode] = {}
    order: list[str] = []
    macros: list[str] = []
    derived_global = derive_defaults(vecs.astype(np.float32)) if n_claims else None
    tau = float("inf")

    if n_claims:
        # Niveau 0 : macro-thèmes (Leiden base sur le graphe global dérivé).
        graph = build_knn_graph(vecs, k=derived_global.k, threshold=derived_global.threshold)
        membership = run_leiden(graph, resolution=resolution, seed=seed).membership
        by_cluster: dict[int, list[int]] = {}
        for i, c in enumerate(membership):
            by_cluster.setdefault(c, []).append(i)
        macro_groups = list(by_cluster.values())

        # Seuil de dispersion DÉRIVÉ de la distribution des dispersions macro. On ne
        # garde que les macros RÉELLEMENT subdivisibles (≥ min_sub_size) : les macros
        # singletons/outliers (1-2 avis) ne peuvent pas être coupés et fausseraient le
        # gap en tirant le seuil vers 0 (→ sur-subdivision de tout le reste).
        floor = derived_global.min_sub_size
        macro_disp = [_node_stats(g, vecs, weights)[1]
                      for g in macro_groups if len(g) >= floor]
        tau = _derive_tau(macro_disp)

        counter = [0]
        # Macros triés par poids social décroissant (ordre d'affichage stable).
        macro_groups.sort(key=lambda g: -_node_stats(g, vecs, weights)[3])
        for grp in macro_groups:
            mid = _build_subtree(grp, None, 0, counter, nodes, order, vecs, weights,
                                 owner, tau, resolution, seed)
            macros.append(mid)

        _name_nodes(nodes, prepared.claim_texts)
        _assign_colors(nodes, macros)
        for node in nodes.values():
            node.representative_claims = _representatives(node, vecs, prepared.claim_texts)

    return ThemeTree(
        nodes=nodes, order=order, macros=macros, dataset=ds.id, prepared=prepared,
        tau=tau, base_resolution=resolution, seed=seed, derived_global=derived_global,
    )


# --------------------------------------------------------------------------- #
# Cache d'arbres EN MÉMOIRE — partagé /analysis ↔ /insights ↔ /citations
# --------------------------------------------------------------------------- #
# L'extraction+embed sont déjà cachés sur disque (claims_endpoint) ; ce cache évite
# en plus de RECONSTRUIRE l'arbre (clustering + UMAP) à chaque appel insights/citations
# sur la même vue. Clé = paramètres qui changent la forme de l'arbre.
_TREE_CACHE: dict[tuple, ThemeTree] = {}


def get_or_build_tree(
    ds,
    *,
    backend: str | None = None,
    model: str | None = None,
    embedder: str = DEFAULT_EMBEDDER,
    min_chars: int | None = None,
    resolution: float = 1.0,
    seed: int = DEFAULT_SEED,
) -> ThemeTree:
    """Renvoie l'arbre de thèmes (depuis le cache mémoire si déjà construit)."""
    key = (ds.id, backend, model, embedder, min_chars, resolution, seed)
    tree = _TREE_CACHE.get(key)
    if tree is None:
        tree = build_theme_tree(
            ds, backend=backend, model=model, embedder=embedder,
            min_chars=min_chars, resolution=resolution, seed=seed,
        )
        _TREE_CACHE[key] = tree
    return tree


def analysis_payload(tree: ThemeTree, *, took_ms: int | None = None) -> dict:
    """Sérialise l'arbre au format du contrat : themes(x,y) + edges + params + backend_used.

    Calcule les positions UMAP 2D des centroïdes (B1) puis les arêtes de co-occurrence,
    et assemble la réponse. `took_ms` (optionnel) trace le coût total côté serveur.
    """
    t0 = perf_counter()
    ids = tree.order
    if ids:
        centroids = np.asarray([tree.nodes[i].centroid for i in ids])
        coords = _project_2d(centroids, tree.seed)
        for nid, (x, y) in zip(ids, coords):
            tree.nodes[nid].x = round(float(x), 4)
            tree.nodes[nid].y = round(float(y), 4)

    themes = [
        {
            "id": n.id,
            "label": n.label,
            "title": n.title or n.label,    # titre court LLM (repli label si non calculé)
            "x": n.x,
            "y": n.y,
            "n_avis": n.n_avis,
            "n_claims": n.n_claims,
            "weight": n.weight,
            "consensus": n.consensus,
            "dispersion": n.dispersion,
            "parent_id": n.parent_id,
            "has_children": n.has_children,
            "color": n.color,       # couleur cluster (source unique : palette.py)
            # extras hors-contrat (le front peut les ignorer) :
            "level": n.depth,
            "keywords": n.keywords,
            "representative_claims": n.representative_claims,
        }
        for n in (tree.nodes[i] for i in ids)
    ]
    edges = _cooccurrence(tree)

    prep = tree.prepared
    dg = tree.derived_global
    n_leaves = sum(1 for n in tree.nodes.values() if not n.children)
    max_depth = max((n.depth for n in tree.nodes.values()), default=0)
    params = {
        **prep.meta(),
        "resolution": tree.base_resolution,
        "seed": tree.seed,
        "n_themes": len(themes),
        "n_macros": len(tree.macros),
        "n_leaves": n_leaves,
        "max_depth": max_depth,
        "adaptive": {
            "dispersion_threshold": (None if tree.tau == float("inf") else round(tree.tau, 4)),
            "note": "subdivision si dispersion > seuil DÉRIVÉ (plus grand écart "
                    "des dispersions macro) ET ≥2 sous-thèmes viables",
        },
        "derived": None if dg is None else {
            "k": dg.k,
            "threshold": round(dg.threshold, 4),
            "min_sub_size": dg.min_sub_size,
            "knn_sim_mean": dg.pool_mean,
            "knn_sim_std": dg.pool_std,
        },
    }
    if took_ms is not None:
        params["took_ms"] = took_ms
    else:
        params["took_ms"] = round((perf_counter() - t0) * 1000)

    return {
        "themes": themes,
        "edges": edges,
        "params": params,
        "backend_used": prep.backend.name,
    }
