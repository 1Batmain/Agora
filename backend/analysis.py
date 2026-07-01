"""Endpoint `/analysis` — carte spatiale des thèmes émergents (B1 + B2 du contrat).

Part des CLAIMS émergents (pipeline ouvert avis→claims→cluster, cf. `claims_endpoint`)
et construit l'objet que le canvas du front affiche :

  - **B1 — relations** : arêtes de **co-occurrence** (deux thèmes liés quand un même
    avis porte des claims tombant dans les deux). Aucune position 2D n'est calculée —
    UMAP a été retiré, le front fait sa propre mise en page (d3-pack).
  - **B2 — hiérarchie VARIANCE-ADAPTATIVE** : pour chaque thème on mesure la
    **dispersion interne** (distance cosinus moyenne au centroïde). Un thème ne se
    subdivise que si sa dispersion dépasse un **seuil DÉRIVÉ** des données (séparation
    par plus grand écart sur la distribution des dispersions macro — ZÉRO magic-number)
    ET que la re-clusterisation de ses propres claims dégage ≥2 sous-thèmes viables.
    Profondeur variable : thèmes homogènes = feuilles, thèmes hétérogènes = subdivisés.

Tout dérive des données (généricité) : aucune liste de thèmes, aucun seuil de corpus
codé en dur. La sortie suit le contrat figé `.agent/queue/front-redesign.md` :

    POST /analysis {dataset, backend?} -> {themes, edges, params, backend_used}
    themes[i] = {id, label, n_avis, n_claims, weight, consensus, dispersion,
                 parent_id|null, has_children}   (pas de x,y : front en d3-pack)
    edges[j]  = {a, b, weight}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from time import perf_counter

import numpy as np

from backend.claims_endpoint import PreparedClaims, prepare_claims
from backend.develop import corpus_idf, rerank_order
from pipeline.claims.pipeline import DEFAULT_EMBEDDER, DEFAULT_SEED, N_REPRESENTATIVE
from pipeline.cluster.adaptive import derive_defaults, derive_k
from pipeline.cluster.knn import build_knn_graph, knn_search
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
    hook: str = ""                  # accroche LLM (phrase d'accroche) précalculée au build
    description: str = ""           # description LLM markdown (relaie les mots-clés)
    convergence: float = 0.0        # accord intra-cluster = consensus_eff (shrinkage pop.)
    keywords: list[str] = field(default_factory=list)
    representative_claims: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    color: str = ""                 # couleur cluster (source unique : palette.py)

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
    root_coarsen: dict | None = None  # diagnostic du coarsening racine (fusion macros)
    claim_idf: dict | None = None     # idf corpus des claims (D1, calculé une fois au build)

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
    subvecs32 = subvecs.astype(np.float32)
    k_eff = derive_k(subvecs.shape[0])
    neighbors = knn_search(subvecs32, k_eff)            # 1 passe k-NN → seuil + graphe
    dd = derive_defaults(subvecs32, k=k_eff, neighbors=neighbors)
    graph = build_knn_graph(subvecs, k=dd.k, threshold=dd.threshold, neighbors=neighbors)

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
# COARSENING racine — fusionne les macro-thèmes dont les centroïdes se recoupent
# --------------------------------------------------------------------------- #
def _coarsen_roots(groups: list[list[int]], vecs: np.ndarray
                   ) -> tuple[list[list[int]], float]:
    """Regroupe les clusters RACINES qui se recoupent → entrée plus grossière, distincte.

    Leiden à sa résolution naturelle (pic de modularité) sur-fragmente l'ENTRÉE quand le
    corpus est mono-sujet : beaucoup de racines aux centroïdes très proches (« se
    recoupent »). On les fusionne par un DOUBLE critère DÉRIVÉ — deux racines fusionnent
    si le cosinus de leurs centroïdes dépasse À LA FOIS :
      1. **μ + σ** de la distribution des similarités inter-centroïdes (la QUEUE HAUTE =
         paires anormalement proches *relativement* à ce corpus ; σ-boundary standard) ;
      2. **min(cohésion_A, cohésion_B)**, où cohésion = similarité moyenne d'un membre à
         son centroïde (= 1 − dispersion). Garde-fou ABSOLU de généricité : on ne fusionne
         que si les centroïdes sont plus proches l'un de l'autre que les membres ne le sont
         de leur propre centroïde — i.e. les deux thèmes se recoupent réellement.

    Le critère 2 est crucial pour la généricité : sur un corpus MULTI-SUJETS bien séparé
    (clusters serrés, centroïdes distants), la similarité inter-centroïde tombe sous la
    cohésion → AUCUNE fusion (on s'abstient). Sur un corpus mono-sujet (tiktok : sim 0.89
    > cohésion 0.83), le critère 1 borne la fusion à la queue haute. Aucun seuil de corpus
    codé en dur. Fusion TRANSITIVE (union-find). Le détail fin reste en drill-down : les
    clusters fusionnés deviennent les enfants du macro.

    Renvoie (super-groupes = listes d'INDICES de `groups`, seuil μ+σ retenu). Repli : si
    <2 groupes, ou si la fusion s'effondre à <2 macros, on s'abstient (chaque groupe seul).
    """
    n = len(groups)
    if n < 2:
        return [[i] for i in range(n)], float("nan")
    cents, cohesion = [], []
    for g in groups:
        s = vecs[g].sum(axis=0)
        nrm = float(np.linalg.norm(s))
        cents.append(s / nrm if nrm > 0 else s)
        cohesion.append(nrm / len(g) if g else 0.0)    # ‖Σv‖/n = cos moyen au centroïde
    sim = np.asarray(cents) @ np.asarray(cents).T
    coh = np.asarray(cohesion)
    iu = np.triu_indices(n, 1)
    pair = sim[iu]
    thr = float(pair.mean() + pair.std())          # μ+σ : un écart-type au-dessus de la moyenne

    floor = thr  # l'UNION doit rester au moins aussi cohésive que la barre de similarité
    # Agglomératif avec GARDE sur l'UNION (anti-chaînage). On fusionne la paire de
    # composantes la PLUS proche au-dessus de `thr`, mais SEULEMENT si la cohésion de
    # l'union reste ≥ `floor`. Sinon la paire est interdite. Sans cette garde, l'ancien
    # union-find transitif chaînait A-B-C… en un macro FOURRE-TOUT (ex. « addiction »
    # happée par « haine »). Ici, fusionner deux thèmes distincts crève la cohésion de
    # l'union → refusé ; seuls les vrais quasi-doublons (union toujours serrée) fusionnent.
    comp_fines: dict[int, list[int]] = {i: [i] for i in range(n)}
    comp_members: dict[int, list[int]] = {i: list(groups[i]) for i in range(n)}
    comp_cent: dict[int, np.ndarray] = {i: cents[i] for i in range(n)}
    forbidden: set[frozenset[int]] = set()
    while True:
        ids = list(comp_fines)
        best, best_sim = None, thr
        for ia in range(len(ids)):
            for ib in range(ia + 1, len(ids)):
                x, y = ids[ia], ids[ib]
                if frozenset((x, y)) in forbidden:
                    continue
                s = float(comp_cent[x] @ comp_cent[y])
                if s > best_sim:
                    best, best_sim = (x, y), s
        if best is None:
            break
        x, y = best
        union = comp_members[x] + comp_members[y]
        su = vecs[union].sum(axis=0)
        nu = float(np.linalg.norm(su))
        hu = nu / len(union) if union else 0.0
        if hu >= floor:                            # l'union reste cohésive → fusion
            comp_fines[x] += comp_fines[y]
            comp_members[x] = union
            comp_cent[x] = su / nu if nu > 0 else su
            del comp_fines[y], comp_members[y], comp_cent[y]
            forbidden = {p for p in forbidden if y not in p}
        else:                                      # fusionner crèverait la cohésion → interdit
            forbidden.add(frozenset((x, y)))
    supers = list(comp_fines.values())
    if len(supers) < 2:                            # rien de fusionnable proprement → on s'abstient
        return [[i] for i in range(n)], thr
    return supers, thr


# --------------------------------------------------------------------------- #
# Construction de l'arbre (récursive, variance-adaptative)
# --------------------------------------------------------------------------- #
def _build_subtree(members: list[int], parent_id: str | None, depth: int,
                   counter: list[int], nodes: dict[str, ThemeNode], order: list[str],
                   vecs: np.ndarray, weights: np.ndarray, owner: list[int],
                   tau: float, base_resolution: float, seed: int,
                   forced_children: list[list[int]] | None = None) -> str:
    """Crée le nœud de `members`, le subdivise si hétérogène (dispersion > τ), récursif.

    `forced_children` (optionnel) impose la partition des enfants au lieu de la dériver
    par variance — sert au COARSENING racine : un macro fusionné prend pour enfants les
    clusters fins qu'il regroupe (détail préservé en drill-down). Les enfants, eux,
    re-subdivisent normalement (variance-adaptatif).
    """
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

    child_groups = forced_children
    if child_groups is None and depth < MAX_DEPTH and dispersion > tau:
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
    # top_k élevé : on garde jusqu'à 25 mots-clés c-TF-IDF par nœud (le label n'en
    # affiche que label_k ; le reste alimente le bandeau scrollable de la synthèse).
    names = name_clusters(cluster_docs, top_k=25, corpus_stopwords=corpus_stop)
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


def _shrinkage_k(pops: list[int]) -> float:
    """Force de shrinkage bayésien = population MÉDIANE d'un macro-thème (≥1).

    Source UNIQUE du `k` partagé par l'indice de consensus global, la convergence
    intra-cluster et la teinte de la carte → une seule définition de « combien
    d'évidence il faut pour qu'un accord compte ».
    """
    k = len(pops)
    if k == 0:
        return 1.0
    s = sorted(pops)
    mid = k // 2
    median = (s[mid - 1] + s[mid]) / 2 if k % 2 == 0 else s[mid]
    return max(1.0, float(median))


def _assign_convergence(nodes: dict[str, ThemeNode], macros: list[str]) -> None:
    """Convergence intra-cluster de chaque nœud = consensus_eff (shrinkage population).

    consensus_eff = (N / (N + k)) · consensus, prior 0, k = population macro médiane
    (cf. `_shrinkage_k`) — MÊME formule que l'indice de consensus global et que la
    teinte de la carte. Un thème à peu d'avis ne peut pas afficher un accord fort
    (évidence insuffisante) ; normalisé [0..1].
    """
    if not macros:
        return
    kk = _shrinkage_k([max(0, nodes[m].n_avis) for m in macros])
    for node in nodes.values():
        n = max(0, node.n_avis)
        node.convergence = round((n / (n + kk)) * node.consensus, 4) if n else 0.0


def _representatives(node: ThemeNode, vecs: np.ndarray, claim_texts: list[str],
                     k: int = N_REPRESENTATIVE, idf: dict | None = None) -> list[str]:
    """Représentants du nœud : arguments DÉVELOPPÉS on-topic (D1), sans doublon littéral.

    Re-ranking `centralité(garde-fou) × développement` (`backend.develop`) : on ne
    surface plus le médoïde court et générique mais l'argument étoffé, la centralité
    restant en garde-fou anti hors-sujet. `idf` = idf corpus des claims (calculé une
    fois au build) ; recalculé localement si absent (repli).
    """
    if not node.members:
        return []
    sims = vecs[node.members] @ node.centroid
    texts = [claim_texts[ci] for ci in node.members]
    if idf is None:
        idf = corpus_idf(texts)
    order = rerank_order(node.members, sims, texts, idf)
    reps: list[str] = []
    for j in order:
        t = texts[j]
        if any(t.lower() == e.lower() for e in reps):
            continue
        reps.append(t)
        if len(reps) >= k:
            break
    return reps


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
    seen: set[frozenset[str]] = set()           # dé-doublonne les fratries identiques
    for sibling_ids in groups:
        if len(sibling_ids) < 2:
            continue
        key = frozenset(sibling_ids)             # (incrémental : macros == enfants racine)
        if key in seen:
            continue
        seen.add(key)
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
    extract_progress=None,
) -> ThemeTree:
    """Extrait/charge les claims puis construit l'arbre variance-adaptatif.

    `prepared` permet de réutiliser une extraction+embed déjà faite (insights/citations
    rejouent l'arbre sans re-préparer). `extract_progress(done, total)` suit l'extraction
    LLM (longue). Ne sérialise PAS le payload du contrat (c'est le rôle de `analysis_payload`).
    """
    if prepared is None:
        kw = {} if min_chars is None else {"min_chars": min_chars}
        prepared = prepare_claims(ds, backend=backend, model=model, embedder=embedder,
                                  progress=extract_progress, **kw)

    vecs = prepared.claim_vecs
    weights = prepared.claim_weight
    owner = prepared.claim_owner
    n_claims = len(prepared.claim_texts)

    nodes: dict[str, ThemeNode] = {}
    order: list[str] = []
    macros: list[str] = []
    # idf corpus des claims — calculé UNE fois, partagé par tous les nœuds (D1) et /citations.
    claim_idf = corpus_idf(prepared.claim_texts) if n_claims else None
    derived_global = derive_defaults(vecs.astype(np.float32)) if n_claims else None
    tau = float("inf")
    root_coarsen: dict | None = None

    if n_claims:
        # Niveau 0 brut : clusters FINS (Leiden base sur le graphe global dérivé).
        graph = build_knn_graph(vecs, k=derived_global.k, threshold=derived_global.threshold)
        membership = run_leiden(graph, resolution=resolution, seed=seed).membership
        by_cluster: dict[int, list[int]] = {}
        for i, c in enumerate(membership):
            by_cluster.setdefault(c, []).append(i)
        fine_groups = list(by_cluster.values())

        # COARSENING de l'ENTRÉE : fusionne les racines aux centroïdes trop proches
        # (μ+σ, dérivé) → moins de macros, plus distincts. Les clusters fins fusionnés
        # deviennent les enfants (drill-down). Aucun effet si rien ne se recoupe.
        super_groups, merge_thr = _coarsen_roots(fine_groups, vecs)

        # Seuil de dispersion DÉRIVÉ de la distribution des dispersions des clusters
        # FINS. On ne garde que ceux RÉELLEMENT subdivisibles (≥ min_sub_size) : les
        # singletons/outliers (1-2 avis) ne peuvent pas être coupés et fausseraient le
        # gap en tirant le seuil vers 0 (→ sur-subdivision de tout le reste).
        floor = derived_global.min_sub_size
        macro_disp = [_node_stats(g, vecs, weights)[1]
                      for g in fine_groups if len(g) >= floor]
        tau = _derive_tau(macro_disp)

        counter = [0]
        # Macros (super-groupes) triés par poids social décroissant ; à l'intérieur, les
        # clusters fins fusionnés sont triés de même (ordre d'affichage stable).
        def _w(members: list[int]) -> float:
            return _node_stats(members, vecs, weights)[3]
        merged = [sorted((fine_groups[i] for i in sg), key=lambda g: -_w(g))
                  for sg in super_groups]
        merged.sort(key=lambda fine: -_w([m for g in fine for m in g]))
        for fine in merged:
            union = [m for g in fine for m in g]
            forced = fine if len(fine) >= 2 else None    # ≥2 fins fusionnés ⇒ drill-down
            mid = _build_subtree(union, None, 0, counter, nodes, order, vecs, weights,
                                 owner, tau, resolution, seed, forced_children=forced)
            macros.append(mid)
        root_coarsen = {
            "n_fine": len(fine_groups), "n_macros": len(macros),
            "merge_threshold": (None if merge_thr != merge_thr else round(merge_thr, 4)),
            "criterion": "fusion racines si cos(centroïdes) > μ+σ des sims inter-centroïdes ET > min(cohésion) (garde-fou généricité)",
        }

        _name_nodes(nodes, prepared.claim_texts)
        _assign_colors(nodes, macros)
        _assign_convergence(nodes, macros)
        for node in nodes.values():
            node.representative_claims = _representatives(
                node, vecs, prepared.claim_texts, idf=claim_idf)

    return ThemeTree(
        nodes=nodes, order=order, macros=macros, dataset=ds.id, prepared=prepared,
        tau=tau, base_resolution=resolution, seed=seed, derived_global=derived_global,
        root_coarsen=root_coarsen, claim_idf=claim_idf,
    )


# --------------------------------------------------------------------------- #
# Cache d'arbres EN MÉMOIRE — partagé /analysis ↔ /insights ↔ /citations
# --------------------------------------------------------------------------- #
# L'extraction+embed sont déjà cachés sur disque (claims_endpoint) ; ce cache évite
# en plus de RECONSTRUIRE l'arbre (clustering + hiérarchie) à chaque appel insights/citations
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


# --------------------------------------------------------------------------- #
# Indices GLOBAUX du dataset — « capter le débat d'un coup d'œil »
# --------------------------------------------------------------------------- #
# Quatre index DÉRIVÉS, normalisés [0..1], calculés sur les thèmes MACRO (le niveau
# que la carte montre par défaut). Choisis pour être ORTHOGONAUX : deux lisent la
# distribution des TAILLES (diversité = équilibre, concentration = domination), un
# lit l'ACCORD interne (consensus, pondéré-évidence comme la carte), un lit la
# STRUCTURE hiérarchique (facettes). Tout est dérivé des données — aucun seuil de
# corpus, aucun magic-number arbitraire.
def _gini(values: list[float]) -> float:
    """Coefficient de Gini d'une distribution positive (0 = égal, → 1 = concentré).

    Formulation par différences absolues moyennes, normalisée par 2·n·moyenne.
    Renvoie 0.0 sur cas dégénéré (≤1 valeur ou somme nulle).
    """
    n = len(values)
    if n < 2:
        return 0.0
    total = sum(values)
    if total <= 0:
        return 0.0
    s = sorted(values)
    # Σ (2i − n − 1) xᵢ  /  (n · Σx)   (i de 1..n) — identité de la moyenne des écarts.
    cum = sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(s))
    return max(0.0, min(1.0, cum / (n * total)))


def _dataset_stats(tree: ThemeTree, *, macro_ids: list[str] | None = None) -> dict:
    """Indices globaux dérivés du dataset — DONNÉE PURE (la copie FR vit côté front).

    Calculés sur les thèmes MACRO. Chaque index porte `{key, value, detail}` : `value`
    ∈ [0..1] et `detail` chiffré porte TOUS les nombres dont le front a besoin pour
    reconstruire libellé + explication. Aucune chaîne d'UI ici.

    `macro_ids` permet de fixer le niveau « macro » d'affichage (utile au mode
    incrémental où la racine unique est un super-nœud : on lit alors ses enfants).
    Repli sur `tree.macros`.
    """
    macros = [tree.nodes[mid] for mid in (macro_ids if macro_ids is not None else tree.macros)]
    pops = [max(0, m.n_avis) for m in macros]
    total_avis = sum(pops)
    k = len(macros)

    totals = {
        "participants": total_avis,     # = n_avis : « nombre de participants » (exposé clair)
        "n_avis": total_avis,
        "n_claims": sum(m.n_claims for m in macros),
        "n_themes": k,
        "n_leaves": sum(1 for n in tree.nodes.values() if not n.children),
        "max_depth": max((n.depth for n in tree.nodes.values()), default=0),
    }
    if k == 0 or total_avis == 0:
        return {"totals": totals, "indices": []}

    shares = [p / total_avis for p in pops]

    # 1) EFFUSION (variété des avis) — équitabilité de Pielou = entropie de Shannon
    #    normalisée par ln(K). 0 = un thème écrase tout (débat monolithique) ; 1 = voix
    #    réparties également entre sujets. `effective_themes` = exp(H) : « nombre
    #    effectif de sujets » (Hill q=1). Affinage de l'ancienne « diversité ».
    nz = [s for s in shares if s > 0]
    h = -sum(s * np.log(s) for s in nz)
    evenness = float(h / np.log(k)) if k > 1 else 0.0
    eff_themes = float(np.exp(h))

    # 2) CONCENTRATION — part des voix dans le thème dominant (lecture immédiate).
    #    0 = aucune domination ; 1 = tout dans un seul thème. Gini en détail.
    top_share = max(shares)

    # 3) CONSENSUS GLOBAL — moyenne pondérée-population du consensus_eff (shrinkage
    #    bayésien, MÊME formule que la carte/convergence intra : prior bas, k = population
    #    médiane). Un thème à peu d'avis ne peut pas peser comme un fort accord.
    kk = _shrinkage_k(pops)                     # force de shrinkage ~ taille typique
    eff = [(p / (p + kk)) * m.consensus for p, m in zip(pops, macros)]  # PRIOR = 0
    consensus_global = sum(p * e for p, e in zip(pops, eff)) / total_avis
    consensus_global = float(max(0.0, min(1.0, consensus_global)))

    # 4) STRUCTURATION — part des voix dans des thèmes SUBDIVISÉS (à facettes). Lit la
    #    hiérarchie variance-adaptative : 0 = débat plat (sujets simples) ; 1 = gros
    #    thèmes tous multi-facettes. Dimension orthogonale aux tailles et à l'accord.
    structured_avis = sum(p for p, m in zip(pops, macros) if m.children)
    structuration = structured_avis / total_avis

    indices = [
        {
            "key": "effusion",
            "value": round(evenness, 4),
            "detail": {"effective_themes": round(eff_themes, 2), "n_themes": k},
        },
        {
            "key": "concentration",
            "value": round(top_share, 4),
            "detail": {"top_share": round(top_share, 4), "gini": round(_gini(pops), 4)},
        },
        {
            "key": "consensus",
            "value": round(consensus_global, 4),
            "detail": {"shrinkage_k": round(kk, 2), "prior": 0.0},
        },
        {
            "key": "structuration",
            "value": round(structuration, 4),
            "detail": {
                "share": round(structuration, 4),   # part des voix en thèmes à facettes
                "structured_macros": sum(1 for m in macros if m.children),
                "n_macros": k,
            },
        },
    ]
    return {"totals": totals, "indices": indices}


def _dataset_context(dataset_id: str) -> str:
    """Intro lisible du dataset (description + contexte de collecte) pour la vue globale.

    Générique : lit le champ optionnel ``context`` du descripteur d'ingestion
    ``pipeline/ingest/descriptors/<id>.json`` (repli sur ``label``, sinon vide).
    Aucun texte corpus-spécifique dans le code — tout vient du descripteur.
    """
    import json
    from pathlib import Path
    desc = (Path(__file__).resolve().parent.parent
            / "pipeline" / "ingest" / "descriptors" / f"{dataset_id}.json")
    try:
        d = json.loads(desc.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return (d.get("context") or d.get("label") or "").strip()


def theme_dict(n: ThemeNode) -> dict:
    """Sérialise un nœud au format du contrat (SANS x,y — UMAP retiré, front en d3-pack)."""
    return {
        "id": n.id,
        "label": n.label,
        "title": n.title or n.label,    # titre court LLM (repli label si non calculé)
        "hook": n.hook,                 # accroche LLM (phrase d'accroche)
        "description": n.description,   # description LLM markdown (relaie les mots-clés)
        "n_avis": n.n_avis,
        "n_claims": n.n_claims,
        "weight": n.weight,
        "consensus": n.consensus,
        "convergence": n.convergence,   # accord intra-cluster (consensus_eff pondéré pop.)
        "dispersion": n.dispersion,
        "parent_id": n.parent_id,
        "has_children": n.has_children,
        "color": n.color,               # couleur cluster (source unique : palette.py)
        # extras hors-contrat (le front peut les ignorer) :
        "level": n.depth,
        "keywords": n.keywords,
        "representative_claims": n.representative_claims,
    }


def analysis_payload(tree: ThemeTree, *, took_ms: int | None = None,
                     macro_ids: list[str] | None = None) -> dict:
    """Sérialise l'arbre au format du contrat : themes + edges + params + backend_used.

    Plus de positions UMAP (front en d3-pack) : seules les arêtes de co-occurrence et la
    hiérarchie sont calculées. `took_ms` (optionnel) trace le coût total côté serveur.
    `macro_ids` fixe le niveau macro d'affichage pour les indices globaux (incrémental).
    """
    t0 = perf_counter()
    ids = tree.order

    themes = [theme_dict(tree.nodes[i]) for i in ids]
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
        "root_coarsen": tree.root_coarsen,   # fusion des racines trop proches (entrée grossière)
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
        "dataset_stats": _dataset_stats(tree, macro_ids=macro_ids),   # indices globaux dérivés (UI : coup d'œil)
        "dataset_context": _dataset_context(tree.dataset),  # intro vue globale (descripteur)
        "backend_used": prep.backend.name,
    }
