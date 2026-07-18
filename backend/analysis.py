"""Endpoint `/analysis` — carte spatiale des thèmes émergents (B1 + B2 du contrat).

Part des CLAIMS émergents (pipeline ouvert avis→claims→cluster, cf. `claims_endpoint`)
et construit l'objet que le canvas du front affiche :

  - **B1 — relations** : arêtes de **co-occurrence** (deux thèmes liés quand un même
    avis porte des claims tombant dans les deux). Aucune position 2D n'est calculée —
    UMAP a été retiré, le front fait sa propre mise en page (d3-pack).
  - **B2 — hiérarchie MESURÉE** : l'arbre suit la CHAÎNE D'EMBOÎTEMENT complète
    (`pipeline.cluster.layers`) — on balaie k (le zoom) et on lit, dans l'emboîtement des
    partitions, AUTANT DE NIVEAUX que la chaîne en dégage de propres (on ne fixe pas le
    nombre : tiktok 4→9→16, rép-num 5→9→17→31…). Les clusters les plus fins sont les feuilles ;
    plus aucune re-clusterisation Leiden par nœud (cf. `HIERARCHY_LAYERS.md`).

Tout dérive des données (généricité) : aucune liste de thèmes, aucun seuil de corpus
codé en dur. La sortie suit le contrat figé `.agent/queue/front-redesign.md` :

    POST /analysis {dataset, backend?} -> {themes, edges, params, backend_used}
    themes[i] = {id, label, n_avis, n_claims, weight, consensus, dispersion,
                 parent_id|null, has_children}   (pas de x,y : front en d3-pack)
    edges[j]  = {a, b, weight}
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from itertools import combinations
from time import perf_counter

import numpy as np

from backend.claims_endpoint import PreparedClaims, prepare_claims
from backend.develop import corpus_idf, rerank_order
from pipeline.claims.pipeline import (
    DEFAULT_EMBEDDER,
    DEFAULT_RESOLUTION,
    DEFAULT_SEED,
    N_REPRESENTATIVE,
)
from pipeline.cluster import layers
from pipeline.cluster.adaptive import derive_defaults
from pipeline.cluster.naming import derive_corpus_stopwords, name_clusters
from pipeline.cluster.palette import color_for

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
    # Avis le plus REPRÉSENTATIF du thème (score composite au build, cf `_hero_avis`) —
    # remplace le médoïde centroïde (générique sur corpus anisotrope). None si indéterminé.
    hero_avis_id: str | None = None
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
                   forced_children: list | None = None) -> str:
    """Crée le nœud de `members` et, récursivement, ses enfants forcés.

    `forced_children` : liste de nœuds `(membres, sous-enfants)` — la hiérarchie MESURÉE par
    la chaîne d'emboîtement, de profondeur QUELCONQUE. On ne fixe PAS le nombre de niveaux :
    tant que la chaîne dégage un étage qui s'emboîte proprement, il devient une profondeur de
    l'arbre. Un nœud sans enfants forcés est une FEUILLE — plus de re-clusterisation Leiden
    (l'ancien `_subdivide`, piloté par `derive_k`, cf. `HIERARCHY_LAYERS.md`).
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

    if forced_children:
        # Enfants triés par poids social décroissant (ordre d'affichage stable).
        for child_members, child_kids in sorted(
                forced_children, key=lambda c: -_node_stats(c[0], vecs, weights)[3]):
            cid = _build_subtree(child_members, node_id, depth + 1, counter, nodes, order,
                                 vecs, weights, owner, forced_children=child_kids or None)
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
# Hero représentatif — score COMPOSITE (remplace le médoïde centroïde, générique sur
# corpus mono-sujet anisotrope, cf research/cluster_merge_note.md)
# --------------------------------------------------------------------------- #
HERO_MIN_CHARS = 200
HERO_MAX_CHARS = 1200


def _lisibilite(text: str) -> float:
    """Score de lisibilité [0,1] d'un avis : fenêtre `HERO_MIN..HERO_MAX` caractères.

    Ni trop court (peu informatif/générique), ni trop long (illisible en hero). La
    lisibilité FR est assurée EN AVAL par le front (affiche `text_fr` traduit)."""
    n = len(text or "")
    if n <= 0:
        return 0.0
    if n < HERO_MIN_CHARS:
        return n / HERO_MIN_CHARS
    if n > HERO_MAX_CHARS:
        return HERO_MAX_CHARS / n
    return 1.0


def _hero_avis(node: ThemeNode, prepared, avis_total: np.ndarray) -> str | None:
    """Avis le plus REPRÉSENTATIF du thème par score COMPOSITE (générique).

    Pour chaque avis ayant ≥1 claim dans le thème :
      hero = argmax( pureté × couverture × représentativité × lisibilité )
    sur les claims du thème :
      - pureté           = part des claims de l'avis qui tombent DANS le thème ;
      - couverture       = nb de claims de l'avis dans le thème ;
      - représentativité = sim moyenne de ses claims à TOUS les claims du thème
                           (embeddings ; moyenne des vecteurs, PAS le centroïde normalisé) ;
      - lisibilité       = fenêtre 200–1200 chars (`_lisibilite`).
    `avis_total[a]` = nb TOTAL de claims de l'avis `a` (dénominateur de la pureté)."""
    members = node.members
    if not members:
        return None
    vecs = prepared.claim_vecs
    owner = prepared.claim_owner
    avis = prepared.avis
    by_avis: dict[int, list[int]] = {}
    for ci in members:
        by_avis.setdefault(owner[ci], []).append(ci)
    theme_mean = vecs[members].mean(axis=0)          # moyenne des embeddings (≠ centroïde normalisé)
    best_id: str | None = None
    best_score = -1.0
    for aidx, cis in by_avis.items():
        if aidx >= len(avis):
            continue
        total = int(avis_total[aidx]) if aidx < len(avis_total) else len(cis)
        purete = (len(cis) / total) if total else 0.0
        couverture = len(cis)
        repres = max(0.0, float(np.mean(vecs[cis] @ theme_mean)))
        lis = _lisibilite(getattr(avis[aidx], "text", "") or "")
        score = purete * couverture * repres * lis
        if score > best_score:
            best_score = score
            best_id = str(getattr(avis[aidx], "id", "") or "") or None
    return best_id


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
# Cœur PARTAGÉ de construction de la forêt de macros (claims /analysis ↔ idées Console)
# --------------------------------------------------------------------------- #
def _build_macro_forest(
    fine_groups: list[list[int]],
    vecs,
    weights,
    owner: list[int],
    texts: list[str],
    *,
    resolution: float = DEFAULT_RESOLUTION,
    seed: int = DEFAULT_SEED,
    hierarchy: list | None = None,
) -> tuple[dict, list, list, float]:
    """Construit et remplit la forêt de thèmes à partir d'une hiérarchie emboîtée.

    Deux sources de hiérarchie (`hierarchy` = liste de nœuds `(membres, sous-enfants)`) :
      - `/analysis` la passe TELLE QUELLE, la partition PLATE au pic de modularité (chaque cluster = un thème, profondeur 0).
      - la Console live la laisse à `None` → coarsening racine (`_coarsen_roots`) à 2 niveaux :
        c'est un explorateur k MANUEL, il ne balaie pas la chaîne.
    Puis `_build_subtree` matérialise l'arbre, et nommage c-TF-IDF / couleurs par macro /
    convergence intra. Agnostique au TYPE de membre (claims pour `/analysis`, idées pour la
    Console). Renvoie `(nodes, order, macros, merge_thr)`.
    """
    nodes: dict = {}
    order: list = []
    macros: list = []

    def _w(members: list[int]) -> float:
        return _node_stats(members, vecs, weights)[3]

    if hierarchy is None:
        # COARSENING de l'ENTRÉE : fusionne les racines aux centroïdes trop proches (μ+σ,
        # dérivé) → forêt à 2 niveaux (macro → clusters fins, ou macro-feuille si un seul fin).
        super_groups, merge_thr = _coarsen_roots(fine_groups, vecs)
        hierarchy = []
        for sg in super_groups:
            fines = [fine_groups[i] for i in sg]
            if len(fines) >= 2:
                hierarchy.append(([m for g in fines for m in g], [(g, []) for g in fines]))
            else:
                hierarchy.append((fines[0], []))
    else:
        merge_thr = float("nan")

    counter = [0]
    for top_members, top_children in sorted(hierarchy, key=lambda nd: -_w(nd[0])):
        mid = _build_subtree(top_members, None, 0, counter, nodes, order, vecs, weights,
                             owner, forced_children=top_children or None)
        macros.append(mid)

    _name_nodes(nodes, texts)
    _assign_colors(nodes, macros)
    _assign_convergence(nodes, macros)
    return nodes, order, macros, merge_thr


# --------------------------------------------------------------------------- #
# Point d'entrée : construit l'arbre (sans x,y) — réutilisé par insights/citations
# --------------------------------------------------------------------------- #
def _central_texts(members: list[int], vecs: np.ndarray, texts: list[str], top: int = 15) -> list[str]:
    """Claims les plus proches du centroïde d'un cluster (pour l'étiquette canonique)."""
    sub = vecs[members]
    c = sub.mean(axis=0)
    nrm = np.linalg.norm(c)
    c = c / nrm if nrm else c
    order = np.argsort(-(sub @ c))[:top]
    return [texts[members[i]] for i in order]


def _abstraction(ds, clusters: list[list[int]], vecs: np.ndarray, texts: list[str],
                 *, model: str | None, compute: bool) -> dict | None:
    """Couche macro : relit le cache, ou la CALCULE (LLM) si `compute` et clé dispo.

    Cachée par signature de la partition → cohérente entre build_analysis/opinion/arguments.
    Repli PLAT (None) si : pas de cache, pas de calcul demandé, pas de clé, ou trop peu de thèmes.
    """
    from pathlib import Path

    from pipeline.cluster import abstraction as ab

    path = Path(f"backend/cache/{ds.id}/analysis/abstraction.json")
    cached = ab.load(path, clusters)
    if cached is not None:
        return cached
    if not compute:
        return None
    from pipeline.cluster import mistral_client
    from pipeline.embed.embedder import embed as _embed
    if not mistral_client.available():
        return None
    cluster_texts = [_central_texts(m, vecs, texts) for m in clusters]
    # Abstraction = tâche de nommage/regroupement légère → modèle SMALL (moins cher que
    # l'extraction). `model` (extraction) ignoré ici volontairement.
    result = ab.compute(cluster_texts, chat_fn=mistral_client.chat, embed_fn=_embed,
                        model="mistral-small-latest")
    if result is not None:
        ab.save(path, clusters, result)
    return result


def build_theme_tree(
    ds,
    *,
    backend: str | None = None,
    model: str | None = None,
    embedder: str = DEFAULT_EMBEDDER,
    min_chars: int | None = None,
    resolution: float = DEFAULT_RESOLUTION,
    seed: int = DEFAULT_SEED,
    prepared: PreparedClaims | None = None,
    extract_progress=None,
    abstract: bool = False,
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
    if len(vecs):
        # RECENTRAGE de l'espace : corrige l'anisotropie du modèle d'embedding (les vecteurs
        # vivent dans un cône étroit). Zéro paramètre, +19 % d'ARI sur le gold.
        vecs = layers.centre(vecs).astype(np.float32)
    weights = prepared.claim_weight
    owner = prepared.claim_owner
    n_claims = len(prepared.claim_texts)

    nodes: dict[str, ThemeNode] = {}
    order: list[str] = []
    macros: list[str] = []
    # idf corpus des claims — calculé UNE fois, partagé par tous les nœuds (D1) et /citations.
    claim_idf = corpus_idf(prepared.claim_texts) if n_claims else None
    derived_global = derive_defaults(vecs.astype(np.float32)) if n_claims else None
    root_coarsen: dict | None = None

    if n_claims:
        # COUCHE PLATE au PIC DE MODULARITÉ : on ne balaie plus k (qui changeait le graphe et
        # dégénérait en pure densification), on fixe UN graphe et on balaie la résolution γ —
        # le bouton direct de granularité. Le pic de modularité donne le grain naturel du
        # corpus. Les couches ABSTRAITES au-dessus viendront d'un autre mécanisme (ré-embedding
        # des synthèses de thèmes, cf. `research/synthesis_embed_note.md`) — pas d'un γ plus
        # grossier. Ici l'arbre est donc PLAT : chaque cluster = un thème.
        membership, gmeta = layers.flat_partition(vecs, seed=seed)
        by_cluster: dict[int, list[int]] = {}
        for i, c in enumerate(membership.tolist()):
            by_cluster.setdefault(c, []).append(i)
        clusters = list(by_cluster.values())

        # COUCHE MACRO (abstraction) : regroupe les thèmes redondants sans souder les distincts.
        # Calculée UNE fois au build (LLM, `abstract=True`) et CACHÉE ; relue par les autres
        # étapes → arbre identique partout (cohérence build_analysis/opinion/arguments).
        absres = _abstraction(ds, clusters, vecs, prepared.claim_texts,
                              model=model, compute=abstract)
        if absres:
            groups: dict[int, list[int]] = {}
            for ti, mi in enumerate(absres["assign"]):
                groups.setdefault(mi, []).append(ti)
            hierarchy = [([m for t in tis for m in clusters[t]],
                          [(clusters[t], []) for t in tis]) for tis in groups.values()]
        else:
            hierarchy = [(members, []) for members in clusters]     # plat (repli)

        nodes, order, macros, merge_thr = _build_macro_forest(
            [], vecs, weights, owner, prepared.claim_texts,
            resolution=resolution, seed=seed, hierarchy=hierarchy)
        root_coarsen = {
            "n_macros": len(macros), "flat": absres is None,
            "criterion": ("macros = abstraction (étiquette canonique LLM + affectation embedding)"
                          if absres else "partition plate au pic de modularité (γ balayé, graphe fixe)"),
            "gamma": gmeta["gamma"], "modularity": gmeta["modularity"],
            "n_fine": gmeta["n_clusters"], "gamma_curve": gmeta["curve"],
            "macro_titles": (absres["macros"] if absres else None),
        }

        # nb TOTAL de claims par avis (dénominateur de la pureté du hero) — calculé une fois.
        avis_total = (np.bincount(np.asarray(prepared.claim_owner, dtype=int),
                                  minlength=len(prepared.avis))
                      if prepared.claim_owner else np.zeros(len(prepared.avis), dtype=int))
        for node in nodes.values():
            node.representative_claims = _representatives(
                node, vecs, prepared.claim_texts, idf=claim_idf)
            node.hero_avis_id = _hero_avis(node, prepared, avis_total)

    tree = ThemeTree(
        nodes=nodes, order=order, macros=macros, dataset=ds.id, prepared=prepared,
        base_resolution=resolution, seed=seed, derived_global=derived_global,
        root_coarsen=root_coarsen, claim_idf=claim_idf,
    )
    return tree


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
    resolution: float = DEFAULT_RESOLUTION,
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
        # « cohesion » = COHÉSION SÉMANTIQUE (proximité des claims dans l'espace d'embedding),
        # PAS un accord d'opinion — nom honnête (contrat de métriques). `consensus` reste servi
        # en ALIAS rétro-compat pour les clients existants.
        "cohesion": n.consensus,
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
        # Avis hero (score composite) — le front l'affiche en repli du 1er représentatif.
        "hero_avis_id": n.hero_avis_id,
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
            "note": "hiérarchie = chaîne d'emboîtement (2 niveaux mesurés : macros → thèmes "
                    "fins). Aucune re-subdivision des feuilles, aucun seuil de dispersion : "
                    "cf. .agent/notes/HIERARCHY_LAYERS.md, HIERARCHY_TAU.md",
        },
        "root_coarsen": tree.root_coarsen,   # chaîne d'emboîtement (k balayé) + propreté macro
        "derived": None if dg is None else {   # DIAGNOSTIC du graphe global (non clusterisé : la chaîne l'est)
            "k": dg.k,
            "threshold": round(dg.threshold, 4),
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
