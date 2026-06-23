"""Bac à sable « console de mixage » — RECLUSTER LIVE sans LLM (~1 s / 3000 claims).

Cœur de la LANE 2 du plan de nuit. À partir des claims + embeddings DÉJÀ CACHÉS
(claims_emb + target_emb), rejoue toute la chaîne de structuration en faisant varier
des KNOBS, et renvoie en plus une **decision-trace** chiffrée (« verre, pas boîte
noire ») :

    blend(α) → graphe k-NN(k) → Leiden(resolution) → subdivision variance-adaptative
    (τ × tau_mult) → coarsening des racines (seuil μ+σ × coarsen_mult)

Aucun appel modèle/LLM : on réutilise les vecteurs cachés (le knob α mélange juste
claim↔cible déjà embeddés). Labels = **c-TF-IDF** passif (pas de naming réglable).

Réutilise les primitives de `backend.analysis` (`_node_stats`, `_subdivide`,
`_derive_tau`) et de `pipeline.cluster.*` (defaults dérivés, k-NN, Leiden, naming)
pour garantir une structure cohérente avec l'analyse servie (à knobs neutres).

API :
  - `get_prepared(ds)`      → PreparedClaims caché en mémoire (1ʳᵉ fois : lit le disque).
  - `recluster_payload(...)`→ payload du CONTRAT (`params/clusters/trace/ms`).
  - `explain_cluster(...)` / `explain_pair(...)` → voisinage + critères (cf. contrat).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from backend.analysis import _derive_tau, _node_stats, _subdivide, MAX_DEPTH
from backend.claims_endpoint import PreparedClaims, prepare_claims
from pipeline.claims.pipeline import DEFAULT_SEED, blend_embeddings
from pipeline.cluster.adaptive import derive_defaults
from pipeline.cluster.knn import build_knn_graph, knn_search
from pipeline.cluster.leiden_cluster import run_leiden
from pipeline.cluster.naming import derive_corpus_stopwords, name_clusters

N_SAMPLE_CLAIMS = 4          # claims d'exemple par cluster (les plus centrales, dédupées)


# --------------------------------------------------------------------------- #
# Cache MÉMOIRE : PreparedClaims par dataset (lecture disque une seule fois) +
# dernier état de recluster (pour /explain, qui ne reçoit pas les knobs).
# --------------------------------------------------------------------------- #
_PREP_CACHE: dict[str, PreparedClaims] = {}
_LAST_STATE: dict[str, "SandboxState"] = {}


def get_prepared(ds) -> PreparedClaims:
    """PreparedClaims d'un dataset (claims+embeddings+cibles cachés), mémoïsé.

    1ʳᵉ fois : lit les caches disque (claims.json + claims_emb.npz + target_emb.npz)
    via `prepare_claims`. ZÉRO extraction LLM tant que les claims sont cachés (ils le
    sont après un build) ; sinon `prepare_claims` lèverait `OllamaUnavailable`.

    On passe le MODÈLE d'extraction du build (`AGORA_EXTRACT_MODEL`) pour que la clé de
    cache claims matche (sinon `prepare_claims` retomberait sur le modèle API par défaut
    et ré-extrairait tout). `target_emb` est généré au 1ᵉʳ appel si absent (embed nomic
    des cibles, CPU, aucun LLM) puis caché.
    """
    prep = _PREP_CACHE.get(ds.id)
    if prep is None:
        from backend.build_analysis import EXTRACT_MODEL
        prep = prepare_claims(ds, model=EXTRACT_MODEL)
        _PREP_CACHE[ds.id] = prep
    return prep


def invalidate(dataset_id: str) -> None:
    """Purge les caches mémoire d'un dataset (à appeler après un rebuild)."""
    _PREP_CACHE.pop(dataset_id, None)
    _LAST_STATE.pop(dataset_id, None)


# --------------------------------------------------------------------------- #
# Structures internes
# --------------------------------------------------------------------------- #
@dataclass
class _Node:
    id: str
    parent_id: str | None
    depth: int
    members: list[int]
    centroid: np.ndarray
    dispersion: float
    cohesion: float                 # = 1 − dispersion (cos moyen au centroïde)
    n_claims: int
    n_avis: int
    children: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    sample_claims: list[str] = field(default_factory=list)


@dataclass
class SandboxState:
    """Dernier recluster d'un dataset — sert /explain (qui ne reçoit pas les knobs)."""
    nodes: dict[str, _Node]
    order: list[str]
    coarsen_threshold: float        # seuil de fusion effectif (μ+σ × coarsen_mult)
    coarsen_base: float             # μ+σ brut (avant coarsen_mult)
    tau: float                      # seuil de subdivision effectif (× tau_mult)
    params: dict


# --------------------------------------------------------------------------- #
# Coarsening des racines (avec knob coarsen_mult) + trace des paires
# --------------------------------------------------------------------------- #
def _centroid_cohesion(members: list[int], vecs: np.ndarray) -> tuple[np.ndarray, float]:
    s = vecs[members].sum(axis=0)
    nrm = float(np.linalg.norm(s))
    cent = s / nrm if nrm > 0 else s
    coh = nrm / len(members) if members else 0.0
    return cent, coh


def _coarsen(fine_groups: list[list[int]], vecs: np.ndarray, coarsen_mult: float):
    """Fusionne les clusters FINS aux centroïdes trop proches (μ+σ × coarsen_mult).

    DOUBLE critère DÉRIVÉ (cf. `backend.analysis._coarsen_roots`) : deux racines
    fusionnent si `cos(centroïdes) > seuil` ET `> min(cohésion_a, cohésion_b)`. Le
    knob `coarsen_mult` déplace le seuil μ+σ (×1 = comportement servi ; <1 = fusionne
    plus ; >1 = fusionne moins). Renvoie `(supers, info, pair_records, cents, cohs)` —
    `pair_records` = TOUTES les paires de fins avec leur décision (pour la trace).
    """
    n = len(fine_groups)
    cents: list[np.ndarray] = []
    cohs: list[float] = []
    for g in fine_groups:
        c, h = _centroid_cohesion(g, vecs)
        cents.append(c)
        cohs.append(h)
    if n < 2:
        info = {"base_threshold": None, "threshold": None, "coarsen_mult": coarsen_mult,
                "n_fine": n, "n_macros": n}
        return [[i] for i in range(n)], info, [], cents, cohs

    C = np.asarray(cents)
    sim = C @ C.T
    coh = np.asarray(cohs)
    iu = np.triu_indices(n, 1)
    pair = sim[iu]
    base_thr = float(pair.mean() + pair.std())
    thr = base_thr * float(coarsen_mult)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(n):
        for b in range(a + 1, n):
            if sim[a, b] > thr and sim[a, b] > min(coh[a], coh[b]):
                parent[find(a)] = find(b)

    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    supers = list(comps.values())
    if len(supers) < 2:                         # sur-fusion → on s'abstient (chaque fin seul)
        supers = [[i] for i in range(n)]

    # Trace : toutes les paires de fins, avec décision merged = même racine in fine.
    root = {i: find(i) for i in range(n)}
    pair_records = []
    for a in range(n):
        for b in range(a + 1, n):
            pair_records.append({
                "a_fine": a, "b_fine": b,
                "sim": round(float(sim[a, b]), 4),
                "threshold": round(thr, 4),
                "cohesion_min": round(float(min(coh[a], coh[b])), 4),
                "merged": (root[a] == root[b]) and (len(supers) > 1),
            })
    info = {"base_threshold": round(base_thr, 4), "threshold": round(thr, 4),
            "coarsen_mult": coarsen_mult, "n_fine": n, "n_macros": len(supers)}
    return supers, info, pair_records, cents, cohs


# --------------------------------------------------------------------------- #
# Construction récursive de l'arbre (variance-adaptative, knob tau_mult)
# --------------------------------------------------------------------------- #
@dataclass
class _Ctx:
    vecs: np.ndarray
    weights: np.ndarray
    owner: list[int]
    tau: float
    resolution: float
    seed: int
    counter: list[int]
    nodes: dict[str, _Node]
    order: list[str]


def _make_node(members, parent_id, depth, ctx) -> _Node:
    nid = f"n{ctx.counter[0]}"
    ctx.counter[0] += 1
    centroid, dispersion, _consensus, _w = _node_stats(members, ctx.vecs, ctx.weights)
    node = _Node(
        id=nid, parent_id=parent_id, depth=depth, members=members,
        centroid=centroid, dispersion=round(float(dispersion), 4),
        cohesion=round(1.0 - float(dispersion), 4),
        n_claims=len(members), n_avis=len({ctx.owner[i] for i in members}),
    )
    ctx.nodes[nid] = node
    ctx.order.append(nid)
    return node


def _build(members, parent_id, depth, ctx, *, force_no_subdiv=False) -> str:
    """Crée le nœud puis le subdivise si hétérogène (dispersion > τ effectif)."""
    node = _make_node(members, parent_id, depth, ctx)
    if not force_no_subdiv and depth < MAX_DEPTH and node.dispersion > ctx.tau:
        groups = _subdivide(members, ctx.vecs, ctx.resolution, ctx.seed)
        if groups:
            for grp in groups:
                cid = _build(grp, node.id, depth + 1, ctx)
                node.children.append(cid)
    return node.id


# --------------------------------------------------------------------------- #
# Naming c-TF-IDF + échantillons de claims
# --------------------------------------------------------------------------- #
def _name_and_sample(ctx: _Ctx, claim_texts: list[str]) -> None:
    ids = list(ctx.nodes.keys())
    idx_of = {nid: i for i, nid in enumerate(ids)}
    docs = {idx_of[nid]: [claim_texts[i] for i in ctx.nodes[nid].members] for nid in ids}
    corpus_stop, _ = derive_corpus_stopwords(claim_texts)
    names = name_clusters(docs, corpus_stopwords=corpus_stop)
    for nid in ids:
        node = ctx.nodes[nid]
        node.keywords = names.get(idx_of[nid], {}).get("keywords", [])
        # Claims d'exemple = les plus proches du centroïde, sans doublon littéral.
        sims = ctx.vecs[node.members] @ node.centroid
        order = np.argsort(-sims)
        reps: list[str] = []
        for j in order:
            t = claim_texts[node.members[j]]
            if any(t.lower() == e.lower() for e in reps):
                continue
            reps.append(t)
            if len(reps) >= N_SAMPLE_CLAIMS:
                break
        node.sample_claims = reps


# --------------------------------------------------------------------------- #
# Point d'entrée : recluster + payload du contrat
# --------------------------------------------------------------------------- #
def recluster_payload(
    ds,
    *,
    alpha: float | None = None,
    k: int | None = None,
    resolution: float | None = None,
    coarsen_mult: float | None = None,
    tau_mult: float | None = None,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Recluster live d'un dataset selon les knobs → payload du CONTRAT `/sandbox`.

    Tous les knobs `None` → défauts (α=0, k/resolution dérivés, coarsen_mult=1,
    tau_mult=1) = la structure SERVIE. Mémoïse l'état pour `/explain`.
    """
    t0 = perf_counter()
    prep = get_prepared(ds)

    # Défauts dérivés (neutres) si non fournis.
    alpha = 0.0 if alpha is None else float(alpha)
    resolution = 1.0 if resolution is None else float(resolution)
    coarsen_mult = 1.0 if coarsen_mult is None else float(coarsen_mult)
    tau_mult = 1.0 if tau_mult is None else float(tau_mult)

    texts = prep.claim_texts
    weights = prep.claim_weight
    owner = prep.claim_owner
    n_claims = len(texts)

    # 1) BLEND α (vectorisé) — claim↔cible sur les vecteurs cachés (aucun ré-embed).
    vecs = blend_embeddings(prep.claim_vecs, prep.target_vecs, prep.target_mask, alpha)
    vecs = np.ascontiguousarray(vecs, dtype=np.float64)

    if n_claims == 0:
        return {"params": {"alpha": alpha, "k": k, "resolution": resolution,
                           "coarsen_mult": coarsen_mult, "tau_mult": tau_mult,
                           "derived": {}},
                "n_claims": 0, "ms": round((perf_counter() - t0) * 1000),
                "clusters": [], "trace": {"pairs": [], "nodes": []}}

    # 2) Défauts dérivés du graphe (k respecté si fourni), k-NN, Leiden. Le voisinage
    #    k-NN est calculé UNE fois (faiss exact) et réutilisé pour le seuil ET le graphe
    #    (au lieu de deux passes O(n²) — cf. knn_search/pool_from_neighbors).
    vecs32 = vecs.astype(np.float32)
    from pipeline.cluster.adaptive import derive_k
    k_eff = derive_k(n_claims) if k is None else int(k)
    neighbors = knn_search(vecs32, k_eff)
    derived = derive_defaults(vecs32, k=k_eff, neighbors=neighbors)
    graph = build_knn_graph(vecs, k=k_eff, threshold=derived.threshold,
                            neighbors=neighbors)
    membership = run_leiden(graph, resolution=resolution, seed=seed).membership
    by_cluster: dict[int, list[int]] = {}
    for i, c in enumerate(membership):
        by_cluster.setdefault(c, []).append(i)
    fine_groups = list(by_cluster.values())

    # 3) Coarsening des racines (knob coarsen_mult) + trace des paires.
    supers, coarsen_info, pair_records, _cents, _cohs = _coarsen(
        fine_groups, vecs, coarsen_mult)

    # 4) Seuil de dispersion DÉRIVÉ (× tau_mult) — sur les fins réellement subdivisibles.
    floor = derived.min_sub_size
    macro_disp = [_node_stats(g, vecs, weights)[1] for g in fine_groups if len(g) >= floor]
    base_tau = _derive_tau(macro_disp)
    tau_eff = base_tau * tau_mult if base_tau != float("inf") else float("inf")

    # 5) Construction de l'arbre. Macros = super-groupes triés par poids ; un macro qui
    #    regroupe ≥2 fins prend ces fins pour enfants (drill-down), chacun re-subdivisé
    #    par variance. On TRACE le node-id de chaque fin (pour mapper la trace des paires).
    ctx = _Ctx(vecs=vecs, weights=weights, owner=owner, tau=tau_eff,
               resolution=resolution, seed=seed, counter=[0], nodes={}, order=[])

    def _w(members: list[int]) -> float:
        return _node_stats(members, vecs, weights)[3]

    supers_sorted = sorted(
        supers, key=lambda sg: -_w([m for i in sg for m in fine_groups[i]]))
    fine_node_id: dict[int, str] = {}
    for sg in supers_sorted:
        fine_idx = sorted(sg, key=lambda i: -_w(fine_groups[i]))
        if len(fine_idx) >= 2:
            union = [m for i in fine_idx for m in fine_groups[i]]
            macro = _make_node(union, None, 0, ctx)
            for fi in fine_idx:
                cid = _build(fine_groups[fi], macro.id, 1, ctx)
                macro.children.append(cid)
                fine_node_id[fi] = cid
        else:
            fi = fine_idx[0]
            mid = _build(fine_groups[fi], None, 0, ctx)
            fine_node_id[fi] = mid

    # 6) Naming c-TF-IDF + claims d'exemple (passif, aucun LLM).
    _name_and_sample(ctx, texts)

    # 7) Sérialisation : clusters + trace (paires re-étiquetées en node-ids).
    clusters = [{
        "id": n.id, "parent_id": n.parent_id,
        "n_claims": n.n_claims, "n_avis": n.n_avis,
        "keywords": n.keywords, "sample_claims": n.sample_claims,
        "cohesion": n.cohesion,
        "depth": n.depth, "has_children": bool(n.children),
    } for n in (ctx.nodes[i] for i in ctx.order)]

    trace_pairs = [{
        "a": fine_node_id[p["a_fine"]], "b": fine_node_id[p["b_fine"]],
        "sim": p["sim"], "threshold": p["threshold"],
        "cohesion_min": p["cohesion_min"], "merged": p["merged"],
    } for p in pair_records]
    trace_nodes = [{
        "id": n.id,
        "dispersion": n.dispersion,
        "tau": (None if tau_eff == float("inf") else round(tau_eff, 4)),
        "subdivided": bool(n.children),
    } for n in (ctx.nodes[i] for i in ctx.order)]

    params = {
        "alpha": alpha, "k": k_eff, "resolution": resolution,
        "coarsen_mult": coarsen_mult, "tau_mult": tau_mult,
        "derived": {
            "k": derived.k,
            "threshold": round(derived.threshold, 4),
            "min_sub_size": derived.min_sub_size,
            "knn_sim_mean": derived.pool_mean,
            "knn_sim_std": derived.pool_std,
            "tau_base": (None if base_tau == float("inf") else round(base_tau, 4)),
            "tau_effective": (None if tau_eff == float("inf") else round(tau_eff, 4)),
            "coarsen": coarsen_info,
            "n_targets": int(prep.target_mask.sum()),
            "target_coverage": (round(float(prep.target_mask.mean()), 4)
                                if prep.target_mask.size else 0.0),
        },
    }
    ms = round((perf_counter() - t0) * 1000)

    # Mémoïse l'état pour /explain (centroïdes, critères, seuils).
    _LAST_STATE[ds.id] = SandboxState(
        nodes=ctx.nodes, order=ctx.order,
        coarsen_threshold=coarsen_info["threshold"] or 0.0,
        coarsen_base=coarsen_info["base_threshold"] or 0.0,
        tau=tau_eff, params=params,
    )

    return {"params": params, "n_claims": n_claims, "ms": ms,
            "clusters": clusters, "trace": {"pairs": trace_pairs, "nodes": trace_nodes}}


# --------------------------------------------------------------------------- #
# /explain — voisinage + critères, à partir du DERNIER recluster
# --------------------------------------------------------------------------- #
def _state_for(ds) -> SandboxState:
    state = _LAST_STATE.get(ds.id)
    if state is None:
        recluster_payload(ds)                   # défauts neutres → peuple l'état
        state = _LAST_STATE[ds.id]
    return state


def _macro_of(state: SandboxState, nid: str) -> str:
    """Remonte au macro (ancêtre de profondeur 0) contenant `nid` — = « cluster fusionné »."""
    cur = nid
    while state.nodes[cur].parent_id is not None:
        cur = state.nodes[cur].parent_id
    return cur


def explain_cluster(ds, cluster_id: str, k: int = 5) -> dict:
    """k clusters les plus proches (centroïdes) + critères du nœud (cf. contrat)."""
    state = _state_for(ds)
    node = state.nodes.get(cluster_id)
    if node is None:
        return {"error": f"cluster inconnu: {cluster_id!r}"}
    sims = []
    for oid, other in state.nodes.items():
        if oid == cluster_id:
            continue
        sims.append((oid, float(node.centroid @ other.centroid)))
    sims.sort(key=lambda kv: -kv[1])
    neighbors = [{
        "id": oid,
        "sim": round(s, 4),
        "cohesion": state.nodes[oid].cohesion,
        "coarsen_threshold": round(state.coarsen_threshold, 4),
        "would_merge": (s > state.coarsen_threshold
                        and s > min(node.cohesion, state.nodes[oid].cohesion)),
        "same_macro": _macro_of(state, oid) == _macro_of(state, cluster_id),
    } for oid, s in sims[:k]]
    return {
        "cluster": cluster_id,
        "criteria": {
            "dispersion": node.dispersion,
            "cohesion": node.cohesion,
            "tau": (None if state.tau == float("inf") else round(state.tau, 4)),
            "subdivided": bool(node.children),
            "parent_id": node.parent_id,
            "n_claims": node.n_claims,
            "n_avis": node.n_avis,
            "keywords": node.keywords,
        },
        "neighbors": neighbors,
        "params": state.params,
    }


def explain_pair(ds, a: str, b: str) -> dict:
    """sim / seuil / cohésions d'une PAIRE de clusters + décision de fusion (cf. contrat)."""
    state = _state_for(ds)
    na, nb = state.nodes.get(a), state.nodes.get(b)
    if na is None or nb is None:
        missing = a if na is None else b
        return {"error": f"cluster inconnu: {missing!r}"}
    sim = round(float(na.centroid @ nb.centroid), 4)
    coh_min = round(min(na.cohesion, nb.cohesion), 4)
    thr = round(state.coarsen_threshold, 4)
    would = sim > state.coarsen_threshold and sim > min(na.cohesion, nb.cohesion)
    return {
        "pair": [a, b],
        "sim": sim,
        "threshold": thr,
        "coarsen_base_threshold": round(state.coarsen_base, 4),
        "cohesion_a": na.cohesion,
        "cohesion_b": nb.cohesion,
        "cohesion_min": coh_min,
        "would_merge": would,
        "same_macro": _macro_of(state, a) == _macro_of(state, b),
        "explanation": (
            f"cos(centroïdes)={sim} {'>' if would else '≤'} seuil μ+σ×mult={thr} "
            f"ET min(cohésions)={coh_min} → "
            + ("FUSION (les centroïdes se recoupent plus que les membres ne tiennent "
               "à leur propre centroïde)" if would else "PAS de fusion (thèmes distincts)")
        ),
        "params": state.params,
    }
