"""Harnais d'éval — STRUCTURE MACRO en INCRÉMENTAL vs BATCH (Lane E0).

Problème : l'incrémental pur (`AnalysisState` : nearest-attach + split local) fige
la partition MACRO très tôt (le 1er split du root sur un petit échantillon) → trop
PEU de macros (tiktok 4, granddebat 6) vs le Leiden+coarsening BATCH (6 / 22).

On compare l'assignation **claim→macro** incrémentale à celle du batch
(`build_theme_tree`) sur tiktok + granddebat, à des checkpoints (25/50/100 %), pour
TROIS dérivations du niveau macro à partir de l'arbre incrémental (mêmes feuilles) :

  - **baseline** : macros = enfants du root (`effective_macro_ids`, état actuel) ;
  - **option A** : COARSEN/MERGE des FEUILLES via `_coarsen_roots` (μ+σ ET cohésion) ;
  - **option B** : RECOMPUTE PARTIEL = Leiden+coarsening sur les CENTROÏDES des
    feuilles (n_feuilles ≪ n_claims → cheap), miroir du batch sur un sous-problème.

Métriques : V-mesure (homogénéité+complétude) vs batch, nb de macros vs batch, coût
(taille du sous-problème recompute = n_feuilles, vs n_claims du batch). Sensibilité à
l'ordre : 2-3 permutations (seed fixé). ZÉRO LLM (claims + embeddings cachés).

    uv run --extra contender python -m research.inc_macro_eval
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from sklearn.metrics import homogeneity_completeness_v_measure

from backend.analysis import (
    _coarsen_roots,
    _node_stats,
    build_theme_tree,
    macro_of,
)
from backend.build_analysis import load_dataset
from backend.claims_endpoint import PreparedClaims, prepare_claims
from backend.state import AnalysisState
from pipeline.cluster.adaptive import derive_defaults
from pipeline.cluster.knn import build_knn_graph
from pipeline.cluster.leiden_cluster import run_leiden

DATASETS = ("tiktok", "granddebat")
CHECKPOINTS = (0.25, 0.50, 1.00)
SEED = 42


# --------------------------------------------------------------------------- #
# Dérivations du niveau MACRO à partir des feuilles de l'arbre incrémental
# --------------------------------------------------------------------------- #
# Chacune renvoie une liste de SUPER-GROUPES = listes d'indices de feuilles. Le
# claim hérite du macro de sa feuille. Tout dérivé des données (aucun magic-number).

def macros_baseline(state: AnalysisState) -> dict[str, int]:
    """État actuel : macro = enfant du root contenant la feuille (ou le root seul)."""
    macro_ids = state.effective_macro_ids()
    rank = {mid: i for i, mid in enumerate(macro_ids)}
    label: dict[str, int] = {}
    for leaf in state._leaves():
        cur = leaf
        while cur.id not in rank and cur.parent_id is not None:
            cur = state.nodes[cur.parent_id]
        label[leaf.id] = rank.get(cur.id, 0)
    return label


def macros_option_a(state: AnalysisState) -> dict[str, int]:
    """Option A — COARSEN/MERGE des feuilles via `_coarsen_roots` (réutilise le batch).

    On fusionne deux feuilles qui « se recoupent » (cos centroïdes > μ+σ ET >
    min(cohésions)). Fusion transitive (union-find) → super-groupes = macros.
    """
    leaves = state._leaves()
    leaf_groups = [leaf.members for leaf in leaves]
    supers, _thr = _coarsen_roots(leaf_groups, state.mat)
    label: dict[str, int] = {}
    for mi, sg in enumerate(supers):
        for li in sg:
            label[leaves[li].id] = mi
    return label


def _leiden_on_leaves(state: AnalysisState) -> tuple[list, list[list[int]]]:
    """Communautés de FEUILLES par Leiden sur leurs centroïdes (graphe kNN dérivé).

    Sous-problème à n_feuilles points (≪ n_claims) → cheap. Renvoie (leaves, communautés
    = listes d'indices LOCAUX de feuilles).
    """
    leaves = state._leaves()
    if len(leaves) < 2:
        return leaves, [[i] for i in range(len(leaves))]
    cents = np.ascontiguousarray(
        np.asarray([leaf.centroid for leaf in leaves]), dtype=np.float64)
    dd = derive_defaults(cents.astype(np.float32))
    graph = build_knn_graph(cents, k=dd.k, threshold=dd.threshold)
    membership = run_leiden(graph, resolution=state.base_resolution, seed=state.seed).membership
    by_comm: dict[int, list[int]] = {}
    for li, c in enumerate(membership):
        by_comm.setdefault(c, []).append(li)
    return leaves, list(by_comm.values())


def macros_option_b(state: AnalysisState) -> dict[str, int]:
    """Option B (brief) — RECOMPUTE PARTIEL : Leiden PUIS coarsening des centroïdes.

    Miroir du batch (`build_theme_tree`) sur un graphe de n_feuilles points : Leiden →
    communautés de feuilles (rôle des `fine_groups`), puis `_coarsen_roots` les fusionne.
    NB : le coarsening final est CORPUS-DÉPENDANT (cf. rapport) — voir `option_B_leiden`.
    """
    leaves, comms = _leiden_on_leaves(state)
    comm_claim_groups = [[m for li in comm for m in leaves[li].members] for comm in comms]
    supers, _thr = _coarsen_roots(comm_claim_groups, state.mat)
    label: dict[str, int] = {}
    for mi, sg in enumerate(supers):           # sg = indices DE COMMUNAUTÉS
        for ci in sg:
            for li in comms[ci]:
                label[leaves[li].id] = mi
    return label


def macros_option_b_leiden(state: AnalysisState) -> dict[str, int]:
    """Option B' — RECOMPUTE PARTIEL SANS coarsening : Leiden sur centroïdes de feuilles.

    Les communautés de feuilles SONT les macros (on retire le coarsening, qui s'est avéré
    destructeur sur corpus multi-sujets — cf. rapport). Le plus STABLE des deux corpus.
    """
    leaves, comms = _leiden_on_leaves(state)
    label: dict[str, int] = {}
    for mi, comm in enumerate(comms):
        for li in comm:
            label[leaves[li].id] = mi
    return label


STRATEGIES = {
    "baseline": macros_baseline,
    "option_A": macros_option_a,
    "option_B": macros_option_b,
    "option_B_leiden": macros_option_b_leiden,
}


# --------------------------------------------------------------------------- #
# Référence BATCH sur un sous-ensemble de claims (prefix d'une permutation)
# --------------------------------------------------------------------------- #
def _subset_prepared(prep: PreparedClaims, idxs: list[int]) -> PreparedClaims:
    """Vue `PreparedClaims` restreinte aux claims `idxs` (réindexés 0..len-1).

    Les owners sont REMAPPÉS denses (sinon `n_avis` et la co-occurrence cassent), mais
    on ne se sert ici que de claim_vecs/owner/weight/texts pour la partition macro.
    """
    owners = [prep.claim_owner[i] for i in idxs]
    remap = {o: j for j, o in enumerate(dict.fromkeys(owners))}
    return PreparedClaims(
        avis=[], claims_by_id={},
        claim_texts=[prep.claim_texts[i] for i in idxs],
        claim_owner=[remap[o] for o in owners],
        claim_weight=prep.claim_weight[idxs],
        claim_vecs=prep.claim_vecs[idxs],
        claim_start=[], claim_end=[],
        backend=prep.backend, model=prep.model, embedder=prep.embedder,
        min_chars=prep.min_chars, extracted=0, embedded=False, cold_seconds=0.0,
    )


def batch_macro_labels(prep: PreparedClaims, idxs: list[int], dataset: str
                       ) -> tuple[np.ndarray, int]:
    """Labels claim→macro du BATCH sur le sous-ensemble `idxs` (ordre de `idxs`)."""
    from types import SimpleNamespace
    sub = _subset_prepared(prep, idxs)
    tree = build_theme_tree(SimpleNamespace(id=dataset), prepared=sub, seed=SEED)
    macro_rank = {mid: i for i, mid in enumerate(tree.macros)}
    labels = np.empty(len(idxs), dtype=np.int32)
    for nid in tree.order:
        node = tree.nodes[nid]
        if node.children:
            continue                       # seules les feuilles couvrent les claims
        m = macro_of(node, tree.nodes)
        for local_i in node.members:       # local_i = position dans `idxs`
            labels[local_i] = macro_rank.get(m.id, 0)
    return labels, len(tree.macros)


# --------------------------------------------------------------------------- #
# Une mesure : à un checkpoint, compare chaque stratégie incrémentale au batch
# --------------------------------------------------------------------------- #
@dataclass
class Row:
    dataset: str
    perm: int
    checkpoint: float
    m: int                     # nb de claims présents
    n_leaves: int
    batch_macros: int
    strategy: str
    inc_macros: int
    homogeneity: float
    completeness: float
    v_measure: float
    v_ceiling: float = 0.0     # V-mesure des feuilles incrémentales vs macros batch


def _inc_labels(state: AnalysisState, strat_fn, order_prefix: list[int]
                ) -> tuple[np.ndarray, int]:
    """Labels claim→macro incrémentaux, alignés sur l'ordre `order_prefix`."""
    leaf_label = strat_fn(state)
    claim_leaf: dict[int, str] = {}
    for leaf in state._leaves():
        for m in leaf.members:
            claim_leaf[m] = leaf.id
    labels = np.array([leaf_label[claim_leaf[g]] for g in order_prefix], dtype=np.int32)
    return labels, len(set(labels.tolist()))


def _oracle_ceiling(state: AnalysisState, order_prefix: list[int],
                    batch_lbl: np.ndarray) -> float:
    """V-mesure du MEILLEUR regroupement possible des feuilles (oracle majoritaire)."""
    pos = {g: i for i, g in enumerate(order_prefix)}
    leaf_macro: dict[str, int] = {}
    for leaf in state._leaves():
        votes: dict[int, int] = {}
        for m in leaf.members:
            b = int(batch_lbl[pos[m]])
            votes[b] = votes.get(b, 0) + 1
        leaf_macro[leaf.id] = max(votes, key=votes.get)
    oracle_lbl, _ = _inc_labels(state, lambda s: leaf_macro, order_prefix)
    _, _, v = homogeneity_completeness_v_measure(batch_lbl, oracle_lbl)
    return v


def evaluate(dataset: str, prep: PreparedClaims, n_perms: int) -> list[Row]:
    n = len(prep.claim_texts)
    rng = np.random.default_rng(SEED)
    rows: list[Row] = []
    for p in range(n_perms):
        order = list(range(n)) if p == 0 else rng.permutation(n).tolist()
        state = AnalysisState(prep, dataset=dataset, seed=SEED)
        cps = sorted(set(int(round(c * n)) for c in CHECKPOINTS))
        added = 0
        for cp in cps:
            for g in order[added:cp]:
                state.add_claim(g)
            added = cp
            prefix = order[:cp]
            batch_lbl, batch_macros = batch_macro_labels(prep, prefix, dataset)
            n_leaves = len(state._leaves())
            # PLAFOND ORACLE : on assigne chaque feuille incrémentale au macro BATCH
            # qui contient la MAJORITÉ de ses claims (regroupement parfait des feuilles).
            # Aucune dérivation macro générique ne peut faire mieux : c'est la borne sup.
            # de ce que la STRUCTURE DES FEUILLES permet de retrouver du batch.
            v_ceiling = _oracle_ceiling(state, prefix, batch_lbl)
            for sname, fn in STRATEGIES.items():
                inc_lbl, inc_macros = _inc_labels(state, fn, prefix)
                h, c, v = homogeneity_completeness_v_measure(batch_lbl, inc_lbl)
                rows.append(Row(
                    dataset=dataset, perm=p, checkpoint=cp / n, m=cp,
                    n_leaves=n_leaves, batch_macros=batch_macros, strategy=sname,
                    inc_macros=inc_macros, homogeneity=round(h, 4),
                    completeness=round(c, 4), v_measure=round(v, 4),
                    v_ceiling=round(v_ceiling, 4)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=3, help="nb de permutations d'ordre")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--out", default="research/inc_macro_results.json")
    args = ap.parse_args()

    all_rows: list[Row] = []
    for ds in args.datasets:
        t = perf_counter()
        prep = prepare_claims(load_dataset(ds))
        rows = evaluate(ds, prep, args.perms)
        all_rows.extend(rows)
        print(f"[{ds}] {len(prep.claim_texts)} claims · {args.perms} perms · "
              f"{perf_counter() - t:.1f}s")

    # Console : table agrégée sur les permutations (moyenne ± écart-type).
    _print_table(all_rows)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump([r.__dict__ for r in all_rows], fh, ensure_ascii=False, indent=2)
    print(f"\n→ {args.out} ({len(all_rows)} lignes)")


def _print_table(rows: list[Row]) -> None:
    from collections import defaultdict
    agg: dict[tuple, list[Row]] = defaultdict(list)
    for r in rows:
        agg[(r.dataset, r.checkpoint, r.strategy)].append(r)
    print(f"\n{'dataset':<10} {'cp':>5} {'strategy':<10} "
          f"{'batch_M':>7} {'inc_M':>11} {'V-measure':>16} {'ceil':>6} {'n_leaves':>8}")
    print("-" * 86)
    last = None
    for (ds, cp, strat), rs in sorted(agg.items()):
        if last is not None and last != (ds, cp):
            print()
        last = (ds, cp)
        bm = rs[0].batch_macros
        inc = np.array([r.inc_macros for r in rs])
        v = np.array([r.v_measure for r in rs])
        nl = rs[0].n_leaves
        ceil = np.mean([r.v_ceiling for r in rs])
        print(f"{ds:<10} {cp:>5.2f} {strat:<10} {bm:>7} "
              f"{inc.mean():>5.1f}±{inc.std():<4.1f} "
              f"{v.mean():>9.3f}±{v.std():<5.3f} {ceil:>6.3f} {nl:>8}")


if __name__ == "__main__":
    main()
