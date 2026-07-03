"""REMÈDE candidat — passe de FUSION post-clustering gardée par sauce_magique.

Prototype la fusion demandée : paires de macros dont la sim centroïde dépasse un
seuil DÉRIVÉ (gap analysis sur la distribution), fusion GLOUTONNE gardée par
`sauce_magique` (on n'accepte une fusion QUE si le score de la façade ne se dégrade
pas). Deux signaux comparés :
  - `centroid` : sim centroïde brute (le remède naïf demandé) ;
  - `graph`    : ratio densité d'arêtes inter/intra (signal STRUCTUREL honnête).

Mesure la façade AVANT/APRÈS sur tiktok ET granddebat, + non-régression témoin
(aucune fusion ne doit rapprocher deux macros de sens officiel distinct).

    MISTRAL_API_KEY=$(cat var/mistral.key) uv run --extra faiss \
        python research/cluster_merge_remedy.py --dataset tiktok
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.analysis import _node_stats, build_theme_tree  # noqa: E402
from backend.build_analysis import EXTRACT_MODEL, load_dataset  # noqa: E402
from backend.recut import recut_tree, sauce_magique  # noqa: E402
from pipeline.cluster.knn import build_knn_graph  # noqa: E402


def _cache_model(dataset: str) -> str:
    """Modèle du cache claims (clé de cache) — épingle le build, ZÉRO extraction."""
    import json as _json
    from backend.recluster import dataset_dir
    rec = _json.loads((dataset_dir(dataset) / "claims.json").read_text(encoding="utf-8"))
    return rec.get("model") or EXTRACT_MODEL


def facade_view(tree, groups):
    """Vue sauce_magique d'une façade = liste de {n_avis, cohesion} par macro-groupe.

    `groups` = liste de listes d'ids de macros (chaque sous-liste = 1 macro fusionné)."""
    vecs = tree.prepared.claim_vecs.astype(np.float64)
    weights = tree.prepared.claim_weight
    owner = tree.prepared.claim_owner
    view = []
    for grp in groups:
        members = [m for mid in grp for m in tree.nodes[mid].members]
        _, _, consensus, _ = _node_stats(members, vecs, weights)
        n_avis = len({owner[i] for i in members})
        view.append({"n_avis": n_avis, "cohesion": consensus})
    return view


def gap_threshold_high(sorted_vals: np.ndarray, top_frac: float = 0.5) -> float:
    """Seuil = milieu du plus grand gap dans la MOITIÉ HAUTE des sims (queue haute)."""
    n = sorted_vals.size
    if n < 3:
        return float("inf")
    lo = int(n * (1 - top_frac))
    tail = sorted_vals[lo:]
    diffs = np.diff(tail)
    if diffs.size == 0 or diffs.max() <= 0:
        return float("inf")
    i = int(np.argmax(diffs))
    return (tail[i] + tail[i + 1]) / 2.0


def graph_edges(tree):
    vecs = tree.prepared.claim_vecs.astype(np.float64)
    dg = tree.derived_global
    return build_knn_graph(vecs, k=dg.k, threshold=dg.threshold).edges


def pair_graph_ratio(tree, edges, a, b):
    setA, setB = set(tree.nodes[a].members), set(tree.nodes[b].members)
    nA, nB = len(setA), len(setB)
    eAA = eBB = eAB = 0
    for i, j, w in edges:
        ia, ib = i in setA, i in setB
        ja, jb = j in setA, j in setB
        if ia and ja:
            eAA += 1
        elif ib and jb:
            eBB += 1
        elif (ia and jb) or (ib and ja):
            eAB += 1
    dens = lambda e, nn: (e / (nn * (nn - 1) / 2)) if nn > 1 else 0.0
    dAB = eAB / (nA * nB) if nA and nB else 0.0
    intra = (dens(eAA, nA) + dens(eBB, nB)) / 2
    return (dAB / intra) if intra > 0 else float("inf")


def fusion_pass(tree, signal="centroid", verbose=True):
    """Fusion gloutonne gardée par sauce_magique. Renvoie (groups, log)."""
    macros = list(tree.macros)
    vecs = tree.prepared.claim_vecs.astype(np.float64)
    cents = {m: np.asarray(tree.nodes[m].centroid) for m in macros}
    sim = {}
    ids = macros
    n = len(ids)
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(cents[ids[i]] @ cents[ids[j]])
            sim[(ids[i], ids[j])] = s
            vals.append(s)
    sv = np.sort(np.array(vals))
    thr = gap_threshold_high(sv)
    edges = graph_edges(tree) if signal == "graph" else None

    # groupes courants (chaque macro seul)
    groups = [[m] for m in macros]
    cur_view = facade_view(tree, groups)
    cur_score, cur_detail = sauce_magique(cur_view)
    log = {"threshold": round(thr, 4), "signal": signal,
           "before": cur_detail, "merges": [], "rejected": []}

    def group_of(mid):
        for g in groups:
            if mid in g:
                return g
        return None

    # candidats : paires au-dessus du seuil, plus fortes d'abord
    cand = sorted([(s, a, b) for (a, b), s in sim.items() if s >= thr], reverse=True)
    if verbose:
        print(f"  seuil dérivé (gap queue haute) = {thr:.4f} · {len(cand)} paires candidates")
    for s, a, b in cand:
        ga, gb = group_of(a), group_of(b)
        if ga is gb:
            continue
        if signal == "graph":
            gr = pair_graph_ratio(tree, edges, a, b)
            if gr < 0.5:            # pas de vrai recoupement structurel → on saute
                log["rejected"].append({"pair": [a, b], "reason": f"graph_ratio={gr:.3f}<0.5"})
                continue
        # fusion tentée
        trial = [g for g in groups if g is not ga and g is not gb] + [ga + gb]
        tview = facade_view(tree, trial)
        tscore, tdetail = sauce_magique(tview)
        if tscore <= cur_score + 1e-9:      # ne se dégrade pas → accepte
            groups = trial
            cur_score, cur_detail = tscore, tdetail
            log["merges"].append({"pair": [a, b], "sim": round(s, 4),
                                  "score": round(tscore, 4)})
            if verbose:
                print(f"   ✓ FUSION {a}⟷{b} sim={s:.3f}  score {cur_score:.4f}")
        else:
            log["rejected"].append({"pair": [a, b], "sim": round(s, 4),
                                    "reason": f"sauce_magique {cur_score:.4f}→{tscore:.4f} (dégrade)"})
            if verbose:
                print(f"   ✗ rejet  {a}⟷{b} sim={s:.3f}  score {cur_score:.4f}→{tscore:.4f}")
    log["after"] = cur_detail
    log["n_before"] = len(macros)
    log["n_after"] = len(groups)
    return groups, log


def title_embed_check(tree, groups, embedder):
    """Non-régression : pour chaque fusion, sim d'embedding des TITRES des macros
    fusionnés (proche ⇒ légitime ; éloigné ⇒ on a fusionné 2 sens distincts)."""
    from pipeline.claims.pipeline import embed_claim_texts
    merged = [g for g in groups if len(g) > 1]
    if not merged:
        print("  (aucune fusion — rien à vérifier)")
        return
    for g in merged:
        titles = [tree.nodes[m].title or tree.nodes[m].label for m in g]
        vt = embed_claim_texts(titles, embedder=embedder)
        vt = vt / np.linalg.norm(vt, axis=1, keepdims=True)
        sims = vt @ vt.T
        iu = np.triu_indices(len(g), 1)
        print(f"   fusion {g} · sim titres min={sims[iu].min():.3f} :: {titles}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tiktok")
    ap.add_argument("--recut", action="store_true", default=True)
    ap.add_argument("--no-recut", dest="recut", action="store_false")
    args = ap.parse_args()

    ds = load_dataset(args.dataset)
    tree = build_theme_tree(ds, model=_cache_model(args.dataset))
    assert tree.prepared.extracted == 0
    if args.recut:
        rc = recut_tree(tree)
        print(f"# {args.dataset} · recut {rc['avant']['n_clusters']}→{rc['apres']['n_clusters']} macros"
              if rc else f"# {args.dataset} · recut no-op")
    print(f"# façade servie : {len(tree.macros)} macros")

    for signal in ("centroid", "graph"):
        print(f"\n########## FUSION signal={signal} ##########")
        groups, log = fusion_pass(tree, signal=signal)
        print(f"  AVANT  : {log['before']}")
        print(f"  APRÈS  : {log['after']}")
        print(f"  macros {log['n_before']} → {log['n_after']} · "
              f"{len(log['merges'])} fusion(s) acceptée(s), {len(log['rejected'])} rejet(s)")
        if log["merges"]:
            print("  — vérif embeddings des titres (non-régression sens) —")
            title_embed_check(tree, groups, tree.prepared.embedder)


if __name__ == "__main__":
    main()
