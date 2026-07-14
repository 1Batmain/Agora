"""AUDIT — sous-consolidation des clusters (Bob/tiktok) : MESURE avant remède.

Parties :
  1. paires de macros/feuilles suspectes (sim centroïde-centroïde) + distribution.
  2. ANISOTROPIE : cos moyen paires ALÉATOIRES vs intra-cluster vs inter suspects ;
     effet de all-but-the-top (retrait du vecteur moyen + re-norm) sur le contraste.
  3. GRAPHE : densité d'arêtes kNN intra vs inter pour la paire suspecte.

READ-ONLY. Reconstruit l'arbre depuis les caches (zéro LLM, modèle épinglé).

    MISTRAL_API_KEY=$(cat var/mistral.key) uv run --extra faiss \
        python research/cluster_merge.py --dataset tiktok
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.analysis import _node_stats, build_theme_tree  # noqa: E402
from backend.build_analysis import EXTRACT_MODEL, load_dataset  # noqa: E402
from pipeline.cluster.adaptive import derive_defaults  # noqa: E402
from pipeline.cluster.knn import build_knn_graph  # noqa: E402


def _cache_model(dataset: str) -> str:
    """Modèle du cache claims (clé de cache) — épingle le build, ZÉRO extraction."""
    import json as _json
    from backend.recluster import dataset_dir
    rec = _json.loads((dataset_dir(dataset) / "claims.json").read_text(encoding="utf-8"))
    return rec.get("model") or EXTRACT_MODEL


def macro_nodes(tree):
    return [tree.nodes[m] for m in tree.macros]


def leaf_nodes(tree):
    return [n for n in tree.nodes.values() if not n.children]


def cos_matrix(cents: np.ndarray) -> np.ndarray:
    return cents @ cents.T


def dist_summary(vals: np.ndarray) -> dict:
    return {
        "n": int(vals.size),
        "min": round(float(vals.min()), 4),
        "p25": round(float(np.percentile(vals, 25)), 4),
        "median": round(float(np.median(vals)), 4),
        "mean": round(float(vals.mean()), 4),
        "p75": round(float(np.percentile(vals, 75)), 4),
        "p90": round(float(np.percentile(vals, 90)), 4),
        "max": round(float(vals.max()), 4),
        "std": round(float(vals.std()), 4),
    }


def largest_gap(sorted_vals: np.ndarray) -> tuple[float, float]:
    """Milieu et taille du plus grand écart d'une distribution triée (gap analysis)."""
    if sorted_vals.size < 2:
        return float("nan"), 0.0
    diffs = np.diff(sorted_vals)
    i = int(np.argmax(diffs))
    return (sorted_vals[i] + sorted_vals[i + 1]) / 2.0, float(diffs[i])


def part1_pairs(tree, label, nodes, rng):
    cents = np.array([n.centroid for n in nodes], dtype=np.float64)
    sim = cos_matrix(cents)
    n = len(nodes)
    iu = np.triu_indices(n, 1)
    pair = sim[iu]
    print(f"\n=== PARTIE 1 — {label} : {n} nœuds, {pair.size} paires ===")
    print("distribution sim centroïde-centroïde :", json.dumps(dist_summary(pair)))
    thr_musigma = float(pair.mean() + pair.std())
    print(f"seuil coarsening actuel μ+σ = {thr_musigma:.4f}  (fusionne si sim > ça)")
    sv = np.sort(pair)
    gmid, gsize = largest_gap(sv)
    print(f"plus grand gap de la distribution : milieu={gmid:.4f} taille={gsize:.4f}")
    # top paires
    order = np.argsort(-pair)
    print(f"\n  top-12 paires les plus proches ({label}) :")
    for rank in range(min(12, pair.size)):
        idx = order[rank]
        a, b = iu[0][idx], iu[1][idx]
        na, nb = nodes[a], nodes[b]
        print(f"   sim={pair[idx]:.4f}  {na.id}({na.n_avis:>4}) «{(na.title or na.label)[:38]}»"
              f"  ⟷  {nb.id}({nb.n_avis:>4}) «{(nb.title or nb.label)[:38]}»")
    return cents, sim, iu, pair


def anisotropy(tree, A, B, rng, tag=""):
    vecs = tree.prepared.claim_vecs.astype(np.float64)
    n = vecs.shape[0]
    # baseline : paires aléatoires du corpus
    idx = rng.integers(0, n, size=(20000, 2))
    idx = idx[idx[:, 0] != idx[:, 1]]
    rand_cos = np.einsum("ij,ij->i", vecs[idx[:, 0]], vecs[idx[:, 1]])
    # intra A, intra B, inter A-B (échantillonnés)
    def sample_pairs(members, m=8000):
        members = np.asarray(members)
        if members.size < 2:
            return np.array([1.0])
        i = rng.integers(0, members.size, size=(m, 2))
        i = i[i[:, 0] != i[:, 1]]
        return np.einsum("ij,ij->i", vecs[members[i[:, 0]]], vecs[members[i[:, 1]]])

    def sample_inter(ma, mb, m=8000):
        ma, mb = np.asarray(ma), np.asarray(mb)
        i = rng.integers(0, ma.size, size=m)
        j = rng.integers(0, mb.size, size=m)
        return np.einsum("ij,ij->i", vecs[ma[i]], vecs[mb[j]])

    intraA = sample_pairs(A.members)
    intraB = sample_pairs(B.members)
    inter = sample_inter(A.members, B.members)
    cA, cB = np.asarray(A.centroid), np.asarray(B.centroid)
    cc = float(cA @ cB)
    print(f"\n=== PARTIE 2 — ANISOTROPIE {tag}  (A={A.id} «{(A.title or A.label)[:30]}», "
          f"B={B.id} «{(B.title or B.label)[:30]}») ===")
    print(f"   cos paires ALÉATOIRES (baseline corpus)  : {rand_cos.mean():.4f}  (σ={rand_cos.std():.3f})")
    print(f"   cos intra-A                              : {intraA.mean():.4f}")
    print(f"   cos intra-B                              : {intraB.mean():.4f}")
    print(f"   cos INTER A-B                            : {inter.mean():.4f}")
    print(f"   centroïde-centroïde A·B                  : {cc:.4f}")
    contrast = (intraA.mean() + intraB.mean()) / 2 - inter.mean()
    sep_vs_base = inter.mean() - rand_cos.mean()
    print(f"   contraste intra−inter                    : {contrast:.4f}")
    print(f"   inter − baseline aléatoire               : {sep_vs_base:.4f}  "
          f"(>0 ⇒ A,B + proches que 2 claims au hasard)")
    return {"rand": rand_cos.mean(), "intraA": intraA.mean(), "intraB": intraB.mean(),
            "inter": inter.mean(), "cc": cc, "contrast": contrast}


def all_but_the_top(vecs: np.ndarray, d_rm: int = 1) -> np.ndarray:
    """Retire le vecteur moyen (+ d_rm-1 PC dominants) puis re-normalise (Mu et al.)."""
    mu = vecs.mean(axis=0, keepdims=True)
    x = vecs - mu
    if d_rm > 1:
        # retire aussi les PC dominants suivants
        u, s, vt = np.linalg.svd(x - x.mean(0, keepdims=True), full_matrices=False)
        pcs = vt[: d_rm - 1]
        x = x - (x @ pcs.T) @ pcs
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return x / nrm


def anisotropy_abtt(tree, A, B, rng):
    """Rejoue l'anisotropie après all-but-the-top sur les vecteurs claims."""
    vecs = tree.prepared.claim_vecs.astype(np.float64)
    for d_rm in (1, 2, 3):
        v2 = all_but_the_top(vecs, d_rm)
        def sample_pairs(members, m=8000):
            members = np.asarray(members)
            i = rng.integers(0, members.size, size=(m, 2)); i = i[i[:, 0] != i[:, 1]]
            return np.einsum("ij,ij->i", v2[members[i[:, 0]]], v2[members[i[:, 1]]])
        def sample_inter(ma, mb, m=8000):
            ma, mb = np.asarray(ma), np.asarray(mb)
            i = rng.integers(0, ma.size, size=m); j = rng.integers(0, mb.size, size=m)
            return np.einsum("ij,ij->i", v2[ma[i]], v2[mb[j]])
        n = v2.shape[0]
        ridx = rng.integers(0, n, size=(20000, 2)); ridx = ridx[ridx[:, 0] != ridx[:, 1]]
        rc = np.einsum("ij,ij->i", v2[ridx[:, 0]], v2[ridx[:, 1]]).mean()
        iA, iB = sample_pairs(A.members).mean(), sample_pairs(B.members).mean()
        inter = sample_inter(A.members, B.members).mean()
        # centroïdes recalculés dans l'espace ABTT
        def cent(members):
            s = v2[np.asarray(members)].sum(0); return s / np.linalg.norm(s)
        cc = float(cent(A.members) @ cent(B.members))
        contrast = (iA + iB) / 2 - inter
        print(f"   ABTT(d={d_rm}): rand={rc:.4f} intraA={iA:.4f} intraB={iB:.4f} "
              f"inter={inter:.4f} cc={cc:.4f} contraste={contrast:.4f}")


def graph_ratio(graph_edges, A, B):
    """Ratio densité inter/intra d'une paire (graphe global). → décisif Leiden."""
    setA, setB = set(A.members), set(B.members)
    nA, nB = len(setA), len(setB)
    eAA = eBB = eAB = 0
    for i, j, w in graph_edges:
        ia, ib = i in setA, i in setB
        ja, jb = j in setA, j in setB
        if ia and ja:
            eAA += 1
        elif ib and jb:
            eBB += 1
        elif (ia and jb) or (ib and ja):
            eAB += 1
    dens = lambda e, nn: (e / (nn * (nn - 1) / 2)) if nn > 1 else 0.0
    densAB = eAB / (nA * nB) if nA and nB else 0.0
    intra_dens = (dens(eAA, nA) + dens(eBB, nB)) / 2
    ratio = densAB / intra_dens if intra_dens > 0 else float("inf")
    return {"eAA": eAA, "eBB": eBB, "eAB": eAB, "densAB": densAB,
            "intra_dens": intra_dens, "ratio": ratio}


def scan_top_pairs(tree, nodes, rng, topk=12):
    """Pour les top-K paires (sim centroïde brute), compare : sim brute, sim ABTT
    des centroïdes, cos INTER claim-claim, baseline aléatoire, ratio graphe."""
    vecs = tree.prepared.claim_vecs.astype(np.float64)
    dg = tree.derived_global
    graph = build_knn_graph(vecs, k=dg.k, threshold=dg.threshold)
    v_abtt = all_but_the_top(vecs, 1)

    def cent_raw(m):
        s = vecs[np.asarray(m.members)].sum(0); return s / np.linalg.norm(s)

    def cent_abtt(m):
        s = v_abtt[np.asarray(m.members)].sum(0); return s / np.linalg.norm(s)

    # baseline aléatoire (brut + ABTT)
    n = vecs.shape[0]
    ridx = rng.integers(0, n, size=(30000, 2)); ridx = ridx[ridx[:, 0] != ridx[:, 1]]
    base_raw = float(np.einsum("ij,ij->i", vecs[ridx[:, 0]], vecs[ridx[:, 1]]).mean())
    base_abtt = float(np.einsum("ij,ij->i", v_abtt[ridx[:, 0]], v_abtt[ridx[:, 1]]).mean())

    cents = np.array([cent_raw(m) for m in nodes])
    sim = cents @ cents.T
    iu = np.triu_indices(len(nodes), 1)
    pair = sim[iu]
    order = np.argsort(-pair)[:topk]

    def sample_inter(ma, mb, V, m=6000):
        ma, mb = np.asarray(ma), np.asarray(mb)
        i = rng.integers(0, ma.size, size=m); j = rng.integers(0, mb.size, size=m)
        return float(np.einsum("ij,ij->i", V[ma[i]], V[mb[j]]).mean())

    print(f"\n=== PARTIE 3 — SCAN top-{topk} paires macro (base_raw={base_raw:.3f}, "
          f"base_abtt={base_abtt:.3f}) ===")
    print(f"   {'A':>5} {'B':>5}  {'cc_brut':>7} {'cc_abtt':>7} {'inter_cc':>8} "
          f"{'inter-base':>10} {'ratio_graphe':>12}  verdict")
    rows = []
    for idx in order:
        A = nodes[iu[0][idx]]; B = nodes[iu[1][idx]]
        cc_raw = pair[idx]
        cc_a = float(cent_abtt(A) @ cent_abtt(B))
        inter_raw = sample_inter(A.members, B.members, vecs)
        gr = graph_ratio(graph.edges, A, B)
        sep = inter_raw - base_raw
        # verdict heuristique : vrai doublon si graphe se recoupe fortement
        verdict = "FUSION?" if gr["ratio"] > 0.5 else ("distinct" if gr["ratio"] < 0.25 else "ambigu")
        rows.append((A.id, B.id, cc_raw, cc_a, inter_raw, sep, gr["ratio"], verdict))
        print(f"   {A.id:>5} {B.id:>5}  {cc_raw:7.4f} {cc_a:7.4f} {inter_raw:8.4f} "
              f"{sep:10.4f} {gr['ratio']:12.3f}  {verdict}  "
              f"«{(A.title or A.label)[:22]}»⟷«{(B.title or B.label)[:22]}»")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tiktok")
    ap.add_argument("--pair", nargs=2, default=None, help="ids A B de la paire suspecte forcée")
    args = ap.parse_args()

    ds = load_dataset(args.dataset)
    tree = build_theme_tree(ds, model=_cache_model(args.dataset))
    assert tree.prepared.extracted == 0, "extraction aurait dû être 100% cachée"

    macros = macro_nodes(tree)
    leaves = leaf_nodes(tree)
    print(f"# dataset={args.dataset}  n_claims={len(tree.prepared.claim_texts)} "
          f"n_macros={len(macros)} n_leaves={len(leaves)} n_nodes={len(tree.nodes)}")

    part1_pairs(tree, "MACROS", macros, rng)
    _, msim, miu, mpair = None, None, None, None
    part1_pairs(tree, "FEUILLES", leaves, rng)

    # paire suspecte : la + proche des macros, ou celle imposée
    cents = np.array([n.centroid for n in macros], dtype=np.float64)
    sim = cents @ cents.T
    iu = np.triu_indices(len(macros), 1)
    pair = sim[iu]
    if args.pair:
        A = tree.nodes[args.pair[0]]; B = tree.nodes[args.pair[1]]
    else:
        top = int(np.argmax(pair))
        A = macros[iu[0][top]]; B = macros[iu[1][top]]

    res = anisotropy(tree, A, B, rng, tag="(espace brut)")
    print("\n   — all-but-the-top (retrait vecteur moyen + PC, re-norm) —")
    anisotropy_abtt(tree, A, B, rng)
    scan_top_pairs(tree, macros, rng, topk=12)


if __name__ == "__main__":
    main()
