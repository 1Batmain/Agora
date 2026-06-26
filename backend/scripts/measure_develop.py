"""D1 — mesure: corrélation longueur du claim ↔ distance au centroïde.

Confirme (ou non) l'intuition de Bob : les claims courts sont plus proches du
centroïde (génériques), les développés plus loin. Mesuré par feuille puis agrégé.
"""
import sys
import numpy as np
from backend.server import _Dataset
from backend.analysis import get_or_build_tree


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def measure(dataset_id):
    ds = _Dataset(dataset_id)
    tree = get_or_build_tree(ds)
    prep = tree.prepared
    texts = prep.claim_texts
    vecs = prep.claim_vecs

    # 1) Global, par feuille : corrélation intra-feuille moyenne (pondérée par taille).
    leaf_corrs = []
    pooled_len, pooled_dist = [], []
    for node in tree.nodes.values():
        if node.children or not node.members or len(node.members) < 5:
            continue
        members = node.members
        sims = vecs[members] @ node.centroid
        dist = 1.0 - sims
        lens = np.array([len(texts[ci]) for ci in members], float)
        c = pearson(lens, dist)
        if c == c:  # not nan
            leaf_corrs.append((c, len(members)))
        # pool z-scored dist within leaf vs raw length for a global view
        if dist.std() > 0:
            pooled_len.extend(lens.tolist())
            pooled_dist.extend(((dist - dist.mean()) / dist.std()).tolist())

    if leaf_corrs:
        cs = np.array([c for c, _ in leaf_corrs])
        ws = np.array([w for _, w in leaf_corrs], float)
        wmean = float((cs * ws).sum() / ws.sum())
    else:
        wmean = float("nan")
    pooled = pearson(pooled_len, pooled_dist)

    # quartiles de longueur vs distance médiane (pooled, dist z-scoré par feuille)
    print(f"\n=== {dataset_id} ===")
    print(f"feuilles mesurées : {len(leaf_corrs)}  | claims poolés : {len(pooled_len)}")
    print(f"corr intra-feuille longueur↔dist (moyenne pondérée) : {wmean:+.3f}")
    print(f"corr poolée longueur↔dist(z par feuille)            : {pooled:+.3f}")
    if pooled_len:
        pl = np.array(pooled_len); pd = np.array(pooled_dist)
        qs = np.quantile(pl, [0.25, 0.5, 0.75])
        bins = np.digitize(pl, qs)
        for b in range(4):
            m = bins == b
            if m.sum():
                print(f"  quartile L{b} (len≈{pl[m].mean():.0f}) : "
                      f"dist_z médiane={np.median(pd[m]):+.3f} n={m.sum()}")
    return wmean, pooled


if __name__ == "__main__":
    for d in (sys.argv[1:] or ["tiktok", "granddebat"]):
        measure(d)
