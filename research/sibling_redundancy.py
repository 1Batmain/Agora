"""Mesure la REDONDANCE entre thèmes frères — l'axe que la chaîne n'optimise pas.

La chaîne optimise l'EMBOÎTEMENT (fin ⊂ grossier), jamais la DISTINCTIVITÉ entre frères. Or
sur tiktok les feuilles sont redondantes pour un lecteur (~5 clusters d'« addiction » exprimés
dans des vocabulaires différents), et cette redondance N'EST PAS géométrique au sens du
centroïde (recentrés, les centroïdes des addictions sont à 0.38–0.53, comme entre sujets
distincts — cf. session 2026-07-15).

Question : un meilleur signal géométrique — le RECOUVREMENT DE VOISINS kNN entre clusters —
la détecte-t-il ? Deux claims du « même sujet » exprimé autrement peuvent être loin par
centroïde mais partager des voisins. On mesure, pour chaque paire de feuilles (i, j), l'AFFINITÉ
= part des voisins kNN des claims de i qui tombent dans j (symétrisée). Si les frères
redondants (addiction×5, tristesse×4, filles×3) ont une affinité nettement plus forte entre
eux qu'avec les autres, la redondance est détectable dans le graphe — sans LLM.

    uv run --extra embed-contender --extra faiss python research/sibling_redundancy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from backend import analysis as A
from backend.build_analysis import load_dataset
from pipeline.cluster.adaptive import derive_k
from pipeline.cluster.knn import knn_search
from pipeline.cluster.layers import centre

K_AFF = 15   # voisins pris en compte pour l'affinité (indépendant du clustering)


def main(ds: str = "tiktok") -> None:
    model = json.loads(Path(f"backend/cache/{ds}/claims.json").read_text())["model"]
    tree = A.build_theme_tree(load_dataset(ds), model=model, seed=42)
    V = centre(tree.prepared.claim_vecs.astype(np.float64)).astype(np.float32)
    n = len(V)

    leaves = [node for node in tree.nodes.values() if not node.children]
    lab = np.full(n, -1)
    names = []
    for li, node in enumerate(sorted(leaves, key=lambda x: -x.n_avis)):
        for m in node.members:
            lab[m] = li
        names.append((node.n_avis, " ".join(node.label.split()[:3])))
    L = len(leaves)

    # kNN (voisins hors self) → matrice d'affinité inter-feuilles.
    nb = knn_search(V, K_AFF)
    idx = nb.idx[:, 1:]                       # retire self (colonne 0)
    aff = np.zeros((L, L))
    for i in range(n):
        if lab[i] < 0:
            continue
        for j in idx[i]:
            if lab[j] >= 0:
                aff[lab[i], lab[j]] += 1
    row = aff.sum(axis=1, keepdims=True)
    P = aff / np.maximum(row, 1)              # part des voisins de i tombant dans j
    np.fill_diagonal(P, 0.0)                  # on ne veut QUE le débordement inter-cluster
    S = (P + P.T) / 2                         # affinité symétrique

    print(f"=== {ds} : {L} feuilles, affinité kNN inter-cluster (k={K_AFF}) ===\n")
    print("Paires de feuilles les PLUS liées par voisinage (débordement) :")
    pairs = [(S[i, j], i, j) for i in range(L) for j in range(i + 1, L)]
    for s, i, j in sorted(pairs, reverse=True)[:12]:
        print(f"  {s:.3f}  [{names[i][0]:>4}] {names[i][1]:<26} ~ [{names[j][0]:>4}] {names[j][1]}")

    # Groupes « même sujet » repérés à la main (pour le contrôle).
    groupes = {
        "addiction": ["application", "appli", "scroller", "dopamine", "dépendance", "addiction"],
        "tristesse": ["triste", "tristes", "dépression", "angoisse", "suicide", "TCA"],
        "filles/enfants": ["fille", "enfants", "fils", "parental", "corps"],
    }
    def topic(name):
        toks = set(name.lower().split())
        for t, kws in groupes.items():
            if toks & set(kws):
                return t
        return "autre"
    topics = [topic(names[i][1]) for i in range(L)]
    print("\nAffinité MOYENNE intra-sujet vs inter-sujet (contrôle) :")
    intra, inter = [], []
    for i in range(L):
        for j in range(i + 1, L):
            (intra if topics[i] == topics[j] != "autre" else inter).append(S[i, j])
    print(f"  même sujet (addiction/tristesse/filles) : {np.mean(intra):.3f}  (n={len(intra)})")
    print(f"  sujets différents                        : {np.mean(inter):.3f}  (n={len(inter)})")
    print(f"  ratio intra/inter                        : {np.mean(intra)/max(np.mean(inter),1e-9):.2f}×")
    print("\n→ ratio ≫ 1 : la redondance est détectable dans le graphe (kNN), pas au centroïde.")
    print("  ratio ≈ 1 : indétectable géométriquement → un juge SÉMANTIQUE (LLM) est nécessaire.")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["tiktok"]))
