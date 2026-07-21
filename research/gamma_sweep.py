"""Exploration OUVERTE — balayer la RÉSOLUTION γ de Leiden sur un graphe FIXE.

Contexte : le « robinet k » actuel change le GRAPHE à chaque palier (densité 120× de k=6 à
k=2000, seuil adaptatif qui tombe à 0 → pure densification). k est donc un PROXY indirect de
la résolution γ, le vrai bouton de granularité de la modularité. Ici on fixe UN graphe kNN
propre (seuil dérivé, k modéré) et on ne bouge QUE γ.

But — PAS reproduire la chaîne actuelle (on n'a pas la version ultime : clusters redondants,
thèmes peu intuitifs). On EXPLORE : quelle granularité donne les thèmes les plus lisibles et
les moins redondants ? On montre les vrais thèmes (c-TF-IDF) à chaque échelle pour juger à l'œil.

    uv run --extra embed-contender --extra faiss python research/gamma_sweep.py [dataset] [k_graphe]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from backend import analysis as A
from backend.build_analysis import load_dataset
from pipeline.cluster.adaptive import derive_defaults
from pipeline.cluster.knn import build_knn_graph, knn_search
from pipeline.cluster.layers import centre
from pipeline.cluster.leiden_cluster import run_leiden
from pipeline.cluster.naming import derive_corpus_stopwords, name_clusters

GAMMAS = [0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 4.5, 7.0, 10.0]


def _labels(texts, membership, top=4):
    docs: dict[int, list] = {}
    for i, c in enumerate(membership):
        docs.setdefault(int(c), []).append(texts[i])
    stop, _ = derive_corpus_stopwords(texts)
    names = name_clusters(docs, top_k=top, corpus_stopwords=stop)
    sizes = {c: len(d) for c, d in docs.items()}
    return names, sizes


def main(ds: str = "tiktok", k_graph: int = 30) -> None:
    model = json.loads(Path(f"backend/cache/{ds}/claims.json").read_text())["model"]
    prep = A.build_theme_tree(load_dataset(ds), model=model, seed=42).prepared
    texts = prep.claim_texts
    V = centre(prep.claim_vecs.astype(np.float64))
    V32 = V.astype(np.float32)
    n = len(V)

    # UN seul graphe, fixe : γ sera le seul levier.
    nb = knn_search(V32, k_graph)
    dd = derive_defaults(V32, k=k_graph, neighbors=nb)
    g = build_knn_graph(V, k=dd.k, threshold=dd.threshold, neighbors=nb)
    print(f"{ds} : {n} claims · graphe FIXE k={k_graph}, seuil={dd.threshold:.3f}, "
          f"{len(g.edges)} arêtes (deg. moy {2*len(g.edges)/n:.0f})\n")

    print(f"{'γ':>6} {'clusters':>9}   (modularité)")
    parts = {}
    for gamma in GAMMAS:
        m = run_leiden(g, resolution=gamma, seed=42)
        parts[gamma] = np.asarray(m.membership)
        print(f"{gamma:>6} {len(set(m.membership)):>9}   {m.modularity:.3f}")

    # Montre les VRAIS thèmes à quelques granularités (fine, moyenne, grossière) pour juger.
    ncl = {gamma: len(set(p)) for gamma, p in parts.items()}
    cibles = [4, 9, 16, 30]
    vus = set()
    print("\n" + "=" * 70)
    for cible in cibles:
        gamma = min(GAMMAS, key=lambda gm: abs(ncl[gm] - cible))
        if gamma in vus:
            continue
        vus.add(gamma)
        p = parts[gamma]
        names, sizes = _labels(texts, p)
        print(f"\n### γ={gamma} → {ncl[gamma]} thèmes (≈ cible {cible}) ###")
        for c in sorted(sizes, key=lambda x: -sizes[x]):
            print(f"   {sizes[c]:>4}  {' · '.join(names[c]['keywords'][:4])}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "tiktok", int(a[1]) if len(a) > 1 else 30)
