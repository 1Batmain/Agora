"""Banc SBM emboîté (Peixoto/graph-tool) vs notre chaîne d'emboîtement — ÉTAPE 1/3.

EXPORT (env uv). Pour chaque corpus : espace recentré → graphe kNN de RÉFÉRENCE (seuil
dérivé, k = derive_k(N) — UN graphe unique et neutre, ni choisi pour flatter la chaîne ni le
SBM) → notre chaîne de layers → golds. Tout est écrit sous `var/sbm/`. Les étapes 2 (SBM,
python SYSTÈME + graph-tool) et 3 (comparaison, env uv) relisent ces fichiers.

    uv run --extra embed-contender --extra faiss python research/sbm_export.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pipeline.cluster.adaptive import derive_defaults, derive_k
from pipeline.cluster.knn import build_knn_graph, knn_search
from pipeline.cluster.layers import centre, chain
from research.xstance_bench import load_claims

OUT = Path("var/sbm")
OUT.mkdir(parents=True, exist_ok=True)


def _claim_vecs(ds: str) -> np.ndarray:
    return np.load(f"backend/cache/{ds}/claims_emb.npz")["vecs"].astype(np.float64)


def _ref_graph(V: np.ndarray):
    """Graphe kNN de référence (recentré, seuil dérivé, k=derive_k(N))."""
    Vc = centre(V)
    V32 = Vc.astype(np.float32)
    n = len(Vc)
    k = derive_k(n)
    nb = knn_search(V32, k)
    dd = derive_defaults(V32, k=k, neighbors=nb)
    g = build_knn_graph(Vc, k=dd.k, threshold=dd.threshold, neighbors=nb)
    edges = np.array([(a, b) for a, b, _w in g.edges], dtype=np.int64)
    return n, edges, k, dd.threshold


def _enc(a) -> np.ndarray:
    """Encode des étiquettes catégorielles en entiers denses (pour ARI/graph-tool)."""
    u = {v: i for i, v in enumerate(sorted(set(map(str, a))))}
    return np.array([u[str(x)] for x in a], dtype=np.int64)


def export(name: str, V: np.ndarray, golds: dict) -> None:
    n, edges, k, thr = _ref_graph(V)
    np.savez(OUT / f"{name}.graph.npz", n=n, edges=edges, k=k, threshold=thr)

    levels = chain(centre(V))                        # notre chaîne d'emboîtement
    parts = {f"L{i}_n{lv.n_clusters}": np.asarray(lv.membership) for i, lv in enumerate(levels)}
    np.savez(OUT / f"{name}.chain.npz", **parts)
    np.savez(OUT / f"{name}.gold.npz", **golds)

    summary = [(lv.k, lv.n_clusters, round(lv.cleanliness, 4)) for lv in levels]
    (OUT / f"{name}.meta.json").write_text(json.dumps(
        {"n": int(n), "k_ref": int(k), "n_edges": int(len(edges)),
         "chain": summary, "golds": list(golds)}, ensure_ascii=False, indent=2))
    print(f"{name:<8} n={n} edges={len(edges)} chaîne={summary} golds={list(golds)}", flush=True)


def main() -> None:
    export("tiktok", _claim_vecs("tiktok"), {})            # pas de gold externe

    Vx, topics, questions = load_claims("xstance")         # gold 12 topics + 191 questions
    Vx = Vx.astype(np.float64)
    export("xstance", Vx, {"topic12": _enc(topics), "question191": _enc(questions)})

    Vt = _claim_vecs("tiktok")                             # témoin 2 domaines
    Vm = np.vstack([Vt, Vx])
    dom = np.array([0] * len(Vt) + [1] * len(Vx), dtype=np.int64)
    export("mix", Vm, {"domain2": dom})


if __name__ == "__main__":
    main()
