"""A/B — transformation du POIDS des arêtes k-NN avant Leiden (R&D pur).

Aujourd'hui une arête du graphe k-NN porte le **cosinus brut** ; Leiden l'optimise
tel quel (`run_leiden`, `weights="weight"`). On teste ici si **accentuer les liens
forts** par une transformation du poids donne un MEILLEUR partitionnement — sans
toucher au chemin de production.

Transforms (appliqués au poids cosinus `w∈[seuil,1]` de chaque arête, AVANT Leiden) :
  - `raw`    : `w`                    (baseline = production actuelle)
  - `cos2`   : `w²`                   (accentue les liens forts)
  - `cos3`   : `w³`                   (idem, plus agressif)
  - `gauss`  : `exp(-((1-w)²)/(2σ²))` avec σ = écart-type des distances (1−w) du graphe
               (noyau gaussien — écrase les liens faibles, sature les forts)

Métriques, par dataset × transform :
  - `n_clusters`  : granularité de la partition RACINE (le transform change-t-il le grain ?)
  - `Q_raw`       : **modularité de la partition mesurée sur les poids COSINUS BRUTS** —
                    yardstick COMMUN et juste : « cette partition est-elle une meilleure
                    découpe de la structure de similarité originale ? » (comparable A/B).
  - `Q_self`      : modularité sur les poids TRANSFORMÉS (ce que Leiden a réellement
                    optimisé) — pour référence, NON comparable entre transforms.
  - repnum only — **alignement à la taxo officielle** (ARI/NMI/V vs Titre I/II/III),
                    via le chemin `build_live_tree` (macros) avec `run_leiden` monkeypatché
                    pour appliquer le transform partout (racine + sous-arbres).

`k` = défaut dérivé de N (`derive_k`) — le réglage de production ; on isole l'effet du
transform du poids, pas du nombre de voisins (cf. verdict k-sweep : k=12-13 sweet-spot).

Réutilisation EXPLICITE des briques prod (zéro re-embed, zéro LLM) : `load_cache`,
`knn_search`, `derive_defaults`, `build_knn_graph`, `run_leiden`, `build_live_tree`.
AUCUN fichier produit n'est modifié — le transform vit en monkeypatch local.

Lancer :
    uv run --extra contender --extra embed-contender --extra faiss --with fastapi \
        python research/knn_weight_sweep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backend.analysis as analysis_mod
import backend.live_cluster as live_mod
from backend.live_cluster import build_live_tree
from backend.recluster import load_cache
from pipeline.cluster.adaptive import derive_defaults, derive_k
from pipeline.cluster.knn import KnnGraph, build_knn_graph, knn_search
from pipeline.cluster.leiden_cluster import DEFAULT_SEED, run_leiden

import research.k_sweep as ks  # repnum_gold_axes / alignment (gold officiel)

DATASETS = ["tiktok", "granddebat", "xstance", "republique-numerique"]
SEED = DEFAULT_SEED
_SCRATCH = Path("/tmp/claude-1000/-home-bat-agora-worktrees-knn-weight")


# --------------------------------------------------------------------------- #
# Transforms du poids d'arête.  fn(raw_w: np.ndarray, sigma: float) -> np.ndarray
# --------------------------------------------------------------------------- #
def _sigma(raw_w: np.ndarray) -> float:
    """σ du noyau gaussien = écart-type des distances (1−cos) des arêtes (≥ 1e-6)."""
    s = float(np.std(1.0 - raw_w))
    return s if s > 1e-9 else 1e-6


TRANSFORMS = {
    "raw":   lambda w, s: w,
    "cos2":  lambda w, s: w ** 2,
    "cos3":  lambda w, s: w ** 3,
    "gauss": lambda w, s: np.exp(-((1.0 - w) ** 2) / (2.0 * s * s)),
}


def _apply(name: str, raw_w: np.ndarray) -> np.ndarray:
    return np.asarray(TRANSFORMS[name](raw_w, _sigma(raw_w)), dtype=np.float64)


# --------------------------------------------------------------------------- #
# Modularité d'une partition mesurée sur un jeu de poids DONNÉ (yardstick).
# --------------------------------------------------------------------------- #
def _modularity(edges, n: int, membership, weights) -> float:
    import igraph as ig

    g = ig.Graph(n=n)
    if edges:
        g.add_edges([(i, j) for (i, j, _) in edges])
        g.es["weight"] = list(weights)
    try:
        return float(g.modularity(membership, weights="weight" if edges else None))
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- #
# Partition RACINE par transform — graphe identique (mêmes arêtes/seuil), seuls
# les POIDS changent → on isole l'effet du transform sur la coupe de Leiden.
# --------------------------------------------------------------------------- #
def root_eval(vecs: np.ndarray, k: int) -> dict:
    n = vecs.shape[0]
    k = max(2, min(int(k), n - 1))
    v32 = np.ascontiguousarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(v32, axis=1, keepdims=True)
    v32 = v32 / np.where(norms > 0, norms, 1.0)
    vecs64 = v32.astype(np.float64)
    neighbors = knn_search(v32, k)
    derived = derive_defaults(v32, k=k, neighbors=neighbors)
    thr = float(derived.threshold)
    graph = build_knn_graph(vecs64, k=k, threshold=thr, neighbors=neighbors)

    raw_w = np.array([w for (_, _, w) in graph.edges], dtype=np.float64)
    out = {"k": k, "threshold": round(thr, 4), "n_edges": len(graph.edges),
           "avg_degree": round(graph.avg_degree, 2), "sigma": round(_sigma(raw_w), 4)
           if raw_w.size else None, "transforms": {}}

    for name in TRANSFORMS:
        tw = _apply(name, raw_w)
        edges_t = [(i, j, float(t)) for (i, j, _), t in zip(graph.edges, tw)]
        g_t = KnnGraph(n=n, edges=edges_t, k=k, threshold=thr, backend=graph.backend)
        res = run_leiden(g_t, resolution=1.0, seed=SEED)
        q_raw = _modularity(graph.edges, n, res.membership, raw_w)   # yardstick commun
        out["transforms"][name] = {
            "n_clusters": res.n_clusters,
            "q_raw": round(q_raw, 4),
            "q_self": round(float(res.modularity), 4),
        }
    return out


# --------------------------------------------------------------------------- #
# Alignement gold (repnum) via le chemin tree — run_leiden monkeypatché pour
# appliquer le transform sur TOUTES les coupes (racine + sous-arbres).
# --------------------------------------------------------------------------- #
def _patched_run_leiden(name: str):
    def patched(graph, resolution=1.0, seed=DEFAULT_SEED, n_iterations=-1):
        if graph.edges and name != "raw":
            raw_w = np.array([w for (_, _, w) in graph.edges], dtype=np.float64)
            tw = _apply(name, raw_w)
            graph = KnnGraph(
                n=graph.n,
                edges=[(i, j, float(t)) for (i, j, _), t in zip(graph.edges, tw)],
                k=graph.k, threshold=graph.threshold, backend=graph.backend,
            )
        return run_leiden(graph, resolution=resolution, seed=seed, n_iterations=n_iterations)
    return patched


def _macro_of(tree, n: int) -> list:
    macro_of = [None] * n
    for mid in tree.macros:
        for i in tree.nodes[mid].members:
            macro_of[i] = mid
    return macro_of


def gold_align_by_transform(dataset: str, ideas, vecs, weights, k: int) -> dict:
    gold = ks.repnum_gold_axes(ideas)
    res = {}
    for name in TRANSFORMS:
        patched = _patched_run_leiden(name)
        live_mod.run_leiden = patched          # chemin racine
        analysis_mod.run_leiden = patched      # chemin sous-arbres
        try:
            tree = build_live_tree(ideas, vecs, weights, k=k, seed=SEED)
        finally:
            live_mod.run_leiden = run_leiden
            analysis_mod.run_leiden = run_leiden
        macro_of = _macro_of(tree, len(ideas))
        res[name] = {"n_macros": len(tree.macros),
                     "align": ks.alignment(macro_of, gold)}
    return res


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
def sweep_dataset(dataset: str) -> dict:
    print(f"\n=== {dataset} ===", flush=True)
    ideas, vecs, weights = load_cache(dataset)
    n = len(ideas)
    k = derive_k(n)
    print(f"  N={n}  derive_k(N)={k}", flush=True)

    root = root_eval(vecs, k)
    print(f"  graphe racine : edges={root['n_edges']}  deg={root['avg_degree']}  "
          f"thr={root['threshold']}  σ={root['sigma']}", flush=True)
    print(f"  {'transform':<8} {'n_clusters':>10} {'Q_raw':>8} {'Q_self':>8}", flush=True)
    base_qraw = root["transforms"]["raw"]["q_raw"]
    for name, t in root["transforms"].items():
        d = t["q_raw"] - base_qraw
        flag = "" if name == "raw" else f"  (ΔQ_raw {d:+.4f})"
        print(f"  {name:<8} {t['n_clusters']:>10} {t['q_raw']:>8.4f} {t['q_self']:>8.4f}{flag}",
              flush=True)

    gold = None
    if dataset == "republique-numerique":
        gold = gold_align_by_transform(dataset, ideas, vecs, weights, k)
        print("  --- alignement gold (Titre I/II/III) ---", flush=True)
        print(f"  {'transform':<8} {'n_macros':>9} {'ARI':>7} {'NMI':>7} {'V':>7}", flush=True)
        for name, g in gold.items():
            a = g["align"] or {}
            print(f"  {name:<8} {g['n_macros']:>9} {a.get('ari', float('nan')):>7.3f} "
                  f"{a.get('nmi', float('nan')):>7.3f} {a.get('v', float('nan')):>7.3f}",
                  flush=True)

    return {"dataset": dataset, "n": n, "k": k, "root": root, "gold": gold}


def main():
    results = [sweep_dataset(ds) for ds in DATASETS]
    out = _SCRATCH / "knn_weight_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRésultats JSON → {out}", flush=True)

    print("\n=== ARGMAX par dataset (yardstick = Q_raw ; gold = ARI repnum) ===", flush=True)
    for r in results:
        tf = r["root"]["transforms"]
        best = max(tf.items(), key=lambda kv: kv[1]["q_raw"])
        line = (f"{r['dataset']:>22}: best Q_raw = {best[0]} ({best[1]['q_raw']:.4f}) "
                f"vs raw ({tf['raw']['q_raw']:.4f})")
        if r["gold"]:
            ga = {k: (v["align"] or {}).get("ari", float("nan")) for k, v in r["gold"].items()}
            bg = max(ga.items(), key=lambda kv: kv[1])
            line += f" | best ARI = {bg[0]} ({bg[1]:.3f}) vs raw ({ga['raw']:.3f})"
        print(line, flush=True)


if __name__ == "__main__":
    sys.exit(main())
