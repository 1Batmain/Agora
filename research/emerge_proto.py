"""ÉMERGENCE d'arguments par DENSITÉ (proto R&D, lane stance-argmining).

Teste l'idée de Bob : les arguments doivent ÉMERGER de la densité de la donnée — le MÊME
système que les thèmes (clustering d'embeddings de claims), un cran plus fin. Un argument =
un sous-cluster DENSE de claims dans un `thème` ; `n_support` = sa taille ; les claims
idiosyncrasiques = bruit NON surfacé (fail-closed : « assez d'arguments → on analyse, sinon
positions pas assez développées »). Zéro sélection LLM biaisée « meilleur d'un mauvais lot ».

Deux questions mesurées :
  (1) GRANULARITÉ — les claims sont-ils déjà de la taille d'un argument (→ clusteriser les
      claims directement, aucune passe argument) ou verbeux (→ extraire des spans plus fins) ?
  (2) ÉMERGENCE — y a-t-il assez de densité au niveau argument ? combien de feuilles donnent
      ≥2 sous-clusters (clivage candidat) / 1 (consensus) / 0 (sous-développé) ?

Lit `research/emerge_cache/<ds>/` (produit par `emerge_build.py`). N'écrit que sous `research/`.
Zéro LLM (émergence pure densité). Affichage = claim MÉDOÏDE verbatim par cluster.
    uv run python research/emerge_proto.py --dataset republique-numerique
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from backend.analysis import _subdivide, _node_stats
from pipeline.cluster.leiden_cluster import DEFAULT_RESOLUTION, DEFAULT_SEED

CACHE = Path(__file__).resolve().parent / "emerge_cache"
# Résolution Leiden pour le niveau ARGUMENT (plus fin que les thèmes). Surchargeable.
ARG_RESOLUTION = float(__import__("os").environ.get("EMERGE_RESOLUTION", str(DEFAULT_RESOLUTION)))
# Un sous-cluster trop DIFFUS n'est pas un argument franc (même logique que τ des thèmes).
MAX_DISPERSION = float(__import__("os").environ.get("EMERGE_MAX_DISPERSION", "0.5"))


def _load(dataset: str):
    d = CACHE / dataset
    vecs = np.load(d / "claim_vecs.npz")["vecs"].astype(np.float32)
    claims = [json.loads(l) for l in (d / "claims.jsonl").read_text().splitlines()]
    leaves = json.loads((d / "leaves.json").read_text())
    return vecs, claims, leaves


def emerge_leaf(gis: list[int], vecs: np.ndarray, weights: np.ndarray) -> list[list[int]]:
    """Sous-communautés d'arguments (indices GLOBAUX) via le MÊME primitif que les thèmes.

    `_subdivide` (kNN cosinus + Leiden) partitionne la feuille en ≥2 sous-communautés viables,
    ou None si homogène → 1 seul argument (consensus sur un point). Zéro notion de bruit : la
    partition couvre toute la feuille, comme l'arbre de thèmes un cran au-dessus.
    """
    groups = _subdivide(gis, vecs, ARG_RESOLUTION, DEFAULT_SEED)
    if groups is None:
        return [list(gis)]                       # homogène → un argument
    return groups                                # déjà en indices GLOBAUX


def medoid(idx: list[int], vecs: np.ndarray) -> int:
    """Index (dans idx) du claim le plus central du cluster (max cos moyen aux autres)."""
    sub = vecs[idx]
    sims = sub @ sub.T
    return idx[int(np.argmax(sims.mean(axis=1)))]


def dispersion(idx: list[int], vecs: np.ndarray) -> float:
    """1 − cos moyen au centroïde (0 = très serré). Vecteurs L2-normalisés."""
    s = vecs[idx].sum(axis=0)
    return max(0.0, 1.0 - float(np.linalg.norm(s)) / len(idx))


def run(dataset: str) -> dict:
    vecs, claims, leaves = _load(dataset)
    texts = [c["text"] for c in claims]

    # (1) Granularité claim — longueur (proxy « claim = argument ? »).
    lengths = [len(t) for t in texts]
    gran = {"claim_chars_median": int(statistics.median(lengths)),
            "claim_chars_p90": int(np.percentile(lengths, 90)),
            "n_claims": len(texts)}

    # (2) Émergence par feuille via le primitif THÈMES (kNN cosinus + Leiden), un cran plus fin.
    per_leaf = []
    n_multi = n_single = 0
    n_args_total = n_args_franc = 0
    for lf in leaves:
        gis = lf["member_gis"]
        if not gis:
            continue
        communities = emerge_leaf(gis, vecs, None)
        args = []
        for members_gi in sorted(communities, key=len, reverse=True):
            disp = dispersion(members_gi, vecs)
            m = medoid(members_gi, vecs)
            franc = len(members_gi) >= 3 and disp <= MAX_DISPERSION   # argument « franc »
            args.append({"n_support": len(members_gi), "dispersion": round(disp, 3),
                         "franc": bool(franc), "argument": texts[m], "medoid_gi": m})
        args_franc = [a for a in args if a["franc"]]
        n_args_total += len(args)
        n_args_franc += len(args_franc)
        if len(args) >= 2:
            n_multi += 1
        else:
            n_single += 1
        per_leaf.append({"theme_id": lf["theme_id"], "title": lf["title"],
                         "n_claims": len(gis), "n_communities": len(args),
                         "n_arguments_francs": len(args_franc), "arguments": args})

    # Feuille « sous-développée » = aucun argument FRANC (communautés trop diffuses/petites).
    n_underdev = sum(1 for l in per_leaf if l["n_arguments_francs"] == 0)
    summary = {
        "dataset": dataset, "granularity": gran, "arg_resolution": ARG_RESOLUTION,
        "max_dispersion": MAX_DISPERSION, "n_leaves": len(per_leaf),
        "leaves_multi_community": n_multi, "leaves_single_community": n_single,
        "leaves_underdeveloped_no_franc_arg": n_underdev,
        "n_arguments_total": n_args_total, "n_arguments_francs": n_args_franc,
    }
    print(f"[emerge] {dataset} · claims médiane {gran['claim_chars_median']} car "
          f"(p90 {gran['claim_chars_p90']}) · {gran['n_claims']} claims")
    print(f"[emerge] {len(per_leaf)} feuilles · multi-communauté {n_multi} · "
          f"mono {n_single} · sous-développées (0 arg franc) {n_underdev}")
    print(f"[emerge] arguments : {n_args_total} communautés, dont {n_args_franc} francs "
          f"(≥3 claims, dispersion ≤ {MAX_DISPERSION})")
    return {"summary": summary, "leaves": per_leaf}


def main() -> None:
    ap = argparse.ArgumentParser(description="Émergence d'arguments par densité (R&D).")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or str(Path(__file__).parent / f"emerge_proto_{args.dataset}.json")
    result = run(args.dataset)
    Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[emerge] → {out}")


if __name__ == "__main__":
    main()
