"""A/B embedder sur un dataset FR SERVI (défaut tiktok) — arctic-l vs nomic-v2.

Sur les vraies données FR il n'y a PAS de gold (`topic`), donc on NE PEUT PAS mesurer
NMI(thème). On évalue par ce qui reste valable sans vérité terrain :
  - métriques INTERNES (indicatives, cf. le piège e5) : silhouette, modularité, cohérence
    NPMI (français), stabilité bootstrap ;
  - structure : #clusters fins, #macro-thèmes, params dérivés ;
  - accord des deux partitions : NMI(membership_nomic, membership_arctic) ;
  - INSPECTION QUALITATIVE : macro-thèmes côte à côte (taille, mots-clés c-TF-IDF, verbatims).

Réplique EXACTE de la partition de production (`build_theme_tree`) : `derive_defaults`
→ `build_knn_graph` → `run_leiden` → `_build_macro_forest`. Embed EN MÉMOIRE via
`Embedder` (aucune écriture des caches servis). Lecture seule sur `backend/cache/<ds>`.

Usage : uv run python -m research.ab_embedder_fr [--dataset tiktok] [--embedders nomic-v2,arctic-l]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from backend.analysis import _build_macro_forest
from pipeline.cluster.adaptive import derive_defaults
from pipeline.cluster.knn import build_knn_graph
from pipeline.cluster.leiden_cluster import DEFAULT_RESOLUTION, run_leiden
from pipeline.embed.embedder import Embedder
from pipeline.embed.registry import resolve_model_id

from .coherence import per_language_coherence
from .metrics import silhouette
from .quality_bench import DEFAULTS, bootstrap_stability

SEED = 42


def load_claims(dataset: str):
    cache = Path("backend/cache") / dataset
    claims = json.loads((cache / "claims.json").read_text())["claims"]
    ideas = [json.loads(l) for l in (cache / "ideas.jsonl").read_text().splitlines() if l.strip()]
    texts, owner = [], []
    for ai, it in enumerate(ideas):
        for c in claims.get(it["id"], []):
            texts.append(c["text"])
            owner.append(ai)
    return texts, owner


def run_embedder(alias: str, texts, owner):
    weights = np.ones(len(texts), dtype=np.float32)
    print(f"[ab] embed {alias} ({resolve_model_id(alias)}) — {len(texts)} claims…", flush=True)
    vecs = Embedder(model_id=alias).embed(texts).astype(np.float32)
    derived = derive_defaults(vecs)
    graph = build_knn_graph(vecs, k=derived.k, threshold=derived.threshold)
    res = run_leiden(graph, resolution=DEFAULT_RESOLUTION, seed=SEED)
    membership = res.membership
    by: dict[int, list[int]] = {}
    for i, c in enumerate(membership):
        by.setdefault(c, []).append(i)
    fine = list(by.values())
    nodes, order, macros, tau, merge_thr = _build_macro_forest(
        fine, vecs, weights, owner, texts,
        min_sub_size=derived.min_sub_size, resolution=DEFAULT_RESOLUTION, seed=SEED)

    p = dict(DEFAULTS); p["seed"] = SEED
    langs = ["fr"] * len(texts)
    sil = silhouette(vecs, membership)
    coh = per_language_coherence(membership, texts, langs)["overall"]
    stab = bootstrap_stability(vecs, p, 4, 0.8, SEED)
    return {
        "alias": alias, "dim": int(vecs.shape[1]),
        "k": derived.k, "threshold": round(float(derived.threshold), 3),
        "min_sub_size": derived.min_sub_size,
        "n_fine": len(fine), "n_macros": len(macros),
        "modularity": round(float(res.modularity), 3),
        "silhouette": None if sil is None else round(float(sil), 3),
        "coherence": round(float(coh), 3),
        "stability": None if stab is None else round(float(stab), 3),
        "membership": membership, "nodes": nodes, "macros": macros, "texts": texts,
    }


def macro_summary(r, top_n=8, verbatims=2):
    """Retourne [(size, label, keywords, [verbatims])] trié par taille décroissante."""
    rows = []
    for mid in r["macros"]:
        node = r["nodes"][mid]
        members = node.members
        vbs = []
        for i in members[:verbatims]:
            t = r["texts"][i].replace("\n", " ").strip()
            vbs.append(t[:120] + ("…" if len(t) > 120 else ""))
        kw = ", ".join((node.keywords or [])[:6]) or (node.label or "—")
        rows.append((len(members), (node.title or node.label or mid), kw, vbs))
    rows.sort(key=lambda x: -x[0])
    return rows[:top_n]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="tiktok")
    ap.add_argument("--embedders", default="nomic-v2,arctic-l")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    out = args.out or f"research/ab_embedder_{args.dataset}.md"

    texts, owner = load_claims(args.dataset)
    aliases = [a.strip() for a in args.embedders.split(",") if a.strip()]
    results = [run_embedder(a, texts, owner) for a in aliases]

    # accord des partitions
    agree = None
    if len(results) == 2:
        agree = round(float(normalized_mutual_info_score(
            results[0]["membership"], results[1]["membership"])), 3)

    L = ["# A/B embedder — dataset FR servi (pas de gold : métriques INTERNES + inspection)\n",
         f"> Dataset **{args.dataset}** · {len(texts)} claims · clustering de PRODUCTION "
         f"(derive_defaults → knn → Leiden → macro-forest) · embed en mémoire (caches servis intacts).\n",
         "> ⚠️ Sans vérité terrain, silhouette/modularité sont INDICATIVES (piège e5). "
         "La cohérence NPMI (fr) et l'inspection humaine priment.\n",
         "## Scorecard (interne)\n",
         "| Métrique | sens | " + " | ".join(r["alias"] for r in results) + " |",
         "|---|:--:|" + "|".join([":--:"] * len(results)) + "|"]
    def row(lbl, sense, key):
        L.append(f"| {lbl} | {sense} | " + " | ".join(str(r[key]) for r in results) + " |")
    row("Cohérence NPMI (fr)", "↑", "coherence")
    row("Silhouette (cosine)", "↑", "silhouette")
    row("Modularité (Leiden)", "↑", "modularity")
    row("Stabilité (ARI boot)", "↑", "stability")
    row("# clusters fins", "·", "n_fine")
    row("# macro-thèmes", "·", "n_macros")
    row("k dérivé", "·", "k")
    row("seuil dérivé", "·", "threshold")
    row("dimension", "·", "dim")
    if agree is not None:
        L.append(f"\n**Accord des partitions** NMI(nomic, arctic) = **{agree}** "
                 "(1 = clusters identiques ; bas = les deux voient des thèmes différents).\n")

    for r in results:
        L.append(f"\n## Macro-thèmes — {r['alias']} ({r['n_macros']} macros, {r['n_fine']} fins)\n")
        for size, title, kw, vbs in macro_summary(r):
            L.append(f"- **{title}** ({size} claims) — _{kw}_")
            for v in vbs:
                L.append(f"    - « {v} »")

    Path(out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n[ab] rapport:", out)
    for r in results:
        print(f"  {r['alias']:10s} coh={r['coherence']} sil={r['silhouette']} "
              f"mod={r['modularity']} stab={r['stability']} fins={r['n_fine']} macros={r['n_macros']}")
    if agree is not None:
        print(f"  accord NMI(nomic,arctic)={agree}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
