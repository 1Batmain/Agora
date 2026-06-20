"""Pipeline batch NLP → GraphPayload coloré par thème.

  ideas.jsonl → embed (e5-small) → graphe k-NN cosine → Leiden →
  scoring + naming TF-IDF → graph.json {meta, nodes, links, themes}

Usage :
    uv run python -m pipeline.cluster.build [--input PATH] [--out PATH]
        [--k 10] [--threshold 0.80] [--resolution 1.0] [--seed 42]
        [--fixture]   # écrit aussi pipeline/cluster/fixtures/graph.sample.json

Le format de sortie est le GraphPayload du contrat cross-lane (aligné dummy) :
    nodes : GraphNode{ id, type, label, props{...}, cluster_id, color }
    links : GraphLink{ source, target, type, props{weight} }
    themes: Theme{ cluster_id, member_ids, size, weight_sum, diversity,
                   consensus, centroid, label, keywords, color }
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pipeline.cluster import io
from pipeline.cluster.knn import build_knn_graph
from pipeline.cluster.leiden_cluster import run_leiden
from pipeline.cluster.naming import name_clusters
from pipeline.cluster.palette import color_for
from pipeline.cluster.scoring import rank_clusters, score_cluster
from pipeline.embed.embedder import Embedder

REPO_ROOT = io.REPO_ROOT
DEFAULT_OUT = REPO_ROOT / "data" / "graph.json"
FIXTURE_OUT = REPO_ROOT / "pipeline" / "cluster" / "fixtures" / "graph.sample.json"


def build_payload(
    input_path: str | None = None,
    k: int = 10,
    threshold: float = 0.80,
    resolution: float = 1.0,
    seed: int = 42,
    model_id: str | None = None,
    with_hdbscan: bool = False,
) -> dict:
    ideas = io.load_ideas(input_path)
    src = io.resolve_input(input_path)
    n = len(ideas)
    if n == 0:
        raise SystemExit("Aucun avis à traiter.")

    # 1) Embeddings ---------------------------------------------------------
    embedder = Embedder(model_id=model_id) if model_id else Embedder()
    texts = [idea.text_clean or idea.text for idea in ideas]
    vecs = embedder.embed(texts)
    weights = np.array([idea.weight for idea in ideas], dtype=np.float32)

    # 2) Graphe k-NN sémantique --------------------------------------------
    knn = build_knn_graph(vecs, k=k, threshold=threshold)

    # 3) Leiden -------------------------------------------------------------
    leiden = run_leiden(knn, resolution=resolution, seed=seed)
    membership = leiden.membership

    # Contender HDBSCAN (optionnel — juste tracé dans meta pour l'éval).
    hdbscan_meta = None
    if with_hdbscan:
        from pipeline.cluster import hdbscan_contender as hc

        if hc.available():
            res = hc.run_hdbscan(vecs, seed=seed)
            hdbscan_meta = {
                "n_clusters": res.n_clusters,
                "n_noise": res.n_noise,
                "params": res.params,
            }
        else:
            hdbscan_meta = {"available": False}

    # 4) Scoring + naming par communauté -----------------------------------
    members: dict[int, list[int]] = defaultdict(list)
    for idx, cid in enumerate(membership):
        members[cid].append(idx)

    scores = {
        cid: score_cluster(idxs, vecs, weights)
        for cid, idxs in members.items()
    }
    cluster_docs = {
        cid: [ideas[i].text for i in idxs] for cid, idxs in members.items()
    }
    names = name_clusters(cluster_docs)

    # 5) Sérialisation GraphPayload ----------------------------------------
    nodes = []
    for idx, idea in enumerate(ideas):
        cid = int(membership[idx])
        nodes.append({
            "id": idea.id,
            "type": "idea",
            "label": (idea.text[:80] + "…") if len(idea.text) > 80 else idea.text,
            "props": {
                "text": idea.text,
                "text_clean": idea.text_clean or idea.text,
                "ts": idea.ts,
                "lang": idea.lang,
                "author_hash": idea.author_hash,
                "source": idea.source,
                "weight": idea.weight,
            },
            "cluster_id": cid,
            "color": color_for(cid),
        })

    id_by_idx = [idea.id for idea in ideas]
    links = [
        {
            "source": id_by_idx[i],
            "target": id_by_idx[j],
            "type": "knn",
            "props": {"weight": round(float(w), 4)},
        }
        for (i, j, w) in knn.edges
    ]

    themes = []
    for cid in rank_clusters(scores):
        s = scores[cid]
        nm = names.get(cid, {"label": f"thème {cid}", "keywords": []})
        themes.append({
            "cluster_id": cid,
            "member_ids": [ideas[i].id for i in members[cid]],
            "size": s.size,
            "weight_sum": s.weight_sum,
            "diversity": s.diversity,
            "consensus": s.consensus,
            "centroid": s.centroid,
            "label": nm["label"],
            "keywords": nm["keywords"],
            "color": color_for(cid),
        })

    payload = {
        "meta": {
            "model_id": embedder.model_id,
            "embedding_dim": int(vecs.shape[1]),
            "n_nodes": len(nodes),
            "n_links": len(links),
            "n_themes": len(themes),
            "input": str(src.relative_to(REPO_ROOT)) if src.is_relative_to(REPO_ROOT) else str(src),
            "params": {
                "k": k,
                "threshold": threshold,
                "resolution": resolution,
                "seed": seed,
                "knn_backend": knn.backend,
                "avg_degree": round(knn.avg_degree, 3),
            },
            "clustering": {
                "primary": "leiden",
                "leiden": {
                    "n_clusters": leiden.n_clusters,
                    "modularity": leiden.modularity,
                    "resolution": leiden.resolution,
                    "seed": leiden.seed,
                },
                "hdbscan_contender": hdbscan_meta,
            },
        },
        "nodes": nodes,
        "links": links,
        "themes": themes,
    }
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Build GraphPayload (NLP batch).")
    ap.add_argument("--input", default=None, help="ideas.jsonl (sinon auto-résolu)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="chemin graph.json")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default=None, help="override model_id")
    ap.add_argument("--with-hdbscan", action="store_true", help="trace le contender")
    ap.add_argument("--fixture", action="store_true",
                    help="écrit aussi le fixture viz graph.sample.json")
    args = ap.parse_args()

    payload = build_payload(
        input_path=args.input,
        k=args.k,
        threshold=args.threshold,
        resolution=args.resolution,
        seed=args.seed,
        model_id=args.model,
        with_hdbscan=args.with_hdbscan,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.fixture:
        FIXTURE_OUT.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE_OUT.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    m = payload["meta"]
    print(f"✓ {out}")
    print(f"  model_id   : {m['model_id']} (dim {m['embedding_dim']})")
    print(f"  nodes/links: {m['n_nodes']} / {m['n_links']}  (avg_deg {m['params']['avg_degree']})")
    print(f"  leiden     : {m['clustering']['leiden']['n_clusters']} communautés "
          f"(modularité {m['clustering']['leiden']['modularity']}, backend {m['params']['knn_backend']})")
    print("  thèmes     :")
    for t in payload["themes"]:
        print(f"    [{t['cluster_id']}] {t['label']}  "
              f"(n={t['size']}, w={t['weight_sum']}, div={t['diversity']}, cons={t['consensus']})")


if __name__ == "__main__":
    main()
