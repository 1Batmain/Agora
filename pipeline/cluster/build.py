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
from pipeline.cluster.dedup import dedup_near
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
    k: int = 8,
    threshold: float = 0.84,
    resolution: float = 1.5,
    seed: int = 42,
    model_id: str | None = None,
    with_hdbscan: bool = False,
    source: str | None = None,
    lang: str | None = None,
    min_chars: int = 0,
    dedup_threshold: float | None = None,
    max_links: int | None = None,
) -> dict:
    ideas = io.load_ideas(input_path)
    src = io.resolve_input(input_path)

    # 0) Subset réel (ex. consultation TikTok FR) + nettoyage des avis trop
    #    courts/vides. On filtre AVANT d'embedder (moins de calcul) et on trace
    #    le détail dans meta pour audit.
    n_loaded = len(ideas)
    if source:
        ideas = [i for i in ideas if i.source == source]
    n_after_source = len(ideas)
    if lang:
        ideas = [i for i in ideas if i.lang == lang]
    n_after_lang = len(ideas)
    if min_chars:
        ideas = [i for i in ideas if len((i.text_clean or i.text).strip()) >= min_chars]
    n_after_minlen = len(ideas)

    n = len(ideas)
    if n == 0:
        raise SystemExit("Aucun avis à traiter (filtres trop stricts ?).")

    # 1) Embeddings ---------------------------------------------------------
    embedder = Embedder(model_id=model_id) if model_id else Embedder()
    texts = [idea.text_clean or idea.text for idea in ideas]
    vecs = embedder.embed(texts)
    weights = np.array([idea.weight for idea in ideas], dtype=np.float32)

    # 1.5) Déduplication near-dup (les gens répètent) -----------------------
    #      cosine > seuil → on garde 1 représentant, on cumule son `weight`.
    dedup_meta = None
    if dedup_threshold is not None:
        dd = dedup_near(vecs, weights, threshold=dedup_threshold)
        ideas = [ideas[i] for i in dd.keep]
        vecs = np.ascontiguousarray(vecs[dd.keep])
        weights = dd.weights
        dedup_meta = {
            "threshold": dedup_threshold,
            "n_in": dd.n_in,
            "n_out": dd.n_out,
            "n_collapsed": dd.n_collapsed,
        }

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
                "weight": round(float(weights[idx]), 4),
            },
            "cluster_id": cid,
            "color": color_for(cid),
        })

    id_by_idx = [idea.id for idea in ideas]
    # Leiden clustering ci-dessus utilise le graphe k-NN COMPLET. Pour le rendu,
    # on peut plafonner les arêtes AFFICHÉES (les plus fortes en cosine) si elles
    # sont trop denses — sans jamais retirer un nœud (aucune voix sacrifiée).
    edges = knn.edges
    n_links_total = len(edges)
    links_capped = False
    if max_links is not None and n_links_total > max_links:
        edges = sorted(edges, key=lambda e: e[2], reverse=True)[:max_links]
        links_capped = True
    links = [
        {
            "source": id_by_idx[i],
            "target": id_by_idx[j],
            "type": "knn",
            "props": {"weight": round(float(w), 4)},
        }
        for (i, j, w) in edges
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
            "subset": {
                "source": source,
                "lang": lang,
                "min_chars": min_chars,
                "n_loaded": n_loaded,
                "n_after_source": n_after_source,
                "n_after_lang": n_after_lang,
                "n_after_minlen": n_after_minlen,
            },
            "dedup": dedup_meta,
            "params": {
                "k": k,
                "threshold": threshold,
                "resolution": resolution,
                "seed": seed,
                "knn_backend": knn.backend,
                "avg_degree": round(knn.avg_degree, 3),
                "max_links": max_links,
                "n_links_total": n_links_total,
                "links_capped": links_capped,
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
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.84)
    ap.add_argument("--resolution", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default=None, help="override model_id")
    ap.add_argument("--source", default=None,
                    help="ne garder que cette source (ex. tiktok)")
    ap.add_argument("--lang", default=None, help="ne garder que cette langue (ex. fr)")
    ap.add_argument("--min-chars", type=int, default=0,
                    help="retire les avis dont text_clean < N caractères")
    ap.add_argument("--dedup", type=float, default=None, metavar="COSINE",
                    help="déduplique les near-dups (cosine > COSINE, ex. 0.95)")
    ap.add_argument("--max-links", type=int, default=None,
                    help="plafonne les arêtes affichées (garde les + fortes ; tous les nœuds restent)")
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
        source=args.source,
        lang=args.lang,
        min_chars=args.min_chars,
        dedup_threshold=args.dedup,
        max_links=args.max_links,
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
    sub = m.get("subset") or {}
    if sub.get("source") or sub.get("lang"):
        print(f"  subset     : source={sub.get('source')} lang={sub.get('lang')} "
              f"min_chars={sub.get('min_chars')}  "
              f"({sub.get('n_loaded')}→{sub.get('n_after_minlen')})")
    if m.get("dedup"):
        d = m["dedup"]
        print(f"  dedup      : cosine>{d['threshold']}  "
              f"{d['n_in']}→{d['n_out']} (−{d['n_collapsed']} near-dups)")
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
