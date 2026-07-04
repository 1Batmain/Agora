"""Banc QUALITÉ de clustering via API (LMStudio, Nvidia NIM, etc.).

Lit une configuration JSON (`api_models.json` par défaut) pour évaluer la
qualité du clustering sur divers modèles distants.
Le pipeline est identique à `quality_bench.py` (rang-kNN → Leiden),
mais avec un appel réseau via `APIEmbedder` au lieu d'un chargement local de poids.

Sortie : un fichier CSV avec les métriques et métadonnées.

Usage :
    uv run python -m research.api_quality_bench
    uv run python -m research.api_quality_bench --config research/api_models.json --out bench_api_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from pipeline.cluster.knn import build_knn_graph
from pipeline.cluster.leiden_cluster import run_leiden
from pipeline.embed.api_embedder import APIEmbedder

# Import des métriques de l'évaluation existante
from . import coherence, multilingual_data
from .metrics import purity, silhouette

DEFAULTS = {
    "k": 15,
    "threshold": -1.0,
    "resolution": 1.0,
    "seed": 42,
}

@dataclass
class APIModelResult:
    name: str
    source: str
    url: str
    model_path: str
    dim: int
    n_clusters: int
    coherence: float | None
    nmi_lang: float
    lang_purity: float
    nmi_topic: float
    topic_purity: float
    silhouette: float | None
    modularity: float
    encode_seconds: float
    latency_ms_per_text: float
    error: str | None = None


def cluster(vecs: np.ndarray, p: dict) -> tuple[list[int], float]:
    graph = build_knn_graph(vecs, k=p["k"], threshold=p["threshold"])
    res = run_leiden(graph, resolution=p["resolution"], seed=p["seed"])
    return res.membership, res.modularity


def eval_api_model(
    config: dict,
    corpus: multilingual_data.MultiCorpus,
    p: dict,
    embedder_instance=None
) -> APIModelResult:
    """Évalue UN modèle via son API ou une instance native."""
    name = config.get("name", "Unnamed")
    source = config.get("source", "unknown")
    url = config.get("url", "")
    model_path = config.get("path", "")
    api_key = config.get("api_key", "")

    if embedder_instance is not None:
        embedder = embedder_instance
    else:
        embedder = APIEmbedder(
            url=url,
            model_path=model_path,
            api_key=api_key,
            batch_size=32,
            input_type=config.get("input_type"),
            dimensions=config.get("dimensions"),
        )

    t0 = time.perf_counter()
    try:
        vecs = embedder.embed(corpus.texts)
    except Exception as e:
        return APIModelResult(
            name=name, source=source, url=url, model_path=model_path,
            dim=0, n_clusters=0, coherence=0.0, nmi_lang=0.0, lang_purity=0.0,
            nmi_topic=0.0, topic_purity=0.0, silhouette=0.0, modularity=0.0,
            encode_seconds=0.0, latency_ms_per_text=0.0, error=str(e)
        )
    
    encode_s = time.perf_counter() - t0
    dim = int(vecs.shape[1])

    # Clustering
    membership, modularity = cluster(vecs, p)
    n_clusters = len(set(membership))

    # Métriques
    lang_ids = corpus.lang_ids()
    topic_ids = corpus.topic_ids()
    coh = coherence.per_language_coherence(membership, corpus.texts, corpus.langs)

    res = APIModelResult(
        name=name,
        source=source,
        url=url,
        model_path=model_path,
        dim=dim,
        n_clusters=n_clusters,
        coherence=coh["overall"],
        nmi_lang=float(normalized_mutual_info_score(lang_ids, membership)),
        lang_purity=purity(membership, lang_ids),
        nmi_topic=float(normalized_mutual_info_score(topic_ids, membership)),
        topic_purity=purity(membership, topic_ids),
        silhouette=silhouette(vecs, membership),
        modularity=modularity,
        encode_seconds=round(encode_s, 3),
        latency_ms_per_text=round(1000 * encode_s / max(1, corpus.n), 3),
    )
    return res


def write_csv(results: list[APIModelResult], out_path: str) -> None:
    fieldnames = [
        "name", "source", "model_path", "dim", "n_clusters", 
        "nmi_lang", "lang_purity", "nmi_topic", "topic_purity", 
        "coherence", "silhouette", "modularity", 
        "encode_seconds", "latency_ms_per_text", "error"
    ]
    
    # Assurer que le dossier parent existe
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            d = {
                "name": r.name,
                "source": r.source,
                "model_path": r.model_path,
                "dim": r.dim,
                "n_clusters": r.n_clusters,
                "nmi_lang": round(r.nmi_lang, 4) if r.nmi_lang is not None else None,
                "lang_purity": round(r.lang_purity, 4) if r.lang_purity is not None else None,
                "nmi_topic": round(r.nmi_topic, 4) if r.nmi_topic is not None else None,
                "topic_purity": round(r.topic_purity, 4) if r.topic_purity is not None else None,
                "coherence": round(r.coherence, 4) if r.coherence is not None else None,
                "silhouette": round(r.silhouette, 4) if r.silhouette is not None else None,
                "modularity": round(r.modularity, 4) if r.modularity is not None else None,
                "encode_seconds": r.encode_seconds,
                "latency_ms_per_text": r.latency_ms_per_text,
                "error": r.error or ""
            }
            writer.writerow(d)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="research/api_models.json", help="Chemin du JSON contenant la liste des modèles")
    ap.add_argument("--out", default="bench_api_results.csv", help="Chemin du CSV de sortie")
    ap.add_argument("--n-topics", type=int, default=6)
    ap.add_argument("--per-cell", type=int, default=10, help="Limité par défaut pour des requêtes API plus rapides")
    ap.add_argument("--max-per-cell", type=int, default=50)
    ap.add_argument("--min-chars", type=int, default=15)
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--k", type=int, default=DEFAULTS["k"])
    ap.add_argument("--resolution", type=float, default=DEFAULTS["resolution"])
    args = ap.parse_args(argv)

    if not os.path.exists(args.config):
        print(f"Erreur: Fichier de configuration '{args.config}' introuvable.")
        return 1

    with open(args.config, "r", encoding="utf-8") as f:
        models_config = json.load(f)

    # Résolution `${VAR}` -> variable d'environnement pour api_key (jamais de secret en clair dans le JSON).
    for m in models_config:
        key = m.get("api_key", "")
        if key.startswith("${") and key.endswith("}"):
            m["api_key"] = os.environ.get(key[2:-1], "")

    p = dict(DEFAULTS)
    p["k"] = args.k
    p["resolution"] = args.resolution
    p["seed"] = args.seed

    print(f"Chargement du corpus (seed={args.seed}, n_topics={args.n_topics})...")
    corpus = multilingual_data.load_balanced(
        n_topics=args.n_topics, per_cell=args.per_cell, max_per_cell=args.max_per_cell,
        min_chars=args.min_chars, seed=args.seed,
    )
    print(f"[corpus] {corpus.n} commentaires | langues {corpus.lang_counts} | {len(corpus.topic_counts)} thèmes")

    results = []
    for m in models_config:
        print(f"Évaluation de {m.get('name')} (source={m.get('source')}, path={m.get('path')}) via API...")
        res = eval_api_model(m, corpus, p)
        if res.error:
            print(f"  -> ERREUR: {res.error}")
        else:
            print(f"  -> OK: dim={res.dim} clusters={res.n_clusters} coh={res.coherence:.3f} nmi_lang={res.nmi_lang:.3f} nmi_topic={res.nmi_topic:.3f}")
        results.append(res)

    write_csv(results, args.out)
    print(f"Banc terminé. Résultats écrits dans {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
