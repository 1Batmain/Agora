"""Runner R&D — banc qualité mistral-embed (API EU) vs nomic (protocole identique).

Pourquoi c'est architecturalement recevable malgré la prod « sans clé » : le
pipeline embed au BUILD (en dev, avec la clé Mistral) et la prod sert le CACHE
read-only. Un embedder API s'utiliserait donc au build, pas au service — la
souveraineté du *serve* prod est préservée. Reste la question data→API au build
(le corpus quitte la machine) ; Mistral est EU (RGPD), et ici le gold x-stance
est déjà public. À peser pour un dataset FR réel (cf. verdict).

On embed le gold x-stance équilibré via `POST /v1/embeddings` (model mistral-embed),
L2-normalise, puis on réutilise le clustering + métriques du banc PARTAGÉ INCHANGÉ
(`quality_bench.cluster`, `coherence`, `metrics`) → protocole identique côté
partition/mesure. nomic-v2 (témoin servi) passe par l'Embedder partagé (re-validation).

Usage :
    uv run python -m research.run_bench_mistral
    uv run python -m research.run_bench_mistral --model mistral-embed --no-bootstrap
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

import numpy as np

from pipeline.cluster.mistral_client import load_api_key

from . import coherence, multilingual_data, quality_bench
from .metrics import purity, silhouette

EMBED_URL = "https://api.mistral.ai/v1/embeddings"


def embed_mistral(texts: list[str], model: str, key: str,
                  batch_size: int = 64, max_retries: int = 5):
    """Embed via l'API Mistral, batché, retry sur 429/5xx. Renvoie (n,d) float32 L2."""
    out: list[list[float]] = [None] * len(texts)  # type: ignore
    t0 = time.perf_counter()
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        payload = json.dumps({"model": model, "input": batch}).encode()
        req = urllib.request.Request(
            EMBED_URL, data=payload,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
        )
        for attempt in range(max_retries):
            try:
                resp = urllib.request.urlopen(req, timeout=60)
                data = json.load(resp)
                for item in data["data"]:
                    out[start + item["index"]] = item["embedding"]
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503) and attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise
        # petit throttle pour rester sous la limite RPM
        time.sleep(0.2)
        if (start // batch_size) % 5 == 0:
            print(f"  [mistral] {min(start + batch_size, len(texts))}/{len(texts)}",
                  flush=True)
    encode_s = time.perf_counter() - t0
    vecs = np.asarray(out, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms, encode_s


def eval_mistral(corpus, p, n_boot, boot_frac, model, key):
    from sklearn.metrics import normalized_mutual_info_score

    vecs, encode_s = embed_mistral(corpus.texts, model, key)
    membership, modularity = quality_bench.cluster(vecs, p)
    lang_ids, topic_ids = corpus.lang_ids(), corpus.topic_ids()
    coh = coherence.per_language_coherence(membership, corpus.texts, corpus.langs)
    return quality_bench.ModelResult(
        model_id=f"mistral-api:{model}",
        alias=model,
        dim=int(vecs.shape[1]),
        n_clusters=len(set(membership)),
        coherence=coh["overall"],
        coherence_per_lang=coh["per_lang"],
        nmi_lang=float(normalized_mutual_info_score(lang_ids, membership)),
        lang_purity=purity(membership, lang_ids),
        nmi_topic=float(normalized_mutual_info_score(topic_ids, membership)),
        topic_purity=purity(membership, topic_ids),
        silhouette=silhouette(vecs, membership),
        modularity=modularity,
        stability=quality_bench.bootstrap_stability(vecs, p, n_boot, boot_frac, p["seed"]),
        load_seconds=0.0,
        encode_seconds=round(encode_s, 3),
        latency_ms_per_text=round(1000 * encode_s / max(1, corpus.n), 3),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mistral-embed")
    ap.add_argument("--shared", default="nomic-v2",
                    help="modèles via l'Embedder partagé (témoin)")
    ap.add_argument("--n-topics", type=int, default=6)
    ap.add_argument("--max-per-cell", type=int, default=130)
    ap.add_argument("--min-chars", type=int, default=15)
    ap.add_argument("--seed", type=int, default=quality_bench.DEFAULTS["seed"])
    ap.add_argument("--bootstrap", type=int, default=4)
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--boot-frac", type=float, default=0.8)
    ap.add_argument("--out", default="research/quality_report_mistral.md")
    args = ap.parse_args(argv)

    key = load_api_key()
    if not key:
        print("[bench-mistral] AUCUNE clé Mistral (var/mistral.key / MISTRAL_API_KEY).")
        return 2

    p = dict(quality_bench.DEFAULTS)
    p["seed"] = args.seed
    n_boot = 0 if args.no_bootstrap else args.bootstrap

    print("[bench-mistral] corpus x-stance équilibré…")
    corpus = multilingual_data.load_balanced(
        n_topics=args.n_topics, max_per_cell=args.max_per_cell,
        min_chars=args.min_chars, seed=args.seed,
    )
    print(f"[bench-mistral] n={corpus.n} langues={corpus.lang_counts}")

    results = []
    for alias in [s.strip() for s in args.shared.split(",") if s.strip()]:
        print(f"[bench-mistral] témoin partagé : {alias}")
        results.append(quality_bench.eval_model(alias, corpus, p, n_boot, args.boot_frac))

    print(f"[bench-mistral] embedder API : {args.model} (EU)")
    results.append(eval_mistral(corpus, p, n_boot, args.boot_frac, args.model, key))

    quality_bench.compute_composite(results)
    results.sort(key=lambda r: (r.composite is None, -(r.composite or 0)))

    import platform
    bench = {
        "meta": {
            "seed": args.seed, "n_comments": corpus.n,
            "lang_counts": corpus.lang_counts, "topic_counts": corpus.topic_counts,
            "n_topics": len(corpus.topic_counts), "topics": sorted(corpus.topic_counts),
            "min_chars": args.min_chars, "bootstrap": n_boot,
            "bootstrap_frac": args.boot_frac, "cluster_params": p,
            "weights": quality_bench.WEIGHTS, "wall_seconds": None,
            "python": platform.python_version(),
        },
        "results": results,
    }
    path = quality_bench.write_report(bench, args.out)
    print(f"[bench-mistral] rapport : {path}")
    for r in results:
        print(f"  {r.alias:14s} nmi_lang={r.nmi_lang:.3f} nmi_topic={r.nmi_topic:.3f} "
              f"coh={round(r.coherence,3) if r.coherence is not None else None} "
              f"dim={r.dim} lat={r.latency_ms_per_text}ms/txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
