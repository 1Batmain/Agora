"""Runner R&D — banc qualité JINA vs nomic (protocole IDENTIQUE au témoin).

Contexte licence (écrit en tête du verdict `research/bench_jina.md`) :
  - jina-embeddings-v3 (multilingue FR/DE/IT, 570M) = **CC-BY-NC-4.0** → NON-COMMERCIAL,
    rédhibitoire pour Agora. On le MESURE quand même (via le port transformers-natif
    `tomaarsen/jina-embeddings-v3-hf`) pour chiffrer ce qu'on s'interdit.
  - jina-embeddings-v2-base-de (Apache-2.0) = déployable MAIS bilingue DE-EN (pas FR/IT)
    ET son code custom est incompatible avec transformers moderne (`transformers.onnx`,
    `find_pruneable_heads_and_indices`, `config.is_decoder`…) → non runnable ici sans
    vendoriser un vieux transformers (dette). Non benché.

Pourquoi un runner à part et pas le registre + `quality_bench` ? jina-v3 via
sentence-transformers exige `trust_remote_code`, dont le code amont casse aussi sur
ce transformers. Le port NATIF (`AutoModel`) fonctionne, mais ne passe pas par
l'`Embedder` (ST). On l'embed donc via un adaptateur natif + mean-pooling, puis on
réutilise le clustering + les métriques du banc PARTAGÉ INCHANGÉ (`cluster`,
`coherence`, `metrics`) → protocole identique côté partition/mesure.

nomic-v2 (témoin servi) et e5-small (référence du piège « cluster par langue »)
passent, eux, par l'`Embedder` partagé (`eval_model`) → re-validation du témoin.

Usage :
    uv run python -m research.run_bench_jina                       # nomic + e5 + jina-v3
    uv run python -m research.run_bench_jina --shared nomic-v2 --no-bootstrap
"""

from __future__ import annotations

import argparse
import gc
import time

import numpy as np

from pipeline.embed.registry import resolve_model_id

from . import coherence, multilingual_data, quality_bench
from .metrics import purity, silhouette

JINA_V3_NATIVE = "tomaarsen/jina-embeddings-v3-hf"  # port transformers-natif de jina-v3
JINA_V3_LICENSE = "CC-BY-NC-4.0 (NON-COMMERCIAL — rédhibitoire pour Agora)"


def embed_jina_v3_native(texts: list[str], batch_size: int = 32, max_length: int = 512):
    """Embed jina-v3 via le port natif (AutoModel), mean-pooling + L2-norm, CPU.

    On n'active PAS d'adaptateur de tâche spécifique (config générique) — cf. limites
    du verdict : v3 propose des LoRA par tâche ('separation'/'retrieval') qui
    pourraient bouger les marges. Pooling moyen (recommandé par la carte modèle).
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(JINA_V3_NATIVE)
    model = AutoModel.from_pretrained(JINA_V3_NATIVE, dtype=torch.float32)
    model.eval()
    load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    out_vecs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
            h = model(**enc).last_hidden_state          # (b, t, d)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            v = torch.nn.functional.normalize(v, dim=1)
            out_vecs.append(v.cpu().numpy().astype(np.float32))
    encode_s = time.perf_counter() - t0
    vecs = np.vstack(out_vecs)
    del model, tok
    gc.collect()
    return vecs, load_s, encode_s


def eval_jina_v3(corpus, p, n_boot, boot_frac):
    """Même pipeline partition+métriques que `quality_bench.eval_model`, embed natif."""
    from sklearn.metrics import normalized_mutual_info_score

    vecs, load_s, encode_s = embed_jina_v3_native(corpus.texts)
    dim = int(vecs.shape[1])
    membership, modularity = quality_bench.cluster(vecs, p)
    lang_ids, topic_ids = corpus.lang_ids(), corpus.topic_ids()
    coh = coherence.per_language_coherence(membership, corpus.texts, corpus.langs)

    return quality_bench.ModelResult(
        model_id=JINA_V3_NATIVE,
        alias="jina-v3",
        dim=dim,
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
        load_seconds=round(load_s, 3),
        encode_seconds=round(encode_s, 3),
        latency_ms_per_text=round(1000 * encode_s / max(1, corpus.n), 3),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shared", default="nomic-v2,e5-small",
                    help="modèles via l'Embedder partagé (alias, virgules)")
    ap.add_argument("--n-topics", type=int, default=6)
    ap.add_argument("--max-per-cell", type=int, default=130)
    ap.add_argument("--min-chars", type=int, default=15)
    ap.add_argument("--seed", type=int, default=quality_bench.DEFAULTS["seed"])
    ap.add_argument("--bootstrap", type=int, default=4)
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--boot-frac", type=float, default=0.8)
    ap.add_argument("--out", default="research/quality_report_jina.md")
    args = ap.parse_args(argv)

    p = dict(quality_bench.DEFAULTS)
    p["seed"] = args.seed
    n_boot = 0 if args.no_bootstrap else args.bootstrap

    print("[bench-jina] chargement du corpus x-stance équilibré (topic×langue)…")
    corpus = multilingual_data.load_balanced(
        n_topics=args.n_topics, max_per_cell=args.max_per_cell,
        min_chars=args.min_chars, seed=args.seed,
    )
    print(f"[bench-jina] corpus n={corpus.n}  langues={corpus.lang_counts}"
          f"  thèmes={len(corpus.topic_counts)}")

    results = []
    shared = [s.strip() for s in args.shared.split(",") if s.strip()]
    for alias in shared:
        print(f"[bench-jina] modèle partagé (Embedder) : {alias} "
              f"({resolve_model_id(alias)})")
        results.append(quality_bench.eval_model(alias, corpus, p, n_boot, args.boot_frac))

    print(f"[bench-jina] jina-v3 (natif, {JINA_V3_NATIVE}) — licence {JINA_V3_LICENSE}")
    results.append(eval_jina_v3(corpus, p, n_boot, args.boot_frac))

    quality_bench.compute_composite(results)
    results.sort(key=lambda r: (r.composite is None, -(r.composite or 0)))

    import platform
    bench = {
        "meta": {
            "seed": args.seed,
            "n_comments": corpus.n,
            "lang_counts": corpus.lang_counts,
            "topic_counts": corpus.topic_counts,
            "n_topics": len(corpus.topic_counts),
            "topics": sorted(corpus.topic_counts),
            "min_chars": args.min_chars,
            "bootstrap": n_boot,
            "bootstrap_frac": args.boot_frac,
            "cluster_params": p,
            "weights": quality_bench.WEIGHTS,
            "wall_seconds": None,
            "python": platform.python_version(),
        },
        "results": results,
    }
    path = quality_bench.write_report(bench, args.out)
    print(f"[bench-jina] rapport (scorecard brute) : {path}")
    for r in results:
        print(f"  {r.alias:10s} nmi_lang={r.nmi_lang:.3f} nmi_topic={r.nmi_topic:.3f} "
              f"coh={r.coherence if r.coherence is None else round(r.coherence,3)} "
              f"lat={r.latency_ms_per_text}ms/txt composite={r.composite}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
