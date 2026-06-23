"""Banc d'arbitrage **Leiden vs UMAP+HDBSCAN** sur x-stance (eval-as-truth).

Pour chaque question politique (commentaires labellisés FAVOR/AGAINST) :
  embed (e5-small, pipeline.embed) → les DEUX clusterings → métriques vs labels
  (NMI/ARI/pureté/silhouette) + stabilité bootstrap + coût.
On agrège (moyenne ± écart-type) par approche et on écrit `eval/report.md`.

On RÉUTILISE le pipeline mergé (pas de réimplémentation) :
  - `pipeline.embed.embedder.Embedder`
  - `pipeline.cluster.knn.build_knn_graph` + `leiden_cluster.run_leiden`
  - `pipeline.cluster.hdbscan_contender.run_hdbscan`

Usage :
    uv run python -m eval.bench                       # défaut : 8 questions
    uv run python -m eval.bench --sample-questions 12 --bootstrap 5
    uv run python -m eval.bench --no-bootstrap        # plus rapide
"""
from __future__ import annotations

import argparse
import platform
import random
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import adjusted_rand_score

from pipeline.cluster.hdbscan_contender import available as hdbscan_available
from pipeline.cluster.hdbscan_contender import run_hdbscan
from pipeline.cluster.knn import build_knn_graph
from pipeline.cluster.leiden_cluster import run_leiden
from pipeline.embed.embedder import DEFAULT_MODEL_ID, Embedder

from . import data, metrics, report

# --- Paramètres de clustering (alignés sur les défauts pipeline.cluster) -------
DEFAULTS = {
    "k": 8,
    "threshold": 0.84,
    "resolution": 1.0,
    "seed": 42,
    # HDBSCAN/UMAP
    "min_cluster_size": 5,
    "n_neighbors": 15,
    "n_components": 5,
}


@dataclass
class ApproachRuns:
    """Accumule les métriques de chaque question pour une approche."""

    name: str
    nmi: list[float] = field(default_factory=list)
    ari: list[float] = field(default_factory=list)
    purity: list[float] = field(default_factory=list)
    silhouette: list = field(default_factory=list)
    n_clusters: list[int] = field(default_factory=list)
    n_noise: list[int] = field(default_factory=list)
    stability: list = field(default_factory=list)  # ARI inter-runs par question
    cluster_seconds: list[float] = field(default_factory=list)

    def add(self, m: dict, secs: float) -> None:
        self.nmi.append(m["nmi"])
        self.ari.append(m["ari"])
        self.purity.append(m["purity"])
        self.silhouette.append(m["silhouette"])
        self.n_clusters.append(m["n_clusters"])
        self.n_noise.append(m["n_noise"])
        self.cluster_seconds.append(secs)


# --- Clustering : deux fonctions homogènes (vecs -> membership) ----------------
def cluster_leiden(vecs: np.ndarray, p: dict) -> list[int]:
    graph = build_knn_graph(vecs, k=p["k"], threshold=p["threshold"])
    res = run_leiden(graph, resolution=p["resolution"], seed=p["seed"])
    return res.membership


def cluster_hdbscan(vecs: np.ndarray, p: dict) -> list[int]:
    res = run_hdbscan(
        vecs,
        n_neighbors=p["n_neighbors"],
        n_components=p["n_components"],
        min_cluster_size=p["min_cluster_size"],
        seed=p["seed"],
    )
    return res.membership


APPROACHES = {
    "Leiden": cluster_leiden,
    "HDBSCAN": cluster_hdbscan,
}


# --- Stabilité bootstrap -------------------------------------------------------
def bootstrap_stability(
    cluster_fn, vecs: np.ndarray, n_boot: int, frac: float, seed: int
) -> float | None:
    """Accord inter-runs : ARI moyen entre paires de ré-échantillons.

    On tire `n_boot` sous-échantillons (sans remise, fraction `frac`), on
    clusterise chacun, puis pour chaque paire de runs on calcule l'ARI sur
    l'INTERSECTION de leurs indices. Stable = clusterings cohérents → ARI≈1.
    """
    n = vecs.shape[0]
    size = max(2, int(round(frac * n)))
    if size >= n:
        return None
    rng = random.Random(seed)

    runs: list[dict[int, int]] = []  # idx global -> cluster, par run
    for _ in range(n_boot):
        idx = rng.sample(range(n), size)
        labels = cluster_fn(vecs[idx], DEFAULTS)
        runs.append({g: int(c) for g, c in zip(idx, labels)})

    aris: list[float] = []
    for a in range(len(runs)):
        for b in range(a + 1, len(runs)):
            common = sorted(set(runs[a]) & set(runs[b]))
            if len(common) < 2:
                continue
            la = [runs[a][g] for g in common]
            lb = [runs[b][g] for g in common]
            aris.append(float(adjusted_rand_score(la, lb)))
    if not aris:
        return None
    return float(np.mean(aris))


# --- Banc ----------------------------------------------------------------------
def run_bench(
    sample_questions: int,
    seed: int,
    n_boot: int,
    boot_frac: float,
    lang: str | None,
    min_comments: int,
    min_per_class: int,
    with_hdbscan: bool,
) -> dict:
    t_start = time.perf_counter()

    questions = data.load_questions(
        lang=lang, min_comments=min_comments, min_per_class=min_per_class
    )
    if not questions:
        raise SystemExit("Aucune question exploitable (vérifie le filtre lang/min).")

    rng = random.Random(seed)
    pool = list(questions)
    rng.shuffle(pool)
    sampled = pool[: min(sample_questions, len(pool))]

    approaches = ["Leiden"] + (["HDBSCAN"] if with_hdbscan else [])
    runs = {name: ApproachRuns(name) for name in approaches}

    embedder = Embedder(model_id=DEFAULT_MODEL_ID)
    embed_total_s = 0.0
    n_embeddings = 0
    per_question = []

    for q in sampled:
        t0 = time.perf_counter()
        vecs = embedder.embed(q.comments)  # L2-normalisés
        embed_total_s += time.perf_counter() - t0
        n_embeddings += q.n
        truth = q.label_ids()

        row = {
            "question_id": q.question_id,
            "question": q.question,
            "n": q.n,
            "n_favor": q.n_favor,
            "n_against": q.n_against,
        }
        for name in approaches:
            fn = APPROACHES[name]
            tc = time.perf_counter()
            membership = fn(vecs, DEFAULTS)
            secs = time.perf_counter() - tc
            m = metrics.score_against_labels(membership, truth, vecs)
            runs[name].add(m, secs)
            if n_boot > 0:
                stab = bootstrap_stability(fn, vecs, n_boot, boot_frac, seed)
                runs[name].stability.append(stab)
            row[name] = m
        per_question.append(row)

    wall = time.perf_counter() - t_start
    return {
        "meta": {
            "model_id": DEFAULT_MODEL_ID,
            "seed": seed,
            "sample_questions": len(sampled),
            "n_questions_available": len(questions),
            "lang": lang or "all",
            "min_comments": min_comments,
            "min_per_class": min_per_class,
            "n_embeddings": n_embeddings,
            "embed_seconds": round(embed_total_s, 3),
            "wall_seconds": round(wall, 3),
            "bootstrap": n_boot,
            "bootstrap_frac": boot_frac,
            "params": DEFAULTS,
            "approaches": approaches,
            "python": platform.python_version(),
        },
        "runs": runs,
        "per_question": per_question,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-questions", type=int, default=8)
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--bootstrap", type=int, default=5, help="nb de ré-échantillons (0 = off)")
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--boot-frac", type=float, default=0.8)
    ap.add_argument("--lang", default="fr", help="'fr' (défaut) ou 'all'")
    ap.add_argument("--min-comments", type=int, default=40)
    ap.add_argument("--min-per-class", type=int, default=5)
    ap.add_argument("--out", default=str(report.DEFAULT_OUT))
    args = ap.parse_args(argv)

    n_boot = 0 if args.no_bootstrap else args.bootstrap
    lang = None if args.lang == "all" else args.lang

    with_hdbscan = hdbscan_available()
    if not with_hdbscan:
        print(
            "[warn] umap-learn/hdbscan absents — le contender HDBSCAN sera ABSENT "
            "du rapport. Installe-les : uv sync --extra contender"
        )

    print(
        f"[bench] {args.sample_questions} questions, lang={args.lang}, "
        f"bootstrap={n_boot}, seed={args.seed} — embeddings e5-small (CPU)…"
    )
    results = run_bench(
        sample_questions=args.sample_questions,
        seed=args.seed,
        n_boot=n_boot,
        boot_frac=args.boot_frac,
        lang=lang,
        min_comments=args.min_comments,
        min_per_class=args.min_per_class,
        with_hdbscan=with_hdbscan,
    )
    path = report.write_report(results, args.out)
    print(f"[bench] rapport écrit : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
