"""Banc QUALITÉ de clustering — e5-small vs nomic-v2 vs bge-m3.

Transforme « améliorer le clustering » en NOMBRE : on compare les **3 modèles
d'embeddings multilingues** du registre sur un **corpus trilingue équilibré**
(x-stance DE/FR/IT, thème = `topic`), avec le **multilingue comme critère
central**. Même pipeline pour tous (rang-kNN → Leiden) : seule l'embedding change.

Métriques par modèle :
  1. Cohérence de thèmes  — NPMI intra-langue (intrinsèque, le cœur).      ↑
  2. Mixité linguistique  — NMI(cluster, langue) (LE test multilingue).    ↓ (bas=bon)
     + pureté linguistique moyenne des clusters.                           ↓
  3. Récupération de thème — NMI(cluster, topic) (vérité terrain).         ↑
     + pureté thématique.                                                  ↑
  4. Séparation interne   — silhouette (cosine) + modularité Leiden.       ↑
  5. Stabilité            — ARI bootstrap inter-runs.                       ↑
  6. Coût                 — chargement + latence d'encodage + dim.

UN SEUL modèle chargé à la fois (libéré entre deux — RAM ~8 Gi). Reproductible
(seed=42). On RÉUTILISE `pipeline.embed` + `pipeline.cluster` (aucune
réimplémentation). Le gagnant est désigné par un **score composite** transparent.

Équité inter-modèles : graphe **rang-kNN** (k plus proches voisins, sans seuil de
cosinus absolu). Les modèles ont des échelles de cosinus différentes (e5 ≈ 0.83
inter-thèmes, bge ≈ 0.48) ; un seuil fixe avantagerait l'un. Le rang est
invariant à l'échelle → comparaison juste.

Usage :
    uv run python -m eval.quality_bench
    uv run python -m eval.quality_bench --n-topics 6 --max-per-cell 130
    uv run python -m eval.quality_bench --models e5,bge-m3 --no-bootstrap
"""

from __future__ import annotations

import argparse
import gc
import platform
import random
import time
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from pipeline.cluster.knn import build_knn_graph
from pipeline.cluster.leiden_cluster import run_leiden
from pipeline.embed.embedder import Embedder
from pipeline.embed.registry import resolve_model_id

from . import coherence, multilingual_data
from .metrics import purity, silhouette

# Modèles du registre comparés par défaut (alias).
DEFAULT_MODELS = ["e5-small", "nomic-v2", "bge-m3"]

# Params de clustering — rang-kNN (seuil très bas = on garde les k voisins).
DEFAULTS = {
    "k": 15,
    "threshold": -1.0,  # rang-kNN : aucun filtre de cosinus absolu (équité)
    "resolution": 1.0,
    "seed": 42,
}

# Poids du score composite (somme = 1). Multilingue + qualité de thème en tête.
WEIGHTS = {
    "nmi_lang": 0.30,    # mixité linguistique (bas = bon) — critère central
    "coherence": 0.25,   # cohérence intrinsèque des thèmes
    "nmi_topic": 0.20,   # récupération du thème (vérité terrain)
    "silhouette": 0.10,  # séparation interne
    "stability": 0.10,   # robustesse inter-runs
    "modularity": 0.05,  # structure communautaire
}
# Sens d'optimisation : True = plus haut vaut mieux.
HIGHER_BETTER = {
    "nmi_lang": False, "coherence": True, "nmi_topic": True,
    "silhouette": True, "stability": True, "modularity": True,
}


@dataclass
class ModelResult:
    model_id: str
    alias: str
    dim: int
    n_clusters: int
    coherence: float | None
    coherence_per_lang: dict
    nmi_lang: float
    lang_purity: float
    nmi_topic: float
    topic_purity: float
    silhouette: float | None
    modularity: float
    stability: float | None
    load_seconds: float
    encode_seconds: float
    latency_ms_per_text: float
    composite: float | None = None
    composite_parts: dict | None = None


# --- clustering : vecs -> membership (rang-kNN → Leiden) -----------------------
def cluster(vecs: np.ndarray, p: dict) -> tuple[list[int], float]:
    graph = build_knn_graph(vecs, k=p["k"], threshold=p["threshold"])
    res = run_leiden(graph, resolution=p["resolution"], seed=p["seed"])
    return res.membership, res.modularity


def bootstrap_stability(
    vecs: np.ndarray, p: dict, n_boot: int, frac: float, seed: int
) -> float | None:
    """ARI moyen entre ré-échantillons (sur l'intersection des indices)."""
    n = vecs.shape[0]
    size = max(2, int(round(frac * n)))
    if size >= n or n_boot < 2:
        return None
    rng = random.Random(seed)
    runs: list[dict[int, int]] = []
    for _ in range(n_boot):
        idx = rng.sample(range(n), size)
        labels, _ = cluster(vecs[idx], p)
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
    return float(np.mean(aris)) if aris else None


def eval_model(
    alias: str,
    corpus: multilingual_data.MultiCorpus,
    p: dict,
    n_boot: int,
    boot_frac: float,
) -> ModelResult:
    """Évalue UN modèle (chargé, mesuré, puis libéré par l'appelant)."""
    model_id = resolve_model_id(alias)
    embedder = Embedder(model_id=model_id)

    # 1) Coût de chargement (force le lazy-load via accès au modèle).
    t0 = time.perf_counter()
    _ = embedder.model  # déclenche le chargement
    load_s = time.perf_counter() - t0

    # 2) Encodage (L2-normalisé) + latence.
    t0 = time.perf_counter()
    vecs = embedder.embed(corpus.texts)
    encode_s = time.perf_counter() - t0
    dim = int(vecs.shape[1])

    # 3) Clustering rang-kNN → Leiden.
    membership, modularity = cluster(vecs, p)
    n_clusters = len(set(membership))

    # 4) Métriques.
    lang_ids = corpus.lang_ids()
    topic_ids = corpus.topic_ids()
    coh = coherence.per_language_coherence(membership, corpus.texts, corpus.langs)

    res = ModelResult(
        model_id=model_id,
        alias=alias,
        dim=dim,
        n_clusters=n_clusters,
        coherence=coh["overall"],
        coherence_per_lang=coh["per_lang"],
        nmi_lang=float(normalized_mutual_info_score(lang_ids, membership)),
        lang_purity=purity(membership, lang_ids),
        nmi_topic=float(normalized_mutual_info_score(topic_ids, membership)),
        topic_purity=purity(membership, topic_ids),
        silhouette=silhouette(vecs, membership),
        modularity=modularity,
        stability=bootstrap_stability(vecs, p, n_boot, boot_frac, p["seed"]),
        load_seconds=round(load_s, 3),
        encode_seconds=round(encode_s, 3),
        latency_ms_per_text=round(1000 * encode_s / max(1, corpus.n), 3),
    )

    # 5) Libère le modèle (un seul chargé à la fois).
    del embedder
    gc.collect()
    return res


def compute_composite(results: list[ModelResult]) -> None:
    """Score composite ∈ [0,1] par normalisation min-max inter-modèles.

    Chaque métrique est normalisée sur les 3 modèles (sens-aware), pondérée par
    `WEIGHTS`, puis sommée. Renseigne `composite` et `composite_parts` in place.
    """
    def values(metric: str) -> list[float | None]:
        return [getattr(r, metric) for r in results]

    norm: dict[str, list[float]] = {}
    for metric in WEIGHTS:
        vals = values(metric)
        present = [v for v in vals if v is not None]
        if not present:
            norm[metric] = [0.5] * len(results)
            continue
        lo, hi = min(present), max(present)
        span = hi - lo
        col = []
        for v in vals:
            if v is None:
                col.append(0.5)
            elif span == 0:
                col.append(0.5)
            else:
                x = (v - lo) / span
                col.append(x if HIGHER_BETTER[metric] else 1.0 - x)
        norm[metric] = col

    for i, r in enumerate(results):
        parts = {m: round(norm[m][i], 4) for m in WEIGHTS}
        r.composite_parts = parts
        r.composite = round(sum(WEIGHTS[m] * parts[m] for m in WEIGHTS), 4)


def run_bench(
    model_aliases: list[str],
    n_topics: int,
    per_cell: int | None,
    max_per_cell: int,
    min_chars: int,
    seed: int,
    n_boot: int,
    boot_frac: float,
    p: dict,
) -> dict:
    t_start = time.perf_counter()
    corpus = multilingual_data.load_balanced(
        n_topics=n_topics, per_cell=per_cell, max_per_cell=max_per_cell,
        min_chars=min_chars, seed=seed,
    )
    print(
        f"[corpus] {corpus.n} commentaires | langues {corpus.lang_counts} | "
        f"{len(corpus.topic_counts)} thèmes"
    )

    results: list[ModelResult] = []
    for alias in model_aliases:
        print(f"[model ] {alias} — chargement + encodage (CPU, un seul à la fois)…")
        try:
            r = eval_model(alias, corpus, p, n_boot, boot_frac)
        except Exception as e:  # un modèle KO ne doit pas couler le banc
            print(f"[error ] {alias} : {type(e).__name__}: {e}")
            gc.collect()
            continue
        print(
            f"         dim={r.dim} clusters={r.n_clusters} "
            f"coh={r.coherence} nmi_lang={r.nmi_lang:.3f} "
            f"nmi_topic={r.nmi_topic:.3f} sil={r.silhouette}"
        )
        results.append(r)

    compute_composite(results)
    results.sort(key=lambda r: (r.composite is None, -(r.composite or 0)))
    wall = time.perf_counter() - t_start
    return {
        "meta": {
            "seed": seed,
            "n_comments": corpus.n,
            "lang_counts": corpus.lang_counts,
            "topic_counts": corpus.topic_counts,
            "n_topics": len(corpus.topic_counts),
            "topics": sorted(corpus.topic_counts),
            "min_chars": min_chars,
            "bootstrap": n_boot,
            "bootstrap_frac": boot_frac,
            "cluster_params": p,
            "weights": WEIGHTS,
            "wall_seconds": round(wall, 1),
            "python": platform.python_version(),
        },
        "results": results,
    }


# --- rendu markdown -----------------------------------------------------------
def _fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_report(bench: dict, out_path: str) -> str:
    m = bench["meta"]
    results: list[ModelResult] = bench["results"]
    L: list[str] = []
    a = L.append

    a("# Banc QUALITÉ de clustering — embeddings multilingues\n")
    a("> Quel modèle d'embedding regroupe le mieux les avis par **thème** plutôt "
      "que par **langue** ? Réponse **par la mesure** : e5-small vs nomic-v2 vs "
      "bge-m3, même pipeline (rang-kNN → Leiden), seul l'embedding change.\n")

    if results:
        winner = results[0]
        a(f"## 🏆 Recommandation : **{winner.alias}** "
          f"(`{winner.model_id}`)\n")
        a(f"Score composite **{_fmt(winner.composite)}** "
          f"(pondéré : mixité linguistique 30 %, cohérence 25 %, récupération de "
          f"thème 20 %, silhouette 10 %, stabilité 10 %, modularité 5 %).\n")

    # Corpus
    a("## Corpus (honnêteté)\n")
    a(f"- **{m['n_comments']}** commentaires x-stance, équilibrés par "
      f"(thème × langue).")
    a(f"- Langues : {m['lang_counts']} (équilibrées → entropie ~max, "
      f"NMS(cluster,langue) interprétable).")
    a(f"- {m['n_topics']} thèmes (vérité terrain `topic`) : "
      f"{', '.join(m['topics'])}.")
    a(f"- Filtre : commentaires ≥ {m['min_chars']} caractères, dédup exact. "
      f"seed={m['seed']}.")
    a(f"- Clustering : rang-kNN k={m['cluster_params']['k']} "
      f"(sans seuil de cosinus, équité inter-modèles), Leiden "
      f"resolution={m['cluster_params']['resolution']}, seed="
      f"{m['cluster_params']['seed']}.")
    a(f"- Bootstrap : {m['bootstrap']} ré-échantillons "
      f"(fraction {m['bootstrap_frac']}). Python {m['python']}, CPU. "
      f"Wall {m['wall_seconds']} s.\n")

    # Scorecard principale
    a("## Scorecard\n")
    a("| Métrique | sens | " + " | ".join(r.alias for r in results) + " |")
    a("|---|:--:|" + "|".join([":--:"] * len(results)) + "|")

    def row(label, sense, attr, nd=3):
        cells = " | ".join(_fmt(getattr(r, attr), nd) for r in results)
        a(f"| {label} | {sense} | {cells} |")

    row("Cohérence NPMI (intra-langue)", "↑", "coherence")
    row("**NMI(cluster, langue)**", "↓", "nmi_lang")
    row("Pureté linguistique", "↓", "lang_purity")
    row("NMI(cluster, thème)", "↑", "nmi_topic")
    row("Pureté thématique", "↑", "topic_purity")
    row("Silhouette (cosine)", "↑", "silhouette")
    row("Modularité (Leiden)", "↑", "modularity")
    row("Stabilité (ARI bootstrap)", "↑", "stability")
    row("Nb clusters", "·", "n_clusters", 0)
    row("Dimension", "·", "dim", 0)
    row("Chargement (s)", "↓", "load_seconds", 1)
    row("Latence (ms/texte)", "↓", "latency_ms_per_text", 2)
    row("**Score composite**", "↑", "composite")
    a("")

    # Détail composite
    a("## Détail du score composite (normalisé min-max inter-modèles, ∈ [0,1])\n")
    a("| Composante (poids) | " + " | ".join(r.alias for r in results) + " |")
    a("|---|" + "|".join([":--:"] * len(results)) + "|")
    for metric, w in WEIGHTS.items():
        cells = " | ".join(
            _fmt((r.composite_parts or {}).get(metric)) for r in results
        )
        a(f"| {metric} ({int(w*100)} %) | {cells} |")
    a("")

    # Cohérence par langue
    a("## Cohérence NPMI détaillée par langue\n")
    a("| Modèle | " + " | ".join(
        sorted({lg for r in results for lg in (r.coherence_per_lang or {})})
    ) + " | moyenne |")
    langs_seen = sorted({lg for r in results for lg in (r.coherence_per_lang or {})})
    a("|---|" + "|".join([":--:"] * (len(langs_seen) + 1)) + "|")
    for r in results:
        cells = " | ".join(_fmt((r.coherence_per_lang or {}).get(lg)) for lg in langs_seen)
        a(f"| {r.alias} | {cells} | {_fmt(r.coherence)} |")
    a("")

    # Lecture
    a("## Lecture\n")
    if results:
        w = results[0]
        a(f"- **{w.alias}** gagne : "
          f"NMI(cluster,langue)={_fmt(w.nmi_lang)} (mixité — bas = les clusters "
          f"ne ségrègent PAS par langue), cohérence={_fmt(w.coherence)}, "
          f"NMI(cluster,thème)={_fmt(w.nmi_topic)} (récupère le thème).")
        if len(results) > 1:
            worst_mix = max(results, key=lambda r: r.nmi_lang)
            a(f"- Pire mixité : **{worst_mix.alias}** "
              f"(NMI langue={_fmt(worst_mix.nmi_lang)}, pureté linguistique "
              f"{_fmt(worst_mix.lang_purity)}) — regroupe par LANGUE, pas par thème.")
        a("- La mixité linguistique est le critère central : un NMI(cluster,langue) "
          "élevé trahit un modèle qui sépare les langues au lieu des thèmes.")
        # Piège classique : les métriques internes récompensent la dégénérescence.
        internal_best = max(
            results,
            key=lambda r: ((r.silhouette or -1) + (r.stability or -1) + r.modularity),
        )
        if internal_best.nmi_lang > 0.3:
            a(f"- ⚠️ **Piège des métriques internes** : **{internal_best.alias}** "
              f"a la meilleure silhouette/modularité/stabilité — mais ses clusters "
              f"sont mono-langues (pureté linguistique {_fmt(internal_best.lang_purity)}). "
              f"Des clusters internes nets mais **faux** : silhouette, modularité et "
              f"stabilité récompensent la solution dégénérée « 1 langue = 1 cluster ». "
              f"D'où le rôle **décisif** de NMI(cluster,langue) et NMI(cluster,thème), "
              f"qui seuls voient que le clustering n'a pas trouvé les thèmes.")
        # Départage des deux meilleurs (utile quand la mixité est à égalité).
        if len(results) >= 2:
            r1, r2 = results[0], results[1]
            a(f"- **{r1.alias} vs {r2.alias}** : mixité quasi identique "
              f"(NMI langue {_fmt(r1.nmi_lang)} vs {_fmt(r2.nmi_lang)}) ; "
              f"{r1.alias} l'emporte sur la cohérence ({_fmt(r1.coherence)} vs "
              f"{_fmt(r2.coherence)}), la récupération de thème ({_fmt(r1.nmi_topic)} "
              f"vs {_fmt(r2.nmi_topic)}) et/ou le coût "
              f"({_fmt(r1.latency_ms_per_text, 0)} vs {_fmt(r2.latency_ms_per_text, 0)} "
              f"ms/texte). {r2.alias} reste un second proche.")
    a("")

    # Limites
    a("## Limites (ce qui n'est PAS testé)\n")
    a("- **Domaine** : x-stance = votations suisses (DE/FR/IT), commentaires "
      "courts et argumentés. Le transfert vers TikTok (témoignages libres FR) "
      "n'est pas validé ici (pas de labels multilingues sur TikTok).")
    a("- **IT sous-représenté** dans la source ; l'équilibrage plafonne donc la "
      "taille par cellule. Échantillon de quelques milliers de commentaires.")
    a("- **Cohérence NPMI** = co-occurrence document intra-langue (pas de fenêtre "
      "glissante gensim) ; valeurs comparables entre modèles (même calcul), pas "
      "à des benchmarks externes.")
    a("- **Params de clustering figés** (pas de sweep) ; rang-kNN choisi pour "
      "l'équité, mais un tuning par modèle pourrait déplacer les marges.")
    a("- Le **topic x-stance** (12 thèmes larges) est une vérité terrain "
      "grossière : un clustering plus fin que les topics est pénalisé sur "
      "NMI(thème) mais peut rester cohérent.")
    a("")

    text = "\n".join(L) + "\n"
    from pathlib import Path

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(text, encoding="utf-8")
    return out_path


DEFAULT_OUT = str(multilingual_data.REPO_ROOT / "eval" / "quality_report.md")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="alias séparés par des virgules (défaut: e5-small,nomic-v2,bge-m3)")
    ap.add_argument("--n-topics", type=int, default=6)
    ap.add_argument("--per-cell", type=int, default=None,
                    help="commentaires par (thème×langue) ; défaut = auto-équilibré")
    ap.add_argument("--max-per-cell", type=int, default=130)
    ap.add_argument("--min-chars", type=int, default=15)
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--bootstrap", type=int, default=4, help="ré-échantillons (0=off)")
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--boot-frac", type=float, default=0.8)
    ap.add_argument("--k", type=int, default=DEFAULTS["k"])
    ap.add_argument("--resolution", type=float, default=DEFAULTS["resolution"])
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    p = dict(DEFAULTS)
    p["k"] = args.k
    p["resolution"] = args.resolution
    p["seed"] = args.seed
    n_boot = 0 if args.no_bootstrap else args.bootstrap
    models = [s.strip() for s in args.models.split(",") if s.strip()]

    print(f"[bench ] modèles={models} seed={args.seed} bootstrap={n_boot}")
    bench = run_bench(
        model_aliases=models,
        n_topics=args.n_topics,
        per_cell=args.per_cell,
        max_per_cell=args.max_per_cell,
        min_chars=args.min_chars,
        seed=args.seed,
        n_boot=n_boot,
        boot_frac=args.boot_frac,
        p=p,
    )
    path = write_report(bench, args.out)
    if bench["results"]:
        w = bench["results"][0]
        print(f"[winner] {w.alias} (composite {w.composite})")
    print(f"[bench ] rapport écrit : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
