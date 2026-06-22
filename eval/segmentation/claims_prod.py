"""Pipeline CLAIMS en PROD : vraie consultation → claims → embed → thèmes ÉMERGENTS.

    AGORA_OLLAMA_URL="$(cat var/MAC_LOCAL_OLLAMA)" \
    uv run --extra contender --extra embed-contender \
        python -m eval.segmentation.claims_prod
        [--input backend/cache/tiktok/ideas.jsonl]
        [--model ministral-3:latest] [--embedder nomic-v2]
        [--min-chars 12] [--resolution 1.0] [--limit N]
        [--out eval/segmentation/claims_prod_report.md]

Différence avec `claims_pipeline.py` (le banc) : ici il n'y a **AUCUN gold, AUCUNE
taxonomie**. C'est le cas réel — on ne sait PAS d'avance quels thèmes existent. Le
pipeline les fait ÉMERGER du bas (style TalkToTheCity) :

  vraie consultation (avis citoyens bruts) → ministral extrait les CLAIMS atomiques
  → embed nomic-v2 → k-NN+Leiden (défauts DÉRIVÉS) → c-TF-IDF nomme les clusters
  → carte des thèmes : label, taille, poids social, consensus/diversité, claims
    représentatives (les plus proches du centroïde).

Le poids social (`weight`, cumulé lors de la dédup d'ingest) est propagé : un claim
hérite du poids de son avis ; le poids d'un thème = somme. On trie les thèmes par
poids (× consensus) — les préoccupations les plus partagées d'abord.

Corpus par défaut : la VRAIE consultation TikTok (open data Assemblée nationale,
`backend/cache/tiktok/ideas.jsonl`, 1 604 réponses libres FR sur le mal-être / le
harcèlement). Souverain : ministral tourne sur le Mac, la donnée ne sort pas.

ÉCRIT UNIQUEMENT dans `eval/segmentation/` (claims_prod_report.md, claims_prod.json,
caches réutilisés `.cache/ollama/`, `.cache/`).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval.segmentation.claims_pipeline import (
    _normalize_rows,
    claim_prompt,
    extract_claims,
)
from eval.segmentation.small_models import (
    OLLAMA_BASE,
    OllamaStats,
    ollama_chat,
    ollama_warmup,
)
from pipeline.cluster.adaptive import derive_defaults
from pipeline.cluster.knn import build_knn_graph
from pipeline.cluster.leiden_cluster import run_leiden
from pipeline.cluster.naming import derive_corpus_stopwords, name_clusters
from pipeline.cluster.scoring import score_cluster

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_INPUT = REPO / "backend" / "cache" / "tiktok" / "ideas.jsonl"
DEFAULT_REPORT = HERE / "claims_prod_report.md"
DEFAULT_SCORES = HERE / "claims_prod.json"


# --------------------------------------------------------------------------- #
@dataclass
class Avis:
    id: str
    text: str
    weight: float


def load_real_avis(path: Path, min_chars: int, limit: int | None) -> list[Avis]:
    """Charge les avis canoniques (ideas.jsonl d'ingest). Filtre les non-réponses."""
    out: list[Avis] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        text = (r.get("text") or r.get("text_clean") or "").strip()
        if len(text) < min_chars:
            continue
        out.append(Avis(id=str(r.get("id") or f"idea-{len(out)}"),
                        text=text, weight=float(r.get("weight", 1.0) or 1.0)))
        if limit and len(out) >= limit:
            break
    return out


def run_extraction(avis: list[Avis], model: str, stats: OllamaStats,
                   think: bool | None) -> dict[str, list[str]]:
    claims: dict[str, list[str]] = {}
    n = len(avis)
    for i, a in enumerate(avis):
        raw = ollama_chat(claim_prompt(a.text), model=model, think=think, stats=stats)
        cl = extract_claims(raw) or [a.text]
        claims[a.id] = cl
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{model}] claims {i + 1}/{n}")
    return claims


# --------------------------------------------------------------------------- #
@dataclass
class Theme:
    cluster_id: int
    label: str
    keywords: list[str]
    n_claims: int
    n_avis: int
    weight: float
    consensus: float
    diversity: float
    examples: list[str]            # claims représentatives (proches du centroïde)


def build_themes(membership: list[int], claim_vecs: np.ndarray, claim_texts: list[str],
                 claim_weight: np.ndarray, claim_owner: list[int], dup_threshold: float,
                 names: dict, n_examples: int = 4) -> list[Theme]:
    by_cluster: dict[int, list[int]] = {}
    for i, cid in enumerate(membership):
        by_cluster.setdefault(cid, []).append(i)

    themes: list[Theme] = []
    for cid, idx in by_cluster.items():
        sc = score_cluster(idx, claim_vecs, claim_weight, dup_threshold=dup_threshold)
        cent = np.asarray(sc.centroid, dtype=np.float64)
        sims = claim_vecs[idx] @ cent
        order = np.argsort(-sims)
        # exemples = claims les plus centrales, en évitant les quasi-doublons littéraux
        ex: list[str] = []
        for j in order:
            t = claim_texts[idx[j]]
            if any(t.lower() == e.lower() for e in ex):
                continue
            ex.append(t)
            if len(ex) >= n_examples:
                break
        themes.append(Theme(
            cluster_id=cid, label=names.get(cid, {}).get("label", f"thème {cid}"),
            keywords=names.get(cid, {}).get("keywords", []),
            n_claims=len(idx), n_avis=len({claim_owner[i] for i in idx}),
            weight=round(sc.weight_sum, 1), consensus=round(sc.consensus, 3),
            diversity=round(sc.diversity, 3), examples=ex))
    # tri : poids social × consensus (préoccupations partagées ET cohérentes d'abord)
    themes.sort(key=lambda t: -(t.weight * max(t.consensus, 0.0)))
    return themes


# --------------------------------------------------------------------------- #
def _md_table(rows, cols):
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows)
    return "\n".join([head, sep, body])


def build_report(input_path: Path, n_avis: int, n_claims: int, model: str, embedder: str,
                 defaults, resolution: float, n_clusters: int, modularity: float,
                 themes: list[Theme], stats: OllamaStats, think: bool | None,
                 limited: bool) -> str:
    ms = 1000.0 * stats.cold_seconds / n_avis if n_avis else 0.0
    cpa = n_claims / n_avis if n_avis else 0.0
    L = []
    L.append("# Pipeline CLAIMS en PROD — thèmes ÉMERGENTS d'une vraie consultation\n")
    L.append(f"*Source : `{input_path.name}` — **{n_avis} avis citoyens réels** "
             f"(consultation TikTok, open data Assemblée nationale ; réponses libres FR sur "
             f"le mal-être / le harcèlement). **AUCUN gold, AUCUNE taxonomie** : les thèmes "
             f"émergent du bas. Extraction : `{model}` (Mac, souverain), pensée "
             f"{'coupée' if think is False else 'native'}. Embed : `{embedder}`. "
             f"Clustering k-NN+Leiden, défauts DÉRIVÉS (k={defaults.k}, "
             f"seuil={defaults.threshold:.3f}), résolution {resolution} → "
             f"**{n_clusters} thèmes**, modularité {modularity:.3f}.*\n")
    if limited:
        L.append("⚠️ **Run PARTIEL** (`--limit`) — aperçu, pas la consultation entière.\n")
    L.append(f"**{n_claims} claims atomiques** extraites ({cpa:.2f}/avis) puis clusterisées. "
             f"Chaque thème ci-dessous est une préoccupation DÉCOUVERTE — personne ne l'a "
             f"écrite dans une liste. Tri par **poids social × consensus** (les "
             f"préoccupations les plus partagées ET cohérentes d'abord). Le poids = somme "
             f"des poids d'avis (near-dups d'ingest cumulés sur leur représentant).\n")

    # Carte synthétique
    L.append("## Carte des thèmes émergents\n")
    rows = [{"#": i + 1, "thème (c-TF-IDF)": t.label, "claims": t.n_claims,
             "avis": t.n_avis, "poids": t.weight, "consensus": t.consensus,
             "diversité": t.diversity} for i, t in enumerate(themes)]
    L.append(_md_table(rows, ["#", "thème (c-TF-IDF)", "claims", "avis", "poids",
                              "consensus", "diversité"]) + "\n")
    L.append("*consensus = cosinus moyen intra-thème (haut = même intention) ; diversité = "
             "1 − densité de quasi-doublons (haut = mêmes idées, formulations variées).*\n")

    # Détail par thème
    L.append("## Détail — claims représentatives par thème\n")
    for i, t in enumerate(themes):
        kw = ", ".join(t.keywords[:6])
        L.append(f"### {i + 1}. {t.label}\n")
        L.append(f"*{t.n_claims} claims · {t.n_avis} avis · poids {t.weight} · "
                 f"consensus {t.consensus} · diversité {t.diversity}*  \n"
                 f"mots-clés : _{kw}_\n")
        for ex in t.examples:
            L.append(f"- {ex}")
        L.append("")

    # Coût
    L.append("## Coût & souveraineté\n")
    L.append(
        f"- **Extraction ministral (Mac)** : {stats.calls} appels réels + {stats.cache_hits} "
        f"cache, ~{stats.cold_seconds:.0f}s (~{ms:.0f} ms/avis), {stats.eval_tokens:,} "
        f"tokens, {stats.errors} erreurs. Embed + clustering : local, négligeable.\n"
        f"- **Souverain** : la donnée citoyenne ne quitte jamais le réseau privé "
        f"(`{OLLAMA_BASE}`, Tailscale). Coût marginal ~0 € (vs ~2-4 €/consultation en API).\n")

    L.append("## Lecture\n")
    L.append(
        f"- **Les thèmes ont émergé sans aucune taxonomie.** Sur le banc gold (8 thèmes "
        f"connus), claims→cluster reconstruisait une bijection 8↔8 à micro-F1 0.784 ; ICI, "
        f"sans gold, il produit directement la carte des {n_clusters} préoccupations de la "
        f"consultation — c'est le mode d'emploi réel.\n")
    L.append(
        "- **Granularité réglable** : la résolution Leiden fixe le nombre de thèmes "
        "(basse = quelques grands thèmes, haute = sous-facettes fines). Aucun nombre n'est "
        "imposé : on choisit la lentille selon l'usage (synthèse vs exploration fine).\n")
    L.append(
        "- **Tri par poids × consensus** : remonte ce que BEAUCOUP de citoyens disent de "
        "façon COHÉRENTE — le signal d'opinion partagée, pas l'anecdote isolée. La "
        "diversité distingue « 100 personnes, 100 formulations » d'un copier-coller viral.\n")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Claims pipeline en prod (consultation réelle).")
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--model", default="ministral-3:latest")
    ap.add_argument("--embedder", default="nomic-v2")
    ap.add_argument("--min-chars", type=int, default=12)
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(DEFAULT_REPORT))
    ap.add_argument("--scores-out", default=str(DEFAULT_SCORES))
    args = ap.parse_args()

    input_path = Path(args.input)
    avis = load_real_avis(input_path, args.min_chars, args.limit)
    n = len(avis)
    print(f"prod: {input_path.name} — {n} avis réels (min_chars={args.min_chars})")

    # 1) extraction claims (ministral, Mac, cache)
    print(f"ÉTAPE 1 — extraction claims via {args.model} @ {OLLAMA_BASE} …")
    stats = OllamaStats()
    ok, think = ollama_warmup(args.model)
    if not ok:
        raise SystemExit(f"Modèle {args.model} injoignable sur {OLLAMA_BASE}.")
    t0 = time.monotonic()
    claims = run_extraction(avis, args.model, stats, think)
    print(f"  extraction: {stats.cold_seconds:.0f}s cumulés, {stats.calls} appels, "
          f"{stats.cache_hits} cache, {stats.errors} err ({time.monotonic() - t0:.0f}s mur)")

    claim_texts: list[str] = []
    claim_owner: list[int] = []
    claim_weight_l: list[float] = []
    for ai, a in enumerate(avis):
        for ctext in claims[a.id]:
            claim_owner.append(ai)
            claim_texts.append(ctext)
            claim_weight_l.append(a.weight)
    n_claims = len(claim_texts)
    print(f"  {n_claims} claims ({n_claims / n:.2f}/avis)")

    # 2) embed
    print(f"ÉTAPE 2 — embeddings {args.embedder} …")
    from eval.segmentation.embeddings import embed_docs

    claim_vecs = _normalize_rows(embed_docs(claim_texts, model_id=args.embedder).astype(np.float64))
    claim_weight = np.asarray(claim_weight_l, dtype=np.float64)

    # 3) clustering émergent (défauts dérivés)
    print("ÉTAPE 3 — clustering émergent …")
    defaults = derive_defaults(claim_vecs.astype(np.float32))
    print(f"  défauts dérivés: k={defaults.k}, seuil={defaults.threshold:.3f}, "
          f"dup={defaults.dup_threshold:.3f}")
    graph = build_knn_graph(claim_vecs, k=defaults.k, threshold=defaults.threshold)
    res = run_leiden(graph, resolution=args.resolution, seed=args.seed)
    print(f"  {res.n_clusters} thèmes émergents, modularité {res.modularity:.3f}")

    # 4) naming c-TF-IDF + carte des thèmes
    print("ÉTAPE 4 — naming + scoring …")
    by_cluster: dict[int, list[int]] = {}
    for i, cid in enumerate(res.membership):
        by_cluster.setdefault(cid, []).append(i)
    corpus_stop, _ = derive_corpus_stopwords(claim_texts)
    cluster_docs = {cid: [claim_texts[i] for i in idx] for cid, idx in by_cluster.items()}
    names = name_clusters(cluster_docs, corpus_stopwords=corpus_stop)
    themes = build_themes(res.membership, claim_vecs, claim_texts, claim_weight,
                          claim_owner, defaults.dup_threshold, names)

    limited = bool(args.limit)
    report = build_report(input_path, n, n_claims, args.model, args.embedder, defaults,
                          args.resolution, res.n_clusters, res.modularity, themes, stats,
                          think, limited)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"✓ {args.out}")

    Path(args.scores_out).write_text(json.dumps({
        "input": input_path.name, "n_avis": n, "n_claims": n_claims,
        "claims_per_avis": round(n_claims / n, 3) if n else 0.0,
        "model": args.model, "embedder": args.embedder, "limited": limited,
        "resolution": args.resolution, "seed": args.seed,
        "derived_defaults": {"k": defaults.k, "threshold": round(defaults.threshold, 4),
                             "dup_threshold": round(defaults.dup_threshold, 4)},
        "n_clusters": res.n_clusters, "modularity": res.modularity,
        "themes": [{"cluster_id": t.cluster_id, "label": t.label, "keywords": t.keywords,
                    "n_claims": t.n_claims, "n_avis": t.n_avis, "weight": t.weight,
                    "consensus": t.consensus, "diversity": t.diversity,
                    "examples": t.examples} for t in themes],
        "cost": {"ollama_calls": stats.calls, "cache_hits": stats.cache_hits,
                 "errors": stats.errors, "cold_seconds": round(stats.cold_seconds, 2),
                 "ms_per_avis": round(1000.0 * stats.cold_seconds / n, 1) if n else 0.0,
                 "eval_tokens": stats.eval_tokens, "endpoint": OLLAMA_BASE},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ {args.scores_out}")


if __name__ == "__main__":
    main()
