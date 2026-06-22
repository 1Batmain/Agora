"""Pipeline CLAIMS (style TalkToTheCity) : avis → claims atomiques → embed → clustering ÉMERGENT.

    AGORA_OLLAMA_URL="$(cat var/MAC_LOCAL_OLLAMA)" \
    uv run --extra contender python -m eval.segmentation.claims_pipeline
        [--gold eval/segmentation/gold_large.json]
        [--model ministral-3:latest] [--embedder nomic-v2]
        [--resolution 1.0] [--sweep 0.5,0.8,1.0,1.4,2.0]
        [--limit N]
        [--out eval/segmentation/claims_report.md]

IDÉE (TalkToTheCity, ascendant) — PAS de segmentation par frontières et AUCUNE
taxonomie imposée. Pour chaque avis, un LLM LOCAL (`ministral-3` sur le Mac de Bob,
souverain) extrait ses **CLAIMS atomiques** en vocabulaire LIBRE (idées autoportantes).
On **embed** chaque claim (nomic-v2 `search_document:`) et on **clusterise** les claims
de TOUT le corpus (k-NN + Leiden, défauts DÉRIVÉS des données). Les **thèmes ÉMERGENT**
du bas : un cluster = un thème découvert. Nouveau = nouvelle claim = nouveau cluster →
ouvert, robuste à la nouveauté (rien n'est bridé par les 1ers avis ni par une liste fermée).

LE CŒUR — sur `gold_large` (305 avis, 8 thèmes), le clustering ascendant **reconstruit-il
les 8 thèmes SANS les voir** ?

1. **Étiquette gold PAR CLAIM** (pour mesurer, jamais pour clusteriser) : chaque claim
   hérite du thème du SEGMENT gold de son avis source le plus proche (cosine) — pour un
   avis mono, c'est son thème unique. Donne UN label gold par claim, non ambigu.
2. **Mapping cluster → thème** : vote majoritaire des étiquettes-claim de ses claims
   (comme TalkToTheCity nomme ses clusters a posteriori). Le clustering, lui, est
   100% non supervisé.
3. **Récupération multi-label PAR AVIS** : thèmes prédits d'un avis = {thème dominant des
   clusters où tombent ses claims} → P/R/F1 multi-label vs gold (comparable à Mistral 0.928
   et au classifieur 0.939).
4. **Qualité clustering vs gold** : homogénéité / complétude / V-mesure (label-claim vs
   appartenance) ; nb de clusters émergents vs 8.
5. **NOUVEAUTÉ** : clusters dont le centroïde est sémantiquement LOIN des 8 centroïdes-
   thèmes du gold (cosine max < seuil dérivé) = idées candidates que la taxo fermée ratait.
   Liste qualitative — c'est la valeur ajoutée de l'ouverture.
6. **COÛT** : temps ministral total sur le Mac, ms/avis à chaud, nb d'appels ; vs ~2-4 €
   par grosse consultation en API (la donnée ne sort pas du réseau privé Tailscale).

Honnêteté : le clustering ne voit AUCUN thème ; seuls le mapping cluster→thème et la
V-mesure utilisent le gold (évaluation standard « cluster-then-label », cf. accuracy de
clustering). Latence Ollama sur Mac partagé, 1 avis/appel → indicative. Cache disque
réutilisé (`.cache/ollama/`, `.cache/`) → relances gratuites.

ÉCRIT UNIQUEMENT dans `eval/segmentation/` (claims_report.md, claims_scores.json, caches).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from eval.segmentation.llm_seg import parse_json_object, prepare, score_themes
from eval.segmentation.seg_bench import load_gold
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

HERE = Path(__file__).resolve().parent
DEFAULT_GOLD = HERE / "gold_large.json"
DEFAULT_REPORT = HERE / "claims_report.md"
DEFAULT_SCORES = HERE / "claims_scores.json"

# Repères des autres approches (mêmes métriques, même jeu) — cf. small_models_report.md.
MISTRAL_F1 = 0.928       # Mistral-small, choix FERMÉ sur les 8 thèmes (API, données sortent)
CLF_F1 = 0.939           # classifieur MLP/nomic ENTRAÎNÉ sur les 8 thèmes (local)


# --------------------------------------------------------------------------- #
# Chargement avis + segments gold (pour l'étiquette gold par claim)
# --------------------------------------------------------------------------- #
@dataclass
class Avis:
    id: str
    type: str                       # "mono" | "multi"
    text: str
    themes: set[str]                # ensemble des thèmes gold de l'avis
    seg_texts: list[str]            # textes des segments gold
    seg_themes: list[str]           # thème de chaque segment (aligné à seg_texts)


def load_avis(gold_path: Path) -> tuple[list[Avis], dict[str, str]]:
    """Avis + segments gold bruts. La taxonomie n'est lue QUE pour l'évaluation."""
    data = json.loads(gold_path.read_text(encoding="utf-8"))
    taxonomy: dict[str, str] = data.get("taxonomy", {})
    out: list[Avis] = []
    for it in data["items"]:
        if it["type"] == "mono":
            theme = it.get("theme", "?")
            out.append(Avis(it["id"], "mono", it["text"], {theme}, [it["text"]], [theme]))
        else:
            segs = it["segments"]
            st = [s["text"] for s in segs]
            sth = [s["theme"] for s in segs]
            out.append(Avis(it["id"], "multi", data.get("join", " ").join(st),
                            {t for t in sth if t and t != "?"}, st, sth))
    return out, taxonomy


# --------------------------------------------------------------------------- #
# Étape 1 — extraction des CLAIMS atomiques (ministral, prompt OUVERT, cache)
# --------------------------------------------------------------------------- #
CLAIM_SYS = (
    "Tu es un analyste d'avis citoyens, multilingue (FR, DE, IT, EN…). On te donne UN "
    "avis. Extrais ses IDÉES distinctes : chaque préoccupation, opinion ou proposition "
    "autoportante, reformulée en UNE assertion atomique, concise et compréhensible HORS "
    "contexte. Une idée = une assertion. N'invente rien, n'ajoute aucune catégorie ni "
    "étiquette ; reste fidèle à l'avis. Si l'avis ne porte qu'une idée, renvoie une seule "
    'assertion. Réponds STRICTEMENT en JSON : {"claims": ["assertion 1", "assertion 2", …]}.'
)


def claim_prompt(text: str) -> list[dict]:
    return [{"role": "system", "content": CLAIM_SYS},
            {"role": "user", "content": "Avis :\n" + text}]


def extract_claims(raw: str | None) -> list[str]:
    """Parse la réponse ministral → liste de claims (tolère clé alternative)."""
    obj = parse_json_object(raw or "")
    if obj is None:
        return []
    val = obj.get("claims")
    if not isinstance(val, list):                 # le petit modèle a renommé la clé
        for v in obj.values():
            if isinstance(v, list):
                val = v
                break
    if not isinstance(val, list):
        return []
    return [str(x).strip() for x in val if str(x).strip()]


def run_extraction(avis: list[Avis], model: str, stats: OllamaStats,
                   think: bool | None) -> dict[str, list[str]]:
    claims: dict[str, list[str]] = {}
    n = len(avis)
    for i, a in enumerate(avis):
        raw = ollama_chat(claim_prompt(a.text), model=model, think=think, stats=stats)
        cl = extract_claims(raw)
        if not cl:                                # repli : l'avis entier = 1 claim
            cl = [a.text]
        claims[a.id] = cl
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{model}] claims {i + 1}/{n} — '{a.id}': {len(cl)} claims")
    return claims


# --------------------------------------------------------------------------- #
# Étape 2 — embed claims + segments gold (nomic-v2, espace PROD)
# --------------------------------------------------------------------------- #
def _normalize_rows(M: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(M, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return M / nrm


# --------------------------------------------------------------------------- #
# Étape 4a — étiquette gold PAR CLAIM (segment gold le plus proche)
# --------------------------------------------------------------------------- #
def claim_gold_labels(claim_vecs: np.ndarray, claim_owner: list[int], avis: list[Avis],
                      seg_vecs_by_avis: dict[int, np.ndarray]) -> list[str]:
    """Pour chaque claim, thème du segment gold de son avis le plus proche (cosine).

    Avis mono → thème unique (trivial). Avis multi → on assigne le thème du segment
    sémantiquement le plus proche. C'est la SEULE désambiguïsation multi→mono, et elle
    ne sert qu'à l'évaluation (donner un label gold par claim) — jamais au clustering.
    """
    labels: list[str] = []
    for ci, ai in enumerate(claim_owner):
        a = avis[ai]
        if len(a.seg_themes) == 1:
            labels.append(a.seg_themes[0])
            continue
        sv = seg_vecs_by_avis[ai]                 # [n_seg, d] normalisé
        sims = sv @ claim_vecs[ci]
        labels.append(a.seg_themes[int(np.argmax(sims))])
    return labels


# --------------------------------------------------------------------------- #
# Étape 3 — clustering émergent (défauts dérivés, Leiden) + mapping
# --------------------------------------------------------------------------- #
@dataclass
class ClusterRun:
    resolution: float
    membership: list[int]
    n_clusters: int
    modularity: float
    cluster_theme: dict[int, str]    # cluster_id → thème gold dominant
    cluster_purity: dict[int, float]
    homogeneity: float
    completeness: float
    v_measure: float
    micro_f1: float
    macro_f1: float
    exact_set: float
    score: object = None             # ThemeScore complet (résolution principale)


def map_clusters_to_themes(membership: list[int], claim_gold: list[str]
                           ) -> tuple[dict[int, str], dict[int, float]]:
    """Cluster → thème dominant (vote majoritaire des étiquettes-claim), + pureté."""
    from collections import Counter

    by_cluster: dict[int, list[str]] = {}
    for cid, g in zip(membership, claim_gold):
        by_cluster.setdefault(cid, []).append(g)
    theme: dict[int, str] = {}
    purity: dict[int, float] = {}
    for cid, golds in by_cluster.items():
        c = Counter(golds)
        top, cnt = c.most_common(1)[0]
        theme[cid] = top
        purity[cid] = cnt / len(golds)
    return theme, purity


def avis_predicted_themes(avis: list[Avis], claim_owner: list[int],
                          membership: list[int], cluster_theme: dict[int, str]
                          ) -> dict[str, set[str]]:
    """Thèmes prédits d'un avis = {thème dominant des clusters de ses claims}."""
    hyps: dict[str, set[str]] = {a.id: set() for a in avis}
    for ci, ai in enumerate(claim_owner):
        cid = membership[ci]
        hyps[avis[ai].id].add(cluster_theme[cid])
    return hyps


def cluster_once(claim_vecs: np.ndarray, defaults, resolution: float, seed: int,
                 claim_gold: list[str], claim_owner: list[int], avis: list[Avis],
                 prepared, labels: list[str]) -> ClusterRun:
    from sklearn.metrics import homogeneity_completeness_v_measure

    graph = build_knn_graph(claim_vecs, k=defaults.k, threshold=defaults.threshold)
    res = run_leiden(graph, resolution=resolution, seed=seed)
    membership = res.membership
    cluster_theme, purity = map_clusters_to_themes(membership, claim_gold)
    hyps = avis_predicted_themes(avis, claim_owner, membership, cluster_theme)
    sc = score_themes(prepared, hyps, labels)
    h, c, v = homogeneity_completeness_v_measure(claim_gold, membership)
    return ClusterRun(
        resolution=resolution, membership=membership, n_clusters=res.n_clusters,
        modularity=res.modularity, cluster_theme=cluster_theme, cluster_purity=purity,
        homogeneity=h, completeness=c, v_measure=v,
        micro_f1=sc.micro_f1, macro_f1=sc.macro_f1, exact_set=sc.exact_set, score=sc)


# --------------------------------------------------------------------------- #
# Étape 5 — nouveauté (clusters loin des centroïdes-thèmes gold)
# --------------------------------------------------------------------------- #
@dataclass
class NoveltyHit:
    cluster_id: int
    size: int
    max_cos_to_theme: float
    nearest_theme: str
    purity: float
    label: str
    examples: list[str]


def theme_centroids(seg_vecs: np.ndarray, seg_theme: list[str], labels: list[str]
                    ) -> tuple[np.ndarray, list[str]]:
    """Centroïde (normalisé) de chaque thème gold à partir des segments gold."""
    cents, names = [], []
    for t in labels:
        idx = [i for i, th in enumerate(seg_theme) if th == t]
        if not idx:
            continue
        cents.append(seg_vecs[idx].mean(axis=0))
        names.append(t)
    C = _normalize_rows(np.asarray(cents, dtype=np.float64))
    return C, names


def find_novelty(claim_vecs: np.ndarray, membership: list[int], claim_texts: list[str],
                 cluster_purity: dict[int, float], theme_cents: np.ndarray,
                 theme_names: list[str], cutoff: float, min_size: int
                 ) -> tuple[list[NoveltyHit], float]:
    """Clusters dont le centroïde a un cosine max aux 8 thèmes < `cutoff` = hors-taxo."""
    by_cluster: dict[int, list[int]] = {}
    for i, cid in enumerate(membership):
        by_cluster.setdefault(cid, []).append(i)

    # Naming c-TF-IDF de TOUS les clusters (réutilise le naming de prod, mots-vides
    # dérivés du corpus de claims) → labels lisibles pour les hits de nouveauté.
    corpus_stop, _ = derive_corpus_stopwords(claim_texts)
    cluster_docs = {cid: [claim_texts[i] for i in idx] for cid, idx in by_cluster.items()}
    names = name_clusters(cluster_docs, corpus_stopwords=corpus_stop)

    hits: list[NoveltyHit] = []
    for cid, idx in by_cluster.items():
        if len(idx) < min_size:
            continue
        cent = claim_vecs[idx].mean(axis=0)
        cent = cent / (np.linalg.norm(cent) or 1.0)
        sims = theme_cents @ cent
        j = int(np.argmax(sims))
        max_cos = float(sims[j])
        if max_cos < cutoff:
            hits.append(NoveltyHit(
                cluster_id=cid, size=len(idx), max_cos_to_theme=round(max_cos, 3),
                nearest_theme=theme_names[j], purity=round(cluster_purity.get(cid, 0.0), 3),
                label=names.get(cid, {}).get("label", f"cluster {cid}"),
                examples=[claim_texts[i] for i in idx[:3]]))
    hits.sort(key=lambda h: h.max_cos_to_theme)
    return hits, cutoff


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _md_table(rows, cols):
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows)
    return "\n".join([head, sep, body])


def _pct(x):
    return f"{x * 100:.0f}%"


def build_report(gold_path: Path, avis: list[Avis], labels: list[str], model: str,
                 embedder: str, defaults, n_claims: int, claims_per_avis: float,
                 main: ClusterRun, sweep: list[ClusterRun], stats: OllamaStats,
                 think: bool | None, novelty: list[NoveltyHit], novelty_cutoff: float,
                 example_blocks: list[str], limited: bool) -> str:
    n = len(avis)
    n_mono = sum(1 for a in avis if a.type == "mono")
    n_multi = n - n_mono
    ms_per_avis = 1000.0 * stats.cold_seconds / n if n else 0.0
    sc = main.score
    beats_mistral = main.micro_f1 >= MISTRAL_F1
    close = main.micro_f1 >= MISTRAL_F1 - 0.05

    L = []
    L.append("# Pipeline CLAIMS (TalkToTheCity) : avis → claims → embed → clustering ÉMERGENT — rapport\n")
    L.append(f"*Jeu : `{gold_path.name}` — N={n} avis ({n_mono} mono, {n_multi} multi), "
             f"{len(labels)} thèmes gold. LLM d'extraction : **`{model}`** (Mac de Bob, "
             f"Apple Silicon via Tailscale — souverain), température 0, JSON mode, pensée "
             f"{'coupée' if think is False else 'native (non-raisonneur)'}. Embeddings : "
             f"**`{embedder}`** (`search_document:`). Clustering : k-NN+Leiden, défauts "
             f"DÉRIVÉS des données (k={defaults.k}, seuil cosine={defaults.threshold:.3f}), "
             f"résolution {main.resolution}.*\n")
    if limited:
        L.append("⚠️ **Run PARTIEL** (`--limit`) — chiffres non représentatifs.\n")
    L.append("**Question.** Le clustering ASCENDANT (claims libres → clusters) reconstruit-il "
             "les 8 thèmes du gold SANS jamais les voir, aussi bien que Mistral (choix fermé, "
             f"micro-F1 {MISTRAL_F1}) ou le classifieur entraîné ({CLF_F1}) — TOUT EN restant "
             "OUVERT (découvre du hors-taxo) et SOUVERAIN (100% local) ?\n")

    # --- Scorecard ---
    L.append("## Scorecard — récupération des thèmes (multi-label par avis) × ouverture × coût\n")
    rows = [
        {"Approche": "**claims → cluster émergent** (ce run)", "ouverte ?": "**OUI**",
         "taxo vue ?": "**NON**", "micro-F1": round(main.micro_f1, 3),
         "macro-F1": round(main.macro_f1, 3), "exact-set": _pct(main.exact_set),
         "V-mesure": round(main.v_measure, 3), "local": "**oui**", "données sortent": "non"},
        {"Approche": "Mistral-small (choix fermé)", "ouverte ?": "non",
         "taxo vue ?": "oui (prompt)", "micro-F1": MISTRAL_F1, "macro-F1": 0.935,
         "exact-set": "73%", "V-mesure": "—", "local": "non", "données sortent": "**oui**"},
        {"Approche": "classifieur MLP/nomic (entraîné)", "ouverte ?": "non",
         "taxo vue ?": "oui (labels)", "micro-F1": CLF_F1, "macro-F1": 0.940,
         "exact-set": "—", "V-mesure": "—", "local": "oui", "données sortent": "non"},
    ]
    L.append(_md_table(rows, ["Approche", "ouverte ?", "taxo vue ?", "micro-F1", "macro-F1",
                              "exact-set", "V-mesure", "local", "données sortent"]) + "\n")
    delta = main.micro_f1 - MISTRAL_F1
    verdict = ("OUI — et il dépasse Mistral" if delta > 0 else "OUI (à portée)" if close
               else "PARTIELLEMENT" if main.micro_f1 >= 0.80 else "NON")
    L.append(f"**Verdict récupération : {verdict}** — claims→cluster micro-F1="
             f"**{main.micro_f1:.3f}** vs Mistral {MISTRAL_F1} ({delta:+.3f}) et clf {CLF_F1} "
             f"({main.micro_f1 - CLF_F1:+.3f}), **sans jamais voir la taxonomie**. "
             f"{main.n_clusters} clusters émergent pour {len(labels)} thèmes gold ; "
             f"V-mesure {main.v_measure:.3f} (homogénéité {main.homogeneity:.3f}, "
             f"complétude {main.completeness:.3f}).\n")

    # --- Sweep résolution ---
    L.append("## Clustering émergent — sensibilité à la résolution\n")
    L.append("La résolution Leiden règle la granularité : basse → peu de gros clusters, "
             "haute → beaucoup de petits. On cherche celle qui fait émerger ~8 thèmes "
             "cohérents. **Aucun thème n'est donné** ; seul le mapping a posteriori utilise le gold.\n")
    srows = [{"résolution": f"**{r.resolution}**" if r.resolution == main.resolution else r.resolution,
              "n_clusters": r.n_clusters, "modularité": round(r.modularity, 3),
              "homogénéité": round(r.homogeneity, 3), "complétude": round(r.completeness, 3),
              "V-mesure": round(r.v_measure, 3), "micro-F1": round(r.micro_f1, 3),
              "macro-F1": round(r.macro_f1, 3), "exact-set": _pct(r.exact_set)}
             for r in sweep]
    L.append(_md_table(srows, ["résolution", "n_clusters", "modularité", "homogénéité",
                               "complétude", "V-mesure", "micro-F1", "macro-F1", "exact-set"]) + "\n")
    L.append(f"*Défauts du graphe DÉRIVÉS des {n_claims} claims (aucun magic-number corpus) : "
             f"k={defaults.k} (∝ log N), seuil d'arête cosine={defaults.threshold:.3f} "
             f"(μ−σ des k-NN). Seed 42.*\n")

    # --- F1 par thème (résolution principale) ---
    L.append(f"## F1 par thème — résolution {main.resolution} (mapping cluster→thème dominant)\n")
    per = sc.per_theme
    trows = [{"thème": t, "P": round(per[t]["p"], 3), "R": round(per[t]["r"], 3),
              "F1": round(per[t]["f1"], 3), "TP": per[t]["tp"], "FP": per[t]["fp"],
              "FN": per[t]["fn"]} for t in sorted(labels, key=lambda x: -per[x]["f1"])]
    L.append(_md_table(trows, ["thème", "P", "R", "F1", "TP", "FP", "FN"]) + "\n")
    # combien de clusters portent chaque thème (la richesse émergente)
    from collections import Counter
    theme_cluster_count = Counter(main.cluster_theme.values())
    cov = ", ".join(f"`{t}`×{theme_cluster_count.get(t, 0)}" for t in labels)
    L.append(f"Chaque thème gold est porté par plusieurs clusters émergents (sous-facettes) : "
             f"{cov}. Les thèmes gold non couverts (×0) sont ceux qu'aucun cluster ne reçoit "
             f"en dominante — source du rappel manquant.\n")

    # --- Exemples avis multi → claims → clusters → thèmes ---
    if example_blocks:
        L.append("## Exemples — un avis → ses claims → leurs clusters → thèmes émergents\n")
        L.extend(example_blocks)

    # --- Nouveauté ---
    L.append("## NOUVEAUTÉ — clusters hors-taxonomie (la valeur ajoutée de l'ouverture)\n")
    L.append(f"Clusters (taille ≥ 3) dont le centroïde est sémantiquement LOIN des 8 "
             f"centroïdes-thèmes du gold (cosine max < **{novelty_cutoff:.3f}**, seuil dérivé "
             f"= 5ᵉ percentile des cosines claim↔son-centroïde-thème). Ce sont des idées que "
             f"la taxonomie FERMÉE de Mistral/du classifieur ne pouvait pas représenter.\n")
    if novelty:
        nrows = [{"cluster": h.cluster_id, "taille": h.size, "cos→thème": h.max_cos_to_theme,
                  "thème le + proche": h.nearest_theme, "pureté": h.purity,
                  "label (c-TF-IDF)": h.label} for h in novelty[:12]]
        L.append(_md_table(nrows, ["cluster", "taille", "cos→thème", "thème le + proche",
                                   "pureté", "label (c-TF-IDF)"]) + "\n")
        L.append("### Exemples de claims hors-taxo\n")
        for h in novelty[:8]:
            L.append(f"- **cluster {h.cluster_id}** « _{h.label}_ » (n={h.size}, "
                     f"cos→`{h.nearest_theme}`={h.max_cos_to_theme}) :")
            for ex in h.examples:
                L.append(f"    - {ex}")
        L.append("")
    else:
        L.append("*Aucun cluster sous le seuil : à cette granularité, tout le corpus reste "
                 "dans le rayon sémantique des 8 thèmes (corpus de test mono-sujet TikTok ; "
                 "l'ouverture se révélerait davantage sur une consultation à sujets épars).*\n")

    # --- Coût ---
    L.append("## Coût, latence, souveraineté\n")
    api_calls = stats.calls
    cold_min = stats.cold_seconds / 60.0
    L.append(
        f"- **Extraction ministral (Mac, à chaud)** : **{api_calls} appels** réels + "
        f"{stats.cache_hits} servis par le cache `.cache/ollama/` — 1 avis/appel, "
        f"**~{stats.cold_seconds:.0f}s** cumulés (~**{ms_per_avis:.0f} ms/avis**, "
        f"~{cold_min:.1f} min pour {n} avis), {stats.eval_tokens:,} tokens générés, "
        f"{stats.errors} erreurs.\n"
        f"- **{n_claims} claims** extraites au total (**{claims_per_avis:.2f} claims/avis** "
        f"en moyenne) — l'avis est décomposé en idées atomiques avant clustering.\n"
        f"- **Embedding + clustering** : 100% local (nomic-v2 CPU + Leiden), négligeable "
        f"devant l'extraction. Réutilise le cache d'embeddings `.cache/`.\n"
        f"- **Souveraineté** : la donnée citoyenne ne sort JAMAIS du réseau privé "
        f"(`{OLLAMA_BASE}`, Tailscale). À comparer à ~**2-4 € par grosse consultation** en "
        f"API (Mistral EU) où le texte intégral des avis est transmis à un tiers. Local = "
        f"**~0 €** marginal, données souveraines, mais dépend du Mac allumé et de sa latence.\n")

    # --- Verdict ---
    L.append("## Verdict — l'approche OUVERTE tient-elle près de 0.93 en restant ouverte & souveraine ?\n")
    L.append(
        f"- **Récupération des thèmes : {verdict}.** Sans voir AUCUN thème, le clustering "
        f"ascendant de claims atteint micro-F1 **{main.micro_f1:.3f}** "
        f"({delta:+.3f} vs Mistral {MISTRAL_F1}, {main.micro_f1 - CLF_F1:+.3f} vs clf {CLF_F1}). "
        f"{'Il EST compétitif avec les approches fermées' if close else 'Il reste en retrait des approches fermées'} "
        f"— rappelons qu'elles, elles connaissent les 8 thèmes d'avance.\n")
    L.append(
        f"- **Reconstruction non supervisée** : V-mesure **{main.v_measure:.3f}** "
        f"(homogénéité {main.homogeneity:.3f} = les clusters sont purs ; complétude "
        f"{main.completeness:.3f} = un thème est éclaté en plusieurs sous-clusters). "
        f"L'éclatement est INHÉRENT à l'ascendant et utile : il fait émerger des SOUS-"
        f"facettes (TalkToTheCity les garde comme sous-thèmes), au prix de la complétude.\n")
    L.append(
        f"- **Ouverture (le point clé)** : {len(novelty)} cluster(s) hors-taxo détecté(s) — "
        f"l'approche n'est PAS bridée par une liste fermée ni par les 1ers avis : toute idée "
        f"nouvelle crée son cluster. C'est ce que NI Mistral (choix fermé) NI le classifieur "
        f"(8 classes figées) ne peuvent faire.\n")
    L.append(
        f"- **Arbitrage expressivité × qualité × coût** : on échange ~{abs(delta):.0%} de "
        f"micro-F1 {'gagné' if delta > 0 else 'perdu'} contre l'OUVERTURE (découverte de "
        f"nouveauté, granularité sous-thème) et la SOUVERAINETÉ (local, ~0 €, données qui ne "
        f"sortent pas), au prix d'une latence d'extraction (~{ms_per_avis:.0f} ms/avis sur le "
        f"Mac). Pour explorer une consultation sans taxo a priori, c'est l'outil ; pour "
        f"étiqueter vite sur une taxo connue, Mistral/le classifieur restent plus directs.\n")
    L.append(
        "- **Honnêteté** : le clustering est non supervisé, mais le mapping cluster→thème et "
        "la V-mesure utilisent le gold (évaluation « cluster-then-label » standard). "
        "L'étiquette gold par claim vient du segment gold le plus proche (désambiguïsation "
        "multi→mono par embedding). Latence Ollama sur Mac partagé, 1 avis/appel, sans "
        "batching → indicative. Corpus de test mono-sujet (TikTok) : il sous-estime la "
        "nouveauté qu'on verrait sur une consultation à sujets dispersés.\n")
    return "\n".join(L)


def _example_blocks(avis, claims, claim_owner, membership, cluster_theme, claim_index,
                    max_n=3):
    """Quelques avis multi : leurs claims, le cluster de chacune, le thème émergent."""
    blocks = []
    multi = [a for a in avis if a.type == "multi"]
    # ordre par nb de claims décroissant (les plus riches)
    multi.sort(key=lambda a: -len(claims.get(a.id, [])))
    shown = 0
    for a in multi:
        cl = claims.get(a.id, [])
        if len(cl) < 2:
            continue
        blocks.append(f"**{a.id}** — gold : {', '.join(sorted(a.themes))}\n")
        blocks.append(f"> {a.text}\n")
        for j, ctext in enumerate(cl):
            gi = claim_index[(a.id, j)]
            cid = membership[gi]
            blocks.append(f"- claim → cluster {cid} → **{cluster_theme[cid]}** : {ctext}")
        pred = sorted({cluster_theme[membership[claim_index[(a.id, j)]]] for j in range(len(cl))})
        blocks.append(f"- **thèmes émergents de l'avis : {', '.join(pred)}**\n")
        shown += 1
        if shown >= max_n:
            break
    return blocks


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline claims → clustering émergent.")
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--model", default="ministral-3:latest")
    ap.add_argument("--embedder", default="nomic-v2")
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--sweep", default="0.5,0.8,1.0,1.4,2.0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(DEFAULT_REPORT))
    ap.add_argument("--scores-out", default=str(DEFAULT_SCORES))
    args = ap.parse_args()

    gold_path = Path(args.gold)
    avis, taxonomy = load_avis(gold_path)
    if args.limit:
        avis = avis[:args.limit]
    if not taxonomy:
        labels = sorted({t for a in avis for t in a.themes})
    else:
        labels = list(taxonomy)
    n = len(avis)
    n_mono = sum(1 for a in avis if a.type == "mono")
    print(f"gold: {gold_path.name} — {n} avis ({n_mono} mono, {n - n_mono} multi), "
          f"{len(labels)} thèmes")

    # --- Étape 1 : extraction des claims (ministral, Mac) ---
    print(f"ÉTAPE 1 — extraction claims via {args.model} @ {OLLAMA_BASE} …")
    stats = OllamaStats()
    ok, think = ollama_warmup(args.model)
    if not ok:
        raise SystemExit(f"Modèle {args.model} injoignable sur {OLLAMA_BASE} "
                         f"(exporter AGORA_OLLAMA_URL depuis var/MAC_LOCAL_OLLAMA).")
    t0 = time.monotonic()
    claims = run_extraction(avis, args.model, stats, think)
    print(f"  extraction: {stats.cold_seconds:.0f}s cumulés, {stats.calls} appels, "
          f"{stats.cache_hits} cache, {stats.errors} err ({time.monotonic() - t0:.0f}s mur)")

    # Aplatis : claim_texts, claim_owner (index d'avis), claim_index (id,j)->global
    claim_texts: list[str] = []
    claim_owner: list[int] = []
    claim_index: dict[tuple[str, int], int] = {}
    for ai, a in enumerate(avis):
        for j, ctext in enumerate(claims[a.id]):
            claim_index[(a.id, j)] = len(claim_texts)
            claim_owner.append(ai)
            claim_texts.append(ctext)
    n_claims = len(claim_texts)
    claims_per_avis = n_claims / n if n else 0.0
    print(f"  {n_claims} claims ({claims_per_avis:.2f}/avis)")

    # --- Étape 2 : embeddings (claims + segments gold) ---
    print(f"ÉTAPE 2 — embeddings {args.embedder} (claims + segments gold) …")
    from eval.segmentation.embeddings import embed_docs

    claim_vecs = _normalize_rows(embed_docs(claim_texts, model_id=args.embedder).astype(np.float64))
    # segments gold (pour l'étiquette gold par claim + centroïdes-thèmes)
    seg_texts, seg_theme, seg_owner = [], [], []
    for ai, a in enumerate(avis):
        for st, th in zip(a.seg_texts, a.seg_themes):
            seg_owner.append(ai)
            seg_texts.append(st)
            seg_theme.append(th)
    seg_vecs = _normalize_rows(embed_docs(seg_texts, model_id=args.embedder).astype(np.float64))
    seg_vecs_by_avis: dict[int, np.ndarray] = {}
    cursor = 0
    for ai, a in enumerate(avis):
        m = len(a.seg_texts)
        seg_vecs_by_avis[ai] = seg_vecs[cursor:cursor + m]
        cursor += m

    # --- Étape 4a : étiquette gold par claim ---
    claim_gold = claim_gold_labels(claim_vecs, claim_owner, avis, seg_vecs_by_avis)

    # --- Étape 3+4 : clustering émergent + mesures (sweep résolution) ---
    print("ÉTAPE 3 — clustering émergent (défauts dérivés) + mesures …")
    defaults = derive_defaults(claim_vecs.astype(np.float32))
    print(f"  défauts dérivés: k={defaults.k}, seuil={defaults.threshold:.3f} "
          f"(μ={defaults.pool_mean}, σ={defaults.pool_std})")
    prepared = prepare([a for a in load_gold(gold_path)[0] if a.id in {x.id for x in avis}])

    sweep_res = sorted({float(x) for x in args.sweep.split(",") if x.strip()} | {args.resolution})
    runs: list[ClusterRun] = []
    for r in sweep_res:
        run = cluster_once(claim_vecs, defaults, r, args.seed, claim_gold, claim_owner,
                           avis, prepared, labels)
        runs.append(run)
        print(f"  rés {r}: {run.n_clusters} clusters, V={run.v_measure:.3f}, "
              f"micro-F1={run.micro_f1:.3f}, exact={run.exact_set:.3f}")
    main_run = next(r for r in runs if r.resolution == args.resolution)

    # --- Étape 5 : nouveauté ---
    print("ÉTAPE 5 — détection de nouveauté (hors-taxo) …")
    theme_cents, theme_names = theme_centroids(seg_vecs, seg_theme, labels)
    # seuil dérivé : 5e percentile des cosines claim↔centroïde de SON thème gold
    name_to_row = {t: i for i, t in enumerate(theme_names)}
    own_cos = []
    for ci, g in enumerate(claim_gold):
        if g in name_to_row:
            own_cos.append(float(theme_cents[name_to_row[g]] @ claim_vecs[ci]))
    novelty_cutoff = float(np.percentile(own_cos, 5)) if own_cos else 0.5
    novelty, _ = find_novelty(claim_vecs, main_run.membership, claim_texts,
                              main_run.cluster_purity, theme_cents, theme_names,
                              novelty_cutoff, min_size=3)
    print(f"  seuil nouveauté cos={novelty_cutoff:.3f} → {len(novelty)} cluster(s) hors-taxo")

    # --- Exemples ---
    examples = _example_blocks(avis, claims, claim_owner, main_run.membership,
                               main_run.cluster_theme, claim_index)

    # --- Report + scores ---
    limited = bool(args.limit)
    report = build_report(gold_path, avis, labels, args.model, args.embedder, defaults,
                          n_claims, claims_per_avis, main_run, runs, stats, think,
                          novelty, novelty_cutoff, examples, limited)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"✓ {args.out}")

    Path(args.scores_out).write_text(json.dumps({
        "gold": gold_path.name, "n_avis": n, "n_mono": n_mono, "n_multi": n - n_mono,
        "n_labels": len(labels), "model": args.model, "embedder": args.embedder,
        "limited": limited, "seed": args.seed,
        "n_claims": n_claims, "claims_per_avis": round(claims_per_avis, 3),
        "derived_defaults": {"k": defaults.k, "threshold": round(defaults.threshold, 4),
                             "pool_mean": defaults.pool_mean, "pool_std": defaults.pool_std},
        "main_resolution": args.resolution,
        "main": {
            "n_clusters": main_run.n_clusters, "modularity": main_run.modularity,
            "homogeneity": round(main_run.homogeneity, 4),
            "completeness": round(main_run.completeness, 4),
            "v_measure": round(main_run.v_measure, 4),
            "micro_F1": round(main_run.micro_f1, 4), "macro_F1": round(main_run.macro_f1, 4),
            "exact_set": round(main_run.exact_set, 4),
            "per_theme": {t: {k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in d.items()} for t, d in main_run.score.per_theme.items()},
        },
        "sweep": [{"resolution": r.resolution, "n_clusters": r.n_clusters,
                   "modularity": round(r.modularity, 4),
                   "homogeneity": round(r.homogeneity, 4),
                   "completeness": round(r.completeness, 4), "v_measure": round(r.v_measure, 4),
                   "micro_F1": round(r.micro_f1, 4), "macro_F1": round(r.macro_f1, 4),
                   "exact_set": round(r.exact_set, 4)} for r in runs],
        "novelty": {"cutoff_cos": round(novelty_cutoff, 4), "n_hits": len(novelty),
                    "hits": [{"cluster_id": h.cluster_id, "size": h.size,
                              "max_cos_to_theme": h.max_cos_to_theme,
                              "nearest_theme": h.nearest_theme, "purity": h.purity,
                              "label": h.label, "examples": h.examples}
                             for h in novelty[:12]]},
        "cost": {"ollama_calls": stats.calls, "cache_hits": stats.cache_hits,
                 "errors": stats.errors, "cold_seconds": round(stats.cold_seconds, 2),
                 "ms_per_avis": round(1000.0 * stats.cold_seconds / n, 1) if n else 0.0,
                 "eval_tokens": stats.eval_tokens, "endpoint": OLLAMA_BASE},
        "baselines": {"mistral_micro_F1": MISTRAL_F1, "classifier_micro_F1": CLF_F1},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ {args.scores_out}")


if __name__ == "__main__":
    main()
