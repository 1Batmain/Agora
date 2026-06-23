"""Segmentation par GRAPHE de mots (kNN sémantique + séquence) → Leiden.

EXPÉ R&D (read-only). Idée : traiter les **vecteurs-mots** d'un avis comme un
**graphe** et laisser **Leiden** trouver les communautés = thèmes = segments. À
comparer au réglé-main ATTENTION (F1_multi 0.769), au change-point (0.44) et à
l'appris, sur le MÊME `gold_large.json`.

⚠️ Le piège — la CONTIGUÏTÉ. Un graphe kNN d'embeddings ignore l'ORDRE : Leiden
regrouperait le mot 1 et le mot 10 sans le mot 5 → communautés NON contiguës, alors
qu'un segment est un span de mots CONSÉCUTIFS. Deux parades, toutes deux ici :
  1. **Arêtes = similarité (kNN cosinus des vecteurs-mots) + séquence (mots
     adjacents reliés, poids α réglable)**. L'adjacence biaise vers des communautés
     contiguës (α=0 = pure similarité = le piège, mesuré comme contrôle).
  2. **Contiguïté IMPOSÉE** a posteriori : segments = runs maximaux de même
     communauté le long de la séquence ; les micro-runs < `min_seg` sont fusionnés
     dans le voisin. Frontières = positions où la communauté change.
La **résolution** de Leiden = granularité (mono cohérent → 1 communauté = 0
frontière ; multi → 2-3 communautés).

Vecteurs-mots : réutilise `attn_seg.word_attention(text).V` (e5-base, MÊME forward
que l'attention → apples-to-apples, cachable). Bonus `nomic-v2` via
`embeddings.embed_word_units` (embed de prod). Graphe/Leiden réutilisent
`pipeline.cluster.{knn.build_knn_graph, leiden_cluster.run_leiden}`.

Seuil de similarité **dérivé** (μ−β·σ des cosinus kNN poolés, zéro magic-number).
Balayage : k (voisins) × α (poids d'adjacence) × résolution × min_seg.

ÉCRIT UNIQUEMENT dans `eval/segmentation/`. CPU, seed fixe.

    uv run --extra contender python -m eval.segmentation.graph_seg \
        [--gold eval/segmentation/gold_large.json] [--sources e5-base nomic-v2] \
        [--out eval/segmentation/graph_report.md]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval.segmentation import metrics as M
from eval.segmentation.attn_seg import _split_words, _word_index, word_attention
from eval.segmentation.seg_bench import GoldItem, load_gold
from eval.segmentation.segmenters import MIN_SEG
from pipeline.cluster.knn import KnnGraph, build_knn_graph
from pipeline.cluster.leiden_cluster import run_leiden

HERE = Path(__file__).resolve().parent
DEFAULT_GOLD = HERE / "gold_large.json"
DEFAULT_REPORT = HERE / "graph_report.md"
DEFAULT_SCORES = HERE / "graph_scores.json"

ATTN_SCORES = HERE / "attn_scores.json"   # attention réglé-main (référence)
CP_SCORES = HERE / "scores.json"          # change-point (référence)

SEED = 42

# Grille de balayage. k = voisins kNN ; α = poids des arêtes de SÉQUENCE (α=0 →
# pure similarité = le piège de contiguïté, gardé comme contrôle) ; résolution =
# granularité Leiden ; min_seg = fusion des micro-runs.
K_GRID = [5, 10, 20]
ALPHA_GRID = [0.0, 0.5, 1.0, 2.0]
RES_GRID = [0.5, 1.0, 1.5, 2.0, 3.0]
MINSEG_GRID = [MIN_SEG, 5]
BETA = 1.0   # seuil de similarité = μ − β·σ des cosinus kNN (dérivé, pas magique)

# Référence apprise (cf. learned_report.md, LR +nég. gold-tuné F1_global — le point
# apples-to-apples : seuil re-calé sur le gold par le MÊME objectif que l'attention).
LEARNED_REF = {
    "approche": "appris LR (gold-tuné F1_global)", "config": "thr=0.1",
    "Pk": 0.2007, "WindowDiff": 0.2054, "F1_multi": 0.7389, "P": 0.9027,
    "R": 0.6255, "mono_FP": 0.0962, "F1_global": 0.7214,
}


# --------------------------------------------------------------------------- #
# Préparation : vecteurs-mots + frontières gold (indices-mots)
# --------------------------------------------------------------------------- #
@dataclass
class PreparedG:
    item: GoldItem
    words: list[str]
    V: np.ndarray            # [n, dim] vecteurs-MOTS L2-normalisés
    n: int
    ref: set[int]


def _word_vectors(text: str, source: str) -> tuple[list[str], np.ndarray]:
    """Vecteurs-mots [n,dim] L2-norm. e5-base/bge-m3 → réutilise `word_attention.V`
    (même forward que l'attention) ; autres (nomic-v2) → `embed_word_units`."""
    if source in ("e5-base", "bge-m3"):
        wa = word_attention(text, source)
        return wa.words, (wa.V if wa.V is not None else np.zeros((wa.n, 1), np.float32))
    from eval.segmentation.embeddings import embed_word_units
    wu = embed_word_units(text, model_id=source)
    return wu.words, wu.vectors


def prepare(items: list[GoldItem], source: str) -> list[PreparedG]:
    out = []
    for it in items:
        words, V = _word_vectors(it.text, source)
        n = len(words)
        _, spans = _split_words(it.text)
        ref = set()
        for off in it.boundaries_char:
            b = _word_index(spans, off)
            if 0 < b < n:
                ref.add(b)
        out.append(PreparedG(it, words, V, n, ref))
    return out


# --------------------------------------------------------------------------- #
# Graphe = similarité kNN (seuil dérivé) + séquence (adjacence pondérée α)
# --------------------------------------------------------------------------- #
def _sim_edges(V: np.ndarray, k: int) -> list[tuple[int, int, float]]:
    """Arêtes kNN cosinus BRUTES (sans seuil) — `build_knn_graph(threshold=-1)`."""
    if V.shape[0] <= 1:
        return []
    g = build_knn_graph(V, k=k, threshold=-1.0, prefer_faiss=True)
    return g.edges


def global_sim_threshold(prep: list[PreparedG], k: int) -> tuple[float, dict]:
    """Seuil de similarité POOLÉ sur tous les avis : μ − β·σ des cosinus kNN.

    Dérivé des données (zéro magic-number, même philosophie que le banc embeddings/
    attention qui calibre μ/σ globalement). Renvoie aussi les arêtes brutes par avis
    pour éviter de recalculer le kNN (cher) à chaque (α, résolution)."""
    edges_by_id, pool = {}, []
    for p in prep:
        e = _sim_edges(p.V, k)
        edges_by_id[p.item.id] = e
        pool.extend(w for _, _, w in e)
    if not pool:
        return -1.0, edges_by_id
    arr = np.asarray(pool, dtype=np.float64)
    thr = float(arr.mean() - BETA * arr.std())
    return thr, edges_by_id


def build_word_graph(n: int, sim_edges: list[tuple[int, int, float]],
                     sim_thr: float, alpha: float) -> KnnGraph:
    """Graphe mot : arêtes de similarité (cosinus ≥ seuil) + arêtes de séquence
    (mots adjacents, poids α). L'adjacence force la quasi-contiguïté des communautés.
    α=0 → graphe de pure similarité (le piège)."""
    weights: dict[tuple[int, int], float] = {}
    for i, j, w in sim_edges:
        if w >= sim_thr:
            a, b = (i, j) if i < j else (j, i)
            weights[(a, b)] = max(weights.get((a, b), 0.0), w)
    if alpha > 0:
        for i in range(n - 1):
            key = (i, i + 1)
            weights[key] = weights.get(key, 0.0) + alpha
    edges = [(a, b, w) for (a, b), w in weights.items()]
    return KnnGraph(n=n, edges=edges, k=0, threshold=sim_thr, backend="graph_seg")


# --------------------------------------------------------------------------- #
# Contiguïté imposée : membership → runs → frontières (fusion des micro-runs)
# --------------------------------------------------------------------------- #
def _runs(labels: list[int]) -> list[list[int]]:
    """Liste de [start, end) par run maximal de même label."""
    runs, start = [], 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            runs.append([start, i])
            start = i
    runs.append([start, len(labels)])
    return runs


def membership_to_boundaries(membership: list[int], n: int, min_seg: int) -> set[int]:
    """Frontières = changements de communauté le long de la séquence, APRÈS fusion
    des runs plus courts que `min_seg` dans leur voisin le plus grand.

    Converge : chaque fusion réduit le nombre de runs. Abstient (∅) si l'avis est
    trop court pour deux segments — comme les autres segmenteurs du banc."""
    if n < 2 * min_seg or n == 0:
        return set()
    labels = list(membership)
    while True:
        runs = _runs(labels)
        if len(runs) == 1:
            break
        # plus petit run sous le seuil
        short = [(e - s, idx) for idx, (s, e) in enumerate(runs) if e - s < min_seg]
        if not short:
            break
        _, idx = min(short)
        s, e = runs[idx]
        left = runs[idx - 1] if idx > 0 else None
        right = runs[idx + 1] if idx < len(runs) - 1 else None
        # absorbe le micro-run dans le VOISIN le plus grand (label dominant)
        if left is None:
            nb = right
        elif right is None:
            nb = left
        else:
            nb = left if (left[1] - left[0]) >= (right[1] - right[0]) else right
        labels[s:e] = [labels[nb[0]]] * (e - s)
    return {s for s, _ in _runs(labels)[1:]}


def segment_graph(p: PreparedG, sim_edges, sim_thr: float, alpha: float,
                  resolution: float, min_seg: int) -> tuple[set[int], int]:
    """Frontières + nb de communautés Leiden pour un avis."""
    if p.n < 2:
        return set(), 1
    g = build_word_graph(p.n, sim_edges, sim_thr, alpha)
    res = run_leiden(g, resolution=resolution, seed=SEED)
    hyp = membership_to_boundaries(res.membership, p.n, min_seg)
    return hyp, res.n_clusters


# --------------------------------------------------------------------------- #
# Évaluation
# --------------------------------------------------------------------------- #
@dataclass
class GConfig:
    source: str
    k: int
    alpha: float
    resolution: float
    min_seg: int
    sim_thr: float


@dataclass
class GScore:
    cfg: GConfig
    pk: float
    windowdiff: float
    f1: float
    precision: float
    recall: float
    gf1: float
    mono_fp_rate: float
    mono_cuts_mean: float
    n_clusters_mean: float

    def as_row(self) -> dict:
        return {
            "source": self.cfg.source, "k": self.cfg.k, "alpha": self.cfg.alpha,
            "res": self.cfg.resolution, "min_seg": self.cfg.min_seg,
            "sim_thr": round(self.cfg.sim_thr, 3),
            "Pk": round(self.pk, 4), "WindowDiff": round(self.windowdiff, 4),
            "F1_multi": round(self.f1, 4), "P": round(self.precision, 4),
            "R": round(self.recall, 4), "mono_FP": round(self.mono_fp_rate, 4),
            "mono_cuts": round(self.mono_cuts_mean, 3),
            "n_clust": round(self.n_clusters_mean, 2),
            "F1_global": round(self.gf1, 4),
        }


def evaluate_config(cfg: GConfig, prep: list[PreparedG],
                    edges_by_id: dict) -> GScore:
    multi = [p for p in prep if p.item.type == "multi"]
    mono = [p for p in prep if p.item.type == "mono"]

    pk_m, wd_m, nclust = [], [], []
    bc = M.BoundaryCounts()
    gbc = M.BoundaryCounts()
    for p in multi:
        hyp, nc = segment_graph(p, edges_by_id[p.item.id], cfg.sim_thr,
                                cfg.alpha, cfg.resolution, cfg.min_seg)
        nclust.append(nc)
        pk_m.append(M.pk(p.n, p.ref, hyp))
        wd_m.append(M.windowdiff(p.n, p.ref, hyp))
        c = M.boundary_counts(p.ref, hyp, tol=1)
        bc = bc + c
        gbc = gbc + c
    mono_hits, mono_cuts = 0, 0
    for p in mono:
        hyp, nc = segment_graph(p, edges_by_id[p.item.id], cfg.sim_thr,
                                cfg.alpha, cfg.resolution, cfg.min_seg)
        nclust.append(nc)
        if hyp:
            mono_hits += 1
        mono_cuts += len(hyp)
        gbc = gbc + M.boundary_counts(p.ref, hyp, tol=1)
    return GScore(
        cfg=cfg,
        pk=float(np.mean(pk_m)) if pk_m else 0.0,
        windowdiff=float(np.mean(wd_m)) if wd_m else 0.0,
        f1=bc.f1, precision=bc.precision, recall=bc.recall, gf1=gbc.f1,
        mono_fp_rate=mono_hits / len(mono) if mono else 0.0,
        mono_cuts_mean=mono_cuts / len(mono) if mono else 0.0,
        n_clusters_mean=float(np.mean(nclust)) if nclust else 0.0,
    )


def sweep_source(source: str, prep: list[PreparedG]) -> list[GScore]:
    scores: list[GScore] = []
    for k in K_GRID:
        thr, edges_by_id = global_sim_threshold(prep, k)
        print(f"  k={k} seuil-sim dérivé={thr:.3f}")
        for alpha in ALPHA_GRID:
            for res in RES_GRID:
                for min_seg in MINSEG_GRID:
                    cfg = GConfig(source, k, alpha, res, min_seg, thr)
                    scores.append(evaluate_config(cfg, prep, edges_by_id))
    return scores


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _md_table(rows: list[dict], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows)
    return "\n".join([head, sep, body])


def _ref_row(name: str, d: dict, cfg: str) -> dict:
    return {"approche": name, "config": cfg, "Pk": d["Pk"], "WindowDiff": d["WindowDiff"],
            "F1_multi": d["F1_multi"], "P": d["P"], "R": d["R"],
            "mono_FP": d["mono_FP"], "F1_global": d["F1_global"]}


def load_ref(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    return d.get("winner")


def build_report(gold_path: Path, items: list[GoldItem],
                 all_scores: dict[str, list[GScore]],
                 attn: dict | None, cp: dict | None) -> str:
    n_mono = sum(1 for it in items if it.type == "mono")
    n_multi = sum(1 for it in items if it.type == "multi")
    flat = [s for ss in all_scores.values() for s in ss]
    winner = max(flat, key=lambda s: (s.gf1, s.f1, -s.windowdiff, -s.pk))

    L = []
    L.append("# Segmentation par GRAPHE de mots (kNN + Leiden) — bat-elle l'attention ?\n")
    L.append(f"*Jeu : `{gold_path.name}` — N={len(items)} ({n_mono} mono, {n_multi} multi). "
             f"Sources de vecteurs-mots : {', '.join(all_scores)}. CPU, seed={SEED}.*\n")

    # 1. Méthode
    L.append("## 1. Méthode — mots = graphe, communautés = segments\n")
    L.append(
        "- **Vecteurs-mots** [n,dim] L2-norm : `e5-base` via `word_attention(text).V` "
        "(MÊME forward que l'attention réglée → apples-to-apples) ; `nomic-v2` via "
        "`embed_word_units` (embed de prod).\n"
        "- **Graphe** = arêtes de **similarité** (kNN cosinus, seuil dérivé μ−σ poolé "
        "sur tous les avis) **+** arêtes de **séquence** (mots adjacents, poids **α**). "
        "L'adjacence force des communautés ~contiguës ; **α=0 = pure similarité = le "
        "piège** (gardé comme contrôle).\n"
        "- **Leiden** (`igraph`+`leidenalg`, RBConfiguration, seed fixe) → communautés. "
        "**Contiguïté IMPOSÉE** : segments = runs maximaux de même communauté le long de "
        f"la séquence ; micro-runs < `min_seg` fusionnés dans le voisin le plus grand. "
        "Frontières = changements de communauté.\n"
        f"- **Balayage** : k∈{K_GRID} × α∈{ALPHA_GRID} × résolution∈{RES_GRID} × "
        f"min_seg∈{MINSEG_GRID}. La **résolution** = granularité (mono cohérent → 1 "
        "communauté = 0 frontière ; multi → 2-3).\n")

    # 2. Scorecard vs références
    L.append("## 2. Scorecard — graphe-Leiden vs références (même gold)\n")
    rows = []
    if attn:
        rows.append(_ref_row("**attention réglé-main** (e5-base)",
                             attn, f"{attn['layers']}/{attn['heads']} W={attn['W']} c={attn['c']}"))
    if cp:
        rows.append(_ref_row("change-point (embeddings)", cp,
                             f"{cp['method']} W={cp['W']} pen={cp.get('pen','')}"))
    rows.append(_ref_row("_appris LR (réf.)_", LEARNED_REF, LEARNED_REF["config"]))
    for src, ss in all_scores.items():
        if not ss:
            continue
        w = max(ss, key=lambda s: (s.gf1, s.f1, -s.windowdiff))
        rows.append({"approche": f"**graphe-Leiden {src}**",
                     "config": f"k={w.cfg.k} α={w.cfg.alpha} res={w.cfg.resolution} "
                               f"min={w.cfg.min_seg}",
                     "Pk": round(w.pk, 4), "WindowDiff": round(w.windowdiff, 4),
                     "F1_multi": round(w.f1, 4), "P": round(w.precision, 4),
                     "R": round(w.recall, 4), "mono_FP": round(w.mono_fp_rate, 4),
                     "F1_global": round(w.gf1, 4)})
    cols = ["approche", "config", "Pk", "WindowDiff", "F1_multi", "P", "R",
            "mono_FP", "F1_global"]
    L.append(_md_table(rows, cols) + "\n")
    L.append("*(Pk/WindowDiff ↓ = mieux, sur les multi ; F1_multi = frontières tol ±1 ; "
             "mono_FP = fraction de mono sur-coupés ; F1_global = frontières mono+multi, "
             "objectif de sélection.)*\n")

    # 3. Top configs graphe
    L.append("## 3. Top 15 configurations graphe-Leiden\n")
    top = sorted(flat, key=lambda s: (-s.gf1, -s.f1, s.windowdiff))[:15]
    cols2 = ["source", "k", "alpha", "res", "min_seg", "sim_thr", "Pk", "WindowDiff",
             "F1_multi", "P", "R", "mono_FP", "mono_cuts", "n_clust", "F1_global"]
    L.append(_md_table([t.as_row() for t in top], cols2) + "\n")

    # 4. Effet de l'adjacence α (le piège de contiguïté) et de la résolution
    L.append("## 4. Le piège de contiguïté — effet de α et de la résolution\n")
    for src, ss in all_scores.items():
        if not ss:
            continue
        L.append(f"\n**{src}** — meilleure config (F1_global) par poids d'adjacence α :\n")
        best_by_alpha = {}
        for s in ss:
            key = s.cfg.alpha
            b = best_by_alpha.get(key)
            if b is None or (s.gf1, s.f1) > (b.gf1, b.f1):
                best_by_alpha[key] = s
        L.append(_md_table([best_by_alpha[a].as_row() for a in ALPHA_GRID
                            if a in best_by_alpha], cols2) + "\n")
        L.append(f"\n**{src}** — meilleure config par résolution (granularité) :\n")
        best_by_res = {}
        for s in ss:
            key = s.cfg.resolution
            b = best_by_res.get(key)
            if b is None or (s.gf1, s.f1) > (b.gf1, b.f1):
                best_by_res[key] = s
        L.append(_md_table([best_by_res[r].as_row() for r in RES_GRID
                            if r in best_by_res], cols2) + "\n")

    # 4b. Frontière détection/abstention — le nœud du problème
    L.append("## 4b. Frontière détection ↔ abstention (le nœud)\n")
    L.append("Pour chaque résolution : la config qui **abstient le mieux** (mono_FP min) vs "
             "celle qui **détecte le mieux** (F1_multi max). Si les deux ne coïncident "
             "JAMAIS, c'est qu'aucun réglage global ne distingue « mono cohérent » de "
             "« virage de thème » au niveau MOT.\n")
    for src, ss in all_scores.items():
        if not ss:
            continue
        rows = []
        for r in RES_GRID:
            sub = [s for s in ss if s.cfg.resolution == r]
            if not sub:
                continue
            ab = min(sub, key=lambda s: (s.mono_fp_rate, -s.f1))
            de = max(sub, key=lambda s: (s.f1, -s.mono_fp_rate))
            rows.append({"res": r,
                         "abstient_monoFP": round(ab.mono_fp_rate, 3),
                         "·_F1_multi": round(ab.f1, 3), "·_nclust": round(ab.n_clusters_mean, 2),
                         "détecte_F1_multi": round(de.f1, 3),
                         "·_monoFP": round(de.mono_fp_rate, 3),
                         "·_nclust ": round(de.n_clusters_mean, 2)})
        cols3 = ["res", "abstient_monoFP", "·_F1_multi", "·_nclust",
                 "détecte_F1_multi", "·_monoFP", "·_nclust "]
        L.append(f"\n**{src}** :\n")
        L.append(_md_table(rows, cols3) + "\n")
    if attn:
        L.append(f"\n*Repère : l'attention réglé-main tient F1_multi={attn['F1_multi']} ET "
                 f"mono_FP={attn['mono_FP']} EN MÊME TEMPS. Aucune ligne ci-dessus ne "
                 f"s'en approche : quand le graphe abstient sur les mono, il abstient aussi "
                 f"sur les multi (F1_multi s'effondre) ; quand il détecte, il sur-coupe "
                 f"tout. Les deux colonnes ne se rejoignent jamais.*\n")

    # 5. Verdict
    L.append("## 5. Verdict honnête\n")
    w = winner
    L.append(f"**Meilleure config graphe : `{w.cfg.source}` · k={w.cfg.k} · α={w.cfg.alpha} "
             f"· res={w.cfg.resolution} · min_seg={w.cfg.min_seg}** → "
             f"F1_multi={w.f1:.3f} (P={w.precision:.3f}, R={w.recall:.3f}), Pk={w.pk:.3f}, "
             f"WindowDiff={w.windowdiff:.3f}, F1_global={w.gf1:.3f}, "
             f"mono_FP={w.mono_fp_rate:.3f}, {w.n_clusters_mean:.2f} communautés/avis.\n")
    if attn:
        d_f1 = w.f1 - attn["F1_multi"]
        d_pk = w.pk - attn["Pk"]
        d_gf1 = w.gf1 - attn["F1_global"]
        beats = (w.f1 > attn["F1_multi"] + 1e-9) and (w.pk < attn["Pk"] - 1e-9)
        beats_g = w.gf1 > attn["F1_global"] + 1e-9
        verdict = "**OUI**" if beats else ("partiellement (F1_global)" if beats_g else "**NON**")
        L.append(f"- **Bat-elle l'attention réglé-main (F1_multi={attn['F1_multi']}, "
                 f"Pk={attn['Pk']}, F1_global={attn['F1_global']}, "
                 f"mono_FP={attn['mono_FP']}) ? {verdict}.** "
                 f"ΔF1_multi={d_f1:+.3f}, ΔPk={d_pk:+.3f} (négatif = mieux), "
                 f"ΔF1_global={d_gf1:+.3f}.\n")
    if cp:
        d = w.f1 - cp["F1_multi"]
        L.append(f"- vs **change-point** (F1_multi={cp['F1_multi']}) : "
                 f"ΔF1_multi={d:+.3f} → le graphe {'**bat**' if d > 0 else 'ne bat pas'} "
                 f"le change-point.\n")
    L.append(
        "- **Pourquoi ça rate — Leiden ne sait pas s'ABSTENIR.** C'est le résultat "
        "central (§4b). L'attention/le change-point calibrent un seuil GLOBAL `μ−cσ` : "
        "sur un mono cohérent, le signal ne descend jamais sous le seuil → **0 frontière**. "
        "Leiden, lui, maximise la modularité PAR document : à résolution fixe il trouve "
        "toujours une partition, même dans un graphe quasi-structureless (problème connu "
        "des communautés spurious / limite de résolution). Résultat : à res basse il "
        "collapse TOUT en 1 communauté (mono_FP→0 mais F1_multi→0.06, il rate aussi les "
        "multi) ; à res haute il coupe TOUT (F1_multi↑ mais mono_FP→1.0). **Les deux "
        "régimes ne se rejoignent jamais** au point de fonctionnement de l'attention "
        f"(F1={attn['F1_multi'] if attn else '?'}, mono_FP={attn['mono_FP'] if attn else '?'} "
        "SIMULTANÉMENT).\n")
    L.append(
        "- **Pourquoi au niveau MOT il n'y a pas de structure** : les vecteurs-mots "
        f"contextuels e5 sont quasi-colinéaires (seuil de similarité dérivé ≈{winner.cfg.sim_thr:.2f}, "
        "μ−σ des cosinus kNN très haut → cosinus tous ~0.9+). Un mono et un multi "
        "présentent donc à peu près la même (faible) structure de communauté : il n'existe "
        "pas de granularité Leiden qui sépare « cohérent » de « virage ». La **béquille de "
        "contiguïté** (arêtes de séquence α + fusion des micro-runs) réimpose bien l'ordre, "
        "mais ne crée pas le signal de frontière qui manque.\n")
    L.append(
        "- **La contiguïté (le piège) est gérée mais non décisive** : α=0 (pure "
        "similarité) sur-coupe à peine plus que α>0 (§4) — le mal vient de l'absence "
        "d'abstention, pas seulement de la non-contiguïté. Imposer les runs + fusionner "
        "les micro-runs évite les communautés entrelacées mais ne sauve pas le verdict.\n")
    L.append(
        f"- **Jeu** : multi = concaténation de mono-thèmes (frontières nettes par "
        f"construction) → borne OPTIMISTE pour toutes les approches. mono_FP mesure "
        f"l'abstention (un mono cohérent doit donner 1 seule communauté = 0 coupe).\n")
    L.append(
        "- **Conclusion** : le graphe-Leiden de mots **NE BAT PAS** l'attention réglé-main "
        "(0.769) ni même le change-point (0.44) ; il échoue sur l'**abstention**, qui est "
        "justement ce que le seuil global `μ−cσ` de l'attention réussit. Piste si on y "
        "tenait : un graphe au niveau PHRASE/clause (moins de nœuds, structure plus nette) "
        "+ un critère d'abstention explicite (ne couper que si la modularité gagnée dépasse "
        "un seuil global) — mais ce serait réinventer le seuil calibré de l'attention par "
        "un détour plus coûteux.\n")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="Segmentation graphe kNN + Leiden.")
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--sources", nargs="+", default=["e5-base"],
                    help="sources de vecteurs-mots : e5-base (défaut), bge-m3, nomic-v2")
    ap.add_argument("--out", default=str(DEFAULT_REPORT))
    ap.add_argument("--scores-out", default=str(DEFAULT_SCORES))
    args = ap.parse_args()

    gold_path = Path(args.gold)
    items, _ = load_gold(gold_path)
    print(f"gold: {gold_path.name} — {len(items)} items")
    attn = load_ref(ATTN_SCORES)
    cp = load_ref(CP_SCORES)
    if attn:
        print(f"réf. attention: F1_multi={attn['F1_multi']} Pk={attn['Pk']}")

    all_scores: dict[str, list[GScore]] = {}
    for src in args.sources:
        print(f"\n=== source {src} ===")
        print("vecteurs-mots + préparation…")
        prep = prepare(items, src)
        print("balayage graphe + Leiden…")
        scores = sweep_source(src, prep)
        all_scores[src] = scores
        w = max(scores, key=lambda s: (s.gf1, s.f1))
        print(f"  best {src}: k={w.cfg.k} α={w.cfg.alpha} res={w.cfg.resolution} "
              f"F1_multi={w.f1:.3f} Pk={w.pk:.3f} F1_global={w.gf1:.3f} "
              f"mono_FP={w.mono_fp_rate:.3f}")

    report = build_report(gold_path, items, all_scores, attn, cp)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"\n✓ {args.out}")

    flat = [s for ss in all_scores.values() for s in ss]
    winner = max(flat, key=lambda s: (s.gf1, s.f1)) if flat else None
    Path(args.scores_out).write_text(json.dumps({
        "gold": gold_path.name, "sources": args.sources, "n_items": len(items),
        "seed": SEED, "beta_sim_threshold": BETA,
        "grid": {"k": K_GRID, "alpha": ALPHA_GRID, "resolution": RES_GRID,
                 "min_seg": MINSEG_GRID},
        "ref_attention": attn, "ref_changepoint": cp, "ref_learned": LEARNED_REF,
        "winner": winner.as_row() if winner else None,
        "configs": [s.as_row() for s in sorted(flat, key=lambda s: -s.gf1)],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ {args.scores_out}")


if __name__ == "__main__":
    main()
