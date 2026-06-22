"""Benchmark PETITS MODÈLES pour le VRAI but : multi-label de thèmes par avis.

    uv run --extra contender --extra embed-contender \
        python -m eval.segmentation.small_models
        [--gold eval/segmentation/gold_large.json]
        [--embedders nomic-v2,e5-small]
        [--ollama llama3.2:3b,qwen3:4b]
        [--folds 5] [--seed 0] [--theme-batch 6]
        [--no-classifier] [--no-ollama]
        [--out eval/segmentation/small_models_report.md]

Question : un PETIT modèle LOCAL tient-il près de la réf Mistral-small
(micro-F1 **0.928**, macro 0.935, exact-set 73% — `llm_report.md`, NON relancée)
pour le multi-label de thèmes, à un coût SCALABLE (rapide, local, gratuit) ?

Deux familles de candidats, mêmes métriques que Mistral (micro/macro-F1, exact-set)
+ vitesse d'inférence (ms/avis) :

**Candidat 1 — classifieur multi-label sur embedding** (le cheval scalable) :
  vecteur d'avis ENTIER (pooling prod, `embed_docs`) en entrée d'une régression
  logistique one-vs-rest ET d'un petit MLP. Cible = ensemble des thèmes (8 classes).
  CV stratifiée PAR AVIS (sur le nb de thèmes → mêmes proportions mono/multi par
  pli, aucune fuite). Probas hors-pli (OOF) → seuil PAR CLASSE calé pour max-F1.
  Inférence quasi-nulle (embed + produit matriciel).

**Candidat 2 — petit LLM LOCAL via Ollama** (filet souverain) : `llama3.2:3b`
  (non-raisonneur) et `qwen3:4b` (raisonneur — pensée désactivée via `think:false`).
  MÊME prompt fermé que Mistral (réutilise `llm_seg.theme_sys/theme_prompt`),
  choix fermé sur les 8 thèmes. ms/avis CPU = clé de scalabilité sur le VPS.

Honnêteté : le classifieur est entraîné sur NOS 8 thèmes — en prod la taxo est
par-consultation (il faudrait un échantillon labellisé par consultation). Les
seuils sont calés sur les probas OOF servant aussi au score (léger optimisme,
disclosé). Vitesse Ollama indicative (CPU partagé). Cache disque pour Ollama
(relances gratuites, coût à froid mémorisé) ; embeddings re-timés à chaud.

ÉCRIT UNIQUEMENT dans `eval/segmentation/` (small_models_report.md,
small_models_scores.json, .cache/ollama/).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from eval.segmentation.llm_seg import (
    Prepared,
    ThemeScore,
    parse_json_object,
    prepare,
    score_themes,
    theme_prompt,
)
from eval.segmentation.seg_bench import load_gold

HERE = Path(__file__).resolve().parent
DEFAULT_GOLD = HERE / "gold_large.json"
DEFAULT_REPORT = HERE / "small_models_report.md"
DEFAULT_SCORES = HERE / "small_models_scores.json"
OLLAMA_CACHE = HERE / ".cache" / "ollama"
OLLAMA_URL = "http://localhost:11434/api/chat"

# Référence Mistral-small (llm_report.md — NON relancée, citée).
MISTRAL = {"micro_f1": 0.928, "macro_f1": 0.9346, "exact_set": 0.73,
           "model": "mistral-small-latest"}


# --------------------------------------------------------------------------- #
# Candidat 1 — classifieur multi-label sur embedding
# --------------------------------------------------------------------------- #
@dataclass
class ClfResult:
    embedder: str
    head: str                       # "logreg" | "mlp"
    score: ThemeScore
    thresholds: dict                # par classe
    embed_ms_per_avis: float        # coût d'embedding (dominant)
    predict_ms_per_avis: float      # coût de la tête (quasi-nul)
    fit_seconds: float              # entraînement total CV (info)


def _multihot(prepared: list[Prepared], labels: list[str]) -> np.ndarray:
    idx = {t: i for i, t in enumerate(labels)}
    Y = np.zeros((len(prepared), len(labels)), dtype=np.int8)
    for r, p in enumerate(prepared):
        for t in p.gold_themes:
            if t in idx:
                Y[r, idx[t]] = 1
    return Y


def _tune_thresholds(Y: np.ndarray, P: np.ndarray, labels: list[str]) -> np.ndarray:
    """Seuil par classe maximisant le F1 de cette classe sur les probas OOF.

    Léger optimisme assumé : calé sur les mêmes OOF que le score final. Pas de
    fuite d'entraînement (le modèle n'a jamais vu son pli de validation), mais le
    choix du seuil voit le jeu — disclosé dans le rapport.
    """
    thr = np.full(len(labels), 0.5)
    grid = np.linspace(0.05, 0.95, 19)
    for c in range(len(labels)):
        best_f1, best_t = -1.0, 0.5
        y = Y[:, c]
        for t in grid:
            pred = (P[:, c] >= t).astype(np.int8)
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 1.0
            rec = tp / (tp + fn) if (tp + fn) else 1.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thr[c] = best_t
    return thr


def _oof_proba(make_head, X: np.ndarray, Y: np.ndarray, strat: np.ndarray,
               folds: int, seed: int) -> tuple[np.ndarray, float]:
    """Probas hors-pli [N, K] via CV stratifiée par avis. Renvoie (P, fit_seconds)."""
    from sklearn.model_selection import StratifiedKFold

    P = np.zeros_like(Y, dtype=np.float64)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fit_seconds = 0.0
    for tr, va in skf.split(X, strat):
        head = make_head()
        t0 = time.perf_counter()
        head.fit(X[tr], Y[tr])
        fit_seconds += time.perf_counter() - t0
        P[va] = _predict_proba_matrix(head, X[va], Y.shape[1])
    return P, fit_seconds


def _predict_proba_matrix(head, X: np.ndarray, k: int) -> np.ndarray:
    """Probas [n, k] robustes au cas où une classe est absente d'un pli."""
    proba = head.predict_proba(X)
    # MLPClassifier multilabel → ndarray [n, k] direct.
    if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] == k:
        return proba
    # OneVsRest → liste de [n, n_classes_c] (souvent [n,2] = [P(0),P(1)]).
    out = np.zeros((X.shape[0], k), dtype=np.float64)
    for c in range(k):
        pc = proba[c] if isinstance(proba, (list, tuple)) else proba[:, c]
        pc = np.asarray(pc)
        out[:, c] = pc[:, 1] if pc.ndim == 2 and pc.shape[1] == 2 else pc.ravel()
    return out


def _make_logreg():
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    return OneVsRestClassifier(
        LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
        n_jobs=1,
    )


def _make_mlp(seed: int):
    from sklearn.neural_network import MLPClassifier

    # Sans early-stopping : il rognerait un pli de validation sur un jeu déjà petit
    # et s'arrêterait bien trop tôt (micro-F1 s'effondre à ~0.45). alpha régularise.
    return MLPClassifier(hidden_layer_sizes=(128,), max_iter=800,
                         early_stopping=False, alpha=1e-3, random_state=seed)


def _pred_sets(P: np.ndarray, thr: np.ndarray, ids: list[str],
               labels: list[str]) -> dict[str, set[str]]:
    hyps: dict[str, set[str]] = {}
    for r, _id in enumerate(ids):
        hyps[_id] = {labels[c] for c in range(len(labels)) if P[r, c] >= thr[c]}
    return hyps


def run_classifier(prepared: list[Prepared], labels: list[str], embedder: str,
                   folds: int, seed: int) -> list[ClfResult]:
    from eval.segmentation.embeddings import embed_docs

    texts = [p.item.text for p in prepared]
    ids = [p.item.id for p in prepared]
    Y = _multihot(prepared, labels)
    strat = Y.sum(axis=1)  # nb de thèmes (1/2/3) → proportions mono/multi par pli

    embed_docs(texts[:1], model_id=embedder)  # warm-up : sort le chargement modèle du timing
    t0 = time.perf_counter()
    X = embed_docs(texts, model_id=embedder).astype(np.float64)
    embed_ms = 1000.0 * (time.perf_counter() - t0) / len(texts)

    results: list[ClfResult] = []
    heads = {"logreg": _make_logreg, "mlp": lambda: _make_mlp(seed)}
    for name, make in heads.items():
        P, fit_s = _oof_proba(make, X, Y, strat, folds, seed)
        thr = _tune_thresholds(Y, P, labels)
        hyps = _pred_sets(P, thr, ids, labels)
        sc = score_themes(prepared, hyps, labels)
        # Coût d'inférence de la tête : refit plein jeu, time un predict_proba.
        head = make()
        head.fit(X, Y)
        t1 = time.perf_counter()
        _predict_proba_matrix(head, X, len(labels))
        predict_ms = 1000.0 * (time.perf_counter() - t1) / len(texts)
        results.append(ClfResult(
            embedder=embedder, head=name, score=sc,
            thresholds={labels[c]: round(float(thr[c]), 2) for c in range(len(labels))},
            embed_ms_per_avis=round(embed_ms, 2),
            predict_ms_per_avis=round(predict_ms, 4),
            fit_seconds=round(fit_s, 2),
        ))
        print(f"  [{embedder}/{name}] micro-F1={sc.micro_f1:.3f} macro-F1={sc.macro_f1:.3f} "
              f"exact={sc.exact_set:.3f} embed={embed_ms:.1f}ms/avis")
    return results


# --------------------------------------------------------------------------- #
# Candidat 2 — petit LLM local via Ollama (même prompt fermé que Mistral)
# --------------------------------------------------------------------------- #
@dataclass
class OllamaStats:
    calls: int = 0
    cache_hits: int = 0
    errors: int = 0
    cold_seconds: float = 0.0       # latence à froid (miss + 1er coût mémorisé)
    cold_calls: int = 0
    eval_tokens: int = 0


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _ollama_key(model: str, messages: list[dict]) -> Path:
    blob = json.dumps([model, messages], ensure_ascii=False, sort_keys=True)
    h = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]
    return OLLAMA_CACHE / f"{h}.json"


def ollama_chat(messages: list[dict], *, model: str, stats: OllamaStats,
                timeout: float = 600.0) -> str | None:
    """Chat Ollama (JSON mode, temp 0, pensée désactivée) avec cache disque."""
    import httpx

    OLLAMA_CACHE.mkdir(parents=True, exist_ok=True)
    cpath = _ollama_key(model, messages)
    if cpath.exists():
        rec = json.loads(cpath.read_text(encoding="utf-8"))
        stats.cache_hits += 1
        stats.cold_calls += 1
        stats.cold_seconds += float(rec.get("seconds", 0.0))
        stats.eval_tokens += int(rec.get("eval_count", 0))
        return rec["content"]

    payload = {
        "model": model, "messages": messages, "stream": False,
        "format": "json", "think": False,
        "options": {"temperature": 0.0, "num_ctx": 4096},
    }
    t0 = time.monotonic()
    try:
        r = httpx.post(OLLAMA_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001 — on rapporte, on ne masque pas
        stats.errors += 1
        print(f"  ⚠️ ollama[{model}]: {type(exc).__name__}")
        return None
    elapsed = time.monotonic() - t0
    content = (data.get("message") or {}).get("content") or ""
    content = _THINK_RE.sub("", content).strip()
    eval_count = int(data.get("eval_count", 0))
    stats.calls += 1
    stats.cold_calls += 1
    stats.cold_seconds += elapsed
    stats.eval_tokens += eval_count
    cpath.write_text(json.dumps(
        {"content": content, "seconds": round(elapsed, 3), "eval_count": eval_count},
        ensure_ascii=False), encoding="utf-8")
    return content


def run_ollama_themes(prepared: list[Prepared], taxonomy: dict[str, str], model: str,
                      batch_size: int, stats: OllamaStats) -> dict[str, set[str]]:
    valid = set(taxonomy)
    hyps: dict[str, set[str]] = {}
    batches = [prepared[i:i + batch_size] for i in range(0, len(prepared), batch_size)]
    for bi, batch in enumerate(batches):
        raw = ollama_chat(theme_prompt(batch, taxonomy), model=model, stats=stats)
        obj = parse_json_object(raw or "")
        if (bi + 1) % 20 == 0 or bi == 0 or obj is None:
            print(f"  [{model}] thèmes lot {bi + 1}/{len(batches)} ({len(batch)} avis)"
                  f"{' ⚠️ parse échec' if obj is None else ''}")
        for p in batch:
            val = None
            if obj is not None:
                val = obj.get(p.item.id)
                # Batch=1 : les petits modèles ré-écrivent souvent la clé (« avis_id »,
                # texte tronqué…) au lieu de l'id. Avec un seul avis, la liste de thèmes
                # est non ambiguë → on prend la 1re valeur-liste quelle que soit la clé.
                if val is None and len(batch) == 1:
                    for v in obj.values():
                        if isinstance(v, list):
                            val = v
                            break
            hyp: set[str] = set()
            if isinstance(val, list):
                hyp = {str(x).strip() for x in val if str(x).strip() in valid}
            hyps[p.item.id] = hyp
    return hyps


@dataclass
class OllamaResult:
    model: str
    score: ThemeScore
    ms_per_avis: float
    stats: OllamaStats
    batch_size: int


def run_ollama(prepared: list[Prepared], taxonomy: dict[str, str], labels: list[str],
               model: str, batch_size: int) -> OllamaResult:
    stats = OllamaStats()
    hyps = run_ollama_themes(prepared, taxonomy, model, batch_size, stats)
    sc = score_themes(prepared, hyps, labels)
    ms = 1000.0 * stats.cold_seconds / len(prepared) if prepared else 0.0
    print(f"  [{model}] micro-F1={sc.micro_f1:.3f} macro-F1={sc.macro_f1:.3f} "
          f"exact={sc.exact_set:.3f} {ms:.0f}ms/avis ({stats.errors} err)")
    return OllamaResult(model=model, score=sc, ms_per_avis=round(ms, 1),
                        stats=stats, batch_size=batch_size)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _md_table(rows, cols):
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows)
    return "\n".join([head, sep, body])


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def build_report(gold_path: Path, n_items: int, n_mono: int, n_multi: int,
                 labels: list[str], clf: list[ClfResult], olm: list[OllamaResult],
                 folds: int, seed: int, theme_batch: int) -> str:
    L = []
    L.append("# Petits modèles pour le multi-label de thèmes — rapport\n")
    L.append(f"*Jeu : `{gold_path.name}` — N={n_items} ({n_mono} mono, {n_multi} multi), "
             f"{len(labels)} thèmes. Réf **Mistral-small** (`llm_report.md`, NON relancée) : "
             f"micro-F1 **{MISTRAL['micro_f1']}**, macro {MISTRAL['macro_f1']}, "
             f"exact-set {_pct(MISTRAL['exact_set'])}. Seed {seed}, CPU.*\n")
    L.append("**Question** : un petit modèle LOCAL atteint-il une qualité « assez proche "
             "de 0.93 » à coût SCALABLE (rapide, gratuit, données qui ne sortent pas) ? "
             "Si oui → résout coût + souveraineté + échelle, ET rend la segmentation "
             "inutile (on a directement l'ensemble des thèmes par avis).\n")

    # --- Scorecard maître ---
    L.append("## Scorecard — qualité × coût/vitesse\n")
    rows = []
    rows.append({"Modèle": f"**Mistral-small** (réf, API)", "type": "LLM cloud",
                 "micro-F1": MISTRAL["micro_f1"], "macro-F1": MISTRAL["macro_f1"],
                 "exact-set": _pct(MISTRAL["exact_set"]), "ms/avis": "~230*",
                 "local": "non", "données sortent": "**oui**"})
    for r in clf:
        s = r.score
        rows.append({
            "Modèle": f"clf {r.head} / {r.embedder}", "type": "embed+tête",
            "micro-F1": round(s.micro_f1, 3), "macro-F1": round(s.macro_f1, 3),
            "exact-set": _pct(s.exact_set),
            "ms/avis": f"{r.embed_ms_per_avis + r.predict_ms_per_avis:.1f}",
            "local": "**oui**", "données sortent": "non"})
    for r in olm:
        s = r.score
        rows.append({
            "Modèle": f"{r.model} (Ollama)", "type": "LLM local",
            "micro-F1": round(s.micro_f1, 3), "macro-F1": round(s.macro_f1, 3),
            "exact-set": _pct(s.exact_set), "ms/avis": f"{r.ms_per_avis:.0f}",
            "local": "**oui**", "données sortent": "non"})
    cols = ["Modèle", "type", "micro-F1", "macro-F1", "exact-set", "ms/avis",
            "local", "données sortent"]
    L.append(_md_table(rows, cols) + "\n")
    L.append("*\\* Mistral ms/avis = ~70s cumulés / 305 avis ≈ 230 ms/avis amorti "
             "(batché 12/appel, réseau UE) — cf. `llm_report.md`. ms/avis classifieur = "
             "embedding (dominant) + tête (quasi-nul). ms/avis Ollama = latence à froid "
             "cumulée / N (CPU partagé, indicatif).*\n")

    # --- Classifieur : détail ---
    if clf:
        L.append("## Candidat 1 — classifieur multi-label sur embedding (le cheval)\n")
        L.append(f"Vecteur d'avis ENTIER (pooling prod `embed_docs`) → tête multi-label. "
                 f"CV stratifiée par avis ({folds} plis, stratifiés sur le nb de thèmes), "
                 f"probas hors-pli, seuil PAR CLASSE calé pour max-F1. "
                 f"LogReg one-vs-rest (`class_weight=balanced`) et MLP (1×128, "
                 f"early-stopping).\n")
        drows = []
        for r in clf:
            s = r.score
            drows.append({
                "embedder": r.embedder, "tête": r.head,
                "micro-P": round(s.micro_p, 3), "micro-R": round(s.micro_r, 3),
                "micro-F1": round(s.micro_f1, 3), "macro-F1": round(s.macro_f1, 3),
                "exact-set": _pct(s.exact_set),
                "embed ms/avis": r.embed_ms_per_avis,
                "tête ms/avis": r.predict_ms_per_avis})
        L.append(_md_table(drows, ["embedder", "tête", "micro-P", "micro-R", "micro-F1",
                                   "macro-F1", "exact-set", "embed ms/avis",
                                   "tête ms/avis"]) + "\n")
        best = max(clf, key=lambda r: r.score.micro_f1)
        L.append("### F1 par thème — meilleure tête "
                 f"(`{best.head}` / `{best.embedder}`, micro-F1 {best.score.micro_f1:.3f})\n")
        per = best.score.per_theme
        trows = [{"thème": t, "P": round(per[t]["p"], 3), "R": round(per[t]["r"], 3),
                  "F1": round(per[t]["f1"], 3), "seuil": best.thresholds.get(t),
                  "TP": per[t]["tp"], "FP": per[t]["fp"], "FN": per[t]["fn"]}
                 for t in sorted(labels, key=lambda x: -per[x]["f1"])]
        L.append(_md_table(trows, ["thème", "P", "R", "F1", "seuil", "TP", "FP", "FN"]) + "\n")

    # --- Ollama : détail ---
    if olm:
        L.append("## Candidat 2 — petit LLM local via Ollama (filet souverain)\n")
        L.append(f"MÊME prompt fermé que Mistral (`llm_seg.theme_prompt`), choix fermé sur "
                 f"les {len(labels)} thèmes, JSON mode, température 0, pensée désactivée "
                 f"(`think:false`). {theme_batch} avis/appel. Cache disque "
                 f"`.cache/ollama/` (relances gratuites, coût à froid mémorisé).\n")
        orows = []
        for r in olm:
            s = r.score
            orows.append({
                "modèle": r.model,
                "micro-P": round(s.micro_p, 3), "micro-R": round(s.micro_r, 3),
                "micro-F1": round(s.micro_f1, 3), "macro-F1": round(s.macro_f1, 3),
                "exact-set": _pct(s.exact_set), "ms/avis": f"{r.ms_per_avis:.0f}",
                "tokens générés": r.stats.eval_tokens, "erreurs": r.stats.errors})
        L.append(_md_table(orows, ["modèle", "micro-P", "micro-R", "micro-F1", "macro-F1",
                                   "exact-set", "ms/avis", "tokens générés", "erreurs"]) + "\n")
        for r in olm:
            s = r.score
            per = s.per_theme
            L.append(f"### F1 par thème — `{r.model}`\n")
            trows = [{"thème": t, "P": round(per[t]["p"], 3), "R": round(per[t]["r"], 3),
                      "F1": round(per[t]["f1"], 3), "TP": per[t]["tp"],
                      "FP": per[t]["fp"], "FN": per[t]["fn"]}
                     for t in sorted(labels, key=lambda x: -per[x]["f1"])]
            L.append(_md_table(trows, ["thème", "P", "R", "F1", "TP", "FP", "FN"]) + "\n")

    # --- Verdict ---
    L.append("## Verdict — un petit modèle local tient-il près de 0.93 à coût scalable ?\n")
    best_local = None
    cands = [("clf " + r.head + "/" + r.embedder, r.score.micro_f1, r.score.exact_set,
              r.embed_ms_per_avis + r.predict_ms_per_avis) for r in clf]
    cands += [(r.model, r.score.micro_f1, r.score.exact_set, r.ms_per_avis) for r in olm]
    if cands:
        best_local = max(cands, key=lambda c: c[1])
        name, f1, ex, ms = best_local
        gap = MISTRAL["micro_f1"] - f1
        close = gap <= 0.03
        L.append(
            f"- **Meilleur local : `{name}`** — micro-F1 **{f1:.3f}** "
            f"(exact-set {_pct(ex)}), soit **{gap:+.3f}** vs Mistral {MISTRAL['micro_f1']} "
            f"à **{ms:.1f} ms/avis**, 100% local, données qui ne sortent pas.\n")
        L.append(
            f"- **{'OUI' if close else 'PROCHE' if gap <= 0.05 else 'NON'}** — "
            f"{'à ≤0.03 de la réf' if close else f'écart {gap:.3f}'} : "
            f"{'un petit modèle local tient près de 0.93 à coût scalable' if close else 'à arbitrer selon la tolérance qualité/coût'}.\n")
    L.append(
        "- **Le classifieur sur embedding est le candidat scalable** : inférence "
        "dominée par l'embedding (déjà calculé en prod pour le clustering), tête "
        "quasi-gratuite, batch, aucune donnée qui sort. Mais il est entraîné sur NOS "
        "8 thèmes — **en prod la taxo est par-consultation**, donc il faudrait un "
        "échantillon labellisé (par LLM ou humain) par consultation pour le ré-entraîner. "
        "C'est le compromis : très bon marché à l'inférence, mais coût d'amorçage par "
        "consultation.\n")
    L.append(
        "- **Les LLM locaux (Ollama)** ne demandent AUCUN entraînement (zéro-shot, "
        "taxo passée dans le prompt → générique par consultation comme Mistral), mais "
        "la latence CPU (~×10–×100 le classifieur) est le prix de la scalabilité sur VPS "
        "partagé. Le raisonneur `qwen3:4b` (pensée coupée) vs le non-raisonneur "
        "`llama3.2:3b` : voir l'écart qualité/vitesse ci-dessus.\n")
    L.append(
        "- **Honnêteté** : seuils du classifieur calés sur les probas OOF servant aussi "
        "au score (léger optimisme, pas de fuite d'entraînement). Vitesse Ollama "
        "indicative (CPU partagé, 1 requête à la fois). Mistral non relancé (chiffres "
        "`llm_report.md`).\n")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Bench petits modèles — multi-label thèmes.")
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--embedders", default="nomic-v2,e5-small")
    ap.add_argument("--ollama", default="llama3.2:3b,qwen3:4b")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--theme-batch", type=int, default=1,
                    help="avis/appel Ollama (1 = mapping non ambigu + vraie latence/avis)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-classifier", action="store_true")
    ap.add_argument("--no-ollama", action="store_true")
    ap.add_argument("--out", default=str(DEFAULT_REPORT))
    ap.add_argument("--scores-out", default=str(DEFAULT_SCORES))
    args = ap.parse_args()

    gold_path = Path(args.gold)
    items, meta = load_gold(gold_path)
    taxonomy = meta.get("taxonomy", {})
    if not taxonomy:
        labels = sorted({t for it in items for t in it.seg_themes if t and t != "?"})
        taxonomy = {t: t for t in labels}
    labels = list(taxonomy)
    if args.limit:
        items = items[:args.limit]
    prepared = prepare(items)
    n_mono = sum(1 for p in prepared if p.item.type == "mono")
    n_multi = sum(1 for p in prepared if p.item.type == "multi")
    print(f"gold: {gold_path.name} — {len(prepared)} items ({n_mono} mono, {n_multi} multi), "
          f"{len(labels)} thèmes")

    clf: list[ClfResult] = []
    if not args.no_classifier:
        for emb in [e.strip() for e in args.embedders.split(",") if e.strip()]:
            print(f"CANDIDAT 1 — classifieur / {emb}…")
            try:
                clf.extend(run_classifier(prepared, labels, emb, args.folds, args.seed))
            except Exception as exc:  # noqa: BLE001 — rapporte l'échec embedder
                print(f"  ⚠️ {emb} échec: {type(exc).__name__}: {exc}")

    olm: list[OllamaResult] = []
    if not args.no_ollama:
        for model in [m.strip() for m in args.ollama.split(",") if m.strip()]:
            print(f"CANDIDAT 2 — Ollama / {model}…")
            olm.append(run_ollama(prepared, taxonomy, labels, model, args.theme_batch))

    report = build_report(gold_path, len(prepared), n_mono, n_multi, labels,
                          clf, olm, args.folds, args.seed, args.theme_batch)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"✓ {args.out}")

    Path(args.scores_out).write_text(json.dumps({
        "gold": gold_path.name, "n_items": len(prepared), "n_mono": n_mono,
        "n_multi": n_multi, "folds": args.folds, "seed": args.seed,
        "reference_mistral": MISTRAL,
        "classifier": [{
            "embedder": r.embedder, "head": r.head,
            "micro_P": round(r.score.micro_p, 4), "micro_R": round(r.score.micro_r, 4),
            "micro_F1": round(r.score.micro_f1, 4), "macro_F1": round(r.score.macro_f1, 4),
            "exact_set": round(r.score.exact_set, 4),
            "embed_ms_per_avis": r.embed_ms_per_avis,
            "predict_ms_per_avis": r.predict_ms_per_avis,
            "fit_seconds": r.fit_seconds, "thresholds": r.thresholds,
            "per_theme": {t: {k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in d.items()} for t, d in r.score.per_theme.items()},
        } for r in clf],
        "ollama": [{
            "model": r.model, "batch_size": r.batch_size,
            "micro_P": round(r.score.micro_p, 4), "micro_R": round(r.score.micro_r, 4),
            "micro_F1": round(r.score.micro_f1, 4), "macro_F1": round(r.score.macro_f1, 4),
            "exact_set": round(r.score.exact_set, 4), "ms_per_avis": r.ms_per_avis,
            "cold_seconds": round(r.stats.cold_seconds, 2),
            "eval_tokens": r.stats.eval_tokens,
            "calls": r.stats.calls, "cache_hits": r.stats.cache_hits,
            "errors": r.stats.errors,
            "per_theme": {t: {k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in d.items()} for t, d in r.score.per_theme.items()},
        } for r in olm],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ {args.scores_out}")


if __name__ == "__main__":
    main()
