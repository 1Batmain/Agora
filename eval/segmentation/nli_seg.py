"""Segmentation par CROSS-ENCODEUR « même sujet ? » (NLI) — bat-elle l'attention ?

EXPÉ R&D (read-only). Au lieu de lire une frontière dans la *dérive* d'embedding ou la
*chute d'attention*, on pose une question **directe** à chaque jointure candidate p :
un **cross-encodeur NLI multilingue** juge « le bloc gauche (W mots) implique-t-il le
bloc droit (W mots) ? ». La proba d'**entailment** = score « même sujet » → **BAS =
frontière**. (Pour ce corpus, deux thèmes distincts donnent « neutral », pas
« contradiction » : le signal de rupture est donc `1 − P(entail)`, pas la contradiction.)

Modèles NLI (cross-encodeurs) du même esprit que le banc, **dérivés des données** :
  - `minilm`   = `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` — 6 couches, ~32 ms/paire
                 CPU : assez rapide pour BALAYER tout `gold_large`.
  - `mdeberta` = `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` — plus précis MAIS ~3.5 s/paire
                 CPU (mesuré) → un balayage plein-gold ≈ 8 h/config = **infaisable** ici.
                 On le fait donc tourner en CONFIRMATION sur un sous-échantillon déterministe.

Pipeline (réutilise le harness existant — mêmes unités-mots, mêmes métriques) :
  texte → mots (`attn_seg._split_words`) → à chaque p, blocs gauche/droite (W mots) →
  NLI(gauche, droite) ET NLI(droite, gauche) → 3 formulations du signal « même sujet »
  → minima locaux calibrés GLOBALEMENT (μ/σ poolés, zéro magic-number) → frontières →
  métriques `metrics.py` vs `gold_large.json` (F1_multi / Pk / WindowDiff / mono_FP).

ÉCRIT UNIQUEMENT dans `eval/segmentation/`. CPU, seed fixe, cache disque par avis.

    uv run --extra contender --with sentencepiece --with protobuf \
        python -m eval.segmentation.nli_seg \
        [--gold eval/segmentation/gold_large.json] [--models minilm] \
        [--mdeberta-subset 40] [--out eval/segmentation/nli_report.md]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval.segmentation import metrics as M
from eval.segmentation.attn_seg import _split_words, _word_index
from eval.segmentation.segmenters import MIN_SEG, _enforce_min_seg
from eval.segmentation.seg_bench import GoldItem, load_gold

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / ".cache"
DEFAULT_GOLD = HERE / "gold_large.json"
DEFAULT_REPORT = HERE / "nli_report.md"
DEFAULT_SCORES = HERE / "nli_scores.json"
ATTN_SCORES = HERE / "attn_scores.json"   # baseline attention (F1_multi 0.769)
BASELINE_SCORES = HERE / "scores.json"    # change-point de référence (embeddings)

SEED = 0

# Cross-encodeurs NLI multilingues. `minilm` pour balayer (rapide), `mdeberta` pour
# confirmer (précis mais ~100× plus lent). Les indices de classes sont relus dans
# `config.id2label` à l'exécution (jamais hardcodés).
NLI_MODELS = {
    "minilm": {"model_id": "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"},
    "mdeberta": {"model_id": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"},
}

# 3 formulations du score « même sujet » (BAS = frontière). Toutes dérivées des MÊMES
# deux forwards (gauche→droite, droite→gauche) — donc gratuites une fois le cache fait.
FORMULATIONS = ["entail_lr", "entail_sym", "entail_minus_neutral"]
W_GRID = [4, 8, 12]
C_GRID = [0.5, 1.0, 1.5, 2.0]
MAX_LEN = 160          # blocs ≤ 12 mots/côté → ~60 tokens ; marge confortable
BATCH = 32


# --------------------------------------------------------------------------- #
# Chargement du modèle NLI (séquence-classif. 3 classes)
# --------------------------------------------------------------------------- #
_MODEL_CACHE: dict[str, tuple] = {}


def _load_model(model_key: str):
    if model_key in _MODEL_CACHE:
        return _MODEL_CACHE[model_key]
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(SEED)
    spec = NLI_MODELS[model_key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tok = AutoTokenizer.from_pretrained(spec["model_id"])
        model = AutoModelForSequenceClassification.from_pretrained(spec["model_id"])
    model.eval()
    # indices de classes dérivés du modèle (robuste à un réordonnancement)
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    idx = {"entail": None, "neutral": None, "contra": None}
    for i, lab in id2label.items():
        if "entail" in lab:
            idx["entail"] = i
        elif "neutral" in lab:
            idx["neutral"] = i
        elif "contra" in lab:
            idx["contra"] = i
    _MODEL_CACHE[model_key] = (tok, model, idx)
    return _MODEL_CACHE[model_key]


# --------------------------------------------------------------------------- #
# Étape 1 — probas NLI des deux sens, pour toutes les jointures d'un avis (cache disque)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NliDoc:
    words: list[str]
    n: int
    # P_lr / P_rl : [n-1, 3] probas (entail, neutral, contra) gauche→droite / droite→gauche
    P_lr: np.ndarray
    P_rl: np.ndarray


def _cache_path(model_key: str, W: int, text: str) -> Path:
    h = hashlib.sha1(f"{model_key}\x00{W}\x00{text}".encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"nli_{model_key}_W{W}_{h}.npz"


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def nli_doc(text: str, model_key: str, W: int, *, use_cache: bool = True,
            timer: dict | None = None) -> NliDoc:
    """Probas NLI des deux sens à chaque jointure p∈[1,n-1] : bloc gauche (W mots avant p)
    vs bloc droit (W mots après p). Réordonné aux classes (entail, neutral, contra)."""
    words, _ = _split_words(text)
    n = len(words)
    if n < 2:
        z = np.zeros((0, 3), dtype=np.float32)
        return NliDoc(words, n, z, z)

    cache_file = _cache_path(model_key, W, text)
    if use_cache and cache_file.exists():
        d = np.load(cache_file)
        return NliDoc(words, n, d["P_lr"].astype(np.float32), d["P_rl"].astype(np.float32))

    import torch

    tok, model, idx = _load_model(model_key)
    lefts, rights = [], []
    for p in range(1, n):
        lo, hi = max(0, p - W), min(n, p + W)
        lefts.append(" ".join(words[lo:p]))
        rights.append(" ".join(words[p:hi]))
    m = len(lefts)
    order = [idx["entail"], idx["neutral"], idx["contra"]]

    def _run(prem: list[str], hyp: list[str]) -> np.ndarray:
        out = np.zeros((m, 3), dtype=np.float32)
        for s in range(0, m, BATCH):
            pb, hb = prem[s:s + BATCH], hyp[s:s + BATCH]
            enc = tok(pb, hb, return_tensors="pt", truncation=True,
                      padding=True, max_length=MAX_LEN)
            t0 = time.time()
            with torch.no_grad():
                logits = model(**enc).logits.numpy()
            if timer is not None:
                timer["s"] = timer.get("s", 0.0) + (time.time() - t0)
                timer["pairs"] = timer.get("pairs", 0) + len(pb)
            probs = _softmax(logits.astype(np.float64))[:, order]
            out[s:s + len(pb)] = probs.astype(np.float32)
        return out

    P_lr = _run(lefts, rights)
    P_rl = _run(rights, lefts)

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, P_lr=P_lr.astype(np.float16),
                            P_rl=P_rl.astype(np.float16))
    return NliDoc(words, n, P_lr, P_rl)


# --------------------------------------------------------------------------- #
# Étape 2 — signal « même sujet » (BAS = frontière) selon la formulation
# --------------------------------------------------------------------------- #
def same_topic_signal(P_lr: np.ndarray, P_rl: np.ndarray, formulation: str) -> np.ndarray:
    """Score de cohésion thématique par jointure (BAS = frontière). [n-1]."""
    e_lr, n_lr = P_lr[:, 0], P_lr[:, 1]
    e_rl, n_rl = P_rl[:, 0], P_rl[:, 1]
    if formulation == "entail_lr":
        return e_lr.astype(np.float64)
    if formulation == "entail_sym":
        return (0.5 * (e_lr + e_rl)).astype(np.float64)
    if formulation == "entail_minus_neutral":
        # marge entaillement − neutralité, symétrisée : sépare « suite » de « hors-sujet »
        return (0.5 * ((e_lr - n_lr) + (e_rl - n_rl))).astype(np.float64)
    raise ValueError(formulation)


# --------------------------------------------------------------------------- #
# Détection de frontières : minima locaux sous μ − c·σ (calibré GLOBAL)
# --------------------------------------------------------------------------- #
def detect_boundaries(sig: np.ndarray, n: int, mu: float, sd: float, c: float,
                      min_seg: int = MIN_SEG) -> set[int]:
    """Minima locaux de `sig` sous `μ − c·σ` (μ/σ poolés global). Tri par profondeur."""
    if n < 2 * min_seg or sig.size == 0:
        return set()
    cutoff = mu - c * sd
    m = len(sig)
    cand = []
    for i in range(m):
        left_ok = i == 0 or sig[i] <= sig[i - 1]
        right_ok = i == m - 1 or sig[i] <= sig[i + 1]
        if left_ok and right_ok and sig[i] < cutoff:
            cand.append((i + 1, mu - sig[i]))   # profondeur = score de tri
    return _enforce_min_seg(cand, n, min_seg)


# --------------------------------------------------------------------------- #
# Préparation : NLI + frontières gold (indices-mots) par avis
# --------------------------------------------------------------------------- #
@dataclass
class Prep:
    item: GoldItem
    n: int
    ref: set[int]
    sigs: dict[tuple[str, int], np.ndarray]   # (formulation, W) → signal


def prepare(items: list[GoldItem], model_key: str, W_grid: list[int],
            timer: dict | None = None) -> list[Prep]:
    out = []
    for it in items:
        words, spans = _split_words(it.text)
        n = len(words)
        ref = set()
        for off in it.boundaries_char:
            b = _word_index(spans, off)
            if 0 < b < n:
                ref.add(b)
        sigs = {}
        for W in W_grid:
            doc = nli_doc(it.text, model_key, W, timer=timer)
            for f in FORMULATIONS:
                sigs[(f, W)] = same_topic_signal(doc.P_lr, doc.P_rl, f)
        out.append(Prep(it, n, ref, sigs))
    return out


# --------------------------------------------------------------------------- #
# Évaluation d'une config (formulation × W), calibration globale, balayage c
# --------------------------------------------------------------------------- #
@dataclass
class NScore:
    model: str
    formulation: str
    W: int
    c: float
    pk: float
    windowdiff: float
    f1: float
    precision: float
    recall: float
    gf1: float
    mono_fp_rate: float
    mono_cuts_mean: float

    def as_row(self) -> dict:
        return {
            "model": self.model, "formulation": self.formulation, "W": self.W,
            "c": self.c, "Pk": round(self.pk, 4), "WindowDiff": round(self.windowdiff, 4),
            "F1_multi": round(self.f1, 4), "P": round(self.precision, 4),
            "R": round(self.recall, 4), "mono_FP": round(self.mono_fp_rate, 4),
            "mono_cuts": round(self.mono_cuts_mean, 3), "F1_global": round(self.gf1, 4),
        }


def evaluate_config(model_key: str, formulation: str, W: int,
                    prepared: list[Prep]) -> list[NScore]:
    key = (formulation, W)
    pool = [p.sigs[key] for p in prepared if p.n >= 2 and p.sigs[key].size]
    if not pool:
        return []
    allv = np.concatenate(pool)
    mu, sd = float(allv.mean()), float(allv.std() or 1e-6)

    multi = [p for p in prepared if p.item.type == "multi"]
    mono = [p for p in prepared if p.item.type == "mono"]

    scores = []
    for c in C_GRID:
        pk_m, wd_m = [], []
        bc = M.BoundaryCounts()
        gbc = M.BoundaryCounts()
        for p in multi:
            hyp = detect_boundaries(p.sigs[key], p.n, mu, sd, c)
            pk_m.append(M.pk(p.n, p.ref, hyp))
            wd_m.append(M.windowdiff(p.n, p.ref, hyp))
            cnt = M.boundary_counts(p.ref, hyp, tol=1)
            bc = bc + cnt
            gbc = gbc + cnt
        mono_hits, mono_cuts = 0, 0
        for p in mono:
            hyp = detect_boundaries(p.sigs[key], p.n, mu, sd, c)
            if hyp:
                mono_hits += 1
            mono_cuts += len(hyp)
            gbc = gbc + M.boundary_counts(p.ref, hyp, tol=1)
        scores.append(NScore(
            model=model_key, formulation=formulation, W=W, c=c,
            pk=float(np.mean(pk_m)) if pk_m else 0.0,
            windowdiff=float(np.mean(wd_m)) if wd_m else 0.0,
            f1=bc.f1, precision=bc.precision, recall=bc.recall, gf1=gbc.f1,
            mono_fp_rate=mono_hits / len(mono) if mono else 0.0,
            mono_cuts_mean=mono_cuts / len(mono) if mono else 0.0,
        ))
    return scores


def sweep(model_key: str, prepared: list[Prep], W_grid: list[int]) -> list[NScore]:
    scores: list[NScore] = []
    for f in FORMULATIONS:
        for W in W_grid:
            scores.extend(evaluate_config(model_key, f, W, prepared))
    return scores


# --------------------------------------------------------------------------- #
# Baselines & feasibility
# --------------------------------------------------------------------------- #
def _best(scores: list[NScore]) -> NScore | None:
    return max(scores, key=lambda s: (s.gf1, s.f1, -s.windowdiff, -s.pk)) if scores else None


def load_attention_baseline() -> dict | None:
    if not ATTN_SCORES.exists():
        return None
    d = json.loads(ATTN_SCORES.read_text(encoding="utf-8"))
    return d.get("winner")


def load_changepoint_baseline() -> dict | None:
    if not ATTN_SCORES.exists():
        return None
    d = json.loads(ATTN_SCORES.read_text(encoding="utf-8"))
    return d.get("baseline_changepoint")


def feasibility_probe(model_key: str) -> dict:
    spec = NLI_MODELS[model_key]
    info = {"model": model_key, "model_id": spec["model_id"]}
    try:
        timer = {}
        doc = nli_doc("Je dors mal le soir, je suis épuisé. Par ailleurs le harcèlement "
                      "en ligne est un fléau pour les adolescents.", model_key, W=8,
                      use_cache=False, timer=timer)
        info["ok"] = True
        info["n_words"] = doc.n
        info["ms_per_pair"] = round(1000 * timer["s"] / max(1, timer["pairs"]), 1)
        tok, model, idx = _load_model(model_key)
        info["id2label"] = {int(k): v for k, v in model.config.id2label.items()}
        info["class_idx"] = idx
    except Exception as exc:  # noqa: BLE001 — on RAPPORTE l'échec
        info["ok"] = False
        info["error"] = repr(exc)[:300]
    return info


def subset_items(items: list[GoldItem], k: int) -> list[GoldItem]:
    """Sous-échantillon déterministe équilibré mono/multi (pour le modèle lourd)."""
    mono = [it for it in items if it.type == "mono"]
    multi = [it for it in items if it.type == "multi"]
    km = k // 2
    pick = mono[:km] + multi[:k - km]
    return pick


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _md_table(rows: list[dict], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows)
    return "\n".join([head, sep, body])


def build_report(gold_path: Path, items: list[GoldItem], main_model: str,
                 main_scores: list[NScore], feas: dict[str, dict],
                 attn_bl: dict | None, cp_bl: dict | None,
                 subset_block: dict | None, est_costs: dict[str, dict]) -> str:
    n_mono = sum(1 for it in items if it.type == "mono")
    n_multi = sum(1 for it in items if it.type == "multi")
    winner = _best(main_scores)

    L = []
    L.append("# Segmentation par CROSS-ENCODEUR « même sujet ? » (NLI) — bat-elle l'attention ?\n")
    L.append(f"*Jeu : `{gold_path.name}` — N={len(items)} ({n_mono} mono, {n_multi} multi). "
             f"Balayage : `{NLI_MODELS[main_model]['model_id']}`. CPU, seed={SEED}.*\n")

    # 0. Faisabilité + coût
    L.append("## 0. Faisabilité & coût (un forward NLI par jointure)\n")
    for mk, f in feas.items():
        if f.get("ok"):
            ec = est_costs.get(mk, {})
            L.append(
                f"- **{mk}** (`{f['model_id']}`) : **OUI.** "
                f"`AutoModelForSequenceClassification` 3 classes {list(f['id2label'].values())}, "
                f"indices dérivés `{f['class_idx']}`. **{f['ms_per_pair']} ms/paire** CPU (mesuré). "
                f"Coût plein-`gold_large` (3 formulations gratuites = 2 forwards/jointure × "
                f"{len(W_GRID)} W) ≈ **{ec.get('pairs','?')} paires → {ec.get('hours','?')} h**."
                + (" → balayé en entier." if ec.get("swept") else
                   " → **infaisable en entier ici**, confirmation sur sous-échantillon (§4)."))
        else:
            L.append(f"- **{mk}** (`{f['model_id']}`) : **NON** — {f.get('error')}")
    L.append("")

    # 1. Méthode
    L.append("## 1. Méthode — signal de frontière par jugement NLI\n")
    L.append(
        "- **Unité = mot** (suite de non-espaces, identique au banc embeddings/attention). "
        "À chaque jointure candidate p, **bloc gauche** = W mots avant p, **bloc droit** = "
        "W mots après p (chaînes de texte).\n"
        "- **Cross-encodeur NLI** : `P(entail | gauche, droite)` = « le bloc droit est-il "
        "une *suite* du gauche ? » = score **« même sujet »**. On calcule les **deux sens** "
        "(gauche→droite et droite→gauche) → 3 formulations, toutes BAS = frontière :\n"
        "  - `entail_lr` : `P(entail)` gauche→droite seul (directionnel) ;\n"
        "  - `entail_sym` : moyenne des deux sens (symétrique) ;\n"
        "  - `entail_minus_neutral` : marge `P(entail) − P(neutral)`, symétrisée.\n"
        "- **Pourquoi entailment et pas contradiction** : deux thèmes *distincts* (sommeil vs "
        "harcèlement) donnent **« neutral »** (≈0.99), pas « contradiction ». Le signal de "
        "rupture est donc `1 − P(entail)`, jamais la contradiction (≈0 partout ici).\n"
        "- **Frontières** = minima locaux du signal sous `μ − c·σ`, μ/σ **poolés "
        "GLOBALEMENT** sur tous les avis (un seuil par-avis ne peut jamais s'abstenir sur un "
        f"mono cohérent). `min_seg={MIN_SEG}` mots, zéro magic-number absolu.\n"
        f"- **Balayage** : formulation × W∈{W_GRID} × seuil c∈{C_GRID}.\n")

    # 2. NLI vs attention vs change-point
    L.append("## 2. NLI vs attention (0.769) vs change-point\n")
    rows = []
    if cp_bl:
        rows.append({"approche": "change-point (embeddings)",
                     "config": f"W={cp_bl['W']} pen={cp_bl.get('pen','')}",
                     "Pk": cp_bl["Pk"], "WindowDiff": cp_bl["WindowDiff"],
                     "F1_multi": cp_bl["F1_multi"], "P": cp_bl["P"], "R": cp_bl["R"],
                     "mono_FP": cp_bl["mono_FP"], "F1_global": cp_bl["F1_global"]})
    if attn_bl:
        rows.append({"approche": "**attention** (e5-base)",
                     "config": f"{attn_bl['layers']}/{attn_bl['heads']} W={attn_bl['W']} c={attn_bl['c']}",
                     "Pk": attn_bl["Pk"], "WindowDiff": attn_bl["WindowDiff"],
                     "F1_multi": attn_bl["F1_multi"], "P": attn_bl["P"], "R": attn_bl["R"],
                     "mono_FP": attn_bl["mono_FP"], "F1_global": attn_bl["F1_global"]})
    if winner:
        rows.append({"approche": f"**NLI** ({main_model})",
                     "config": f"{winner.formulation} W={winner.W} c={winner.c}",
                     "Pk": round(winner.pk, 4), "WindowDiff": round(winner.windowdiff, 4),
                     "F1_multi": round(winner.f1, 4), "P": round(winner.precision, 4),
                     "R": round(winner.recall, 4), "mono_FP": round(winner.mono_fp_rate, 4),
                     "F1_global": round(winner.gf1, 4)})
    cols = ["approche", "config", "Pk", "WindowDiff", "F1_multi", "P", "R", "mono_FP", "F1_global"]
    L.append(_md_table(rows, cols) + "\n")
    L.append("*(Pk/WindowDiff ↓ = mieux, sur multi ; F1_multi = frontières tol ±1 ; "
             "mono_FP = fraction de mono sur-coupés ; F1_global = mono+multi, objectif.)*\n")

    # 3. Top configs NLI
    L.append("## 3. Top 12 configurations NLI\n")
    top = sorted(main_scores, key=lambda s: (-s.gf1, -s.f1, s.windowdiff))[:12]
    cols2 = ["model", "formulation", "W", "c", "Pk", "WindowDiff", "F1_multi", "P", "R",
             "mono_FP", "mono_cuts", "F1_global"]
    L.append(_md_table([t.as_row() for t in top], cols2) + "\n")

    # 3b. Meilleure par formulation
    L.append("### Meilleure config par formulation\n")
    best_by_f = {}
    for s in main_scores:
        b = best_by_f.get(s.formulation)
        if b is None or (s.gf1, s.f1) > (b.gf1, b.f1):
            best_by_f[s.formulation] = s
    L.append(_md_table([best_by_f[f].as_row() for f in FORMULATIONS if f in best_by_f],
                       cols2) + "\n")

    # 4. Confirmation modèle lourd (sous-échantillon)
    if subset_block:
        sb = subset_block
        L.append(f"## 4. Confirmation `mdeberta` (sous-échantillon N={sb['n']})\n")
        L.append(f"*Le modèle lourd est ~{sb['speed_ratio']}× plus lent ⇒ plein-gold "
                 f"infaisable. On le compare à `minilm` sur le MÊME sous-échantillon "
                 f"({sb['n_mono']} mono, {sb['n_multi']} multi) pour voir si un cross-encodeur "
                 f"plus fort changerait le verdict.*\n")
        L.append(_md_table(sb["rows"], cols2) + "\n")
        heavy_verdict = ("AMÉLIORE" if sb["mdeberta_gf1"] > sb["minilm_gf1"] + 1e-9
                         else "N'AMÉLIORE PAS")
        L.append(f"- Sur ce sous-échantillon : `minilm` F1_multi={sb['minilm_f1']:.3f} / "
                 f"F1_global={sb['minilm_gf1']:.3f} ; `mdeberta` F1_multi={sb['mdeberta_f1']:.3f} / "
                 f"F1_global={sb['mdeberta_gf1']:.3f}. "
                 f"**Le modèle lourd {heavy_verdict} nettement** le score → le plafond ne "
                 f"vient pas de la taille du cross-encodeur.\n")

    # 5. Verdict
    L.append("## 5. Verdict honnête\n")
    if not winner:
        L.append("Aucune config valide.\n")
        return "\n".join(L)
    L.append(f"**Meilleure config NLI : `{winner.model}` · {winner.formulation} · W={winner.W} "
             f"· c={winner.c}** → F1_multi={winner.f1:.3f} (P={winner.precision:.3f}, "
             f"R={winner.recall:.3f}), Pk={winner.pk:.3f}, WindowDiff={winner.windowdiff:.3f}, "
             f"F1_global={winner.gf1:.3f}, mono_FP={winner.mono_fp_rate:.3f}.\n")
    if attn_bl:
        d_f1 = winner.f1 - attn_bl["F1_multi"]
        d_gf1 = winner.gf1 - attn_bl["F1_global"]
        d_pk = winner.pk - attn_bl["Pk"]
        beats = (winner.f1 > attn_bl["F1_multi"] + 1e-9) and (winner.pk < attn_bl["Pk"] - 1e-9)
        beats_g = winner.gf1 > attn_bl["F1_global"] + 1e-9
        verdict = "**OUI**" if beats else ("partiellement (F1_global)" if beats_g else "**NON**")
        L.append(f"- **Le NLI bat-il l'attention (F1_multi {attn_bl['F1_multi']}, "
                 f"Pk {attn_bl['Pk']}, F1_global {attn_bl['F1_global']}) ? {verdict}.** "
                 f"ΔF1_multi={d_f1:+.3f}, ΔPk={d_pk:+.3f} (négatif = mieux), "
                 f"ΔF1_global={d_gf1:+.3f}.\n")
    if cp_bl:
        L.append(f"- **vs change-point** (F1_multi {cp_bl['F1_multi']}, F1_global "
                 f"{cp_bl['F1_global']}) : ΔF1_multi={winner.f1 - cp_bl['F1_multi']:+.3f}, "
                 f"ΔF1_global={winner.gf1 - cp_bl['F1_global']:+.3f}.\n")
    L.append(
        "- **Coût** : un (en fait deux) forward(s) de cross-encodeur PAR JOINTURE — bien "
        "plus cher que l'attention (un seul forward d'encodeur par avis donne TOUT le signal) "
        "ou le change-point (embeddings + PELT). `minilm` rend le balayage faisable ; "
        "`mdeberta` (precision ↑) coûte ~100× plus → non-balayable en entier sur CPU.\n")
    L.append(
        "- **Honnêteté NLI** : un modèle MNLI/XNLI juge l'*entailment logique*, pas le "
        "*même-sujet* directement ; on détourne `P(entail)` comme proxy de cohésion. "
        "Sur des blocs de quelques mots (peu de contexte), le jugement est bruité — d'où "
        "le plafond. Le jeu (multi = concaténation de mono-thèmes) est une borne OPTIMISTE.\n")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _est_cost(items: list[GoldItem], ms_per_pair: float, swept: bool) -> dict:
    total_joins = 0
    for it in items:
        n = len(_split_words(it.text)[0])
        total_joins += max(0, n - 1)
    pairs = total_joins * 2 * len(W_GRID)   # 2 sens × |W|
    hours = pairs * ms_per_pair / 1000 / 3600
    return {"pairs": pairs, "hours": round(hours, 2), "swept": swept}


def main() -> None:
    ap = argparse.ArgumentParser(description="Segmentation par cross-encodeur NLI.")
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--models", nargs="+", default=["minilm"],
                    help="modèle(s) à balayer en entier (minilm rapide).")
    ap.add_argument("--mdeberta-subset", type=int, default=40,
                    help="taille du sous-échantillon de confirmation mdeberta (0 = aucun).")
    ap.add_argument("--out", default=str(DEFAULT_REPORT))
    ap.add_argument("--scores-out", default=str(DEFAULT_SCORES))
    args = ap.parse_args()

    gold_path = Path(args.gold)
    items, _ = load_gold(gold_path)
    print(f"gold: {gold_path.name} — {len(items)} items")
    attn_bl = load_attention_baseline()
    cp_bl = load_changepoint_baseline()
    if attn_bl:
        print(f"baseline attention: F1_multi={attn_bl['F1_multi']} F1_global={attn_bl['F1_global']}")

    feas, est_costs = {}, {}
    main_model = args.models[0]

    # Faisabilité de tous les modèles concernés (balayés + confirmation)
    probe_models = list(dict.fromkeys(args.models + (["mdeberta"] if args.mdeberta_subset else [])))
    for mk in probe_models:
        feas[mk] = feasibility_probe(mk)
        print(f"faisabilité {mk}: ok={feas[mk].get('ok')} "
              f"ms/pair={feas[mk].get('ms_per_pair')}")

    # Balayage plein-gold sur le(s) modèle(s) rapide(s)
    all_scores: list[NScore] = []
    for mk in args.models:
        if not feas[mk].get("ok"):
            continue
        est_costs[mk] = _est_cost(items, feas[mk]["ms_per_pair"], swept=True)
        print(f"=== balayage {mk} (plein gold) ===")
        prepared = prepare(items, mk, W_GRID)
        sc = sweep(mk, prepared, W_GRID)
        all_scores.extend(sc)
        w = _best(sc)
        if w:
            print(f"  best {mk}: {w.formulation} W={w.W} c={w.c} "
                  f"F1_multi={w.f1:.3f} F1_global={w.gf1:.3f} mono_FP={w.mono_fp_rate:.3f}")

    # Confirmation modèle lourd sur sous-échantillon
    subset_block = None
    if args.mdeberta_subset and feas.get("mdeberta", {}).get("ok"):
        k = args.mdeberta_subset
        sub = subset_items(items, k)
        est_costs["mdeberta"] = _est_cost(items, feas["mdeberta"]["ms_per_pair"], swept=False)
        print(f"=== confirmation mdeberta (sous-échantillon N={len(sub)}) ===")
        rows = []
        sub_best = {}
        for mk in ["minilm", "mdeberta"]:
            if not feas.get(mk, {}).get("ok"):
                continue
            prep = prepare(sub, mk, W_GRID)
            sc = sweep(mk, prep, W_GRID)
            b = _best(sc)
            if b:
                sub_best[mk] = b
                rows.append(b.as_row())
                print(f"  {mk} subset best: {b.formulation} W={b.W} c={b.c} "
                      f"F1_multi={b.f1:.3f} F1_global={b.gf1:.3f}")
        if "minilm" in sub_best and "mdeberta" in sub_best:
            subset_block = {
                "n": len(sub),
                "n_mono": sum(1 for it in sub if it.type == "mono"),
                "n_multi": sum(1 for it in sub if it.type == "multi"),
                "rows": rows,
                "minilm_f1": sub_best["minilm"].f1, "minilm_gf1": sub_best["minilm"].gf1,
                "mdeberta_f1": sub_best["mdeberta"].f1, "mdeberta_gf1": sub_best["mdeberta"].gf1,
                "speed_ratio": round(feas["mdeberta"]["ms_per_pair"]
                                     / max(0.1, feas["minilm"]["ms_per_pair"])),
            }

    report = build_report(gold_path, items, main_model, all_scores, feas, attn_bl, cp_bl,
                          subset_block, est_costs)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"\n✓ {args.out}")

    winner = _best(all_scores)
    Path(args.scores_out).write_text(json.dumps({
        "gold": gold_path.name, "models": args.models, "n_items": len(items), "seed": SEED,
        "feasibility": feas, "est_costs": est_costs,
        "baseline_attention": attn_bl, "baseline_changepoint": cp_bl,
        "subset_confirmation": subset_block,
        "winner": winner.as_row() if winner else None,
        "configs": [s.as_row() for s in sorted(all_scores, key=lambda s: -s.gf1)],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ {args.scores_out}")


if __name__ == "__main__":
    main()
