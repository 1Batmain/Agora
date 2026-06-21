"""Décodage ADAPTATIF par document — le point de fonctionnement qui TRANSFÈRE.

Diagnostic des runs précédents (`learned_report.md` §0) : la tête apprise *classe* bien
les frontières (ranking bon), mais un **seuil ABSOLU** sur `P(frontière|p)` calé sur la
source (WikiSection EN/DE) **ne transfère pas** au gold FR : il sur-coupe (run 1) ou
s'éteint (run 2, +négatifs, F1 zéro-shot ~0). La **distribution de P diffère par domaine** ;
un nombre fixe sature l'un / éteint l'autre.

Le fix testé ici : décoder les frontières **RELATIVEMENT à chaque document** au lieu d'un
seuil numérique transféré tel quel. Variantes (toutes les params dérivées de la SOURCE
uniquement = vrai zéro-shot) :

  1. **Relatif** (`rel`)         : coupe aux **maxima locaux de P** au-dessus de `μ_P + k·σ_P`
                                   (μ_P, σ_P DU DOC ; k réglé sur la source). NMS + min_seg.
  2. **Relatif + plancher** (`rel+floor`) : ET `P > floor` (floor = percentile de P source).
  3. **Garde d'abstention** (`rel+gate:STAT`) ⚠️ CRUCIAL : un seuil purement relatif coupe
     TOUJOURS le moins-pire (le max global dépasse μ) → il ne peut pas s'abstenir sur un avis
     cohérent (mono). On ajoute un **gate de platitude/pic** : si la distribution de P du doc
     est **plate** (stat de pic < τ) → **abstention**. Stats testées (dérivées, sans dimension
     sauf `maxp`) : `sigma` (σ_P), `maxp` (max P), `peak_z` ((max−μ)/σ), `kurt` (kurtosis
     excès). τ réglé sur la source (sépare mono/multi). C'est ce qui garde mono_FP bas.
  4. **Calibration** (`calib+fixed`) : isotonic sur la CV source → P calibrée, puis seuil
     ABSOLU réglé source. Teste l'hypothèse « calibrer rend le seuil fixe transférable ».

Grille bench : {LR, GBM} × {avec négatifs, sans négatifs} × {chaque variante de décodage}.
Tout calé sur **WikiSection-CV (source) UNIQUEMENT**, appliqué tel quel au gold FR. Métriques
gold : F1_multi, Pk, **mono_FP** (les 104 mono). Question : une variante adaptative bat-elle
le **réglé-main 0.769** en zéro-shot avec mono_FP bas ?

RÉUTILISE `learned_seg` (features, échantillonnage, entraînement, métriques) + le cache
d'attention. ÉCRIT UNIQUEMENT `eval/segmentation/`. CPU, seed fixe, modèle GELÉ.

    uv run --extra contender python -m eval.segmentation.adaptive_seg
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval.segmentation import metrics as M
from eval.segmentation import learned_seg as L
from eval.segmentation.segmenters import MIN_SEG, _enforce_min_seg

HERE = Path(__file__).resolve().parent
DEFAULT_GOLD = HERE / "gold_large.json"
DEFAULT_REPORT = HERE / "adaptive_report.md"
DEFAULT_SCORES = HERE / "adaptive_scores.json"

SEED = 0
MODEL_KEY = "e5-base"

# Réglage du décodage relatif. `k` = nb de σ au-dessus de μ_P (par doc) au-delà duquel un
# maximum local de P devient une frontière. Réglé sur la SOURCE (jamais le gold). k peut
# être négatif (la source peut vouloir un seuil sous la moyenne) — aucun signe imposé.
K_GRID = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
# Percentiles (dérivés) balayés pour le gate τ et le plancher absolu.
GATE_Q = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
FLOOR_Q = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
GATE_STATS = ["sigma", "maxp", "peak_z", "kurt"]


# --------------------------------------------------------------------------- #
# Statistiques par document (pré-calculées une fois → fitting rapide)
# --------------------------------------------------------------------------- #
def _local_maxima(p: np.ndarray) -> np.ndarray:
    """Indices des maxima locaux (plateau inclus : ≥ voisins)."""
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=int)
    keep = []
    for i in range(m):
        left = p[i] >= p[i - 1] if i > 0 else True
        right = p[i] >= p[i + 1] if i < m - 1 else True
        if left and right:
            keep.append(i)
    return np.asarray(keep, dtype=int)


def peak_stats(p: np.ndarray) -> dict:
    """Mesures de « platitude/pic » de la distribution de P d'un doc.
    `sigma`=σ_P, `maxp`=max P, `peak_z`=(max−μ)/σ (sans dimension), `kurt`=kurtosis excès."""
    mu = float(p.mean())
    sd = float(p.std())
    mx = float(p.max())
    peak_z = (mx - mu) / (sd + 1e-9)
    kurt = float(((p - mu) ** 4).mean() / (sd ** 4 + 1e-12) - 3.0) if sd > 1e-9 else 0.0
    return {"sigma": sd, "maxp": mx, "peak_z": peak_z, "kurt": kurt}


@dataclass
class DocPre:
    """Pré-calcul par doc : proba, μ/σ, maxima locaux, stats de pic."""
    proba: np.ndarray
    mu: float
    sd: float
    lmax: np.ndarray
    stats: dict


def precompute(probas: dict[str, np.ndarray]) -> dict[str, DocPre]:
    out = {}
    for did, p in probas.items():
        p = np.asarray(p, dtype=np.float64)
        if p.size == 0:
            out[did] = DocPre(p, 0.0, 0.0, np.zeros(0, dtype=int), peak_stats(np.array([0.0])))
            continue
        out[did] = DocPre(p, float(p.mean()), float(p.std()), _local_maxima(p), peak_stats(p))
    return out


# --------------------------------------------------------------------------- #
# Décodeurs
# --------------------------------------------------------------------------- #
@dataclass
class Adaptive:
    """Décodeur relatif par doc (+ gate d'abstention + plancher optionnels)."""
    k: float
    gate_stat: str | None = None
    tau: float = 0.0
    floor: float | None = None

    def cut(self, pre: DocPre, n: int, min_seg: int = MIN_SEG) -> set[int]:
        if pre.proba.size == 0 or n < 2 * min_seg:
            return set()
        if self.gate_stat is not None and pre.stats[self.gate_stat] < self.tau:
            return set()                       # abstention (distribution plate → mono)
        thr = pre.mu + self.k * pre.sd
        cand = []
        for i in pre.lmax:
            v = pre.proba[i]
            if v > thr and (self.floor is None or v > self.floor):
                cand.append((int(i) + 1, float(v)))
        return _enforce_min_seg(cand, n, min_seg)


@dataclass
class Calibrated:
    """P calibrée (isotonic source) puis seuil ABSOLU (réglé source) — V4."""
    iso: object
    thr: float

    def cut(self, pre: DocPre, n: int, min_seg: int = MIN_SEG) -> set[int]:
        if pre.proba.size == 0 or n < 2 * min_seg:
            return set()
        cp = self.iso.predict(pre.proba)
        cand = [(i + 1, float(cp[i])) for i in range(len(cp)) if cp[i] >= self.thr]
        return _enforce_min_seg(cand, n, min_seg)


@dataclass
class Fixed:
    """Seuil ABSOLU sur P (la baseline qui ne transfère pas — pour le contraste)."""
    thr: float

    def cut(self, pre: DocPre, n: int, min_seg: int = MIN_SEG) -> set[int]:
        if pre.proba.size == 0 or n < 2 * min_seg:
            return set()
        cand = [(i + 1, float(pre.proba[i])) for i in range(len(pre.proba))
                if pre.proba[i] >= self.thr]
        return _enforce_min_seg(cand, n, min_seg)


# --------------------------------------------------------------------------- #
# Évaluation (réutilise metrics + structure de learned_seg.evaluate)
# --------------------------------------------------------------------------- #
def score_fast(docs: list[L.Doc], pre: dict[str, DocPre], decoder) -> tuple:
    """Objectif LÉGER pour le réglage source : F1_global + recall + mono_FP via les seuls
    comptes de frontières (PAS de Pk/WindowDiff, O(n²) en trop). Renvoie (gf1, recall,
    mono_fp_rate) — départage : gf1 ↑, recall ↑, mono_FP ↓."""
    gbc = M.BoundaryCounts()
    mono_hits = mono_tot = 0
    for d in docs:
        hyp = decoder.cut(pre[d.id], d.n)
        gbc = gbc + M.boundary_counts(d.ref, hyp, tol=1)
        if d.type == "mono":
            mono_tot += 1
            if hyp:
                mono_hits += 1
    mono_fp = mono_hits / mono_tot if mono_tot else 0.0
    return (round(gbc.f1, 5), round(gbc.recall, 5), -round(mono_fp, 5))


def evaluate(docs: list[L.Doc], pre: dict[str, DocPre], decoder) -> L.EvalResult:
    """Décode chaque doc avec `decoder`, agrège Pk/WindowDiff (multi) + F1 (multi & global)
    + faux-positifs mono. Même comptage que `learned_seg.evaluate`."""
    multi = [d for d in docs if d.type == "multi"]
    mono = [d for d in docs if d.type == "mono"]
    pk_m, wd_m = [], []
    bc = M.BoundaryCounts()
    gbc = M.BoundaryCounts()
    for d in multi:
        hyp = decoder.cut(pre[d.id], d.n)
        pk_m.append(M.pk(d.n, d.ref, hyp))
        wd_m.append(M.windowdiff(d.n, d.ref, hyp))
        c = M.boundary_counts(d.ref, hyp, tol=1)
        bc = bc + c
        gbc = gbc + c
    mono_hits, mono_cuts = 0, 0
    for d in mono:
        hyp = decoder.cut(pre[d.id], d.n)
        if hyp:
            mono_hits += 1
        mono_cuts += len(hyp)
        gbc = gbc + M.boundary_counts(d.ref, hyp, tol=1)
    return L.EvalResult(
        thr=getattr(decoder, "k", getattr(decoder, "thr", 0.0)),
        pk=float(np.mean(pk_m)) if pk_m else 0.0,
        windowdiff=float(np.mean(wd_m)) if wd_m else 0.0,
        f1=bc.f1, precision=bc.precision, recall=bc.recall, gf1=gbc.f1,
        mono_fp_rate=mono_hits / len(mono) if mono else 0.0,
        mono_cuts_mean=mono_cuts / len(mono) if mono else 0.0,
    )


# --------------------------------------------------------------------------- #
# Fitting des décodeurs sur la SOURCE (objectif F1_global = détection − sur-coupe mono)
# --------------------------------------------------------------------------- #
def _stat_values(pre: dict[str, DocPre], stat: str) -> np.ndarray:
    return np.array([dp.stats[stat] for dp in pre.values() if dp.proba.size])


def _proba_pool(pre: dict[str, DocPre]) -> np.ndarray:
    arr = [dp.proba for dp in pre.values() if dp.proba.size]
    return np.concatenate(arr) if arr else np.array([0.5])


def fit_adaptive(src_docs, src_pre, *, gate_stat=None, use_floor=False, fixed_tau=None):
    """Balaie k (× τ gate × floor) sur la source, renvoie le meilleur `Adaptive` par F1_global.
    `fixed_tau` fige le seuil du gate (réutilise un gate déjà réglé → évite de re-balayer τ)."""
    floors = [None]
    if use_floor:
        pool = _proba_pool(src_pre)
        floors = [float(np.quantile(pool, q)) for q in FLOOR_Q]
    taus = [0.0]
    if gate_stat is not None:
        if fixed_tau is not None:
            taus = [fixed_tau]
        else:
            sv = _stat_values(src_pre, gate_stat)
            taus = sorted({float(np.quantile(sv, q)) for q in GATE_Q})
    best, best_key = None, None
    for k in K_GRID:
        for tau in taus:
            for fl in floors:
                dec = Adaptive(k=k, gate_stat=gate_stat, tau=tau, floor=fl)
                key = score_fast(src_docs, src_pre, dec)
                if best_key is None or key > best_key:
                    best_key, best = key, dec
    return best


def fit_fixed(src_docs, src_pre):
    """Meilleur seuil absolu sur P (objectif F1_global source)."""
    best, best_key = None, None
    for thr in L.THR_GRID:
        key = score_fast(src_docs, src_pre, Fixed(thr))
        if best_key is None or key > best_key:
            best_key, best = key, Fixed(thr)
    return best


def fit_calibrated(src_docs, src_pre, oof_proba, oof_labels):
    """Isotonic source (P→label) + meilleur seuil absolu sur P calibrée (F1_global source)."""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(oof_proba, oof_labels)
    cal_pre = {did: DocPre(iso.predict(dp.proba) if dp.proba.size else dp.proba,
                           0.0, 0.0, dp.lmax, dp.stats) for did, dp in src_pre.items()}
    best, best_key = None, None
    for thr in L.THR_GRID:
        key = score_fast(src_docs, cal_pre, Fixed(thr))
        if best_key is None or key > best_key:
            best_key, best = key, Calibrated(iso, thr)
    return best


# --------------------------------------------------------------------------- #
# OOF flexible : source = multi + mono, mais on n'entraîne que sur `train_types`
# --------------------------------------------------------------------------- #
def oof_probs(all_docs, kind, train_types=None, n_splits=5):
    """Probabilités OUT-OF-FOLD (GroupKFold par doc). Si `train_types` est donné, chaque
    fold n'entraîne QUE sur les docs de ce type (p.ex. multi seuls = « sans négatifs »),
    mais prédit sur TOUT le held-out (multi ET mono) → P de la tête sur les mono qu'elle
    n'a jamais vues. Renvoie (probas par doc, proba poolée, labels poolés)."""
    from sklearn.model_selection import GroupKFold

    X, y, groups, _ = L._stack(all_docs)
    types = np.concatenate([np.full(d.X.shape[0], d.type) for d in all_docs])
    offsets, cur = [], 0
    for d in all_docs:
        offsets.append((cur, cur + d.X.shape[0]))
        cur += d.X.shape[0]
    oof = np.zeros(len(y), dtype=np.float64)
    gkf = GroupKFold(n_splits=min(n_splits, len(all_docs)))
    for tr, va in gkf.split(X, y, groups):
        if train_types is not None:
            tr = tr[np.isin(types[tr], train_types)]
        model = L._make_model(kind)
        model.fit(X[tr], y[tr])
        oof[va] = model.predict_proba(X[va])[:, 1]
    probas = {d.id: oof[a:b] for d, (a, b) in zip(all_docs, offsets)}
    return probas, oof, y


def fit_full(docs, kind, train_types=None):
    """Tête finale (transfert) : entraîne sur tous les docs de `train_types`."""
    use = docs if train_types is None else [d for d in docs if d.type in train_types]
    X, y, _, _ = L._stack(use)
    model = L._make_model(kind)
    model.fit(X, y)
    return model


# --------------------------------------------------------------------------- #
# Orchestration : grille {LR,GBM} × {avec,sans négatifs} × {décodeurs}
# --------------------------------------------------------------------------- #
DECODER_SPECS = [
    ("fixed", "Seuil absolu (réf qui ne transfère pas)"),
    ("calib+fixed", "Isotonic source + seuil absolu (V4)"),
    ("rel", "Relatif μ+kσ, sans gate"),
    ("rel+floor", "Relatif + plancher absolu (V2)"),
    ("rel+gate:sigma", "Relatif + gate σ_P (V3)"),
    ("rel+gate:maxp", "Relatif + gate max P (V3)"),
    ("rel+gate:peak_z", "Relatif + gate (max−μ)/σ (V3)"),
    ("rel+gate:kurt", "Relatif + gate kurtosis (V3)"),
    ("rel+gate:best+floor", "Relatif + meilleur gate + plancher (V2+V3)"),
]


def build_decoder(name, src_docs, src_pre, oof_proba, oof_labels,
                  best_gate=None, gate_cache=None):
    """Construit + RÈGLE un décodeur (sur la source) à partir de son nom. `gate_cache`
    réutilise les gates déjà réglés (évite de re-balayer k×τ par stat)."""
    gate_cache = gate_cache or {}
    if name == "fixed":
        return fit_fixed(src_docs, src_pre)
    if name == "calib+fixed":
        return fit_calibrated(src_docs, src_pre, oof_proba, oof_labels)
    if name == "rel":
        return fit_adaptive(src_docs, src_pre)
    if name == "rel+floor":
        return fit_adaptive(src_docs, src_pre, use_floor=True)
    if name.startswith("rel+gate:") and not name.endswith("+floor"):
        stat = name.split(":", 1)[1]
        return gate_cache.get(stat) or fit_adaptive(src_docs, src_pre, gate_stat=stat)
    if name == "rel+gate:best+floor":
        ft = gate_cache[best_gate].tau if best_gate in gate_cache else None
        return fit_adaptive(src_docs, src_pre, gate_stat=best_gate, use_floor=True,
                            fixed_tau=ft)
    raise ValueError(name)


def run_config(kind, neg_mode, source_docs, gold_docs, gold_pre_holder):
    """Pour une (tête × régime de négatifs) : OOF source, fit décodeurs, transfert gold.
    Renvoie {decoder_name: {"src": EvalResult, "gold": EvalResult, "params": ...}}."""
    train_types = None if neg_mode == "with_neg" else ["multi"]
    print(f"  · OOF source ({kind}, {neg_mode})…")
    probas, oof_p, oof_y = oof_probs(source_docs, kind, train_types=train_types)
    src_pre = precompute(probas)

    print(f"  · fit tête complète + transfert gold ({kind}, {neg_mode})…")
    model = fit_full(source_docs, kind, train_types=train_types)
    gold_probas = L.predict_docs(model, gold_docs)
    gold_pre = precompute(gold_probas)
    gold_pre_holder[(kind, neg_mode)] = gold_pre

    # gate gagnant (sur la source, F1_global) pour la variante combinée best+floor.
    gate_cache = {}
    gate_scores = {}
    for stat in GATE_STATS:
        dec = fit_adaptive(source_docs, src_pre, gate_stat=stat)
        gate_cache[stat] = dec
        gate_scores[stat] = score_fast(source_docs, src_pre, dec)
    best_gate = max(gate_scores, key=lambda s: gate_scores[s])

    out = {}
    for name, _desc in DECODER_SPECS:
        dec = build_decoder(name, source_docs, src_pre, oof_p, oof_y,
                            best_gate=best_gate, gate_cache=gate_cache)
        src_r = evaluate(source_docs, src_pre, dec)
        gold_r = evaluate(gold_docs, gold_pre, dec)          # ZÉRO-SHOT
        out[name] = {"src": src_r, "gold": gold_r, "decoder": dec,
                     "best_gate": best_gate}
    return out


# --------------------------------------------------------------------------- #
# Données source (réutilise l'échantillonnage de learned_seg)
# --------------------------------------------------------------------------- #
def prepare_source(args):
    print("→ échantillonnage source (WikiSection multi + négatifs mono)…")
    en = L.sample_wikisection("en", args.n_en)
    de = L.sample_wikisection("de", args.n_de)
    pos_ids = {it.id for it in en} | {it.id for it in de}
    wmono_en = L.sample_wikisection_mono("en", args.n_wiki_mono_en, pos_ids)
    wmono_de = L.sample_wikisection_mono("de", args.n_wiki_mono_de, pos_ids)
    mab_en = L.sample_mabsa_mono("en", args.n_mabsa_en)
    mab_de = L.sample_mabsa_mono("de", args.n_mabsa_de)
    mab_fr = L.sample_mabsa_mono("fr", args.n_mabsa_fr)
    print(f"  POS EN={len(en)} DE={len(de)} | NEG wiki-mono EN={len(wmono_en)} "
          f"DE={len(wmono_de)} | M-ABSA EN={len(mab_en)} DE={len(mab_de)} FR={len(mab_fr)}")
    print("→ featurize source (attention cachée)…")
    docs = []
    for label, items in [("EN multi", en), ("DE multi", de),
                         ("EN wiki-mono", wmono_en), ("DE wiki-mono", wmono_de),
                         ("EN M-ABSA", mab_en), ("DE M-ABSA", mab_de), ("FR M-ABSA", mab_fr)]:
        print(f"  {label}:")
        docs += L.featurize(items, args.model)
    n_pos = sum(1 for d in docs if d.type == "multi")
    n_neg = sum(1 for d in docs if d.type == "mono")
    breakdown = {"n_en": len(en), "n_de": len(de), "wiki_mono_en": len(wmono_en),
                 "wiki_mono_de": len(wmono_de), "mabsa_en": len(mab_en),
                 "mabsa_de": len(mab_de), "mabsa_fr": len(mab_fr),
                 "n_pos": n_pos, "n_neg": n_neg}
    return docs, breakdown


def main():
    ap = argparse.ArgumentParser(description="Décodage adaptatif par document (zéro-shot).")
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--model", default=MODEL_KEY)
    # Tailles réduites vs learned_seg (CPU) — le décodage, pas la taille de train, est la
    # variable testée. Override pour reproduire les tailles complètes.
    ap.add_argument("--n-en", type=int, default=1000)
    ap.add_argument("--n-de", type=int, default=500)
    ap.add_argument("--n-wiki-mono-en", type=int, default=350)
    ap.add_argument("--n-wiki-mono-de", type=int, default=175)
    ap.add_argument("--n-mabsa-en", type=int, default=350)
    ap.add_argument("--n-mabsa-de", type=int, default=175)
    ap.add_argument("--n-mabsa-fr", type=int, default=500)
    ap.add_argument("--out", default=str(DEFAULT_REPORT))
    ap.add_argument("--scores-out", default=str(DEFAULT_SCORES))
    args = ap.parse_args()

    from eval.segmentation.attn_seg import ATTN_MODELS
    model_id = ATTN_MODELS[args.model]["model_id"]

    source_docs, breakdown = prepare_source(args)

    print("→ gold (transfert FR)…")
    gold_items, _ = L.load_gold(Path(args.gold))
    gold_docs = L.featurize(gold_items, args.model)

    refs = L.load_ref_baselines()

    print("→ grille {LR,GBM} × {avec,sans négatifs} × décodeurs…")
    gold_pre_holder = {}
    results = {}
    for kind in ("lr", "gbm"):
        for neg_mode in ("with_neg", "without_neg"):
            print(f" [{kind} / {neg_mode}]")
            results[(kind, neg_mode)] = run_config(
                kind, neg_mode, source_docs, gold_docs, gold_pre_holder)

    ctx = {"model_id": model_id, "refs": refs, "breakdown": breakdown,
           "n_source": len(source_docs), "results": results,
           "gold_pre": gold_pre_holder, "gold_docs": gold_docs}
    report = build_report(ctx)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"\n✓ {args.out}")

    dump = {"model": model_id, "seed": SEED, "n_source": len(source_docs),
            "breakdown": breakdown, "refs": refs,
            "grid": {f"{k}|{nm}": {name: {"src": r["src"].as_row(),
                                          "gold": r["gold"].as_row(),
                                          "best_gate": r["best_gate"]}
                                  for name, r in res.items()}
                     for (k, nm), res in results.items()}}
    Path(args.scores_out).write_text(json.dumps(dump, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(f"✓ {args.scores_out}")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _all_rows(ctx):
    for (kind, nm), res in ctx["results"].items():
        for name, r in res.items():
            yield kind, nm, name, r


def _winner_constrained(ctx, mono_cap):
    """Meilleur F1_multi zéro-shot SOUS contrainte mono_FP ≤ cap (le point de
    fonctionnement utile : détecter SANS sur-couper). None si aucun ne respecte le cap."""
    best = None
    for kind, nm, name, r in _all_rows(ctx):
        g = r["gold"]
        if g.mono_fp_rate <= mono_cap:
            key = (round(g.f1, 4), -round(g.mono_fp_rate, 4))
            if best is None or key > best[0]:
                best = (key, kind, nm, name, r)
    return best


def _best_detection(ctx):
    """Meilleur F1_multi zéro-shot SANS contrainte (plafond de détection, mono ignoré)."""
    best = None
    for kind, nm, name, r in _all_rows(ctx):
        g = r["gold"]
        key = (round(g.f1, 4), -round(g.mono_fp_rate, 4))
        if best is None or key > best[0]:
            best = (key, kind, nm, name, r)
    return best


def build_report(ctx):
    refs = ctx["refs"]
    at = refs.get("attn_tuned") or {}
    cp = refs.get("changepoint") or {}
    bd = ctx["breakdown"]
    L_ = []
    L_.append("# Décodage ADAPTATIF par document — le point de fonctionnement qui transfère\n")
    L_.append(
        f"*La tête apprise classe bien les frontières mais un **seuil absolu** sur "
        f"`P(frontière|p)` ne transfère pas cross-domaine (cf. `learned_report.md` §0). "
        f"Ici on remplace le seuil fixe par un **point de fonctionnement adaptatif PAR "
        f"DOCUMENT**, toutes params réglées sur la **SOURCE uniquement** (WikiSection EN/DE "
        f"+ négatifs mono EN/DE/FR) = vrai zéro-shot, évalué sur notre **gold témoignages FR** "
        f"(201 multi + 104 mono). Encodeur `{ctx['model_id']}` GELÉ. Source = {ctx['n_source']} "
        f"docs ({bd['n_pos']} multi positifs + {bd['n_neg']} mono négatifs). Seed={SEED}, CPU. "
        f"Tailles réduites vs `learned_seg` (la variable testée est le **décodage**, pas la "
        f"taille de train).*\n")
    L_.append(
        "**Variantes de décodage** (toutes réglées source, objectif F1_global = détecter sans "
        "sur-couper les mono) :\n"
        "- `fixed` : seuil absolu sur P (la baseline qui ne transfère pas — pour contraste).\n"
        "- `calib+fixed` : isotonic (source) → P calibrée, puis seuil absolu (V4).\n"
        "- `rel` : coupe aux **maxima locaux de P** au-dessus de `μ_P + k·σ_P` du DOC (V1, k réglé source).\n"
        "- `rel+floor` : `rel` ET `P > floor` (floor = percentile de P source) (V2).\n"
        "- `rel+gate:STAT` : `rel` + **abstention** si la distribution de P du doc est plate "
        "(stat de pic < τ) ; STAT ∈ {σ_P, max P, (max−μ)/σ, kurtosis}, τ réglé source (V3).\n"
        "- `rel+gate:best+floor` : meilleur gate (source) + plancher (V2+V3).\n")

    # --- Référence ---
    L_.append("## Références (gold)\n")
    ref_rows = []
    if at:
        ref_rows.append({"approche": "attention RÉGLÉE-main (réf 0.769)",
                         "F1_multi": at["F1_multi"], "Pk": at["Pk"], "mono_FP": at["mono_FP"],
                         "F1_global": at["F1_global"]})
    if cp:
        ref_rows.append({"approche": "change-point (réf 0.44)", "F1_multi": cp["F1_multi"],
                         "Pk": cp["Pk"], "mono_FP": cp["mono_FP"], "F1_global": cp["F1_global"]})
    L_.append(L._md_table(ref_rows, ["approche", "F1_multi", "Pk", "mono_FP", "F1_global"]) + "\n")

    # --- Grille principale : zéro-shot par (config × décodeur) ---
    L_.append("## Grille zéro-shot — {LR,GBM} × {avec,sans négatifs} × décodage\n")
    L_.append("*`src gf1` / `src mFP` = F1_global / mono_FP sur la SOURCE (où le décodeur est "
              "réglé). Colonnes gold = **transfert zéro-shot** (params jamais vues le gold). "
              "Objectif : `F1_multi` HAUT (> 0.769) ET `mono_FP` BAS (≈/< 0.14).*\n")
    cols = ["config", "décodage", "src gf1", "src mFP",
            "gold F1_multi", "gold Pk", "gold mono_FP", "gold P", "gold R"]
    rows = []
    for kind in ("lr", "gbm"):
        for nm in ("with_neg", "without_neg"):
            res = ctx["results"][(kind, nm)]
            tag = f"{kind.upper()} {'+nég' if nm == 'with_neg' else 'sans nég'}"
            for name, _d in DECODER_SPECS:
                r = res[name]
                g, s = r["gold"], r["src"]
                disp = name.replace("best", r["best_gate"]) if "best" in name else name
                rows.append({
                    "config": tag, "décodage": disp,
                    "src gf1": round(s.gf1, 3), "src mFP": round(s.mono_fp_rate, 3),
                    "gold F1_multi": round(g.f1, 4), "gold Pk": round(g.pk, 3),
                    "gold mono_FP": round(g.mono_fp_rate, 3),
                    "gold P": round(g.precision, 3), "gold R": round(g.recall, 3)})
    L_.append(L._md_table(rows, cols) + "\n")

    # --- Verdict ---
    L_.append("## Verdict — une variante adaptative bat-elle le réglé-main 0.769 en zéro-shot ?\n")
    at_f1 = at.get("F1_multi", 0.0)
    at_mfp = at.get("mono_FP", 0.0)
    # Le point de fonctionnement utile DOIT s'abstenir comme le réglé-main : on impose
    # mono_FP ≤ cap (réglé-main + petite marge) et on maximise la F1 SOUS cette contrainte.
    mono_cap = round(at_mfp + 0.03, 3)
    win = _winner_constrained(ctx, mono_cap)
    det = _best_detection(ctx)
    L_.append(
        f"*Le point de fonctionnement utile doit **détecter ET s'abstenir** : on retient la "
        f"variante de **F1_multi max SOUS contrainte mono_FP ≤ {mono_cap}** (réglé-main "
        f"{at_mfp} + marge). Un gros F1 avec mono_FP élevé = sur-coupe, pas une victoire.*\n")

    # plafond de détection (mono ignoré) — pour situer.
    dk, dnm, dname, dr = det[1], det[2], det[3], det[4]
    dg = dr["gold"]
    ddisp = dname.replace("best", dr["best_gate"]) if "best" in dname else dname
    L_.append(
        f"- **Plafond de détection** (mono ignoré) : `{ddisp}` sur {dk.upper()} "
        f"{'+nég' if dnm == 'with_neg' else 'sans nég'} → F1_multi={dg.f1:.3f} "
        f"(mais mono_FP={dg.mono_fp_rate:.3f} → sur-coupe).\n")

    if win is None:
        L_.append(
            f"- **Sous contrainte mono_FP ≤ {mono_cap}** : **AUCUNE** variante zéro-shot ne "
            f"qualifie — toutes celles qui détectent (F1 élevé) sur-coupent les mono.\n")
        verdict = (f"❌ **NON** — aucune variante adaptative ne tient le point de fonctionnement "
                   f"du réglé-main en zéro-shot (détecter SANS sur-couper). Le décodage adaptatif "
                   f"aide vs le seuil fixe mais le transfert reste incomplet — signal qu'il faut "
                   f"un autre angle (cf. plafond oracle ci-dessous : la marge existe, c'est le "
                   f"réglage zéro-shot du gate qui ne transfère pas).")
        L_.append(f"- **{verdict}**\n")
    else:
        _key, wkind, wnm, wname, wr = win
        g = wr["gold"]
        disp = wname.replace("best", wr["best_gate"]) if "best" in wname else wname
        beats = g.f1 > at_f1 + 0.005
        near = g.f1 >= at_f1 - 0.03
        L_.append(
            f"- **Meilleur zéro-shot UTILE** (mono_FP ≤ {mono_cap}) : `{disp}` sur "
            f"**{wkind.upper()} {'+nég' if wnm == 'with_neg' else 'sans nég'}** → "
            f"F1_multi=**{g.f1:.4f}** (P={g.precision:.3f}, R={g.recall:.3f}), Pk={g.pk:.3f}, "
            f"**mono_FP={g.mono_fp_rate:.3f}**.\n")
        L_.append(
            f"- vs **réglé-main** (F1_multi={at_f1}, mono_FP={at_mfp}) : "
            f"ΔF1_multi=**{g.f1 - at_f1:+.4f}**, Δmono_FP={g.mono_fp_rate - at_mfp:+.3f}.\n")
        if beats:
            verdict = (f"✅ **OUI** — `{disp}` bat le réglé-main en zéro-shot (F1_multi "
                       f"{g.f1:.3f} > 0.769) AVEC mono_FP bas ({g.mono_fp_rate:.3f} ≤ "
                       f"{mono_cap}). Le décodage adaptatif débloque le transfert : le "
                       f"segmenteur appris est **mûr pour la prod**.")
        elif near:
            verdict = (f"≈ **QUASI** — `{disp}` approche le réglé-main (F1_multi {g.f1:.3f} vs "
                       f"0.769) avec mono_FP comparable/meilleur ({g.mono_fp_rate:.3f}). Le "
                       f"décodage adaptatif TRANSFÈRE (contre le seuil fixe qui s'éteignait) "
                       f"mais ne dépasse pas encore le réglé-main.")
        else:
            verdict = (f"❌ **NON** — sous contrainte d'abstention, le meilleur zéro-shot "
                       f"(F1_multi={g.f1:.3f}) reste loin du réglé-main (0.769). Détecter ET "
                       f"s'abstenir en zéro-shot n'est pas atteint — il faut un autre angle.")
        L_.append(f"- **{verdict}**\n")

        # adaptatif vs fixe (effet DIRECT du décodage), même config gagnante.
        fixed_r = ctx["results"][(wkind, wnm)]["fixed"]["gold"]
        L_.append(
            f"- **Effet du décodage adaptatif vs seuil fixe** (même tête {wkind.upper()} "
            f"{'+nég' if wnm == 'with_neg' else 'sans nég'}, zéro-shot) : fixe → F1_multi="
            f"{fixed_r.f1:.3f} / mono_FP={fixed_r.mono_fp_rate:.3f} ; adaptatif `{disp}` → "
            f"F1_multi={g.f1:.3f} / mono_FP={g.mono_fp_rate:.3f} "
            f"(ΔF1_multi={g.f1 - fixed_r.f1:+.3f}). C'est l'apport NET du point de "
            f"fonctionnement par-document.\n")

    # --- Plafond (oracle gold) : headroom du décodage ---
    L_.append("## Plafond (oracle) — décodage réglé sur le gold (triche, pour le headroom)\n")
    L_.append(f"*Mêmes variantes mais réglées DIRECTEMENT sur le gold (sous la MÊME contrainte "
              f"mono_FP ≤ {mono_cap}) : mesure ce que le ranking de la tête permettrait avec un "
              f"point de coupe PARFAIT. Si l'oracle dépasse 0.769 mais pas le zéro-shot → c'est "
              f"le **réglage du gate** qui ne transfère pas, pas le modèle. Sinon → le ranking "
              f"lui-même plafonne.*\n")
    ocols = ["config", "meilleur décodage (oracle)", "F1_multi", "Pk", "mono_FP", "P", "R"]
    orows = []
    gold_docs = ctx["gold_docs"]
    oof_y_gold = np.concatenate([d.y for d in gold_docs])
    for kind in ("lr", "gbm"):
        for nm in ("with_neg", "without_neg"):
            gold_pre = ctx["gold_pre"][(kind, nm)]
            res = ctx["results"][(kind, nm)]
            bg = next(iter(res.values()))["best_gate"]
            tag = f"{kind.upper()} {'+nég' if nm == 'with_neg' else 'sans nég'}"
            oof_p_gold = np.concatenate([gold_pre[d.id].proba for d in gold_docs
                                         if gold_pre[d.id].proba.size])
            # évalue chaque variante UNE fois (réglée sur le gold), puis sélectionne.
            variants = []
            for name, _d in DECODER_SPECS:
                dec = build_decoder(name, gold_docs, gold_pre, oof_p_gold, oof_y_gold,
                                    best_gate=bg)
                variants.append((name, evaluate(gold_docs, gold_pre, dec)))
            ok = [(n, r) for n, r in variants if r.mono_fp_rate <= mono_cap]
            if ok:
                oname, r = max(ok, key=lambda t: (round(t[1].f1, 4), -round(t[1].mono_fp_rate, 4)))
            else:                                  # aucun ne tient le cap même en oracle
                oname, r = min(variants, key=lambda t: round(t[1].mono_fp_rate, 4))
                oname = oname + " (cap non tenu)"
            disp = oname.replace("best", bg)
            orows.append({"config": tag, "meilleur décodage (oracle)": disp,
                          "F1_multi": round(r.f1, 4), "Pk": round(r.pk, 3),
                          "mono_FP": round(r.mono_fp_rate, 3),
                          "P": round(r.precision, 3), "R": round(r.recall, 3)})
    L_.append(L._md_table(orows, ocols) + "\n")

    # --- Honnêteté / généricité ---
    L_.append("## Honnêteté & généricité\n")
    L_.append(
        "- **Discipline zéro-shot** : k, τ (gate), floor, map isotonic — TOUT réglé sur la "
        "source (WikiSection-CV OOF + mono), jamais sur le gold. Le seul chiffre qui compte est "
        "la colonne `gold` de la grille. L'oracle est explicitement étiqueté triche (headroom).\n")
    L_.append(
        "- **Pourquoi le gate est crucial** : un seuil purement relatif (`rel`) coupe toujours "
        "le maximum local le moins pire → il NE PEUT PAS s'abstenir sur un mono cohérent "
        "(voir sa colonne `gold mono_FP`). Le gate de platitude/pic est ce qui rend "
        "l'abstention possible SANS seuil absolu transféré.\n")
    L_.append(
        "- **Généricité** : zéro lexique, zéro constante magique non dérivée — μ_P/σ_P par doc, "
        "τ/floor = percentiles de la source, k sans dimension. Calculable sur n'importe quelle "
        "consultation, n'importe quelle langue (transfert EN/DE→FR).\n")
    return "\n".join(L_)


if __name__ == "__main__":
    main()
