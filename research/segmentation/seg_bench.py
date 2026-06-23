"""Banc de segmentation sémantique — 3 segmenteurs × fenêtre W × seuil, vs gold.

    uv run --with ruptures python -m eval.segmentation.seg_bench
        [--gold eval/segmentation/gold.json] [--model nomic-v2]
        [--out eval/segmentation/report.md] [--no-changepoint]

Pipeline : token-embeddings (nomic-v2 par défaut) → vecteurs-MOTS → balayage des
config (segmenteur × W × seuil) → Pk / WindowDiff / F1-frontières / faux-positifs-mono
→ scorecard du gagnant + report.md (faisabilité, tableaux, exemples).

ÉCRIT UNIQUEMENT dans `eval/segmentation/` (report.md, scores.json, .cache/).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from eval.segmentation import metrics as M
from eval.segmentation import segmenters as S
from eval.segmentation.embeddings import embed_word_units, feasibility_probe

HERE = Path(__file__).resolve().parent
DEFAULT_GOLD = HERE / "gold.json"
DEFAULT_REPORT = HERE / "report.md"
DEFAULT_SCORES = HERE / "scores.json"


# --------------------------------------------------------------------------- #
# Chargement du gold (2 formats : gold.json simple, gold_large.json explicite)
# --------------------------------------------------------------------------- #
@dataclass
class GoldItem:
    id: str
    type: str               # "mono" | "multi"
    text: str
    boundaries_char: list[int]   # offsets char des frontières internes (vide si mono)
    seg_themes: list[str] = field(default_factory=list)


def load_gold(path: Path) -> tuple[list[GoldItem], dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    join = data.get("join", " ")
    items: list[GoldItem] = []
    for it in data["items"]:
        if it["type"] == "mono":
            items.append(GoldItem(it["id"], "mono", it["text"], [], [it.get("theme", "?")]))
            continue
        # multi : soit texte+frontières explicites (gold_large), soit à reconstruire.
        if "text" in it and "boundaries_char" in it:
            text = it["text"]
            bch = list(it["boundaries_char"])
            themes = [s["theme"] for s in it.get("segments", [])]
        else:
            segs = it["segments"]
            parts = [s["text"] for s in segs]
            text = join.join(parts)
            bch, off = [], 0
            for p in parts[:-1]:
                off += len(p)
                bch.append(off)         # frontière AVANT le séparateur de jointure
                off += len(join)
            themes = [s["theme"] for s in segs]
        items.append(GoldItem(it["id"], "multi", text, bch, themes))
    meta = {k: v for k, v in data.items() if k != "items"}
    return items, meta


# --------------------------------------------------------------------------- #
# Évaluation d'une config sur tout le jeu
# --------------------------------------------------------------------------- #
@dataclass
class Prepared:
    item: GoldItem
    U: np.ndarray
    n: int
    ref: set[int]            # frontières gold en indices-MOTS


def prepare(items: list[GoldItem], model_id: str) -> list[Prepared]:
    prepared = []
    for it in items:
        wu = embed_word_units(it.text, model_id=model_id)
        n = len(wu)
        ref = set()
        for off in it.boundaries_char:
            b = wu.boundary_word_index(off)
            if 0 < b < n:
                ref.add(b)
        prepared.append(Prepared(it, wu.vectors, n, ref))
    return prepared


@dataclass
class ConfigScore:
    method: str
    W: int
    thr: float
    pk: float                # moyenne sur multi
    windowdiff: float        # moyenne sur multi
    f1: float                # micro sur multi (tol ±1)
    precision: float
    recall: float
    gf1: float               # F1 GLOBAL (mono+multi) : coupes mono = faux positifs
    mono_fp_rate: float      # fraction de mono avec ≥1 coupe
    mono_cuts_mean: float    # nb moyen de coupes par mono
    pk_all: float            # Pk moyen sur TOUS les items (mono inclus)

    @property
    def score(self) -> float:
        return self.gf1       # objectif = F1 global (récompense le rappel, pénalise les FP mono)

    def as_row(self) -> dict:
        return {
            "method": self.method, "W": self.W,
            S.THRESH_NAME[self.method]: self.thr,
            "Pk": round(self.pk, 4), "WindowDiff": round(self.windowdiff, 4),
            "F1_multi": round(self.f1, 4), "P": round(self.precision, 4),
            "R": round(self.recall, 4), "mono_FP": round(self.mono_fp_rate, 4),
            "mono_cuts": round(self.mono_cuts_mean, 3),
            "F1_global": round(self.gf1, 4),
        }


def evaluate(method: str, W: int, thr: float, prepared: list[Prepared],
             gstats: S.GlobalStats) -> ConfigScore:
    multi = [p for p in prepared if p.item.type == "multi"]
    mono = [p for p in prepared if p.item.type == "mono"]

    pk_m, wd_m = [], []
    bc = M.BoundaryCounts()        # frontières multi (F1 multi)
    gbc = M.BoundaryCounts()       # frontières GLOBAL (mono+multi)
    for p in multi:
        hyp = S.segment(method, p.U, W, thr, gstats)
        pk_m.append(M.pk(p.n, p.ref, hyp))
        wd_m.append(M.windowdiff(p.n, p.ref, hyp))
        c = M.boundary_counts(p.ref, hyp, tol=1)
        bc = bc + c
        gbc = gbc + c

    mono_hits, mono_cuts, pk_all = 0, 0, []
    for p in mono:
        hyp = S.segment(method, p.U, W, thr, gstats)
        if hyp:
            mono_hits += 1
        mono_cuts += len(hyp)
        pk_all.append(M.pk(p.n, p.ref, hyp))
        gbc = gbc + M.boundary_counts(p.ref, hyp, tol=1)  # ref vide → coupes = FP
    pk_all += pk_m

    mono_fp = mono_hits / len(mono) if mono else 0.0
    return ConfigScore(
        method=method, W=W, thr=thr,
        pk=float(np.mean(pk_m)) if pk_m else 0.0,
        windowdiff=float(np.mean(wd_m)) if wd_m else 0.0,
        f1=bc.f1, precision=bc.precision, recall=bc.recall,
        gf1=gbc.f1,
        mono_fp_rate=mono_fp,
        mono_cuts_mean=mono_cuts / len(mono) if mono else 0.0,
        pk_all=float(np.mean(pk_all)) if pk_all else 0.0,
    )


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _md_table(rows: list[dict], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows)
    return "\n".join([head, sep, body])


def _segments_str(boundaries: set[int], words: list[str]) -> str:
    """Rend le texte avec « ⟂ » aux frontières (indices-mots)."""
    out = []
    for i, w in enumerate(words):
        if i in boundaries:
            out.append("⟂")
        out.append(w)
    return " ".join(out)


def build_report(gold_path: Path, gold_meta: dict, items: list[GoldItem],
                 prepared: list[Prepared], scores: list[ConfigScore],
                 feasibility: dict, model_id: str, changepoint_on: bool,
                 gstats: S.GlobalStats) -> str:
    n_mono = sum(1 for it in items if it.type == "mono")
    n_multi = sum(1 for it in items if it.type == "multi")
    winner = max(scores, key=lambda s: (s.score, -s.windowdiff, -s.pk))

    L = []
    L.append("# Banc de segmentation sémantique — rapport\n")
    L.append(f"*Jeu : `{gold_path.name}` — N={len(items)} ({n_mono} mono, {n_multi} multi). "
             f"Embeddings : `{model_id}`. Seed fixé, CPU.*\n")

    # --- Étape 0 : faisabilité ---
    L.append("## 0. Faisabilité des token-embeddings\n")
    if feasibility.get("ok"):
        L.append(f"**nomic-v2 : OUI.** Token-embeddings récupérés via "
                 f"`SentenceTransformer.encode(text, output_value='token_embeddings')` "
                 f"→ `last_hidden_state` `[n_tokens, {feasibility['dim']}]` AVANT pooling. "
                 f"Préfixe doc `{feasibility['doc_prefix']!r}`, `trust_remote_code="
                 f"{feasibility['trust_remote_code']}`. Alignement token→offset char exact "
                 f"via `offset_mapping` (tokens spéciaux + préfixe retirés). "
                 f"**Aucun repli e5/bge nécessaire.**\n")
    else:
        L.append(f"**nomic-v2 : NON** — {feasibility.get('error')}. "
                 f"Repli documenté disponible (e5-base / bge-m3, token-embeddings garantis "
                 f"par sentence-transformers ; relancer avec `--model e5` ou `--model bge-m3`).\n")

    # --- Méthode ---
    L.append("## 1. Méthode\n")
    L.append("- **Unité** = mot (suite de non-espaces, langue-agnostique). Vecteur-mot = "
             "moyenne des token-embeddings du mot, L2-normalisé. Fenêtre glissante W = "
             "moyenne des vecteurs-mots.\n"
             "- **Segmenteurs** : (1) *TextTiling-cosine* — minima locaux de cos(bloc-"
             "gauche, bloc-droite) sous `mu_bloc - c.sigma_bloc` ; (2) *Centroïde live* — "
             "coupe quand `cos(mot, centroïde courant)` < `mu_nouveaute - alpha.sigma` ; "
             "(3) *Change-point* — `ruptures` PELT/rbf, pénalité balayée.\n"
             "- **Seuils dérivés ET calibrés GLOBALEMENT** (mu/sigma poolés sur TOUS les "
             "avis, pas par-document) : un seuil purement relatif à un avis ne peut jamais "
             "s'abstenir sur un mono cohérent (il coupe toujours au point le moins pire). "
             "Coefficients sans dimension, aucun magic-number absolu. `min_seg=%d` mots.\n"
             % S.MIN_SEG)
    L.append("- **Métriques** : Pk & WindowDiff (↓, sur multi), F1 des frontières (tol ±1 "
             "mot, micro sur multi), **taux de faux-positifs mono** (fraction de mono "
             "produisant ≥1 coupe — métrique clé). **Objectif de sélection = F1 GLOBAL** "
             "(frontières sur mono+multi : toute coupe d'un mono est un faux positif → "
             "le segmenteur « ne rien couper » n'est pas favorisé).\n")
    if not changepoint_on:
        L.append("- ⚠️ *Change-point désactivé (`ruptures` indisponible) — relancer avec "
                 "`uv run --with ruptures …`.*\n")

    # --- Meilleure config par méthode ---
    L.append("## 2. Meilleure config par segmenteur\n")
    best_by_method = {}
    for s in scores:
        b = best_by_method.get(s.method)
        if b is None or (s.score, -s.windowdiff) > (b.score, -b.windowdiff):
            best_by_method[s.method] = s
    cols = ["method", "W", "Pk", "WindowDiff", "F1_multi", "P", "R", "mono_FP",
            "mono_cuts", "F1_global"]
    rows = [best_by_method[m].as_row() for m in S.SEGMENTERS if m in best_by_method]
    L.append(_md_table(rows, cols) + "\n")
    L.append(f"*(Pk/WindowDiff = moyenne sur les {n_multi} multi ; mono_FP/mono_cuts sur les "
             f"{n_mono} mono.)*\n")

    # --- Top configs global ---
    L.append("## 3. Top 12 configurations (toutes méthodes)\n")
    top = sorted(scores, key=lambda s: (-s.score, s.windowdiff, s.pk))[:12]
    cols2 = ["method", "W", "Pk", "WindowDiff", "F1_multi", "P", "R", "mono_FP",
             "mono_cuts", "F1_global"]
    L.append(_md_table([t.as_row() for t in top], cols2) + "\n")

    # --- Gagnant ---
    tn = S.THRESH_NAME[winner.method]
    L.append("## 4. Gagnant\n")
    L.append(f"**`{winner.method}` · W={winner.W} · {tn}={winner.thr}** — "
             f"F1 global={winner.gf1:.3f} ; F1 multi={winner.f1:.3f} "
             f"(P={winner.precision:.3f}, R={winner.recall:.3f}), "
             f"Pk={winner.pk:.3f}, WindowDiff={winner.windowdiff:.3f}, "
             f"faux-positifs mono={winner.mono_fp_rate:.3f} "
             f"({winner.mono_cuts_mean:.2f} coupe/mono).\n")

    # --- Exemples ---
    L.append("## 5. Exemples (avis multi → frontières)\n")
    multi_prepared = [p for p in prepared if p.item.type == "multi"]
    for p in multi_prepared[:3]:
        words = embed_word_units(p.item.text, model_id=model_id).words
        hyp = S.segment(winner.method, p.U, winner.W, winner.thr, gstats)
        L.append(f"**{p.item.id}** (thèmes : {', '.join(p.item.seg_themes)})\n")
        L.append(f"- gold : {_segments_str(p.ref, words)}")
        L.append(f"- prédit : {_segments_str(hyp, words)}\n")

    # --- Honnêteté ---
    miss = 1.0 - winner.recall
    L.append("## 6. Limites — verdict honnête\n")
    L.append(
        f"- **La segmentation par embeddings reste MÉDIOCRE sur des transitions "
        f"naturelles.** Même la meilleure config (`{winner.method}` W={winner.W}) ne "
        f"récupère que **R={winner.recall:.2f}** des frontières gold (soit **~{miss*100:.0f}% "
        f"de frontières ratées**) pour une précision P={winner.precision:.2f}, et **sur-coupe** "
        f"les mono ({winner.mono_fp_rate*100:.0f}% des mono reçoivent ≥1 coupe parasite, "
        f"{winner.mono_cuts_mean:.2f} coupe/mono). Le signal token-level capte mal les "
        f"virages de thème quand la transition n'est pas lexicalement marquée.\n")
    L.append(
        f"- **Jeu (N={len(items)}) : multi = concaténation de segments mono-thème.** "
        f"Frontières nettes par construction → ces chiffres sont déjà une **borne "
        f"optimiste** ; sur des avis vraiment continus, attendre pire.\n")
    L.append(
        "- **Implication pour la prod** : avant de câbler un segmenteur, soit relever le "
        "rappel (signal plus riche : phrases/clauses, modèle supervisé, marqueurs "
        "discursifs), soit assumer qu'on découpe surtout les avis franchement multi-thèmes "
        "et qu'on tolère la sur-coupe des mono en aval (dédup/agrégation thématique).\n")
    L.append(f"- Registre unique (consultation TikTok FR) — pas de garantie cross-domaine ; "
             f"seuils dérivés par config mais grille discrète (W∈{S.W_GRID}).\n")
    if gold_meta.get("_doc"):
        L.append(f"- *gold `_doc` : {gold_meta['_doc'][:200]}…*\n")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="Banc de segmentation sémantique.")
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--model", default="nomic-v2")
    ap.add_argument("--out", default=str(DEFAULT_REPORT))
    ap.add_argument("--scores-out", default=str(DEFAULT_SCORES))
    ap.add_argument("--no-changepoint", action="store_true",
                    help="désactive le segmenteur ruptures (si non installé)")
    args = ap.parse_args()

    gold_path = Path(args.gold)
    items, gold_meta = load_gold(gold_path)
    print(f"gold: {gold_path.name} — {len(items)} items")

    feasibility = feasibility_probe(args.model)
    print(f"faisabilité {args.model}: ok={feasibility.get('ok')} "
          f"dim={feasibility.get('dim')}")

    changepoint_on = not args.no_changepoint and S._ruptures_available()
    if not args.no_changepoint and not changepoint_on:
        print("⚠️ ruptures indisponible — change-point ignoré (relancer avec --with ruptures)")

    print("embeddings + préparation…")
    prepared = prepare(items, args.model)
    gstats = S.compute_global_stats([p.U for p in prepared], S.W_GRID)

    scores: list[ConfigScore] = []
    for method, W, thr in S.iter_configs(include_changepoint=changepoint_on):
        scores.append(evaluate(method, W, thr, prepared, gstats))
    print(f"{len(scores)} configs évaluées")

    winner = max(scores, key=lambda s: (s.score, -s.windowdiff, -s.pk))
    print(f"GAGNANT: {winner.method} W={winner.W} "
          f"{S.THRESH_NAME[winner.method]}={winner.thr} "
          f"F1={winner.f1:.3f} Pk={winner.pk:.3f} mono_FP={winner.mono_fp_rate:.3f}")

    report = build_report(gold_path, gold_meta, items, prepared, scores,
                          feasibility, args.model, changepoint_on, gstats)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"✓ {args.out}")

    Path(args.scores_out).write_text(
        json.dumps({
            "gold": gold_path.name, "model": args.model, "n_items": len(items),
            "feasibility": feasibility,
            "winner": winner.as_row(),
            "configs": [s.as_row() for s in sorted(scores, key=lambda s: -s.score)],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ {args.scores_out}")


if __name__ == "__main__":
    main()
