"""Segmentation par ATTENTION avec encodeur **e5-LARGE** — relève-t-il le plafond ?

EXPÉ R&D (read-only). Rejoue STRICTEMENT le segmenteur attention réglé-main
(`attn_seg.py` : `word_attention` → `cross_signal` → minima calibrés global) en
remplaçant l'encodeur `intfloat/multilingual-e5-base` (12 couches) par
`intfloat/multilingual-e5-large` (24 couches, 16 têtes). Question unique : un encodeur
plus gros relève-t-il le plafond du signal de cohésion thématique porté par l'attention,
ou plafonne-t-on quel que soit le modèle ?

On NE TOUCHE PAS la méthode : même grille (jeux-de-couches × têtes mean/local × W × c),
même calibration μ/σ poolée globalement, même `gold_large.json`. Seul le MODÈLE change.
Le contrôle décisif reste le réglé-main e5-base (F1_multi 0.769 / Pk 0.149 / mono_FP 0.144)
mesuré sur le MÊME jeu par `attn_seg.py` (lu depuis `attn_scores.json`).

ÉCRIT UNIQUEMENT dans `eval/segmentation/`. CPU, seed fixe.

    uv run --extra contender python -m eval.segmentation.e5large_seg \
        [--gold eval/segmentation/gold_large.json] \
        [--out eval/segmentation/e5large_report.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.segmentation import attn_seg as A
from eval.segmentation.seg_bench import load_gold

HERE = Path(__file__).resolve().parent
DEFAULT_GOLD = HERE / "gold_large.json"
DEFAULT_REPORT = HERE / "e5large_report.md"
DEFAULT_SCORES = HERE / "e5large_scores.json"
BASE_ATTN_SCORES = HERE / "attn_scores.json"   # réglé-main e5-base (attn_seg.py)

MODEL = "e5-large"
SEED = A.SEED


def load_base_reference() -> tuple[dict | None, list[dict]]:
    """Réglé-main e5-base depuis `attn_scores.json` : (winner global, configs e5-base).

    Le winner du banc attention EST e5-base (e5-base > bge-m3) ; on l'utilise comme
    plafond à battre. On garde aussi toutes les configs e5-base pour la comparaison
    couche-par-couche (même jeu-de-couches → comparaison appariée)."""
    if not BASE_ATTN_SCORES.exists():
        return None, []
    d = json.loads(BASE_ATTN_SCORES.read_text(encoding="utf-8"))
    cfgs = [c for c in d.get("configs", []) if c.get("model") == "e5-base"]
    win = d.get("winner")
    if win and win.get("model") != "e5-base":
        win = max(cfgs, key=lambda c: (c["F1_global"], c["F1_multi"]), default=None)
    return win, cfgs


def best_e5base_by_layerset(cfgs: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for c in cfgs:
        k = c["layers"]
        b = out.get(k)
        if b is None or (c["F1_global"], c["F1_multi"]) > (b["F1_global"], b["F1_multi"]):
            out[k] = c
    return out


def build_report(gold_path, items, scores, info, feas, emb_ctrl,
                 base_win: dict | None, base_cfgs: list[dict]) -> str:
    n_mono = sum(1 for it in items if it.type == "mono")
    n_multi = sum(1 for it in items if it.type == "multi")
    winner = max(scores, key=lambda s: (s.gf1, s.f1, -s.windowdiff, -s.pk))

    L: list[str] = []
    L.append("# Segmentation par ATTENTION — e5-LARGE relève-t-il le plafond ?\n")
    L.append(f"*Jeu : `{gold_path.name}` — N={len(items)} ({n_mono} mono, {n_multi} multi). "
             f"Encodeur : `{A.ATTN_MODELS[MODEL]['model_id']}` "
             f"({feas.get('n_layers')} couches, {feas.get('n_heads')} têtes). "
             f"Méthode IDENTIQUE au réglé-main e5-base (`attn_seg.py`) : seul le modèle "
             f"change. CPU, seed={SEED}.*\n")

    # 0. Faisabilité
    L.append("## 0. Faisabilité de l'extraction d'attention sur e5-large\n")
    if feas.get("ok"):
        L.append(f"- **OUI.** `intfloat/multilingual-e5-large` est un **XLM-R large** "
                 f"standard : `AutoModel(attn_implementation='eager')` + "
                 f"`output_attentions=True` → tuple de `[batch, heads={feas['n_heads']}, "
                 f"seq, seq]` × **{feas['n_layers']} couches**. Aucun hook custom requis "
                 f"(contrairement à nomic). Réduction token→mot par `offset_mapping`, "
                 f"préfixe `'passage: '` retiré comme pour e5-base.\n")
    else:
        L.append(f"- **NON** — {feas.get('error')}\n")
        return "\n".join(L)

    # 1. Comparaison frontale e5-large vs e5-base réglé-main
    L.append("## 1. e5-large vs e5-base réglé-main (la comparaison décisive)\n")
    rows = []
    if base_win:
        rows.append({"encodeur": "**e5-base** (réglé-main, plafond)",
                     "config": f"{base_win['layers']}/{base_win['heads']} "
                               f"W={base_win['W']} c={base_win['c']}",
                     "Pk": base_win["Pk"], "WindowDiff": base_win["WindowDiff"],
                     "F1_multi": base_win["F1_multi"], "P": base_win["P"],
                     "R": base_win["R"], "mono_FP": base_win["mono_FP"],
                     "F1_global": base_win["F1_global"]})
    w = winner
    rows.append({"encodeur": f"**e5-large** (cette expé)",
                 "config": f"{w.cfg.layer_set}/{w.cfg.head_agg} W={w.cfg.W} c={w.c}",
                 "Pk": round(w.pk, 4), "WindowDiff": round(w.windowdiff, 4),
                 "F1_multi": round(w.f1, 4), "P": round(w.precision, 4),
                 "R": round(w.recall, 4), "mono_FP": round(w.mono_fp_rate, 4),
                 "F1_global": round(w.gf1, 4)})
    if emb_ctrl:
        e = emb_ctrl
        rows.append({"encodeur": "_embedding-trajectoire e5-large_ (contrôle)",
                     "config": f"{e['method']} W={e['W']}",
                     "Pk": e["Pk"], "WindowDiff": e["WindowDiff"],
                     "F1_multi": e["F1_multi"], "P": e["P"], "R": e["R"],
                     "mono_FP": e["mono_FP"], "F1_global": e["F1_global"]})
    cols = ["encodeur", "config", "Pk", "WindowDiff", "F1_multi", "P", "R",
            "mono_FP", "F1_global"]
    L.append(A._md_table(rows, cols) + "\n")
    L.append("*(Pk/WindowDiff ↓ = mieux, sur multi ; F1_multi = frontières tol ±1 ; "
             "mono_FP = fraction de mono sur-coupés ↓ ; F1_global = mono+multi, objectif "
             "de sélection.)*\n")

    # 2. Top configs e5-large
    L.append("## 2. Top 15 configurations e5-large\n")
    top = sorted(scores, key=lambda s: (-s.gf1, -s.f1, s.windowdiff))[:15]
    cols2 = ["model", "layers", "heads", "W", "c", "Pk", "WindowDiff", "F1_multi",
             "P", "R", "mono_FP", "mono_cuts", "F1_global"]
    L.append(A._md_table([t.as_row() for t in top], cols2) + "\n")

    # 3. Comparaison appariée par jeu de couches (e5-large vs e5-base, même jeu)
    L.append("## 3. Couche-par-couche — e5-large vs e5-base (même jeu, meilleur c/W)\n")
    L.append(f"Têtes locales sélectionnées par jeu de couches : "
             f"`{info['local_heads']}` (sur {info['n_heads']} têtes).\n")
    best_large = {}
    for s in scores:
        k = s.cfg.layer_set
        b = best_large.get(k)
        if b is None or (s.gf1, s.f1) > (b.gf1, b.f1):
            best_large[k] = s
    best_base = best_e5base_by_layerset(base_cfgs)
    rows3 = []
    for k in ["early", "lowmid", "mid", "late", "midlate", "all"]:
        sl = best_large.get(k)
        sb = best_base.get(k)
        if sl is None:
            continue
        d_gf1 = (sl.gf1 - sb["F1_global"]) if sb else None
        rows3.append({
            "jeu": k,
            "large F1_g": round(sl.gf1, 4), "large F1_m": round(sl.f1, 4),
            "large Pk": round(sl.pk, 4), "large mono_FP": round(sl.mono_fp_rate, 4),
            "base F1_g": sb["F1_global"] if sb else "—",
            "base F1_m": sb["F1_multi"] if sb else "—",
            "ΔF1_global": f"{d_gf1:+.4f}" if d_gf1 is not None else "—",
        })
    cols3 = ["jeu", "large F1_g", "large F1_m", "large Pk", "large mono_FP",
             "base F1_g", "base F1_m", "ΔF1_global"]
    L.append(A._md_table(rows3, cols3) + "\n")

    # 4. Verdict
    L.append("## 4. Verdict honnête\n")
    L.append(f"**Meilleure config e5-large : `{w.cfg.layer_set}/{w.cfg.head_agg}` · "
             f"W={w.cfg.W} · c={w.c}** → F1_multi={w.f1:.3f} "
             f"(P={w.precision:.3f}, R={w.recall:.3f}), Pk={w.pk:.3f}, "
             f"WindowDiff={w.windowdiff:.3f}, F1_global={w.gf1:.3f}, "
             f"mono_FP={w.mono_fp_rate:.3f}.\n")
    if base_win:
        d_f1 = w.f1 - base_win["F1_multi"]
        d_pk = w.pk - base_win["Pk"]
        d_gf1 = w.gf1 - base_win["F1_global"]
        d_fp = w.mono_fp_rate - base_win["mono_FP"]
        # « bat » = strictement meilleur sur F1_multi ET Pk (objectif réglé-main).
        beats = (w.f1 > base_win["F1_multi"] + 1e-9) and (w.pk < base_win["Pk"] - 1e-9)
        beats_g = w.gf1 > base_win["F1_global"] + 1e-9
        verdict = "**OUI**" if beats else ("partiellement (F1_global)" if beats_g else "**NON**")
        L.append(f"- **e5-large bat-il e5-base réglé-main ? {verdict}.** "
                 f"vs e5-base (F1_multi={base_win['F1_multi']}, Pk={base_win['Pk']}, "
                 f"mono_FP={base_win['mono_FP']}, F1_global={base_win['F1_global']}) : "
                 f"**ΔF1_multi={d_f1:+.3f}**, **ΔPk={d_pk:+.3f}** (négatif = mieux), "
                 f"ΔF1_global={d_gf1:+.3f}, Δmono_FP={d_fp:+.3f}.\n")
    if emb_ctrl:
        e = emb_ctrl
        d_f1e = w.f1 - e["F1_multi"]
        d_pke = w.pk - e["Pk"]
        beats_same = (w.f1 > e["F1_multi"] + 1e-9) and (w.pk < e["Pk"] - 1e-9)
        L.append(f"- **Contrôle MÊME encodeur** : la trajectoire d'embedding de e5-large "
                 f"lui-même (`{e['method']}` W={e['W']}) donne F1_multi={e['F1_multi']}, "
                 f"Pk={e['Pk']}, F1_global={e['F1_global']}. L'attention e5-large fait "
                 f"ΔF1_multi={d_f1e:+.3f}, ΔPk={d_pke:+.3f} → l'attention "
                 f"{'**bat**' if beats_same else 'ne bat pas nettement'} sa propre "
                 f"trajectoire d'embedding.\n")
    L.append("- **Coût du modèle large (honnêteté ressources)** : e5-large = 24 couches "
             "/ 16 têtes / ~560M params vs e5-base 12 couches / 12 têtes / ~278M. "
             "L'extraction d'attention matérialise `[24, 16, n, n]` poids par avis (×2 "
             "couches, ×~1.3 têtes vs base) : RAM et latence du forward nettement "
             "supérieures, cache disque plus lourd. À ne payer QUE si le gain de "
             "segmentation ci-dessus le justifie.\n")
    L.append("- **Jeu** : multi = concaténation de mono-thèmes (frontières nettes par "
             "construction) → borne OPTIMISTE, identique pour e5-base et e5-large : la "
             "comparaison reste équitable.\n")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="Segmentation par attention — e5-large.")
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--out", default=str(DEFAULT_REPORT))
    ap.add_argument("--scores-out", default=str(DEFAULT_SCORES))
    args = ap.parse_args()

    gold_path = Path(args.gold)
    items, _ = load_gold(gold_path)
    print(f"gold: {gold_path.name} — {len(items)} items")

    base_win, base_cfgs = load_base_reference()
    if base_win:
        print(f"réglé-main e5-base: {base_win['layers']}/{base_win['heads']} "
              f"W={base_win['W']} c={base_win['c']} F1_multi={base_win['F1_multi']} "
              f"Pk={base_win['Pk']} F1_global={base_win['F1_global']}")

    feas = A.feasibility_probe(MODEL)
    print(f"faisabilité e5-large: {feas.get('ok')} "
          f"(L={feas.get('n_layers')} H={feas.get('n_heads')})")
    if not feas.get("ok"):
        Path(args.out).write_text(
            build_report(gold_path, items, [], {}, feas, None, base_win, base_cfgs),
            encoding="utf-8")
        print(f"✗ extraction impossible — voir {args.out}")
        return

    print("extraction attention + préparation (e5-large, peut être long sur CPU)…")
    prepared = A.prepare(items, MODEL)
    print("contrôle trajectoire d'embedding (même encodeur)…")
    emb_ctrl = A.emb_control(prepared)
    if emb_ctrl:
        print(f"  embedding e5-large: {emb_ctrl['method']} W={emb_ctrl['W']} "
              f"F1_multi={emb_ctrl['F1_multi']} Pk={emb_ctrl['Pk']} "
              f"F1_global={emb_ctrl['F1_global']}")
    print("balayage configs attention…")
    scores, info = A.sweep_model(MODEL, prepared)
    w = max(scores, key=lambda s: (s.gf1, s.f1))
    print(f"  best e5-large: {w.cfg.layer_set}/{w.cfg.head_agg} W={w.cfg.W} c={w.c} "
          f"F1_multi={w.f1:.3f} Pk={w.pk:.3f} F1_global={w.gf1:.3f}")

    report = build_report(gold_path, items, scores, info, feas, emb_ctrl,
                          base_win, base_cfgs)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"\n✓ {args.out}")

    winner = max(scores, key=lambda s: (s.gf1, s.f1))
    Path(args.scores_out).write_text(json.dumps({
        "gold": gold_path.name, "model": MODEL, "n_items": len(items), "seed": SEED,
        "feasibility": feas,
        "base_reference_e5base": base_win,
        "embedding_control": emb_ctrl,
        "winner": winner.as_row(),
        "info": {"n_layers": info["n_layers"], "n_heads": info["n_heads"],
                 "local_heads": info["local_heads"]},
        "configs": [s.as_row() for s in sorted(scores, key=lambda s: -s.gf1)],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ {args.scores_out}")


if __name__ == "__main__":
    main()
