"""Rapport COMPARATIF graphe-nomic vs graphe-e5base vs attention/change-point.

EXPÉ R&D (read-only). Le balayage brut est produit par `graph_seg.py --sources
nomic-v2` (→ `graph_nomic_scores.json`). Ce script ne RÉ-ÉVALUE rien : il LIT les
JSON de scores déjà produits et compose un rapport honnête centré sur LA question
du brief — « les vecteurs-mots NOMIC-v2 relèvent-ils le graphe de mots, qui a
échoué en e5-base (F1_multi 0.31) ? bat-il l'attention (0.769) ? ».

Pourquoi un rapport dédié et non `graph_seg.build_report` : (1) le brief demande
explicitement la comparaison graphe-nomic **vs graphe-e5base**, que le rapport
auto ne contient pas (il ne charge qu'une source à la fois) ; (2) la phrase
templatée du §5 de `build_report` attribue à TOUTE source le diagnostic e5
« cosinus ~0.9+ » alors que nomic a un seuil dérivé bien plus bas (~0.53–0.63 =
mots MOINS colinéaires) — fait central et honnête de cette expé. On le corrige ici.

Sources de vérité (commitées) :
  - `graph_nomic_scores.json` : balayage nomic-v2 (winner + 120 configs).
  - `graph_scores.json`       : balayage e5-base (référence graphe).
  - `attn_scores.json`, `scores.json` : attention réglé-main, change-point.

ÉCRIT UNIQUEMENT dans `eval/segmentation/`.

    uv run python -m eval.segmentation.graph_nomic_report \
        [--out eval/segmentation/graph_nomic_report.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOMIC_SCORES = HERE / "graph_nomic_scores.json"
E5_SCORES = HERE / "graph_scores.json"
ATTN_SCORES = HERE / "attn_scores.json"
CP_SCORES = HERE / "scores.json"
DEFAULT_OUT = HERE / "graph_nomic_report.md"

RES_GRID = [0.5, 1.0, 1.5, 2.0, 3.0]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _winner_block(d: dict) -> dict | None:
    return d.get("winner")


def _md_table(rows: list[dict], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows)
    return "\n".join([head, sep, body])


def _ref_row(name: str, d: dict, cfg: str) -> dict:
    return {"approche": name, "config": cfg, "Pk": d["Pk"], "WindowDiff": d["WindowDiff"],
            "F1_multi": d["F1_multi"], "P": d["P"], "R": d["R"],
            "mono_FP": d["mono_FP"], "F1_global": d["F1_global"]}


def _graph_row(name: str, w: dict) -> dict:
    return {"approche": name,
            "config": f"k={w['k']} α={w['alpha']} res={w['res']} min={w['min_seg']} "
                      f"thr={w['sim_thr']}",
            "Pk": w["Pk"], "WindowDiff": w["WindowDiff"], "F1_multi": w["F1_multi"],
            "P": w["P"], "R": w["R"], "mono_FP": w["mono_FP"], "F1_global": w["F1_global"]}


def build() -> str:
    nomic = _load(NOMIC_SCORES)
    e5 = _load(E5_SCORES)
    attn = _winner_block(_load(ATTN_SCORES)) if ATTN_SCORES.exists() else None
    cp = _winner_block(_load(CP_SCORES)) if CP_SCORES.exists() else None

    nw = nomic["winner"]
    e5w = e5["winner"]
    n_items = nomic.get("n_items", "?")
    seed = nomic.get("seed", 42)
    configs = nomic["configs"]  # toutes, triées par -F1_global

    L: list[str] = []
    L.append("# Graphe de mots (kNN + Leiden) sur embeddings **NOMIC-v2** — relève-t-il le graphe ? bat-il l'attention ?\n")
    L.append(f"*Jeu : `gold_large.json` — N={n_items}. Vecteurs-mots : **nomic-v2** "
             f"(`embed_word_units`, embed de prod) vs **e5-base** (réf. graphe). CPU, "
             f"seed={seed}. Balayage par `graph_seg.py` ; ce rapport lit les JSON de "
             f"scores et compare (read-only).*\n")

    # 0. Réponse en une ligne
    L.append("## 0. Réponse courte\n")
    L.append(
        f"**NON — deux fois.** nomic-v2 fait *marginalement* mieux que e5-base au niveau "
        f"graphe (F1_multi {nw['F1_multi']:.3f} vs {e5w['F1_multi']:.3f} ; "
        f"F1_global {nw['F1_global']:.3f} vs {e5w['F1_global']:.3f} ; "
        f"Pk {nw['Pk']:.3f} vs {e5w['Pk']:.3f}) mais reste **à un gouffre** de "
        f"l'attention réglé-main (F1_multi {attn['F1_multi'] if attn else '?'}). "
        f"La cause d'échec est IDENTIQUE à e5 : **mono_FP={nw['mono_FP']:.3f}** — Leiden "
        f"sur-coupe quasiment tous les mono. Des embeddings de mots meilleurs (moins "
        f"colinéaires : seuil dérivé ~{nw['sim_thr']} vs ~{e5w['sim_thr']} en e5) ne "
        f"créent PAS la capacité d'**abstention** qui manque.\n")

    # 1. Méthode (rappel court)
    L.append("## 1. Méthode (rappel)\n")
    L.append(
        "Identique à `graph_seg.py`, seule la source de vecteurs-mots change. Mots = "
        "nœuds ; arêtes = **similarité** (kNN cosinus, seuil dérivé μ−σ poolé, zéro "
        "magic-number) **+ séquence** (mots adjacents, poids **α** → quasi-contiguïté). "
        "**Leiden** → communautés ; **contiguïté imposée** (runs maximaux ; micro-runs "
        f"< min_seg fusionnés). Frontières = changements de communauté. "
        f"Balayage : k∈{nomic['grid']['k']} × α∈{nomic['grid']['alpha']} × "
        f"résolution∈{nomic['grid']['resolution']} × min_seg∈{nomic['grid']['min_seg']} "
        f"= {len(configs)} configs.\n")

    # 2. Scorecard
    L.append("## 2. Scorecard — graphe-nomic vs graphe-e5base vs réglé-main (même gold)\n")
    rows = []
    if attn:
        rows.append(_ref_row("**attention réglé-main** (e5-base)", attn,
                             f"{attn['layers']}/{attn['heads']} W={attn['W']} c={attn['c']}"))
    if cp:
        rows.append(_ref_row("change-point (embeddings)", cp,
                             f"{cp['method']} W={cp['W']} pen={cp.get('pen','')}"))
    lr = nomic.get("ref_learned")
    if lr:
        rows.append(_ref_row("_appris LR (réf.)_", lr, lr.get("config", "")))
    rows.append(_graph_row("graphe-Leiden **e5-base** (réf.)", e5w))
    rows.append(_graph_row("**graphe-Leiden nomic-v2**", nw))
    cols = ["approche", "config", "Pk", "WindowDiff", "F1_multi", "P", "R",
            "mono_FP", "F1_global"]
    L.append(_md_table(rows, cols) + "\n")
    L.append("*(Pk/WindowDiff ↓ = mieux, sur les multi ; F1_multi = frontières tol ±1 ; "
             "mono_FP = fraction de mono sur-coupés = mesure d'abstention ; F1_global = "
             "frontières mono+multi, objectif de sélection.)*\n")

    # 3. nomic vs e5base — le delta, ligne à ligne
    L.append("## 3. nomic-v2 vs e5-base au niveau graphe — qu'est-ce qui bouge ?\n")
    drows = [
        {"métrique": "F1_multi (frontières multi)", "e5-base": e5w["F1_multi"],
         "nomic-v2": nw["F1_multi"], "Δ (nomic−e5)": round(nw["F1_multi"] - e5w["F1_multi"], 4)},
        {"métrique": "F1_global (objectif)", "e5-base": e5w["F1_global"],
         "nomic-v2": nw["F1_global"], "Δ (nomic−e5)": round(nw["F1_global"] - e5w["F1_global"], 4)},
        {"métrique": "Pk (↓)", "e5-base": e5w["Pk"], "nomic-v2": nw["Pk"],
         "Δ (nomic−e5)": round(nw["Pk"] - e5w["Pk"], 4)},
        {"métrique": "WindowDiff (↓)", "e5-base": e5w["WindowDiff"],
         "nomic-v2": nw["WindowDiff"], "Δ (nomic−e5)": round(nw["WindowDiff"] - e5w["WindowDiff"], 4)},
        {"métrique": "mono_FP (↓, abstention)", "e5-base": e5w["mono_FP"],
         "nomic-v2": nw["mono_FP"], "Δ (nomic−e5)": round(nw["mono_FP"] - e5w["mono_FP"], 4)},
        {"métrique": "seuil-sim dérivé (μ−σ)", "e5-base": e5w["sim_thr"],
         "nomic-v2": nw["sim_thr"], "Δ (nomic−e5)": round(nw["sim_thr"] - e5w["sim_thr"], 4)},
    ]
    L.append(_md_table(drows, ["métrique", "e5-base", "nomic-v2", "Δ (nomic−e5)"]) + "\n")
    L.append(
        f"- **Ce qui s'améliore** : Pk/WindowDiff baissent nettement "
        f"({e5w['Pk']:.3f}→{nw['Pk']:.3f}) et la sur-coupe est un peu moins violente "
        f"(le winner nomic met {nw['n_clust']} communautés/avis, e5 en mettait "
        f"{e5w['n_clust']}). Le **seuil de similarité dérivé chute** "
        f"({e5w['sim_thr']}→{nw['sim_thr']}) : les vecteurs-mots nomic sont **moins "
        f"colinéaires** que les e5 (μ−σ des cosinus kNN bien plus bas → plus d'« écart » "
        f"exploitable). C'est le seul vrai signe que nomic a plus de structure au niveau mot.\n")
    L.append(
        f"- **Ce qui ne bouge PAS** : mono_FP reste à **{nw['mono_FP']:.3f}** (e5 : "
        f"{e5w['mono_FP']:.3f}). C'est LE point de bascule : nomic ne gagne quasi rien "
        f"sur l'abstention. F1_global ne grimpe que de "
        f"{nw['F1_global'] - e5w['F1_global']:+.3f} et reste **3× sous** l'attention.\n")

    # 4. Top configs nomic
    L.append("## 4. Top 12 configurations graphe-nomic\n")
    cols2 = ["k", "alpha", "res", "min_seg", "sim_thr", "Pk", "WindowDiff",
             "F1_multi", "P", "R", "mono_FP", "mono_cuts", "n_clust", "F1_global"]
    L.append(_md_table(configs[:12], cols2) + "\n")

    # 5. Le nœud : abstention vs détection (recalculé depuis les 120 configs nomic)
    L.append("## 5. Le nœud — abstention ↔ détection, par résolution (nomic-v2)\n")
    L.append("Pour chaque résolution : la config qui **abstient le mieux** (mono_FP min) "
             "vs celle qui **détecte le mieux** (F1_multi max). Si les deux ne coïncident "
             "JAMAIS, aucun réglage global ne sépare « mono cohérent » de « virage de "
             "thème » au niveau MOT.\n")
    arows = []
    for r in RES_GRID:
        sub = [c for c in configs if c["res"] == r]
        if not sub:
            continue
        ab = min(sub, key=lambda c: (c["mono_FP"], -c["F1_multi"]))
        de = max(sub, key=lambda c: (c["F1_multi"], -c["mono_FP"]))
        arows.append({"res": r,
                      "abstient_monoFP": round(ab["mono_FP"], 3),
                      "·_F1_multi": round(ab["F1_multi"], 3),
                      "·_nclust": ab["n_clust"],
                      "détecte_F1_multi": round(de["F1_multi"], 3),
                      "·_monoFP": round(de["mono_FP"], 3),
                      "·_nclust ": de["n_clust"]})
    L.append(_md_table(arows, ["res", "abstient_monoFP", "·_F1_multi", "·_nclust",
                               "détecte_F1_multi", "·_monoFP", "·_nclust "]) + "\n")
    if attn:
        L.append(
            f"*Repère : l'attention tient F1_multi={attn['F1_multi']} ET "
            f"mono_FP={attn['mono_FP']} EN MÊME TEMPS. Aucune ligne ci-dessus ne s'en "
            f"approche : à res basse nomic abstient (mono_FP→0) mais rate AUSSI les multi "
            f"(F1_multi→{arows[0]['·_F1_multi'] if arows else '?'}) ; dès qu'il détecte "
            f"(F1_multi max ~0.33) il re-coupe TOUS les mono (mono_FP→1.0). Les deux "
            f"colonnes ne se rejoignent jamais — exactement comme en e5.*\n")

    # 6. Verdict
    L.append("## 6. Verdict honnête\n")
    beats_attn = attn and (nw["F1_multi"] > attn["F1_multi"]) and (nw["Pk"] < attn["Pk"])
    beats_cp = cp and nw["F1_multi"] > cp["F1_multi"]
    L.append(
        f"- **Bat-il l'attention (F1_multi={attn['F1_multi'] if attn else '?'}, "
        f"Pk={attn['Pk'] if attn else '?'}, mono_FP={attn['mono_FP'] if attn else '?'}) ? "
        f"{'OUI' if beats_attn else '**NON**'}.** "
        f"ΔF1_multi={nw['F1_multi'] - attn['F1_multi']:+.3f}, "
        f"ΔPk={nw['Pk'] - attn['Pk']:+.3f}, "
        f"ΔF1_global={nw['F1_global'] - attn['F1_global']:+.3f}.\n")
    if cp:
        L.append(
            f"- **Bat-il le change-point (F1_multi={cp['F1_multi']}) ? "
            f"{'OUI' if beats_cp else '**NON**'}.** "
            f"ΔF1_multi={nw['F1_multi'] - cp['F1_multi']:+.3f}.\n")
    L.append(
        f"- **nomic relève-t-il le graphe ?** À peine. F1_global "
        f"{e5w['F1_global']:.3f}→{nw['F1_global']:.3f} ({nw['F1_global'] - e5w['F1_global']:+.3f}), "
        f"Pk {e5w['Pk']:.3f}→{nw['Pk']:.3f}. Mieux, mais dans le même régime d'échec : "
        f"le verdict graphe (NON) est inchangé.\n")
    L.append(
        f"- **Pourquoi nomic ne sauve pas le graphe — l'abstention, pas la colinéarité.** "
        f"On pouvait croire que l'échec e5 venait de vecteurs-mots trop colinéaires "
        f"(seuil dérivé ~{e5w['sim_thr']}, cosinus ~0.9+ → graphe quasi-structureless). "
        f"nomic INFIRME cette explication-là : son seuil dérivé tombe à ~{nw['sim_thr']} "
        f"(mots bien moins colinéaires, donc plus de structure de similarité disponible) "
        f"— et POURTANT mono_FP reste à {nw['mono_FP']:.3f}. Le vrai mal n'est donc pas "
        f"le manque de signal de similarité, mais que **Leiden ne sait pas s'abstenir** : "
        f"à résolution fixe il maximise la modularité PAR document et trouve toujours "
        f"une partition (≥2 communautés) même sur un mono cohérent. C'est structurel à "
        f"l'objectif Leiden, pas une affaire d'embedding.\n")
    L.append(
        f"- **Ce que l'attention réussit et que le graphe ne peut pas imiter** : un "
        f"seuil GLOBAL `μ−cσ` calibré sur tout le corpus. Sur un mono, le signal ne "
        f"descend jamais sous ce seuil → **0 frontière** (mono_FP={attn['mono_FP'] if attn else '?'}). "
        f"Leiden n'a aucun équivalent global : sa « résolution » est un curseur de "
        f"granularité par-document, pas un seuil d'abstention transférable (§5).\n")
    L.append(
        "- **Conclusion** : le graphe-Leiden de mots sur **nomic-v2 NE BAT NI** "
        f"l'attention réglé-main ({attn['F1_multi'] if attn else '0.769'}) **NI** le "
        f"change-point ({cp['F1_multi'] if cp else '0.44'}) ; il bat seulement, et de peu, "
        f"le graphe-e5base ({e5w['F1_multi']}). De meilleurs embeddings de mots déplacent "
        f"le seuil de similarité mais ne créent pas l'abstention — qui est le verrou. "
        f"Piste (déjà notée pour e5, inchangée) : graphe au niveau PHRASE/clause + "
        f"critère d'abstention explicite (ne couper que si le gain de modularité dépasse "
        f"un seuil global), ce qui reviendrait à réinventer le seuil calibré de "
        f"l'attention par un détour plus coûteux.\n")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rapport comparatif graphe-nomic vs e5base.")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    Path(args.out).write_text(build(), encoding="utf-8")
    print(f"✓ {args.out}")


if __name__ == "__main__":
    main()
