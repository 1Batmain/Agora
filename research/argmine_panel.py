"""PANEL AVEUGLE — argument mining : CURRENT (paraphrase) vs V-CLUSTER vs V-SELECT.

Tranche le compromis que le contrôle verbatim ne mesure pas : le format verbatim SEUL
(décision Bob) coûte-t-il en LISIBILITÉ / REPRÉSENTATIVITÉ face à la paraphrase servie, et
V-CLUSTER (medoïde offline) ou V-SELECT (LLM sélectionne) rend-il le mieux le débat ?

Protocole (aligné sur `research/stance_panel_judge.py` / verdict stance_target_ab) :
  * Juge `mistral-large-latest`, T=0. Pour chaque thème (intersection des 3 méthodes), on
    montre les 3 jeux d'arguments ANONYMISÉS X/Y/Z (ordre mélangé, seedé) + le titre du thème.
  * Le juge RANGE X/Y/Z (1er = meilleur) sur deux critères EXPLICITES : (1) FIDÉLITÉ à la voix
    citoyenne, (2) REPRÉSENTATIVITÉ + clarté des arguments principaux du débat.
  * 3 passes par thème, consignes légèrement variées (neutre / accent fidélité / accent
    distinctivité) pour ne pas jouer 3× le même biais. Décision = agrégat (plurality des 1ers
    + rang moyen). Aveugle : la clé méthode↔lettre n'est dépouillée qu'APRÈS les votes.

Lecture read-only des caches servis + `research/argmine_verbatim_results.json`. N'écrit que
sous `research/`. Repro :
    MISTRAL_API_KEY=$(cat var/mistral.key) uv run python research/argmine_panel.py \
        --dataset lutte-contre-les-fausses-informations
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from pipeline.cluster import mistral_client

ROOT = Path(__file__).resolve().parent.parent
CACHED = ROOT / "cached_data"
SEED = 42
JUDGE_MODEL = "mistral-large-latest"

_VARIANTS = [
    ("neutre", "Juge globalement la meilleure restitution du débat."),
    ("fidélité", "Pondère surtout la FIDÉLITÉ : les arguments doivent sonner comme la voix "
                 "réelle des citoyens, sans reformulation qui ajoute ou lisse."),
    ("distinctivité", "Pondère surtout la RICHESSE : le jeu doit couvrir les idées PRINCIPALES "
                      "et DISTINCTES du camp, sans redite ni argument passe-partout."),
]


def _fmt_set(args: list[dict]) -> str:
    return "\n".join(f"  - [{a.get('stance','?')}] {a['argument'][:200]}" for a in args) or "  (aucun)"


def judge_theme(title: str, sets: dict[str, list[dict]], accent: str) -> list[str] | None:
    """Renvoie le classement des lettres (1er→dernier), ou None si échec."""
    letters = sorted(sets)  # X, Y, Z (déjà mélangés à l'attribution)
    blocks = "\n\n".join(f"JEU {L} :\n{_fmt_set(sets[L])}" for L in letters)
    system = (
        "Tu es analyste de consultations citoyennes. On te montre le TITRE d'un thème et "
        f"{len(letters)} JEUX d'arguments (chacun issu d'une méthode différente, anonymisée). "
        "Chaque jeu prétend restituer les arguments principaux d'un camp. CLASSE les jeux du "
        "MEILLEUR au moins bon sur DEUX critères : (1) FIDÉLITÉ à la voix citoyenne réelle, "
        "(2) REPRÉSENTATIVITÉ et clarté des arguments principaux du débat. " + accent + " "
        'Réponds en JSON strict : {"rang":["<lettre1>","<lettre2>",...]} — meilleur d\'abord, '
        "toutes les lettres présentes, UNIQUEMENT les lettres (pas d'objet, pas de "
        "justification, pas de recopie des arguments), rien d'autre."
    )
    user = f"TITRE : {title}\n\n{blocks}"
    try:
        raw = mistral_client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=JUDGE_MODEL, temperature=0.0, max_tokens=400, json_mode=True)
        rang = json.loads(raw).get("rang", [])
    except (mistral_client.MistralError, json.JSONDecodeError, TypeError):
        return None

    def _letter(x) -> str:
        # large renvoie soit "Z", soit {"jeu"/"lettre"/"set":"Z", ...} — on extrait la lettre.
        if isinstance(x, dict):
            x = x.get("jeu") or x.get("lettre") or x.get("set") or x.get("nom") or ""
        return str(x).strip().upper()

    rang = [L for L in (_letter(x) for x in rang) if L in letters]
    # garde-fou : classement complet et sans doublon
    if sorted(set(rang)) != sorted(letters) or len(rang) != len(letters):
        return None
    return rang


def run(dataset: str) -> dict:
    base = CACHED / dataset
    cur_themes = json.loads((base / "analysis" / "arguments.json").read_text())["themes"]
    current = {t["theme_id"]: t["arguments"] for t in cur_themes}
    title_of = {t["theme_id"]: t.get("title", t["theme_id"]) for t in cur_themes}
    res = json.loads((ROOT / "research" / "argmine_verbatim_results.json").read_text())
    vcluster = {t["theme_id"]: t["arguments"] for t in res["methods"]["vcluster"]["themes"]}
    vselect = {t["theme_id"]: t["arguments"] for t in res["methods"]["vselect"]["themes"]}

    methods = {"current": current, "vcluster": vcluster, "vselect": vselect}
    themes = sorted(set(current) & set(vcluster) & set(vselect))
    print(f"[panel] {dataset} · {len(themes)} thèmes en intersection des 3 méthodes")

    rng = random.Random(SEED)
    votes: list[dict] = []
    # Agrégats : points de rang (1er=len-1 … dernier=0, style Borda) + comptes de 1res places.
    borda: Counter = Counter()
    firsts: Counter = Counter()
    n_ballots = 0
    for tid in themes:
        title = title_of.get(tid, tid)
        # attribution ALÉATOIRE lettre↔méthode, par thème (aveugle)
        order = ["current", "vcluster", "vselect"]
        rng.shuffle(order)
        letter_of = {m: L for m, L in zip(order, ["X", "Y", "Z"])}
        method_of = {L: m for m, L in letter_of.items()}
        sets = {letter_of[m]: methods[m][tid] for m in order}
        for accent_name, accent in _VARIANTS:
            rank = judge_theme(title, sets, accent)
            if rank is None:
                continue
            n_ballots += 1
            ranked_methods = [method_of[L] for L in rank]
            firsts[ranked_methods[0]] += 1
            for pos, meth in enumerate(ranked_methods):
                borda[meth] += (len(ranked_methods) - 1 - pos)
            votes.append({"theme_id": tid, "accent": accent_name,
                          "rank_methods": ranked_methods,
                          "letter_of": letter_of})

    summary = {
        "dataset": dataset, "n_themes": len(themes), "n_ballots": n_ballots,
        "firsts": dict(firsts),
        "borda": dict(borda),
        "mean_rank": {m: round(1 + (n_ballots * 2 - borda[m]) / n_ballots, 3)
                      if n_ballots else None for m in methods},
    }
    print(f"[panel] bulletins : {n_ballots}")
    print(f"[panel] 1res places : {dict(firsts)}")
    print(f"[panel] Borda (haut=mieux) : {dict(borda)}")
    print(f"[panel] rang moyen (bas=mieux) : {summary['mean_rank']}")
    return {"summary": summary, "votes": votes}


def main() -> None:
    ap = argparse.ArgumentParser(description="Panel aveugle argument mining (R&D).")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=str(Path(__file__).parent / "argmine_panel_votes.json"))
    args = ap.parse_args()
    if not mistral_client.available():
        raise SystemExit("Pas de clé Mistral (MISTRAL_API_KEY). Abandon.")
    result = run(args.dataset)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[panel] → {args.out}")


if __name__ == "__main__":
    main()
