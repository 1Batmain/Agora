"""PANEL AVEUGLE — cible de clivage (a) actuel vs (b) distinctif+contraste, sur CE corpus.

La métrique fit (research/cleavage_ab.py) est FLAT-à-négative pour (b) ici — mais elle l'était
AUSSI sur tiktok (0.801→0.800), où c'est le PANEL AVEUGLE qui a tranché en faveur de (b) (7-2).
Ce script rejoue ce panel sur lutte-contre-les-fausses-informations pour trancher le TRANSFERT.

Protocole identique à `research/stance_panel_judge.py` : juge mistral-large T=0, 3 passes de
consignes variées, cibles X/Y anonymisées (ordre seedé), titre + 3 contributions de contexte.
Décision de paire = majorité des 3 passes. Aveugle : clé (a/b)↔lettre dépouillée après.

Read-only sur caches + `cleavage_ab_deriv.<ds>.json` (les dérivations déjà payées). N'écrit que
sous `research/`.
    MISTRAL_API_KEY=$(cat var/mistral.key) uv run python research/cleavage_ab_panel.py \
        --dataset lutte-contre-les-fausses-informations
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from pipeline.cluster import mistral_client

ROOT = Path(__file__).resolve().parent.parent
CACHED = ROOT / "cached_data"
SEED = 42
JUDGE_MODEL = "mistral-large-latest"

_ACCENTS = [
    "Juge globalement la meilleure cible.",
    "Accent : la cible doit capter le débat CENTRAL du thème, pas une facette bruyante.",
    "Accent : la cible doit rester POLAIRE et débattable, sans être un énoncé passe-partout "
    "interchangeable avec un thème voisin.",
]


def judge_pair(title: str, reps: list[str], x: str, y: str, accent: str) -> str | None:
    ctx = "\n".join(f"- {r[:160]}" for r in reps[:3])
    system = (
        "Tu es analyste de consultations citoyennes. On te donne le TITRE d'un thème, 3 "
        "contributions de CONTEXTE, et DEUX cibles candidates X et Y (des propositions polaires "
        "sur lesquelles les citoyens seraient POUR ou CONTRE). Choisis la MEILLEURE : celle qui "
        "capte le débat CENTRAL du thème EN restant polaire, débattable et spécifique à CE thème. "
        + accent + ' Réponds en JSON strict : {"choix":"X"} ou {"choix":"Y"} — rien d\'autre.'
    )
    user = f"TITRE : {title}\n\nCONTEXTE :\n{ctx}\n\nCIBLE X : {x}\nCIBLE Y : {y}"
    try:
        raw = mistral_client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=JUDGE_MODEL, temperature=0.0, max_tokens=60, json_mode=True)
        c = str(json.loads(raw).get("choix", "")).strip().upper()
    except (mistral_client.MistralError, json.JSONDecodeError, TypeError):
        return None
    return c if c in ("X", "Y") else None


def run(dataset: str) -> dict:
    base = CACHED / dataset
    rows = json.loads((ROOT / "research" / f"cleavage_ab_deriv.{dataset}.json").read_text())
    an = {t["id"]: t for t in json.loads((base / "analysis" / "analysis.json").read_text())["themes"]}
    rng = random.Random(SEED)

    per_pair: list[dict] = []
    wins = Counter()          # décisions de paire (majorité)
    for r in rows:
        # cibles identiques → paire ininformative, on saute
        if r["prop_a"].strip().lower() == r["prop_b"].strip().lower():
            continue
        reps = an.get(r["theme_id"], {}).get("representative_claims") or []
        # X/Y ← a/b mélangé (aveugle)
        swap = rng.random() < 0.5
        x_meth, y_meth = ("b", "a") if swap else ("a", "b")
        x = r["prop_b"] if swap else r["prop_a"]
        y = r["prop_a"] if swap else r["prop_b"]
        votes = []
        for accent in _ACCENTS:
            c = judge_pair(r["title"], reps, x, y, accent)
            if c is None:
                continue
            votes.append(x_meth if c == "X" else y_meth)
        if not votes:
            continue
        maj = Counter(votes).most_common(1)[0][0]
        wins[maj] += 1
        per_pair.append({"theme_id": r["theme_id"], "votes": votes, "winner": maj,
                         "prop_a": r["prop_a"], "prop_b": r["prop_b"]})

    summary = {"dataset": dataset, "n_pairs": len(per_pair),
               "b_wins": wins["b"], "a_wins": wins["a"]}
    print(f"[cleavage-panel] {len(per_pair)} paires jugées")
    print(f"[cleavage-panel] (b) gagne {wins['b']} · (a) gagne {wins['a']}")
    return {"summary": summary, "pairs": per_pair}


def main() -> None:
    ap = argparse.ArgumentParser(description="Panel aveugle cible (a) vs (b) — transfert (R&D).")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=str(Path(__file__).parent / "cleavage_ab_panel_votes.json"))
    args = ap.parse_args()
    if not mistral_client.available():
        raise SystemExit("Pas de clé Mistral. Abandon.")
    result = run(args.dataset)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[cleavage-panel] → {args.out}")


if __name__ == "__main__":
    main()
