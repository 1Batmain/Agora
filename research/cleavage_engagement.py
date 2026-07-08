"""CIBLE (c) « pour/contre-cadrée » vs (a) actuelle — bench ENGAGEMENT (idée Bob).

Hypothèse : dériver la cible SOUS LA CONTRAINTE « telle que chaque témoignage soit clairement
POUR ou CONTRE » donne au LLM un critère opérationnel (au lieu de l'abstrait « résume le débat »)
→ cible moins diffuse → MOINS de « nuance » (abstention) → PLUS d'engagement. C'est le maillon
faible mesuré (33 % de feuilles `impur` sur lutte). La métrique qui tranche = l'ENGAGEMENT, pas le
fit (plat pour la variante (b)).

Protocole (contrôlé, entrées identiques) : par feuille, on dérive la cible (a) [prompt actuel]
ET (c) [pour/contre-cadré], on RE-CLASSE la stance des MÊMES claims envers chacune, et on compare
%nuance / engagement / #impur. Réutilise le prompt/stance SERVIS (`backend.build_opinion`).

Lit `research/emerge_cache/lutte.../` (via `emerge_build.py`, membres par feuille y compris
impur). N'écrit que sous `research/`. Zéro touche prod.
    MISTRAL_API_KEY=$(cat var/mistral.key) \
    uv run python research/cleavage_engagement.py --dataset lutte-contre-les-fausses-informations
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from backend.build_opinion import (
    MODEL, MIN_CLAIMS, MIN_ENGAGEMENT, cleavage_system, _chat_retry, run_stance, aggregate,
)
from pipeline.cluster import mistral_client

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "research" / "emerge_cache"
REP_FOR_DERIVE = 14
CAP_CLAIMS = 40         # borne le coût stance par feuille


# Variante (c) : la contrainte pour/contre DEVIENT le critère de dérivation (idée Bob).
def cleavage_c(title: str) -> str:
    return (
        "Tu es analyste de consultations citoyennes. On te donne le TITRE d'un THÈME, ses "
        "MOTS-CLÉS et des CONTRIBUTIONS verbatim. Formule UNE PROPOSITION D'ACTION telle que, "
        "pour CHAQUE contribution, on puisse répondre SANS AMBIGUÏTÉ à : « ce témoignage est-il "
        "POUR ou CONTRE cette proposition ? ». La proposition doit donc être CONCRÈTE, BINAIRE et "
        "TRANCHANTE (on peut y être clairement favorable OU défavorable), et rester CENTRALE au "
        f"thème « {title} ». ÉVITE toute formulation vague, générale ou consensuelle qui "
        "laisserait des gens « sans avis ». COURTE (≤12 mots), à l'infinitif ou nominale — ex. "
        "« rendre le vote obligatoire », « réduire le nombre d'élus ». "
        "Réponds en JSON strict : {\"objet\":\"<proposition>\",\"justif\":\"<≤14 mots>\"}."
    )


def derive(system: str, title: str, keywords: list[str], texts: list[str]) -> str:
    kw = ", ".join(keywords[:10])
    contribs = "\n".join(f"- {t[:160]}" for t in texts[:REP_FOR_DERIVE])
    user = f"MOTS-CLÉS : {kw}\n\nCONTRIBUTIONS :\n{contribs}"
    try:
        raw = _chat_retry([{"role": "system", "content": system},
                           {"role": "user", "content": user}], model=MODEL, max_tokens=200)
        return str(json.loads(raw).get("objet", "")).strip() or title
    except (mistral_client.MistralError, json.JSONDecodeError, TypeError):
        return title


def _central_order(gis: list[int], vecs: np.ndarray) -> list[int]:
    s = vecs[gis].sum(axis=0); n = np.linalg.norm(s)
    c = s / n if n > 0 else s
    return [gis[i] for i in np.argsort(-(vecs[gis] @ c))]


def run(dataset: str) -> dict:
    d = CACHE / dataset
    vecs = np.load(d / "claim_vecs.npz")["vecs"].astype(np.float32)
    claims = [json.loads(l) for l in (d / "claims.jsonl").read_text().splitlines()]
    texts_all = [c["text"] for c in claims]
    leaves = json.loads((d / "leaves.json").read_text())

    rows = []
    for lf in leaves:
        gis = lf["member_gis"]
        if len(gis) < MIN_CLAIMS:
            continue
        ordered = _central_order(gis, vecs)
        rep = [texts_all[g] for g in ordered[:REP_FOR_DERIVE]]
        capped = ordered[:CAP_CLAIMS]
        items = [(i, texts_all[g]) for i, g in enumerate(capped)]
        title, kw = lf["title"], lf["keywords"]

        out = {"theme_id": lf["theme_id"], "title": title, "n": len(items)}
        for tag, system in (("a", cleavage_system(title)), ("c", cleavage_c(title))):
            cible = derive(system, title, kw, rep)
            st = run_stance(cible, items, model=MODEL)
            counts = Counter(st[i]["stance"] for i, _ in items if i in st)
            agg = aggregate(lf["theme_id"], cible, counts, len(items))
            out[tag] = {"cible": cible, "nuance": counts.get("nuance", 0),
                        "engagement": agg["engagement"], "profil": agg["profil"],
                        "pct_nuance": round(counts.get("nuance", 0) / len(items), 3)}
        rows.append(out)
        print(f"  [{out['theme_id']}] n={out['n']:2d} "
              f"| a eng={out['a']['engagement']:.2f} nu={out['a']['pct_nuance']:.2f} ({out['a']['profil'][:4]}) "
              f"| c eng={out['c']['engagement']:.2f} nu={out['c']['pct_nuance']:.2f} ({out['c']['profil'][:4]})")

    def _agg(tag):
        eng = [r[tag]["engagement"] for r in rows]
        return {"engagement_mean": round(statistics.mean(eng), 3),
                "engagement_median": round(statistics.median(eng), 3),
                "pct_nuance_mean": round(statistics.mean(r[tag]["pct_nuance"] for r in rows), 3),
                "n_impur": sum(1 for r in rows if r[tag]["profil"] == "impur"),
                "n_clivant": sum(1 for r in rows if r[tag]["profil"] == "clivant"),
                "n_consensuel": sum(1 for r in rows if r[tag]["profil"] == "consensuel")}
    wins_c = sum(1 for r in rows if r["c"]["engagement"] > r["a"]["engagement"] + 0.02)
    wins_a = sum(1 for r in rows if r["a"]["engagement"] > r["c"]["engagement"] + 0.02)
    summary = {"dataset": dataset, "n_leaves": len(rows), "min_engagement_threshold": MIN_ENGAGEMENT,
               "a": _agg("a"), "c": _agg("c"),
               "leaves_c_more_engaged": wins_c, "leaves_a_more_engaged": wins_a}
    print("\n=== ENGAGEMENT (a) actuel vs (c) pour/contre-cadré ===")
    print(f"(a): eng moy {summary['a']['engagement_mean']} · %nuance {summary['a']['pct_nuance_mean']} "
          f"· impur {summary['a']['n_impur']}/{len(rows)} (clivant {summary['a']['n_clivant']} consensuel {summary['a']['n_consensuel']})")
    print(f"(c): eng moy {summary['c']['engagement_mean']} · %nuance {summary['c']['pct_nuance_mean']} "
          f"· impur {summary['c']['n_impur']}/{len(rows)} (clivant {summary['c']['n_clivant']} consensuel {summary['c']['n_consensuel']})")
    print(f"par feuille : (c) + engagée {wins_c} · (a) + engagée {wins_a}")
    return {"summary": summary, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description="Bench engagement cible (a) vs (c) pour/contre — R&D.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not mistral_client.available():
        raise SystemExit("Pas de clé Mistral. Abandon.")
    out = args.out or str(Path(__file__).parent / f"cleavage_engagement_{args.dataset}.json")
    res = run(args.dataset)
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"[engagement] → {out}")


if __name__ == "__main__":
    main()
