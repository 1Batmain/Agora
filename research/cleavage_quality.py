"""QUALITÉ de l'engagement — (a) sur-classe-t-il, ou (c) sous-classe-t-il ? (objection Bob).

Le bench `cleavage_engagement.py` a montré (c) < (a) en engagement. Objection Bob : moins
d'engagement ≠ moins bien si les décisions restantes sont de meilleure QUALITÉ (ne pas forcer
d'opinion si la position n'est pas claire). Ce script mesure la QUALITÉ, pas la quantité :

  1. CONFIANCE : sur les claims où (a) TRANCHE mais (c) s'ABSTIENT, quelle est la confiance
     auto-déclarée de (a) ? (calibrée d'après le gold : high ≫ low en accuracy.)
  2. JUGE AVEUGLE : pour ces mêmes désaccords, un juge neutre statue « position CLAIRE sur la
     cible, ou réellement sans avis ? ». « pas claire » ⇒ (a) SUR-classe (Bob a raison) ;
     « claire » ⇒ (a) capte une vraie position que la cible étroite de (c) ratait.

Réutilise les cibles DÉJÀ dérivées (`cleavage_engagement_<ds>.json`) — pas de re-dérivation.
Lit `emerge_cache/<ds>/`. Zéro touche prod.
    MISTRAL_API_KEY=$(cat var/mistral.key) \
    uv run python research/cleavage_quality.py --dataset lutte-contre-les-fausses-informations
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from backend.build_opinion import MODEL, run_stance, _chat_retry
from pipeline.cluster import mistral_client

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "research" / "emerge_cache"
CAP_CLAIMS = 40


_JUDGE_SYSTEM = (
    "Tu es analyste de consultations citoyennes. On te donne une CIBLE (proposition d'action "
    "débattable) et UNE contribution citoyenne. Question : la contribution prend-elle une "
    "POSITION CLAIRE sur CETTE action — la soutenir OU s'y opposer de façon défendable — ou "
    "est-elle réellement SANS position claire sur cette action précise (hors-sujet, purement "
    "descriptive, ambivalente) ? Juge la position sur l'ACTION, pas la tonalité générale. "
    'Réponds en JSON strict : {"claire":true|false}.'
)


def judge_clear(cible: str, text: str) -> bool | None:
    messages = [{"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": f"CIBLE : {cible}\n\nCONTRIBUTION : {text}"}]
    try:
        raw = _chat_retry(messages, model=MODEL, max_tokens=20)   # backoff sur 429 (RPM large bas)
        return bool(json.loads(raw).get("claire"))
    except (mistral_client.MistralError, json.JSONDecodeError, TypeError):
        return None


def _central_order(gis, vecs):
    s = vecs[gis].sum(axis=0); n = np.linalg.norm(s)
    c = s / n if n > 0 else s
    return [gis[i] for i in np.argsort(-(vecs[gis] @ c))]


def run(dataset: str) -> dict:
    d = CACHE / dataset
    vecs = np.load(d / "claim_vecs.npz")["vecs"].astype(np.float32)
    claims = [json.loads(l) for l in (d / "claims.jsonl").read_text().splitlines()]
    texts_all = [c["text"] for c in claims]
    eng = json.loads((ROOT / "research" / f"cleavage_engagement_{dataset}.json").read_text())

    disagreements = []            # (a) tranche, (c) nuance
    conf_a_extra = Counter()      # confiance de (a) sur ces claims
    leaves = {l["theme_id"]: l for l in json.loads((d / "leaves.json").read_text())}
    for row in eng["rows"]:
        lf = leaves[row["theme_id"]]
        ordered = _central_order(lf["member_gis"], vecs)[:CAP_CLAIMS]
        items = [(i, texts_all[g]) for i, g in enumerate(ordered)]
        st_a = run_stance(row["a"]["cible"], items, model=MODEL)
        st_c = run_stance(row["c"]["cible"], items, model=MODEL)
        for i, g in enumerate(ordered):
            sa, sc = st_a.get(i), st_c.get(i)
            if not sa or not sc:
                continue
            if sa["stance"] in ("favorable", "defavorable") and sc["stance"] == "nuance":
                conf_a_extra[sa.get("confidence", "low")] += 1
                disagreements.append({"theme_id": row["theme_id"], "cible_a": row["a"]["cible"],
                                      "text": texts_all[g], "stance_a": sa["stance"],
                                      "conf_a": sa.get("confidence", "low")})
    print(f"[quality] {len(disagreements)} claims où (a) tranche & (c) s'abstient")
    print(f"[quality] confiance de (a) sur ces claims : {dict(conf_a_extra)}")

    # Juge aveugle : (a) a-t-il raison de trancher (position claire sur cible_a) ?
    clear = notclear = 0
    for dz in disagreements:
        v = judge_clear(dz["cible_a"], dz["text"])
        if v is None:
            continue
        dz["judge_claire"] = v
        clear += v; notclear += (not v)
    total = clear + notclear
    summary = {
        "dataset": dataset, "n_disagreements": len(disagreements),
        "conf_a_on_extra": dict(conf_a_extra),
        "judge_a_clear": clear, "judge_a_notclear": notclear,
        "a_overclassif_rate": round(notclear / total, 3) if total else None,
    }
    print(f"[quality] juge : (a) position CLAIRE {clear}/{total} · PAS claire {notclear}/{total} "
          f"→ taux de sur-classement de (a) = {summary['a_overclassif_rate']}")
    return {"summary": summary, "disagreements": disagreements}


def main() -> None:
    ap = argparse.ArgumentParser(description="Qualité engagement (a) vs (c) — R&D.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not mistral_client.available():
        raise SystemExit("Pas de clé Mistral. Abandon.")
    out = args.out or str(Path(__file__).parent / f"cleavage_quality_{args.dataset}.json")
    res = run(args.dataset)
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"[quality] → {out}")


if __name__ == "__main__":
    main()
