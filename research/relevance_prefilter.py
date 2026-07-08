"""PRÉ-FILTRE DE PERTINENCE avant stance — corrige le sur-classement par association.

Défaut mesuré (`cleavage_quality_note.md`) : le STANCE servi classe « favorable » des
témoignages TANGENTIELS (même thème, action différente) parce que la cible est large et la
consigne anti-abstention pousse à trancher → 79 % de sur-classement sur les cas litigieux,
CONFIANT (la confiance ne le détecte pas).

Correctif testé, CIBLE-AGNOSTIQUE : avant de classer la stance, un pré-filtre demande « ce
témoignage PORTE-t-il sur CETTE action précise ? » (pertinence, indépendante du pour/contre).
Non-pertinent → abstention forcée (pas de stance). On garde la cible LARGE (a) telle quelle.

Mesure : sur lutte (cible (a) déjà dérivée), comparer stance BASELINE vs FILTRÉE. Les claims
qui basculent décidé→abstenu sont jugés en aveugle : le pré-filtre retire-t-il bien les
sur-classements (juge « pas de position claire ») SANS jeter de vraies positions ?

Réutilise `cleavage_engagement_<ds>.json` (cibles a) + `emerge_cache/<ds>/`. Zéro touche prod.
    MISTRAL_API_KEY=$(cat var/mistral.key) \
    uv run python research/relevance_prefilter.py --dataset lutte-contre-les-fausses-informations
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from backend.build_opinion import MODEL, run_stance, _chat_retry
from research.cleavage_quality import judge_clear, _central_order, CAP_CLAIMS
from pipeline.cluster import mistral_client

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "research" / "emerge_cache"
BATCH = 10


_REL_SYSTEM = (
    "Tu es analyste de consultations citoyennes. On te donne une ACTION (proposition débattable) "
    "et des CONTRIBUTIONS citoyennes numérotées. Pour CHAQUE contribution, indique si elle PORTE "
    "SUR CETTE ACTION précise — si elle en parle, la vise, ou prend clairement position dessus — "
    "INDÉPENDAMMENT du fait qu'elle soit pour ou contre. Réponds NON (porte=false) si la "
    "contribution parle d'une AUTRE mesure/action (même sur le même thème général), ou est "
    "purement descriptive/générale sans viser cette action. Ne juge PAS le pour/contre, juste la "
    'PERTINENCE. Réponds en JSON strict : {"results":[{"i":<int>,"porte":true|false}]}.'
)


def relevance_batch(cible: str, items: list[tuple[int, str]]) -> dict[int, bool]:
    out: dict[int, bool] = {}
    for start in range(0, len(items), BATCH):
        batch = items[start:start + BATCH]
        lines = "\n".join(f"[{i}] {t}" for i, t in batch)
        messages = [{"role": "system", "content": _REL_SYSTEM},
                    {"role": "user", "content": f"ACTION : {cible}\n\nCONTRIBUTIONS :\n{lines}"}]
        try:
            raw = _chat_retry(messages, model=MODEL, max_tokens=800)
            for rec in json.loads(raw).get("results", []):
                out[int(rec["i"])] = bool(rec.get("porte"))
        except (mistral_client.MistralError, json.JSONDecodeError, TypeError, ValueError, KeyError):
            for i, _ in batch:
                out.setdefault(i, True)   # repli prudent : garder (ne pas filtrer sur échec)
    for i, _ in items:
        out.setdefault(i, True)
    return out


def run(dataset: str) -> dict:
    d = CACHE / dataset
    vecs = np.load(d / "claim_vecs.npz")["vecs"].astype(np.float32)
    claims = [json.loads(l) for l in (d / "claims.jsonl").read_text().splitlines()]
    texts_all = [c["text"] for c in claims]
    leaves = {l["theme_id"]: l for l in json.loads((d / "leaves.json").read_text())}
    eng = json.loads((ROOT / "research" / f"cleavage_engagement_{dataset}.json").read_text())

    base_dec = base_n = filt_dec = 0
    flipped = []                # décidé baseline → abstenu par le filtre
    kept_sample = []            # décidé ET pertinent (pour précision maintenue)
    for row in eng["rows"]:
        cible = row["a"]["cible"]
        lf = leaves[row["theme_id"]]
        ordered = _central_order(lf["member_gis"], vecs)[:CAP_CLAIMS]
        items = [(i, texts_all[g]) for i, g in enumerate(ordered)]
        rel = relevance_batch(cible, items)
        st = run_stance(cible, items, model=MODEL)
        for i, g in enumerate(ordered):
            s = st.get(i)
            if not s:
                continue
            base_n += 1
            decided = s["stance"] in ("favorable", "defavorable")
            base_dec += decided
            if decided and not rel.get(i, True):
                flipped.append({"theme_id": row["theme_id"], "cible": cible,
                                "text": texts_all[g], "stance": s["stance"],
                                "conf": s.get("confidence", "low")})
            elif decided:
                filt_dec += 1
                kept_sample.append({"cible": cible, "text": texts_all[g]})

    # Juge aveugle sur les BASCULES : le filtre a-t-il retiré des sur-classements (pas clairs) ?
    good_removals = bad_removals = 0
    for f in flipped:
        v = judge_clear(f["cible"], f["text"])
        if v is None:
            continue
        f["judge_claire"] = v
        good_removals += (not v)   # pas clair → bon retrait
        bad_removals += v          # clair → mauvais retrait (vraie position jetée)
    tot_rm = good_removals + bad_removals

    # Précision maintenue : juge un échantillon des GARDÉS-décidés.
    import random
    rng = random.Random(42)
    samp = rng.sample(kept_sample, min(20, len(kept_sample)))
    kept_clear = sum(1 for k in samp if judge_clear(k["cible"], k["text"]))

    summary = {
        "dataset": dataset, "n_claims": base_n,
        "baseline_decided": base_dec, "filtered_decided": filt_dec,
        "removed_by_filter": len(flipped),
        "removals_good_notclear": good_removals, "removals_bad_clear": bad_removals,
        "removal_precision": round(good_removals / tot_rm, 3) if tot_rm else None,
        "kept_sample": len(samp), "kept_judged_clear": kept_clear,
        "kept_precision": round(kept_clear / len(samp), 3) if samp else None,
        "conf_of_removed": dict(Counter(f["conf"] for f in flipped)),
    }
    print(f"[relevance] {base_n} claims · baseline décidés {base_dec} → filtrés {filt_dec} "
          f"(retirés {len(flipped)})")
    print(f"[relevance] retraits : {good_removals} sur-classements (pas clairs) · "
          f"{bad_removals} vraies positions jetées → précision du filtre {summary['removal_precision']}")
    print(f"[relevance] confiance des retirés : {summary['conf_of_removed']}")
    print(f"[relevance] précision des GARDÉS (échantillon {len(samp)}) : "
          f"{kept_clear}/{len(samp)} clairs = {summary['kept_precision']}")
    return {"summary": summary, "flipped": flipped}


def main() -> None:
    ap = argparse.ArgumentParser(description="Pré-filtre de pertinence avant stance — R&D.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not mistral_client.available():
        raise SystemExit("Pas de clé Mistral. Abandon.")
    out = args.out or str(Path(__file__).parent / f"relevance_prefilter_{args.dataset}.json")
    res = run(args.dataset)
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"[relevance] → {out}")


if __name__ == "__main__":
    main()
