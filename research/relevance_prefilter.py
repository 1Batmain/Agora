"""PRÉ-FILTRE DE PERTINENCE avant stance — pipeline caché, paramétrable (calibration + corpus).

Correctif du sur-classement par association (`cleavage_quality_note.md`) : avant la stance,
« ce témoignage PORTE-t-il sur CETTE action ? » → non-pertinent = abstention (garde la cible
large). Ce script CACHE la dérivation de cible + la stance par (dataset, modèle), puis teste
des variantes de filtre (strict/soft) et des modèles (small vs large) sans re-payer le lourd.

  base_cache(dataset, model) → relevance_base_<ds>_<model>.json (cible + stance par feuille, caché)
  puis relevance(variant) + juge aveugle → métriques précision/rappel.

Zéro touche prod. Lit `emerge_cache/<ds>/`.
    MISTRAL_API_KEY=$(cat var/mistral.key) uv run python research/relevance_prefilter.py \
        --dataset lutte-contre-les-fausses-informations --model mistral-large-latest --variant strict
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from backend.build_opinion import MIN_CLAIMS, run_stance, _chat_retry, cleavage_system
from research.cleavage_quality import judge_clear, _central_order, CAP_CLAIMS
from pipeline.cluster import mistral_client

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "research" / "emerge_cache"
BATCH = 10
REP_FOR_DERIVE = 14

_REL_INTRO = ("Tu es analyste de consultations citoyennes. On te donne une ACTION (proposition "
              "débattable) et des CONTRIBUTIONS numérotées. Pour CHAQUE contribution, indique si "
              "elle PORTE SUR CETTE ACTION précise, INDÉPENDAMMENT du pour/contre. Ne juge PAS le "
              "pour/contre, juste la PERTINENCE.\n")
_REL_STRICT = (_REL_INTRO +
    "Réponds NON (porte=false) si la contribution parle d'une AUTRE mesure/action (même sur le "
    "même thème général) ou est purement descriptive/générale sans viser cette action.\n"
    'Réponds en JSON strict : {"results":[{"i":<int>,"porte":true|false}]}.')
_REL_SOFT = (_REL_INTRO +
    "Réponds NON (porte=false) SEULEMENT si la contribution vise une action CLAIREMENT DIFFÉRENTE, "
    "ou est purement descriptive SANS aucune implication pour cette action. En cas de doute, ou si "
    "elle touche même INDIRECTEMENT cette action, réponds OUI.\n"
    'Réponds en JSON strict : {"results":[{"i":<int>,"porte":true|false}]}.')
_REL = {"strict": _REL_STRICT, "soft": _REL_SOFT}


def _derive(title: str, kw: list[str], texts: list[str], model: str) -> str:
    contribs = "\n".join(f"- {t[:160]}" for t in texts[:REP_FOR_DERIVE])
    user = f"MOTS-CLÉS : {', '.join(kw[:10])}\n\nCONTRIBUTIONS :\n{contribs}"
    try:
        raw = _chat_retry([{"role": "system", "content": cleavage_system(title)},
                           {"role": "user", "content": user}], model=model, max_tokens=200)
        return str(json.loads(raw).get("objet", "")).strip() or title
    except (mistral_client.MistralError, json.JSONDecodeError, TypeError):
        return title


def base_cache(dataset: str, model: str) -> list[dict]:
    path = ROOT / "research" / f"relevance_base_{dataset}_{model}.json"
    if path.exists():
        print(f"[relevance] base cache HIT ({model})")
        return json.loads(path.read_text())
    d = CACHE / dataset
    vecs = np.load(d / "claim_vecs.npz")["vecs"].astype(np.float32)
    claims = [json.loads(l) for l in (d / "claims.jsonl").read_text().splitlines()]
    texts_all = [c["text"] for c in claims]
    leaves = json.loads((d / "leaves.json").read_text())
    out = []
    for lf in leaves:
        if len(lf["member_gis"]) < MIN_CLAIMS:
            continue
        ordered = _central_order(lf["member_gis"], vecs)[:CAP_CLAIMS]
        rep = [texts_all[g] for g in ordered[:REP_FOR_DERIVE]]
        cible = _derive(lf["title"], lf["keywords"], rep, model)
        items = [(i, texts_all[g]) for i, g in enumerate(ordered)]
        st = run_stance(cible, items, model=model)
        rec = {"theme_id": lf["theme_id"], "cible": cible, "claims": [
            {"i": i, "text": texts_all[ordered[i]], "stance": st[i]["stance"],
             "conf": st[i].get("confidence", "low")} for i, _ in items if i in st]}
        out.append(rec)
        print(f"  base [{rec['theme_id']}] {len(rec['claims'])} claims · {model}")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def relevance_batch(cible: str, items: list[tuple[int, str]], variant: str) -> dict[int, bool]:
    out: dict[int, bool] = {}
    for start in range(0, len(items), BATCH):
        batch = items[start:start + BATCH]
        lines = "\n".join(f"[{i}] {t}" for i, t in batch)
        try:
            raw = _chat_retry([{"role": "system", "content": _REL[variant]},
                               {"role": "user", "content": f"ACTION : {cible}\n\nCONTRIBUTIONS :\n{lines}"}],
                              model="mistral-large-latest", max_tokens=800)
            for rec in json.loads(raw).get("results", []):
                out[int(rec["i"])] = bool(rec.get("porte"))
        except (mistral_client.MistralError, json.JSONDecodeError, TypeError, ValueError, KeyError):
            pass
    for i, _ in items:
        out.setdefault(i, True)   # repli prudent : garder
    return out


def run(dataset: str, model: str, variant: str) -> dict:
    base = base_cache(dataset, model)
    flipped, kept = [], []
    base_dec = 0
    for rec in base:
        items = [(c["i"], c["text"]) for c in rec["claims"]]
        rel = relevance_batch(rec["cible"], items, variant)
        for c in rec["claims"]:
            if c["stance"] not in ("favorable", "defavorable"):
                continue
            base_dec += 1
            if not rel.get(c["i"], True):
                flipped.append({"cible": rec["cible"], "text": c["text"], "conf": c["conf"]})
            else:
                kept.append({"cible": rec["cible"], "text": c["text"]})
    good = bad = 0
    for f in flipped:
        v = judge_clear(f["cible"], f["text"])
        if v is None:
            continue
        f["judge_claire"] = v
        good += (not v); bad += v
    rng = random.Random(42)
    samp = rng.sample(kept, min(25, len(kept)))
    kept_clear = sum(1 for k in samp if judge_clear(k["cible"], k["text"]))
    tot_rm = good + bad
    summary = {
        "dataset": dataset, "model": model, "variant": variant,
        "baseline_decided": base_dec, "filtered_decided": len(kept),
        "removed": len(flipped), "removal_good_notclear": good, "removal_bad_clear": bad,
        "removal_precision": round(good / tot_rm, 3) if tot_rm else None,
        "kept_sample": len(samp), "kept_clear": kept_clear,
        "kept_precision": round(kept_clear / len(samp), 3) if samp else None,
        "conf_removed": dict(Counter(f["conf"] for f in flipped)),
    }
    print(f"\n[{dataset} · {model} · {variant}] décidés {base_dec}→{len(kept)} (retirés {len(flipped)}) "
          f"· retrait précis {summary['removal_precision']} (bon {good}/mauvais {bad}) "
          f"· gardés clairs {summary['kept_precision']} · conf retirés {summary['conf_removed']}")
    return {"summary": summary, "flipped": flipped}


def main() -> None:
    ap = argparse.ArgumentParser(description="Pré-filtre pertinence — calibration/corpus (R&D).")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="mistral-large-latest")
    ap.add_argument("--variant", default="strict", choices=["strict", "soft"])
    args = ap.parse_args()
    if not mistral_client.available():
        raise SystemExit("Pas de clé Mistral. Abandon.")
    res = run(args.dataset, args.model, args.variant)
    out = ROOT / "research" / f"relevance_run_{args.dataset}_{args.model}_{args.variant}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"[relevance] → {out}")


if __name__ == "__main__":
    main()
