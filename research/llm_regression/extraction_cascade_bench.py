"""Bench de RÉGRESSION de l'EXTRACTION — cascade d'escalade (idée de Bob).

On teste un GROS échantillon d'avis sur le PLUS PETIT modèle, et on ne remonte au modèle du
dessus QUE les avis où le petit échoue. Assertion : si le petit modèle extrait proprement, le
gros ferait au moins aussi bien → inutile de le tester dessus. On mesure ainsi, par avis, le
PLUS PETIT modèle qui tient — donc directement la distribution de coût.

Critère d'échec = OBJECTIF (pas de juge), fondé sur l'invariant verbatim du pipeline :
l'extraction est EXTRACTIVE, `align_spans` REJETTE tout span non retrouvé dans l'avis, et un
avis dont RIEN ne s'ancre retombe sur « avis entier = 1 claim ». Donc pour chaque (avis, modèle)
on enregistre les signaux BRUTS :
  - n_specs      : nb de portions proposées par le LLM (parse)
  - n_anchored   : nb ancrées verbatim (align_spans) = claims atomiques réels
  - fallback     : True si n_anchored == 0 → repli avis-entier (extraction ratée)
  - anchor_rate  : n_anchored / n_specs (bas = le modèle paraphrase au lieu de recopier)
  - missing      : True si l'avis est absent/malformé dans la réponse (batch)
On garde le BRUT → le seuil PASS/FAIL se règle APRÈS coup sur la distribution (robustesse).

Verdict : taux de PASS par palier + le plus petit modèle suffisant par avis. Résilient (écrit
au fil de l'eau ; le backend Mistral retente déjà les 429).

    MISTRAL_API_KEY=$(cat var/mistral.key) uv run --extra contender --extra embed-contender \
      --extra faiss python -u research/extraction_cascade_bench.py [N_total] [batch]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from types import SimpleNamespace

from backend.recluster import _read_descriptor_file, load_cache
from pipeline.claims.backend import ApiBackend
from pipeline.claims.extract import extract_claims

# Échelle du moins cher au plus cher.
LADDER = ["ministral-3b-latest", "ministral-8b-latest", "mistral-small-latest",
          "mistral-medium-latest", "mistral-large-latest"]
# Échantillon multi-dataset (robustesse : domaines + longueurs variés).
MIX = {"tiktok": 0.35, "republique-numerique": 0.35, "granddebat": 0.30}
OUT = Path("research/llm_regression/extraction_cascade_results.json")


def _sample(n_total: int) -> list[dict]:
    """Avis {id, text, dataset, question}, échantillon déterministe (tête de liste par dataset)."""
    out = []
    for ds, frac in MIX.items():
        cap = int(round(n_total * frac))
        try:
            ideas, _v, _w = load_cache(ds)
        except Exception as e:
            print(f"  (skip {ds}: {e})"); continue
        q = (_read_descriptor_file(ds) or {}).get("question", "") or None
        for it in ideas[:cap]:
            t = (getattr(it, "text_clean", None) or it.text or "").strip()
            if len(t) >= 30:
                out.append({"id": str(it.id), "text": t, "dataset": ds, "question": q})
    return out


def _extract_batch(avis: list[dict], model: str, batch: int) -> list[dict]:
    """Extrait `avis` avec `model` via le VRAI `extract_claims` (batching + max_tokens dynamique +
    repli mono-avis inclus → mesure le comportement RÉEL de prod) → signaux par avis.

    Signal objectif = REPLI avis-entier : `extract_claims` ne colle « avis entier = 1 claim » que
    si RIEN ne s'ancre verbatim (le modèle n'a pas su extraire de portion). C'est l'échec net."""
    be = ApiBackend(model=model)
    total = len(avis)
    rows = []
    done = 0
    # extract_claims prend UNE question ; on groupe par question, puis on tronçonne pour la progression.
    byq: dict[str | None, list[dict]] = {}
    for a in avis:
        byq.setdefault(a["question"], []).append(a)
    for q, group in byq.items():
        for i in range(0, len(group), batch * 4):
            sub = group[i:i + batch * 4]
            objs = [SimpleNamespace(id=a["id"], text=a["text"]) for a in sub]
            res = extract_claims(objs, backend=be, question=q, batch_size=batch)
            for a, o in zip(sub, objs):
                cs = res.get(o.id, [])
                whole = len(cs) == 1 and cs[0].text.strip() == o.text.strip()
                # aussi suspect : 1 claim couvrant ~tout l'avis (quasi-repli sans match exact)
                near = len(cs) == 1 and len(cs[0].text) >= 0.9 * len(o.text)
                rows.append({"id": a["id"], "dataset": a["dataset"], "n_claims": len(cs),
                             "fallback": whole, "near_whole": whole or near})
            done += len(sub)
            print(f"      … {done}/{total} avis extraits", flush=True)
    return rows


def _passes(row: dict) -> bool:
    # PASS = le modèle a extrait au moins un claim ATOMIQUE (pas le repli avis-entier).
    return not row["near_whole"]


def main(n_total: int = 2000, batch: int = 15) -> None:
    avis = _sample(n_total)
    print(f"[cascade] {len(avis)} avis · échelle {LADDER}", flush=True)
    results = {"n_avis": len(avis), "ladder": LADDER, "mix": MIX,
               "by_model": {}, "min_model_hist": {}, "raw": {}}
    by_id = {a["id"]: a for a in avis}
    current = list(avis)                                       # avis restant à faire passer

    for model in LADDER:
        if not current:
            break
        print(f"\n=== palier {model} : {len(current)} avis ===", flush=True)
        rows = _extract_batch(current, model, batch)
        passers, failers = [], []
        for r in rows:
            results["raw"].setdefault(r["id"], {})[model] = r
            (passers if _passes(r) else failers).append(r)
        n_fb = sum(1 for r in rows if r["fallback"])
        results["by_model"][model] = {
            "tested": len(rows), "pass": len(passers), "fail": len(failers),
            "fallback_avis_entier": n_fb,
            "pass_rate": round(len(passers) / max(len(rows), 1), 3),
        }
        for r in passers:                                      # 1er (= plus petit) modèle qui tient
            results["min_model_hist"][r["id"]] = model
        print(f"    pass={len(passers)}/{len(rows)} ({results['by_model'][model]['pass_rate']:.0%}) "
              f"· repli-avis-entier={n_fb} · escaladés={len(failers)}", flush=True)
        current = [by_id[r["id"]] for r in failers]
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    # avis qui échouent MÊME sur large = cas durs
    results["residuel_dur"] = [r["id"] for r in current] if current else []
    # distribution finale : combien d'avis servis par chaque palier (le plus petit suffisant)
    dist = {}
    for m in results["min_model_hist"].values():
        dist[m] = dist.get(m, 0) + 1
    results["distribution_min_model"] = dist
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    print("\n=== VERDICT (extraction, plancher verbatim) ===")
    tot = len(avis)
    cum = 0
    for m in LADDER:
        k = dist.get(m, 0)
        cum += k
        print(f"  {m:24} suffit pour {k:>4} avis ({k/tot:.0%})  · cumulé {cum/tot:.0%}")
    print(f"  résiduel dur (échoue même sur large) : {len(results['residuel_dur'])} ({len(results['residuel_dur'])/tot:.0%})")
    print(f"\nBrut → {OUT}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    main(n, b)
