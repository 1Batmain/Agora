"""Face-à-face EXTRACTION — couverture + qualité sur les MÊMES avis (verdict décisionnel).

La cascade mesurait le PLANCHER (le modèle extrait-il quelque chose de fidèle ?). Ici on répond à
la vraie question « peut-on descendre l'extraction sans PERTE » : on fait tourner CHAQUE modèle
candidat sur les MÊMES avis atomisables et on compare :
  - COUVERTURE (objectif) : nb de claims atomiques extraits par avis.
  - QUALITÉ (juge medium, EN AVEUGLE) : chaque jeu de claims noté 0-5 sur fidélité + complétude
    + atomicité, étiquettes A/B/C/D mélangées, noms de modèles cachés.

Échantillon = avis ATOMISABLES (≥1 modèle a extrait des claims dans la cascade) — sinon on
compare des mono-claim (tous les modèles à égalité). Robuste : N grand, 3 datasets. Écrit au fil.

    MISTRAL_API_KEY=$(cat var/mistral.key) uv run --extra contender --extra embed-contender \
      --extra faiss python -u research/extraction_headtohead_bench.py [N] [batch]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.recluster import _read_descriptor_file, load_cache
from pipeline.claims.backend import ApiBackend
from pipeline.claims.extract import extract_claims
from pipeline.cluster import mistral_client as MC

# Candidats = l'échelle SOUS large (la question : jusqu'où descendre). medium EXCLU des candidats
# car il est le JUGE → éviter tout biais d'auto-jugement (medium notant medium).
CANDIDATS = ["ministral-3b-latest", "ministral-8b-latest", "mistral-large-latest"]
PRICE_OUT = {"ministral-3b-latest": 0.04, "ministral-8b-latest": 0.10,
             "mistral-medium-latest": 2.0, "mistral-large-latest": 6.0}
JUDGE = "mistral-medium-latest"
CASCADE = Path("research/llm_regression/extraction_cascade_results.json")
OUT = Path("research/llm_regression/extraction_h2h_results.json")
LETTERS = "ABCD"


def _atomizable_ids() -> dict[str, list[str]]:
    """ids atomisables (≥1 modèle a extrait des claims dans la cascade), groupés par dataset."""
    r = json.loads(CASCADE.read_text())
    raw = r["raw"]
    byds: dict[str, list[str]] = {}
    for aid, per in raw.items():
        if any(not v["fallback"] for v in per.values()):
            ds = next(iter(per.values()))["dataset"]
            byds.setdefault(ds, []).append(aid)
    return byds


def _sample(n: int) -> list[dict]:
    """N avis atomisables {id, text, dataset, question}, proportionnels aux datasets de la cascade."""
    byds = _atomizable_ids()
    tot = sum(len(v) for v in byds.values())
    out = []
    for ds, ids in byds.items():
        cap = max(1, round(n * len(ids) / tot))
        ideas, _v, _w = load_cache(ds)
        txt = {str(it.id): (getattr(it, "text_clean", None) or it.text) for it in ideas}
        q = (_read_descriptor_file(ds) or {}).get("question") or None
        for aid in ids[:cap]:
            if aid in txt:
                out.append({"id": aid, "text": txt[aid], "dataset": ds, "question": q})
    return out


def _extract_all(avis: list[dict], model: str, batch: int) -> dict[str, list[str]]:
    """{avis_id: [textes de claims]} pour `model` (vrai extract_claims)."""
    be = ApiBackend(model=model)
    out: dict[str, list[str]] = {}
    byq: dict[str | None, list[dict]] = {}
    for a in avis:
        byq.setdefault(a["question"], []).append(a)
    done = 0
    for q, group in byq.items():
        for i in range(0, len(group), batch * 4):
            sub = group[i:i + batch * 4]
            objs = [SimpleNamespace(id=a["id"], text=a["text"]) for a in sub]
            res = extract_claims(objs, backend=be, question=q, batch_size=batch)
            for a in sub:
                out[a["id"]] = [c.text for c in res.get(a["id"], [])]
            done += len(sub)
            print(f"    [{model.split('-')[0]}] {done}/{len(avis)}", flush=True)
    return out


def _judge(avis_text: str, labelled: list[tuple[str, list[str]]]) -> dict[str, int]:
    """Note EN AVEUGLE chaque jeu de claims 0-5 (fidélité + complétude + atomicité)."""
    blocks = []
    for L, claims in labelled:
        lst = "\n".join(f"    - {c}" for c in claims) or "    (aucun claim)"
        blocks.append(f"  {L}.\n{lst}")
    sets = "\n".join(blocks)
    system = (
        "Tu JUGES des extractions de claims (portions verbatim recopiées d'un avis citoyen). "
        "Tu notes chaque jeu de 0 à 5 selon : FIDÉLITÉ (portions bien tirées de l'avis, rien "
        "d'inventé), COMPLÉTUDE (capte les points clés de l'avis, n'en oublie pas), ATOMICITÉ "
        "(claims distincts et atomiques, PAS un seul bloc = tout l'avis). Tu ne sais PAS quel "
        "modèle a produit quel jeu. Réponds STRICTEMENT en JSON {\"A\":n,\"B\":n,...} et rien d'autre."
    )
    user = f"AVIS :\n{avis_text[:1500]}\n\nJEUX DE CLAIMS :\n{sets}\n\nNotes JSON :"
    raw = MC.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                  model=JUDGE, max_tokens=120, temperature=0.0)
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    return {L: int(v) for L, v in d.items() if isinstance(v, (int, float))}


def main(n: int = 200, batch: int = 8) -> None:
    avis = _sample(n)
    print(f"[h2h] {len(avis)} avis atomisables · candidats {CANDIDATS}", flush=True)
    print("\n--- extraction par modèle ---", flush=True)
    claims_by_model = {m: _extract_all(avis, m, batch) for m in CANDIDATS}

    results = {"n_avis": len(avis), "candidats": CANDIDATS, "per_avis": [],
               "scores": {m: [] for m in CANDIDATS}, "n_claims": {m: [] for m in CANDIDATS}}
    print("\n--- jugement en aveugle ---", flush=True)
    for ti, a in enumerate(avis):
        order = CANDIDATS[ti % len(CANDIDATS):] + CANDIDATS[:ti % len(CANDIDATS)]
        labelled = [(LETTERS[i], claims_by_model[m].get(a["id"], [])) for i, m in enumerate(order)]
        lab2model = {LETTERS[i]: m for i, m in enumerate(order)}
        notes = _judge(a["text"], labelled)
        row = {"id": a["id"], "dataset": a["dataset"], "by_model": {}}
        for L, m in lab2model.items():
            nc = len(claims_by_model[m].get(a["id"], []))
            sc = notes.get(L)
            row["by_model"][m] = {"n_claims": nc, "score": sc}
            results["n_claims"][m].append(nc)
            if sc is not None:
                results["scores"][m].append(sc)
        results["per_avis"].append(row)
        if (ti + 1) % 10 == 0:
            print(f"    jugé {ti+1}/{len(avis)}", flush=True)
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    print("\n=== VERDICT face-à-face (extraction) ===")
    _lsc = results["scores"]["mistral-large-latest"]
    base = round(sum(_lsc) / len(_lsc), 2) if _lsc else None      # référence = large, calculée d'abord
    for m in CANDIDATS:
        sc = results["scores"][m]
        avg = round(sum(sc) / len(sc), 2) if sc else None
        ncm = round(sum(results["n_claims"][m]) / max(len(results["n_claims"][m]), 1), 2)
        ret = f"{round(100*avg/base)}%" if (avg and base) else "—"
        print(f"  {m:24} qualité={avg}/5 (rétention {ret}) · couverture={ncm} claims/avis · prix×{round(PRICE_OUT[m]/PRICE_OUT['ministral-3b-latest'])}")
    results["summary_note"] = "rétention = score / large ; couverture = nb moyen de claims/avis"
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nBrut → {OUT}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    main(n, b)
