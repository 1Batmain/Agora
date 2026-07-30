"""Re-extraction 3b vs large AVEC textes sauvés — pour (1) test OBJECTIF de surdécoupage,
(2) jugement par l'agent (lecture d'un échantillon), sans juge LLM.

Hypothèse de Bob : ministral-3b SURDÉCOUPE (plus de claims mais des fragments, pas plus de
couverture). Test objectif, mêmes avis atomisables que le face-à-face :
  - n_claims / avis            (déjà : 3b 2.39 vs large 1.97)
  - longueur MOYENNE d'un claim (surdécoupage → claims plus courts)
  - COUVERTURE totale = somme des chars de claims / chars de l'avis
      • surdécoupage  : n↑, longueur↓, couverture ≈ égale (même texte, plus de morceaux)
      • + thorough    : n↑, couverture↑ (capte plus de l'avis)
  - RATIO fragment : part des claims d'un jeu qui sont sous-chaîne d'un autre claim du MÊME jeu
    (redondance de fragmentation).
Écrit textes + métriques ; dump lisible d'un échantillon pour jugement humain/agent.

    MISTRAL_API_KEY=$(cat var/mistral.key) uv run --extra contender --extra embed-contender \
      --extra faiss python -u research/extraction_judge_me.py [N] [batch]
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.recluster import _read_descriptor_file, load_cache
from pipeline.claims.backend import ApiBackend
from pipeline.claims.extract import extract_claims

MODELS = ["ministral-3b-latest", "mistral-large-latest"]
CASCADE = Path("research/llm_regression/extraction_cascade_results.json")
OUT = Path("research/llm_regression/extraction_texts.json")
DUMP = Path("research/llm_regression/extraction_sample_to_judge.txt")


def _sample(n: int) -> list[dict]:
    r = json.loads(CASCADE.read_text())["raw"]
    byds: dict[str, list[str]] = {}
    for aid, per in r.items():
        if any(not v["fallback"] for v in per.values()):
            byds.setdefault(next(iter(per.values()))["dataset"], []).append(aid)
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


def _extract(avis: list[dict], model: str, batch: int) -> dict[str, list[str]]:
    be = ApiBackend(model=model)
    out: dict[str, list[str]] = {}
    byq: dict[str | None, list[dict]] = {}
    for a in avis:
        byq.setdefault(a["question"], []).append(a)
    done = 0
    for q, group in byq.items():
        for i in range(0, len(group), batch * 4):
            sub = group[i:i + batch * 4]
            res = extract_claims([SimpleNamespace(id=a["id"], text=a["text"]) for a in sub],
                                 backend=be, question=q, batch_size=batch)
            for a in sub:
                out[a["id"]] = [c.text for c in res.get(a["id"], [])]
            done += len(sub)
            print(f"    [{model.split('-')[0]}] {done}/{len(avis)}", flush=True)
    return out


def _frag_ratio(claims: list[str]) -> float:
    """Part des claims qui sont sous-chaîne d'un AUTRE claim du même jeu (fragmentation)."""
    if len(claims) < 2:
        return 0.0
    frag = 0
    for i, c in enumerate(claims):
        c2 = c.strip()
        if c2 and any(j != i and c2 in o and len(c2) < len(o) for j, o in enumerate(claims)):
            frag += 1
    return frag / len(claims)


def main(n: int = 200, batch: int = 8) -> None:
    avis = _sample(n)
    print(f"[judge-me] {len(avis)} avis · {MODELS}", flush=True)
    ext = {m: _extract(avis, m, batch) for m in MODELS}

    data = {"n_avis": len(avis), "models": MODELS, "avis": []}
    metrics = {m: {"n_claims": [], "claim_len": [], "coverage": [], "frag": []} for m in MODELS}
    for a in avis:
        rec = {"id": a["id"], "dataset": a["dataset"], "text": a["text"], "claims": {}}
        for m in MODELS:
            cs = ext[m].get(a["id"], [])
            rec["claims"][m] = cs
            metrics[m]["n_claims"].append(len(cs))
            for c in cs:
                metrics[m]["claim_len"].append(len(c))
            cover = sum(len(c) for c in cs) / max(len(a["text"]), 1)
            metrics[m]["coverage"].append(cover)
            metrics[m]["frag"].append(_frag_ratio(cs))
        data["avis"].append(rec)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    print("\n=== MÉTRIQUES OBJECTIVES DE SURDÉCOUPAGE ===")
    for m in MODELS:
        M = metrics[m]
        print(f"  {m:22} n_claims/avis={st.mean(M['n_claims']):.2f} · "
              f"long_claim méd={st.median(M['claim_len']):.0f} moy={st.mean(M['claim_len']):.0f} chars · "
              f"couverture={st.mean(M['coverage']):.2f}× · fragment={st.mean(M['frag']):.1%}")

    # dump lisible : 40 avis (les plus divergents en n_claims) pour jugement par l'agent
    div = sorted(data["avis"],
                 key=lambda r: abs(len(r["claims"][MODELS[0]]) - len(r["claims"][MODELS[1]])),
                 reverse=True)[:40]
    lines = []
    for r in div:
        lines.append(f"===== avis {r['id']} ({r['dataset']}) =====")
        lines.append(f"AVIS: {r['text'][:600]}")
        for m in MODELS:
            lines.append(f"  [{m}] {len(r['claims'][m])} claims:")
            for c in r["claims"][m]:
                lines.append(f"     • {c}")
        lines.append("")
    DUMP.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nTextes → {OUT} · échantillon lisible (40 avis divergents) → {DUMP}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    main(n, b)
