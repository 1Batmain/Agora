"""Prompt d'extraction ADAPTÉ PAR MODÈLE — corriger le défaut de chacun sans changer de modèle.

Constat (étape 4) : avec le MÊME prompt, `mistral-large` SOUS-découpe (méga-claims empilant des
sujets distincts) et `ministral-3b` SUR-fragmente parfois (coupe une phrase courte, claim d'un mot).
Ici on ajoute au prompt système un NUDGE ciblé par modèle et on compare baseline vs adapté :
  - 3b   : anti-fragmentation (un claim = une idée complète autonome ; jamais couper une phrase).
  - large: anti-sous-découpage (une énumération de propositions distinctes = un claim CHACUNE).

Mono-avis (pas de batch → pas de souci de max_tokens ; on teste le PROMPT, pas le débit). Échantillon
= avis atomisables tirés de la cascade, biaisé vers les cas durs (courts pour 3b, listes pour large).
Métriques objectives (fragmentation, méga-claim) + dump lisible pour jugement par l'agent.

    MISTRAL_API_KEY=$(cat var/mistral.key) uv run --extra contender --extra embed-contender \
      --extra faiss python -u research/llm_regression/extraction_prompt_tune.py [N]
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.recluster import _read_descriptor_file, load_cache
from pipeline.claims.backend import ApiBackend
from pipeline.claims.extract import claim_sys, parse_claims
from pipeline.claims.ollama import OllamaStats
from pipeline.claims.span import align_spans

# Nudge ajouté au prompt système, PAR modèle. Vide = baseline (prompt actuel inchangé).
NUDGE = {
    "ministral-3b-latest": (
        "\n\nGRANULARITÉ (IMPORTANT pour toi) : chaque claim doit être une IDÉE COMPLÈTE et "
        "AUTONOME — une proposition qui tient debout seule. NE COUPE JAMAIS une phrase en "
        "morceaux, ne crée JAMAIS un claim d'un ou deux mots ni un fragment isolé entre "
        "guillemets. Sur un avis COURT qui ne porte qu'une idée, renvoie UN SEUL claim entier."
    ),
    "mistral-large-latest": (
        "\n\nGRANULARITÉ (IMPORTANT pour toi) : quand l'avis ÉNUMÈRE plusieurs propositions, "
        "mesures ou griefs DISTINCTS (souvent une liste), donne UN CLAIM PAR proposition — ne "
        "les empile JAMAIS dans un seul claim fourre-tout. Un long avis multi-sujets doit "
        "produire PLUSIEURS claims. Ne regroupe que ce qui porte VRAIMENT la même idée."
    ),
}
MODELS = ["ministral-3b-latest", "mistral-large-latest"]
CASCADE = Path("research/llm_regression/extraction_cascade_results.json")
OUT = Path("research/llm_regression/prompt_tune_texts.json")
DUMP = Path("research/llm_regression/prompt_tune_to_judge.txt")


def _sample(n: int) -> list[dict]:
    """Avis atomisables, biaisés vers les cas durs : moitié COURTS (stress 3b), moitié LONGS/listes
    (stress large)."""
    r = json.loads(CASCADE.read_text())["raw"]
    ids_by_ds: dict[str, list[str]] = {}
    for aid, per in r.items():
        if any(not v["fallback"] for v in per.values()):
            ids_by_ds.setdefault(next(iter(per.values()))["dataset"], []).append(aid)
    rows = []
    for ds, ids in ids_by_ds.items():
        ideas, _v, _w = load_cache(ds)
        txt = {str(it.id): (getattr(it, "text_clean", None) or it.text) for it in ideas}
        q = (_read_descriptor_file(ds) or {}).get("question") or None
        for aid in ids:
            if aid in txt:
                rows.append({"id": aid, "text": txt[aid], "dataset": ds, "question": q,
                             "len": len(txt[aid])})
    # Fenêtres UTILES (pas les extrêmes) : court-mais-multi-clause pour la fragmentation 3b ;
    # listes de propositions de taille moyenne pour le sous-découpage large (pas les pavés légaux 10k).
    short = [a for a in rows if 80 <= a["len"] <= 300]
    long = [a for a in rows if 450 <= a["len"] <= 2500]
    short.sort(key=lambda a: a["len"])
    long.sort(key=lambda a: a["len"], reverse=True)     # les plus longs de la fenêtre (listes riches)
    return short[:n // 2] + long[:n - n // 2]


def _extract(text: str, question: str | None, model: str, nudge: str) -> list[str]:
    be = ApiBackend(model=model)
    sys_prompt = claim_sys(question) + nudge
    raw = be.complete([{"role": "system", "content": sys_prompt},
                       {"role": "user", "content": "Avis :\n" + text}], stats=OllamaStats())
    claims = align_spans(text, parse_claims(raw))
    return [c.text for c in claims]


def _metrics(claims: list[str], avis_len: int) -> dict:
    n = len(claims)
    lens = [len(c) for c in claims]
    words = [len(c.split()) for c in claims]
    frag = sum(1 for w in words if w <= 3)                        # claims ≤3 mots = fragments
    mega = 1 if (n == 1 and lens and lens[0] >= 0.7 * avis_len and avis_len > 300) else 0
    return {"n": n, "med_len": st.median(lens) if lens else 0,
            "frag_court": frag, "mega": mega}


def main(n: int = 30) -> None:
    avis = _sample(n)
    print(f"[prompt-tune] {len(avis)} avis (moitié courts / moitié longs) · {MODELS}", flush=True)
    data = {"n_avis": len(avis), "avis": []}
    agg = {m: {c: {"n": [], "frag": [], "mega": [], "medlen": []} for c in ("base", "tune")}
           for m in MODELS}
    for i, a in enumerate(avis):
        rec = {"id": a["id"], "dataset": a["dataset"], "len": a["len"], "text": a["text"], "out": {}}
        for m in MODELS:
            base = _extract(a["text"], a["question"], m, "")
            tune = _extract(a["text"], a["question"], m, NUDGE[m])
            for label, cs in (("base", base), ("tune", tune)):
                mt = _metrics(cs, a["len"])
                agg[m][label]["n"].append(mt["n"]); agg[m][label]["frag"].append(mt["frag_court"])
                agg[m][label]["mega"].append(mt["mega"]); agg[m][label]["medlen"].append(mt["med_len"])
                rec["out"][f"{m}/{label}"] = cs
        data["avis"].append(rec)
        print(f"  {i+1}/{len(avis)} ({a['dataset']}, {a['len']}c)", flush=True)
        OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    print("\n=== MÉTRIQUES baseline vs adapté (par modèle) ===")
    for m in MODELS:
        for c in ("base", "tune"):
            A = agg[m][c]
            print(f"  {m:22} {c:4} · n_claims={st.mean(A['n']):.2f} · frag(≤3mots)={sum(A['frag'])} "
                  f"· méga-claims={sum(A['mega'])} · long_méd={st.median(A['medlen']):.0f}")
    # dump lisible : baseline vs adapté côte à côte
    lines = []
    for r in data["avis"]:
        lines.append(f"===== {r['id']} ({r['dataset']}, {r['len']}c) =====")
        lines.append(f"AVIS: {r['text'][:500]}")
        for m in MODELS:
            for c in ("base", "tune"):
                cs = r["out"][f"{m}/{c}"]
                lines.append(f"  [{m} · {c}] {len(cs)} claims:")
                for x in cs:
                    lines.append(f"     • {x}")
        lines.append("")
    DUMP.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nTextes → {OUT} · dump lisible → {DUMP}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
