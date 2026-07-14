"""Combien de PRISES DE POSITION l'extraction laisse-t-elle de côté ? (harnais T-N7)

La couverture brute du texte (≈88 % sur tiktok) n'est PAS le bon chiffre : le prompt
d'extraction écarte VOLONTAIREMENT le narratif, le cadrage et les politesses (règle 2,
« SÉLECTIVITÉ »). Compter les caractères perdus, ou les clauses en « mais », mesure la
mauvaise chose — la moitié de ces clauses sont des fragments (« mais oufff », « mais très
peu ») qu'il serait NUISIBLE de capturer.

Ce qui compte : parmi les segments non couverts, combien portent un grief, une opinion,
une proposition ou un vécu évalué ? Un juge LLM tranche, segment par segment.

Mesure de RÉFÉRENCE (tiktok, 2419 claims, mistral-large, 2026-07-09) :
    296 segments non couverts ≥30 car. → 144 portent une position (49 %)
    soit ≈ 6,0 % du volume de claims. Les 152 autres sont du narratif, écarté à raison.

Biais connus, dans les DEUX sens :
  - sous-estime : les segments < 30 caractères ne sont pas examinés ;
  - sur-estime  : le juge est un LLM, il compte parfois un fragment tronqué comme position
                  (« Et ce qui me heurte au plus profond, c'est que »).
Un échantillon annoté À LA MAIN reste la seule vérité terrain. Ce harnais donne l'ordre de
grandeur et, surtout, permet de COMPARER deux extractions (avant/après) à protocole égal.

Usage :
    uv run --extra embed-contender --extra faiss python research/extraction_coverage.py [dataset]
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pipeline.claims.ollama import parse_json_object
from pipeline.cluster import mistral_client

MIN_SEG = 30          # en deçà, un segment ne peut guère porter une position
BATCH = 10
JUDGE_MODEL = "mistral-large-latest"


def uncovered_segments(txt: str, spans: list[tuple[int, int]]) -> list[str]:
    """Portions de l'avis qu'AUCUN span de claim ne recouvre, ≥ MIN_SEG caractères."""
    cov = bytearray(len(txt))
    for s, e in spans:
        for k in range(max(0, s), min(e, len(txt))):
            cov[k] = 1
    out, i = [], 0
    while i < len(txt):
        if cov[i]:
            i += 1
            continue
        j = i
        while j < len(txt) and not cov[j]:
            j += 1
        seg = txt[i:j].strip(" \n\t,;:.–-")
        if len(seg) >= MIN_SEG:
            out.append(seg)
        i = j
    return out


def _sys_prompt(question: str) -> str:
    return (
        "Tu juges des EXTRAITS de témoignages citoyens, laissés de côté par une extraction "
        f"automatique. La consultation posait : « {question} »\n\n"
        "Pour CHAQUE extrait numéroté, dis s'il porte une PRISE DE POSITION du citoyen : "
        "un grief, une opinion, une proposition, un vécu évalué (« ça m'a détruit », "
        "« je ne veux pas l'avoir », « c'est utile pour apprendre »).\n"
        "Réponds `false` si c'est du pur NARRATIF, du CADRAGE, une politesse, une annonce, "
        "ou une donnée factuelle sans jugement (« j'ai 15 ans », « j'utilise depuis 5 ans »).\n\n"
        'JSON STRICT : {"1": {"position": true, "pourquoi": "…"}, "2": {"position": false, …}}'
    )


def main(dataset: str = "tiktok") -> None:
    base = Path(f"backend/cache/{dataset}")
    question = json.loads((base / "meta.json").read_text())["question"]
    claims = json.loads((base / "claims.json").read_text())["claims"]

    segs: list[tuple[str, str]] = []
    n_claims = 0
    for line in (base / "ideas.jsonl").read_text().splitlines():
        idea = json.loads(line)
        aid = idea["id"]
        txt = idea["props"].get("text_clean") or ""
        cl = claims.get(aid) or []
        n_claims += len(cl)
        spans = [tuple(s) for c in cl for s in c.get("spans", []) if s and s[0] >= 0]
        segs.extend((aid, s) for s in uncovered_segments(txt, spans))

    print(f"{dataset} : {n_claims} claims · {len(segs)} segments non couverts ≥{MIN_SEG} car.")
    sys_p = _sys_prompt(question)
    lots = [segs[i:i + BATCH] for i in range(0, len(segs), BATCH)]

    def judge(lot):
        body = "\n\n".join(f"#{i}\n{s}" for i, (_a, s) in enumerate(lot, 1))
        raw = mistral_client.chat(
            [{"role": "system", "content": sys_p}, {"role": "user", "content": body}],
            model=JUDGE_MODEL, temperature=0.0, max_tokens=1400, json_mode=True)
        obj = parse_json_object(raw) or {}
        return [(aid, s, bool((obj.get(str(i)) or {}).get("position")))
                for i, (aid, s) in enumerate(lot, 1)]

    res: list[tuple[str, str, bool]] = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(judge, lot) for lot in lots]
        for k, f in enumerate(as_completed(futs), 1):
            try:
                res.extend(f.result())
            except Exception as e:  # noqa: BLE001
                print(f"  lot en échec : {type(e).__name__}: {e}")
            if k % 10 == 0:
                print(f"  {k}/{len(lots)} lots", flush=True)

    lost = [r for r in res if r[2]]
    print(f"\nsegments jugés       : {len(res)}")
    print(f"PORTENT une position : {len(lost)} ({len(lost) / max(len(res), 1):.0%})")
    print(f"narratif / cadrage   : {len(res) - len(lost)} (écarté À RAISON par la règle 2)")
    print(f"\n→ positions perdues ≈ {len(lost)}, soit {len(lost) / max(n_claims, 1):.1%} "
          f"du volume de claims")
    print(f"appels LLM perdus : {mistral_client.get_exhausted()}")
    print("\nRéférence tiktok 2026-07-09 : 144 / 296 (49 %) ≈ 6,0 % du volume.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tiktok")
