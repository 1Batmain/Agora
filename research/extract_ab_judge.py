"""LLM-JUGE neutre (mistral-large) pour l'A/B extraction : complétude · segmentation · bruit.

Donné l'avis + les claims de A et de B, ANONYMISÉS en « lot 1 / lot 2 » (ordre ALTERNÉ
pour annuler le biais de position), le juge dit quel lot capte les positions le plus
COMPLÈTEMENT, avec la meilleure SEGMENTATION, sans BRUIT ni SUR-FRAGMENTATION. On agrège
les victoires par dimension après dé-anonymisation.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pipeline.claims.ollama import parse_json_object
from pipeline.cluster import mistral_client

CACHE_DIR = Path(__file__).resolve().parents[1] / "research" / "extract_ab_cache"
MODEL = "mistral-large-latest"

JUDGE_SYS = (
    "Tu es un évaluateur RIGOUREUX et NEUTRE de l'extraction d'opinions citoyennes. On te "
    "donne UN avis, puis DEUX lots de claims (« lot 1 » et « lot 2 ») extraits de cet avis "
    "par deux systèmes différents. Un bon lot capture les PRISES DE POSITION de l'avis "
    "(griefs, opinions, propositions) en les recopiant verbatim.\n\n"
    "Juge les deux lots sur TROIS dimensions INDÉPENDANTES :\n"
    "• COMPLÉTUDE : quel lot capture le PLUS de prises de position RÉELLES de l'avis, sans "
    "en oublier (surtout sur un avis qui aborde plusieurs thèmes) ?\n"
    "• SEGMENTATION : quel lot sépare le MIEUX les idées distinctes (un claim = une idée), "
    "sans fusionner deux thèmes ni couper une idée unique ?\n"
    "• BRUIT : quel lot a le MOINS de bruit (passages narratifs/cadrage sans position, "
    "redites, sur-fragmentation d'une même idée) ? Ici « gagner » = être le plus PROPRE.\n\n"
    "Pour chaque dimension réponds « 1 », « 2 » ou « tie ». Sois exigeant : ne déclare "
    "« tie » que si les lots sont vraiment équivalents. Réponds STRICTEMENT en JSON : "
    "{\"completude\": \"1|2|tie\", \"segmentation\": \"1|2|tie\", \"bruit\": \"1|2|tie\", "
    "\"justif\": \"une phrase courte\"}."
)


def _fmt(claims: list[dict]) -> str:
    if not claims:
        return "(aucun claim)"
    out = []
    for i, c in enumerate(claims, 1):
        tgt = c.get("target")
        tag = f"  [cible: {{}}]" if tgt else ""
        out.append(f"{i}. « {c['text']} »" + (tag.format(tgt) if tgt else ""))
    return "\n".join(out)


def _complete(messages):
    for attempt in range(6):
        try:
            return mistral_client.chat(messages, model=MODEL, temperature=0.0,
                                       max_tokens=400, json_mode=True, timeout=120)
        except mistral_client.MistralError as exc:
            if exc.status in {0, 429, 500, 502, 503, 504} and attempt < 5:
                time.sleep(min(40.0, 2.0 * (2 ** attempt)))
                continue
            return None
    return None


def pick_judge_set(avis, A, B, n=40):
    """~40 avis, grand débat multi-thèmes prioritaire (≥2 claims dans un bras)."""
    scored = []
    for a in avis:
        na, nb = len(A[a.id]["claims"]), len(B[a.id]["claims"])
        multi = max(na, nb) >= 2
        # priorité : granddebat multi-thèmes, puis le reste
        rank = (0 if (a.ds == "granddebat" and multi) else 1 if multi else 2)
        scored.append((rank, -max(na, nb), a))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [a for _, _, a in scored[:n]]


def run_judge(avis, A, B, n=40):
    cache = CACHE_DIR / "judge.json"
    sel = pick_judge_set(avis, A, B, n)

    def judge_one(idx_a):
        idx, a = idx_a
        # Ordre ALTERNÉ : pair → A=lot1 ; impair → B=lot1.
        a_is_lot1 = (idx % 2 == 0)
        ca, cb = A[a.id]["claims"], B[a.id]["claims"]
        lot1, lot2 = (ca, cb) if a_is_lot1 else (cb, ca)
        user = (f"AVIS :\n{a.text}\n\n--- LOT 1 ---\n{_fmt(lot1)}\n\n"
                f"--- LOT 2 ---\n{_fmt(lot2)}")
        raw = _complete([{"role": "system", "content": JUDGE_SYS},
                         {"role": "user", "content": user}])
        obj = parse_json_object(raw or "") or {}

        def deanon(v):
            if v in ("1", "2"):
                lot1_is_A = a_is_lot1
                winner_lot1 = (v == "1")
                return "A" if (winner_lot1 == lot1_is_A) else "B"
            return "tie"

        return {
            "id": a.id, "ds": a.ds, "a_is_lot1": a_is_lot1,
            "n_a": len(ca), "n_b": len(cb),
            "completude": deanon(obj.get("completude")),
            "segmentation": deanon(obj.get("segmentation")),
            "bruit": deanon(obj.get("bruit")),
            "justif": obj.get("justif", ""),
        }

    verdicts = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(judge_one, (i, a)) for i, a in enumerate(sel)]
        for k, fut in enumerate(as_completed(futs), 1):
            verdicts.append(fut.result())
            print(f"[juge] {k}/{len(sel)}")

    tally = {dim: {"A": 0, "B": 0, "tie": 0} for dim in ("completude", "segmentation", "bruit")}
    for v in verdicts:
        for dim in tally:
            tally[dim][v[dim]] += 1

    out = {"n": len(verdicts), "tally": tally, "verdicts": verdicts}
    cache.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(tally, ensure_ascii=False, indent=2))
    return out
