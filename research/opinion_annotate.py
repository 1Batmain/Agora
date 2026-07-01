"""Annotation MANUELLE (gold) de l'échantillon stance sur la cible T2 (objet de clivage),
pour mesurer le taux d'erreur LLM. Les labels GOLD ci-dessous ont été posés à la main en
relisant chaque claim vs sa cible (favorable / defavorable / nuance), INDÉPENDAMMENT du
label LLM. « nuance » = le claim n'adresse pas LA mesure précise de la cible.

Lancement : python research/opinion_annotate.py
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

RES = Path(__file__).resolve().parent
annot = json.loads((RES / "opinion_proto_annot.json").read_text(encoding="utf-8"))

# Gold posé à la main (index → stance). Voir opinion_proto_note.md §taux d'erreur.
GOLD = {
    0: "favorable", 1: "nuance", 2: "nuance", 3: "nuance", 4: "nuance",
    5: "nuance", 6: "favorable", 7: "favorable", 8: "favorable", 9: "nuance",
    10: "nuance", 11: "nuance", 12: "nuance", 13: "nuance", 14: "favorable",
    15: "nuance", 16: "nuance", 17: "nuance", 18: "nuance", 19: "favorable",
    20: "nuance", 21: "nuance", 22: "nuance", 23: "nuance", 24: "nuance",
    25: "nuance", 26: "nuance", 27: "favorable", 28: "nuance", 29: "nuance",
    30: "nuance", 31: "nuance", 32: "nuance", 33: "nuance", 34: "favorable",
    35: "favorable", 36: "favorable", 37: "nuance", 38: "nuance", 39: "nuance",
    40: "nuance", 41: "nuance", 42: "favorable", 43: "favorable", 44: "favorable",
}

assert len(GOLD) == len(annot), (len(GOLD), len(annot))

errors = []
for i, r in enumerate(annot):
    r["gold"] = GOLD[i]
    if GOLD[i] != r["llm_stance"]:
        errors.append((i, r["llm_stance"], GOLD[i], r["claim"][:70]))

n = len(annot)
agree = n - len(errors)
gold_dist = Counter(GOLD.values())
llm_dist = Counter(r["llm_stance"] for r in annot)

# Précision/rappel par classe (gold = vérité).
def pr(label):
    tp = sum(1 for i, r in enumerate(annot) if r["llm_stance"] == label and GOLD[i] == label)
    fp = sum(1 for i, r in enumerate(annot) if r["llm_stance"] == label and GOLD[i] != label)
    fn = sum(1 for i, r in enumerate(annot) if r["llm_stance"] != label and GOLD[i] == label)
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    return tp, fp, fn, prec, rec

print(f"N = {n}   accord = {agree}/{n} = {agree/n*100:.1f}%   erreur = {len(errors)/n*100:.1f}%")
print(f"GOLD dist : {dict(gold_dist)}")
print(f"LLM  dist : {dict(llm_dist)}")
print("\nPar classe (gold=vérité)  tp/fp/fn  prec  rec")
for lab in ("favorable", "defavorable", "nuance"):
    tp, fp, fn, prec, rec = pr(lab)
    ps = f"{prec:.2f}" if prec is not None else "—"
    rs = f"{rec:.2f}" if rec is not None else "—"
    print(f"  {lab:11} {tp:2}/{fp:2}/{fn:2}   {ps}  {rs}")

print("\nERREURS :")
for i, llm, gold, claim in errors:
    print(f"  #{i:2} LLM={llm:11} GOLD={gold:11} | {claim}")

(RES / "opinion_proto_annot.json").write_text(
    json.dumps(annot, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n✓ gold écrit dans opinion_proto_annot.json")
