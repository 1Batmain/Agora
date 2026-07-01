#!/usr/bin/env python3
"""Validation de NOTRE passe de stance contre le gold xstance (FAVOR/AGAINST).

Pour chaque commentaire xstance, on lance `stance_batch` (STANCE_SYSTEM, mistral-small)
avec cible = la PROPRE question fermée du commentaire, puis on mappe :
  favorable -> FAVOR, defavorable -> AGAINST, nuance -> abstention.
On compare au gold `props.label`. Sortie : research/stance_validation_raw.jsonl
(une ligne par commentaire) + impression d'un résumé.
"""
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.build_opinion import stance_batch, MODEL, BATCH
from pipeline.cluster import mistral_client

DATA = "backend/cache/xstance/ideas.jsonl"
OUT = "research/stance_validation_raw.jsonl"
MODEL_NAME = MODEL


def load_items():
    rows = []
    for line in open(DATA, encoding="utf-8"):
        p = json.loads(line)["props"]
        rows.append({
            "id": json.loads(line)["id"],
            "text": (p.get("text_clean") or p.get("text") or "").strip(),
            "question": p["question"],
            "lang": p["lang"],
            "topic": p.get("topic", ""),
            "gold": p["label"],
        })
    return rows


def main():
    rows = load_items()
    # Index global stable pour chaque commentaire.
    for gi, r in enumerate(rows):
        r["gi"] = gi

    # Regroupe par question (la cible), puis batch de BATCH.
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["question"]].append(r)

    tasks = []  # (cible, [(gi, text), ...])
    for q, group in by_q.items():
        for s in range(0, len(group), BATCH):
            chunk = group[s:s + BATCH]
            tasks.append((q, [(c["gi"], c["text"]) for c in chunk]))

    print(f"[validate] {len(rows)} commentaires, {len(by_q)} questions, "
          f"{len(tasks)} batches, modèle={MODEL_NAME}", flush=True)

    preds = {}  # gi -> dict(stance, confidence, justif)
    done = 0

    def work(task):
        cible, items = task
        try:
            return cible, stance_batch(cible, items, model=MODEL_NAME)
        except (mistral_client.MistralError, json.JSONDecodeError):
            # repli unitaire
            out = {}
            for i, text in items:
                try:
                    out.update(stance_batch(cible, [(i, text)], model=MODEL_NAME))
                except (mistral_client.MistralError, json.JSONDecodeError):
                    out[i] = {"stance": "nuance", "confidence": "low", "justif": "(échec LLM)"}
            return cible, out

    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(work, t): t for t in tasks}
        for fut in as_completed(futs):
            cible, got = fut.result()
            # garantir une entrée par item du batch (repli unitaire si manquant)
            items = dict(futs[fut][1])
            for gi, text in items.items():
                if gi not in got:
                    try:
                        got.update(stance_batch(cible, [(gi, text)], model=MODEL_NAME))
                    except (mistral_client.MistralError, json.JSONDecodeError):
                        got[gi] = {"stance": "nuance", "confidence": "low", "justif": "(échec LLM)"}
            preds.update(got)
            done += 1
            if done % 20 == 0:
                print(f"[validate] {done}/{len(tasks)} batches", flush=True)

    STANCE2GOLD = {"favorable": "FAVOR", "defavorable": "AGAINST", "nuance": "ABSTAIN"}
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            p = preds.get(r["gi"], {"stance": "nuance", "confidence": "low", "justif": "(manquant)"})
            f.write(json.dumps({
                "id": r["id"], "lang": r["lang"], "topic": r["topic"],
                "question": r["question"], "gold": r["gold"],
                "stance": p["stance"], "pred": STANCE2GOLD[p["stance"]],
                "confidence": p["confidence"], "justif": p.get("justif", ""),
            }, ensure_ascii=False) + "\n")
    print(f"[validate] écrit {OUT} ({len(rows)} lignes)", flush=True)


if __name__ == "__main__":
    main()
