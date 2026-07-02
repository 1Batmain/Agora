"""Backfill du coût d'EXTRACTION dans cost.json — ESTIMATION honnête, marquée comme telle.

Les builds d'extraction ont précédé l'instrumentation de `mistral_client.chat` : le coût
mesuré (`/cost`) ne comptait que l'enrichissement (~0.06 $) alors que l'extraction
mistral-large est le poste dominant (~×100). On l'estime ICI depuis les données réelles :
  - tokens d'ENTRÉE ≈ chars(texte des avis)/4 + prompt système par lot (BATCH avis/appel) ;
  - tokens de SORTIE ≈ chars(JSON de claims produit)/4.
La phase est nommée `extraction_estimee` et le doc porte `extraction_note` — l'UI et l'API
disent que c'est une estimation, pas une mesure. (Audit 2026-07 : « /cost sous-estime ×100 ».)
"""
import json
from pathlib import Path

from backend import cost

CACHE = Path("backend/cache")
SYS_PROMPT_TOKENS = 900   # prompt système claims v2 (~3.6k chars)
BATCH = 8                 # avis par appel (pipeline.claims.extract.BATCH_SIZE)

for ddir in sorted(CACHE.iterdir()):
    claims_p = ddir / "claims.json"
    ideas_p = ddir / "ideas.jsonl"
    if not (claims_p.exists() and ideas_p.exists() and (ddir / "analysis").exists()):
        continue
    ds = ddir.name
    doc = json.loads(claims_p.read_text())
    model = doc.get("model", "mistral-large-latest")
    claims = doc.get("claims", {})
    n_avis = len(claims)
    # Entrée : texte des avis (source de l'extraction) + prompts système par lot.
    in_chars = 0
    for line in open(ideas_p, encoding="utf-8"):
        try:
            in_chars += len((json.loads(line).get("props") or {}).get("text") or "")
        except json.JSONDecodeError:
            continue
    n_batches = max(1, (n_avis + BATCH - 1) // BATCH)
    prompt_tokens = in_chars // 4 + n_batches * SYS_PROMPT_TOKENS
    # Sortie : le JSON de claims produit (texte + spans + cibles).
    out_chars = len(json.dumps(claims, ensure_ascii=False))
    completion_tokens = out_chars // 4
    usage = {
        "calls": n_batches,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "by_model": {model: {"calls": n_batches, "prompt_tokens": prompt_tokens,
                             "completion_tokens": completion_tokens}},
    }
    doc2 = cost.record_phase(
        ds, "extraction_estimee", usage,
        extra={"extraction_note": (
            "Phase 'extraction_estimee' = ESTIMATION a posteriori (chars/4 sur les avis et "
            "claims réels), PAS une mesure : l'extraction a précédé l'instrumentation. "
            "Les phases suivantes sont mesurées à l'appel.")})
    t = doc2["total"]
    print(f"{ds:24} extraction ~{prompt_tokens//1000}k in / {completion_tokens//1000}k out "
          f"→ total {t['total_tokens']//1000}k tokens ≈ {t['estimated_usd']} USD")
