"""Ré-extraction EXTRACT v3 d'un dataset entier (mistral-large, BATCHÉ) → claims.json.

Rejoue l'extraction des claims sur TOUS les avis avec le prompt stance-v3 et le batching
N avis/appel, écrit le cache `claims.json` (réutilisé tel quel par un futur rebuild), et
imprime les métriques corpus : couverture cible, verbatim (claims ET cibles), claims/avis,
multi-spans, vitesse. À lancer après un changement du prompt (cache claims invalidé).

Usage : uv run python -m backend.reextract_v3 --dataset tiktok --batch 8
"""

from __future__ import annotations

import argparse
from time import perf_counter

from backend.build_analysis import load_dataset, EXTRACT_MODEL
from backend.claims_endpoint import (
    CLAIMS_NAME, DEFAULT_MIN_CHARS, _avis_from_ideas, _save_claims_cache, dataset_dir,
)
from pipeline.claims.backend import resolve_backend
from pipeline.claims.extract import BATCH_SIZE, extract_claims
from pipeline.claims.ollama import OllamaStats


def main() -> None:
    ap = argparse.ArgumentParser(description="Ré-extraction v3 batchée d'un dataset → claims.json + métriques.")
    ap.add_argument("--dataset", default="tiktok")
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--model", default=EXTRACT_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="0 = tous les avis (sinon tronque, debug)")
    ap.add_argument("--no-save", action="store_true", help="ne pas écrire claims.json (dry-run)")
    args = ap.parse_args()

    ds = load_dataset(args.dataset)
    avis = _avis_from_ideas(ds.ideas, DEFAULT_MIN_CHARS)
    if args.limit:
        avis = avis[: args.limit]
    be = resolve_backend("api", model=args.model)
    stats = OllamaStats()
    n = len(avis)
    print(f"Ré-extraction {args.dataset} : {n} avis · {be.model} · batch={args.batch}", flush=True)

    def progress(done: int, total: int) -> None:
        if done == total or done % 80 == 0:
            el = perf_counter() - t0
            print(f"  {done}/{total} · {el:.0f}s · {el / done:.2f}s/avis · "
                  f"{stats.calls} appels · {stats.errors} err", flush=True)

    t0 = perf_counter()
    claims_by_id = extract_claims(avis, backend=be, stats=stats,
                                  progress=progress, batch_size=args.batch)
    secs = perf_counter() - t0

    by_text = {a.id: a.text for a in avis}
    n_claims = n_vb = n_multi = n_target = n_target_vb = n_whole = 0
    for aid, claims in claims_by_id.items():
        text = by_text[aid]
        for c in claims:
            n_claims += 1
            n_vb += c.is_verbatim(text)
            n_multi += len(c.spans) > 1
            n_whole += c.spans == ((0, len(text)),)
            if c.target is not None:
                n_target += 1
                ts, te = c.target
                n_target_vb += 0 <= ts < te <= len(text)

    if not args.no_save:
        path = dataset_dir(args.dataset) / CLAIMS_NAME
        _save_claims_cache(path, be.model, claims_by_id)
        print(f"\n💾 claims.json écrit : {path}")

    print("\n" + "─" * 64)
    print(f"avis={n} · claims={n_claims} ({n_claims / n:.2f}/avis) · multi-span={n_multi}")
    print(f"VERBATIM claims : {n_vb}/{n_claims} ({100.0 * n_vb / n_claims:.2f}%)")
    print(f"CIBLE           : {n_target}/{n_claims} ({100.0 * n_target / n_claims:.1f}%) "
          f"· cibles verbatim {n_target_vb}/{n_target} "
          f"({100.0 * n_target_vb / n_target if n_target else 100:.1f}%)")
    print(f"avis-entier (repli/sélection courte) : {n_whole}")
    print(f"VITESSE : {secs:.0f}s total · {secs / n:.2f}s/avis · "
          f"{stats.calls} appels LLM · {stats.errors} err")


if __name__ == "__main__":
    main()
