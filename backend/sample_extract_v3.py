"""Mesure EXTRACT v3 : couverture cible (stance) + verbatim + vitesse, BATCH vs 1-par-1.

Compare, sur un même échantillon réel (mistral-large par défaut), l'extraction :
  - BATCH (N avis/appel, défaut `BATCH_SIZE`) — le mode de prod v3 ;
  - SINGLE (1 avis/appel) — le mode v2, témoin qualité.
N'écrit AUCUN cache (extraction isolée). Vérifie que le batching ne dégrade pas la
qualité et chiffre la couverture cible + le gain de vitesse.

Usage : uv run python -m backend.sample_extract_v3 --dataset tiktok --n 40 --batch 8
"""

from __future__ import annotations

import argparse
from time import perf_counter

from backend.build_analysis import load_dataset, EXTRACT_MODEL
from backend.claims_endpoint import DEFAULT_MIN_CHARS, _avis_from_ideas
from pipeline.claims.backend import resolve_backend
from pipeline.claims.extract import BATCH_SIZE, extract_claims
from pipeline.claims.ollama import OllamaStats


def _measure(avis, claims_by_id: dict) -> dict:
    """Agrège les métriques v3 sur un dict ``{avis_id: [Claim]}``."""
    by_text = {a.id: a.text for a in avis}
    n_claims = n_verbatim = n_multi = n_target = n_target_vb = 0
    for aid, claims in claims_by_id.items():
        text = by_text[aid]
        for c in claims:
            n_claims += 1
            n_verbatim += c.is_verbatim(text)
            n_multi += len(c.spans) > 1
            if c.target is not None:
                n_target += 1
                ts, te = c.target
                n_target_vb += 0 <= ts < te <= len(text)
    return {
        "n_avis": len(avis), "n_claims": n_claims,
        "claims_per_avis": n_claims / len(avis) if avis else 0.0,
        "verbatim": n_verbatim, "verbatim_pct": 100.0 * n_verbatim / n_claims if n_claims else 100.0,
        "multi": n_multi,
        "target": n_target, "target_pct": 100.0 * n_target / n_claims if n_claims else 0.0,
        "target_vb": n_target_vb,
    }


def _run(avis, *, model: str, batch_size: int) -> tuple[dict, float, OllamaStats]:
    be = resolve_backend("api", model=model)
    stats = OllamaStats()
    t0 = perf_counter()
    claims_by_id = extract_claims(avis, backend=be, stats=stats, batch_size=batch_size)
    return claims_by_id, perf_counter() - t0, stats


def _print(label: str, m: dict, secs: float, stats: OllamaStats) -> None:
    print(f"\n── {label} ── {secs:.1f}s · {stats.calls} appels · {stats.errors} err "
          f"· {secs / m['n_avis']:.2f}s/avis")
    print(f"   claims={m['n_claims']} ({m['claims_per_avis']:.2f}/avis) · "
          f"multi-span={m['multi']}")
    print(f"   VERBATIM {m['verbatim']}/{m['n_claims']} ({m['verbatim_pct']:.1f}%)")
    print(f"   CIBLE    {m['target']}/{m['n_claims']} ({m['target_pct']:.1f}%) · "
          f"cibles verbatim={m['target_vb']}/{m['target']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Mesure EXTRACT v3 (couverture cible + vitesse, batch vs single).")
    ap.add_argument("--dataset", default="tiktok")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--model", default=EXTRACT_MODEL)
    ap.add_argument("--single", action="store_true", help="ne lance QUE le batch (skip le témoin 1-par-1)")
    args = ap.parse_args()

    ds = load_dataset(args.dataset)
    avis = _avis_from_ideas(ds.ideas, DEFAULT_MIN_CHARS)[: args.n]
    print(f"Échantillon : {len(avis)} avis · modèle {args.model} · batch={args.batch}")

    cb, secs, stats = _run(avis, model=args.model, batch_size=args.batch)
    mb = _measure(avis, cb)
    _print(f"BATCH (n={args.batch})", mb, secs, stats)

    if not args.single:
        cs, secs1, stats1 = _run(avis, model=args.model, batch_size=1)
        ms = _measure(avis, cs)
        _print("SINGLE (1/appel)", ms, secs1, stats1)
        speedup = secs1 / secs if secs else 0.0
        print(f"\n⚡ vitesse : batch {secs:.1f}s vs single {secs1:.1f}s → ×{speedup:.1f} plus rapide")
        print(f"📊 cible : batch {mb['target_pct']:.0f}% vs single {ms['target_pct']:.0f}% · "
              f"verbatim batch {mb['verbatim_pct']:.0f}% / single {ms['verbatim_pct']:.0f}%")


if __name__ == "__main__":
    main()
