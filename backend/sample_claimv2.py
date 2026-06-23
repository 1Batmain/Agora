"""Validation ÉCHANTILLON de claim-v2 : extraction réelle (mistral-large) sur 5-10 avis.

Vérifie l'acceptance « parts verbatim + target verbatim 100% » AVANT de ré-extraire
tout le dataset. N'écrit RIEN dans les caches (extraction isolée). Affiche, par avis,
les claims (parts + cible) et un récap verbatim.

Usage : uv run python -m backend.sample_claimv2 --dataset tiktok --n 8
"""

from __future__ import annotations

import argparse

from backend.build_analysis import load_dataset, EXTRACT_MODEL
from backend.claims_endpoint import DEFAULT_MIN_CHARS, _avis_from_ideas
from pipeline.claims.backend import resolve_backend
from pipeline.claims.extract import claim_prompt, parse_claims
from pipeline.claims.ollama import OllamaStats
from pipeline.claims.span import align_spans


def main() -> None:
    ap = argparse.ArgumentParser(description="Valide claim-v2 sur un échantillon (extraction réelle).")
    ap.add_argument("--dataset", default="tiktok")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--model", default=EXTRACT_MODEL)
    args = ap.parse_args()

    ds = load_dataset(args.dataset)
    avis = _avis_from_ideas(ds.ideas, DEFAULT_MIN_CHARS)[: args.n]
    be = resolve_backend("api", model=args.model)
    stats = OllamaStats()
    print(f"Échantillon : {len(avis)} avis · modèle {be.model}\n")

    n_claims = n_anchored = n_verbatim = n_multi = n_target = n_target_ok = 0
    for a in avis:
        raw = be.complete(claim_prompt(a.text), stats=stats)
        specs = parse_claims(raw)
        claims = align_spans(a.text, specs)
        print(f"━━ avis {a.id} ({len(a.text)} car.) → {len(specs)} specs, {len(claims)} claims")
        print(f"   {a.text[:160].replace(chr(10), ' ')}…")
        for c in claims:
            n_claims += 1
            anchored = c.anchored
            vb = c.is_verbatim(a.text)
            n_anchored += anchored
            n_verbatim += vb
            if len(c.spans) > 1:
                n_multi += 1
            tgt_txt = None
            if c.target is not None:
                n_target += 1
                ts, te = c.target
                tgt_txt = a.text[ts:te]
                if tgt_txt and tgt_txt in a.text:
                    n_target_ok += 1
            flag = "✓" if vb else "✗ NON-VERBATIM"
            parts_disp = " | ".join(a.text[s:e] for s, e in c.spans)
            print(f"     {flag} [{len(c.spans)} span] {parts_disp[:140]}")
            print(f"          cible: {tgt_txt!r}")
        print()

    print("─" * 60)
    print(f"claims={n_claims} · ancrés={n_anchored} · verbatim={n_verbatim} "
          f"({100.0*n_verbatim/n_claims if n_claims else 100:.1f}%)")
    print(f"multi-spans={n_multi} · avec cible={n_target} · cibles verbatim={n_target_ok}")
    print(f"appels LLM={stats.calls} · erreurs={stats.errors} · {stats.cold_seconds:.1f}s")
    if n_claims and n_verbatim == n_claims and n_target == n_target_ok:
        print("✅ ACCEPTANCE ÉCHANTILLON : 100% parts verbatim + 100% cibles verbatim")
    else:
        print("⚠️  échantillon NON conforme (voir ci-dessus)")


if __name__ == "__main__":
    main()
