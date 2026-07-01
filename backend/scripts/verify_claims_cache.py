"""Vérifie le `claims.json` SERVI d'un dataset (pas une ré-extraction).

Contrôle d'acceptance UNIFICATION : sur le cache d'extraction RÉELLEMENT servi par
les endpoints (`backend/cache/<dataset>/claims.json`), recompute contre le texte des
avis :
  - **verbatim** : chaque claim (join de ses spans) ET sa cible sont des sous-chaînes
    exactes de l'avis (`Claim.is_verbatim`) → doit être 100 % (gate dur PAR AVIS) ;
  - **couverture cible** : part des claims portant une cible (stance-ready) ;
  - profil : claims/avis, multi-span, avis pris en entier (repli/sélection courte).

Lecture seule, zéro LLM, zéro écriture. Sort un code non-nul si un claim n'est pas
verbatim (utilisable en CI/gate).

Usage : uv run python -m backend.scripts.verify_claims_cache --dataset xstance
"""

from __future__ import annotations

import argparse
import json
import sys

from backend.build_analysis import load_dataset
from backend.claims_endpoint import CLAIMS_NAME, DEFAULT_MIN_CHARS, _avis_from_ideas
from backend.recluster import CACHE_DIR
from pipeline.claims.span import as_claim


def verify(dataset: str) -> dict:
    ds = load_dataset(dataset)
    avis = _avis_from_ideas(ds.ideas, DEFAULT_MIN_CHARS)
    text_by_id = {a.id: a.text for a in avis}

    path = CACHE_DIR / dataset / CLAIMS_NAME
    if not path.exists():
        raise FileNotFoundError(f"claims.json absent pour {dataset!r} : {path}")
    rec = json.loads(path.read_text(encoding="utf-8"))
    claims = rec.get("claims", {})

    n_av_with = sum(1 for lst in claims.values() if lst)
    n_claims = n_vb = n_multi = n_target = n_target_vb = n_whole = 0
    bad = []
    for aid, lst in claims.items():
        text = text_by_id.get(aid)
        if text is None:
            continue
        for cd in (lst or []):
            c = as_claim(cd)
            n_claims += 1
            ok = c.is_verbatim(text)
            n_vb += ok
            if not ok and len(bad) < 5:
                bad.append((aid, c.text[:60]))
            n_multi += len(c.spans) > 1
            n_whole += c.spans == ((0, len(text)),)
            if c.target is not None:
                n_target += 1
                ts, te = c.target
                n_target_vb += 0 <= ts < te <= len(text)

    return {
        "dataset": dataset, "model": rec.get("model"),
        "n_avis": len(avis), "n_avis_with_claims": n_av_with,
        "n_claims": n_claims, "n_multi_span": n_multi, "n_whole_avis": n_whole,
        "verbatim": n_vb, "verbatim_pct": 100.0 * n_vb / max(n_claims, 1),
        "n_target": n_target, "target_cov_pct": 100.0 * n_target / max(n_claims, 1),
        "target_verbatim": n_target_vb,
        "target_verbatim_pct": 100.0 * n_target_vb / max(n_target, 1),
        "bad_verbatim": bad,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Vérifie le claims.json servi (verbatim + couverture cible).")
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    r = verify(args.dataset)
    print(f"=== {r['dataset']} ===  modèle={r['model']}")
    print(f"avis avec claims : {r['n_avis_with_claims']} / {r['n_avis']} "
          f"({100 * r['n_avis_with_claims'] / max(r['n_avis'], 1):.1f}%)")
    print(f"claims           : {r['n_claims']} "
          f"({r['n_claims'] / max(r['n_avis_with_claims'], 1):.2f}/avis-avec) · "
          f"multi-span={r['n_multi_span']} · avis-entier={r['n_whole_avis']}")
    print(f"VERBATIM claims  : {r['verbatim']}/{r['n_claims']} ({r['verbatim_pct']:.2f}%)")
    print(f"CIBLE (couv.)    : {r['n_target']}/{r['n_claims']} ({r['target_cov_pct']:.1f}%) · "
          f"cibles verbatim {r['target_verbatim']}/{r['n_target']} ({r['target_verbatim_pct']:.1f}%)")
    if r["bad_verbatim"]:
        print("NON-VERBATIM exemples:", r["bad_verbatim"])
    sys.exit(0 if r["verbatim"] == r["n_claims"] else 1)


if __name__ == "__main__":
    main()
