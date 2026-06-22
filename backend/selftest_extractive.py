"""Auto-test EXTRACTIF — garantie « zéro hallucination » + provenance cohérente.

Deux niveaux, sans réseau :

  1. UNITAIRE — `align_spans` (`pipeline.claims.span`) : ancrage exact, repli tolérant
     aux espaces, rejet d'une portion absente, offsets distincts pour des répétitions.
  2. INTÉGRATION — sur l'extraction CACHÉE d'un dataset (`claims.json`), vérifie que
     100% des claims sont des sous-chaînes EXACTES de leur avis : pour chaque claim
     ancré, `avis_text[start:end] == claim.text` ET `claim.text in avis_text`. C'est
     la garantie « zéro hallucination » mesurée sur le vrai corpus.

Usage :
    uv run python -m backend.selftest_extractive            # unitaire seul
    uv run python -m backend.selftest_extractive --dataset tiktok   # + intégration
"""

from __future__ import annotations

import argparse
import json
import sys

from backend.build_analysis import load_dataset
from backend.claims_endpoint import DEFAULT_MIN_CHARS, _avis_from_ideas, _load_claims_cache
from backend.recluster import dataset_dir
from pipeline.claims.span import align_spans, as_claim, whole_avis_claim


def test_align_spans() -> None:
    avis = ("Je trouve que TikTok est dangereux pour les jeunes.  Il faut\n"
            "  une régulation stricte, vraiment une régulation stricte.")
    cands = [
        "TikTok est dangereux pour les jeunes",   # exact
        "Il faut une régulation stricte",          # tolérant aux espaces (saut de ligne)
        "une régulation stricte",                  # 1ʳᵉ occurrence
        "une régulation stricte",                  # 2ᵉ occurrence → offset distinct
        "ASBESTOS interdit partout",               # absent → rejeté
    ]
    claims = align_spans(avis, cands)

    assert all(c.is_verbatim(avis) for c in claims), "span non verbatim"
    assert all("ASBESTOS" not in c.text for c in claims), "portion inventée acceptée"
    # Les deux occurrences de 'une régulation stricte' ont des offsets différents.
    occ = [c for c in claims if c.text == "une régulation stricte"]
    assert len(occ) == 2 and occ[0].start != occ[1].start, "répétitions non distinguées"
    # Repli + normalisation cache (dict) round-trip.
    w = whole_avis_claim(avis)
    assert w.is_verbatim(avis) and as_claim(w.to_dict()) == w
    print("✓ unitaire align_spans : exact / espaces / rejet / répétitions / repli")


def test_provenance(dataset: str) -> int:
    """Vérifie que 100% des claims cachés sont des sous-chaînes exactes. Renvoie n_claims."""
    claims_path = dataset_dir(dataset) / "claims.json"
    if not claims_path.exists():
        print(f"⚠ {dataset} : claims.json absent — lance le build d'abord.")
        return -1
    rec = json.loads(claims_path.read_text(encoding="utf-8"))
    by_id = _load_claims_cache(claims_path, rec.get("model"))

    # Texte d'avis EXACTEMENT comme vu à l'extraction (mêmes filtres min_chars).
    ds = load_dataset(dataset)
    text_by_id = {a.id: a.text for a in _avis_from_ideas(ds.ideas, DEFAULT_MIN_CHARS)}

    n_claims = anchored = exact = whole = 0
    bad: list[str] = []
    for aid, claims in by_id.items():
        text = text_by_id.get(aid)
        if text is None:
            continue
        for c in claims:
            n_claims += 1
            if c.start == 0 and c.end == len(text) and c.text == text:
                whole += 1                      # repli avis-entier (trivialement verbatim)
            if not c.anchored:
                continue
            anchored += 1
            if text[c.start:c.end] == c.text and c.text in text:
                exact += 1
            else:
                bad.append(f"{aid}:{c.start}-{c.end}")

    assert not bad, f"{len(bad)} claims NON verbatim (ex: {bad[:3]})"
    pct = 100.0 * exact / anchored if anchored else 100.0
    print(f"✓ provenance {dataset} : {n_claims} claims ({whole} replis avis-entier) · "
          f"{anchored} ancrés, {exact} verbatim = {pct:.1f}% sous-chaînes exactes")
    return n_claims


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-test extractif (verbatim, zéro hallucination).")
    ap.add_argument("--dataset", default=None, help="vérifie aussi la provenance précalculée d'un dataset")
    args = ap.parse_args()

    test_align_spans()
    if args.dataset:
        n = test_provenance(args.dataset)
        if n < 0:
            sys.exit(2)
    print("OK")


if __name__ == "__main__":
    main()
