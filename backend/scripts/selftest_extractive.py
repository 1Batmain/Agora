"""Auto-test EXTRACTIF — garantie « zéro hallucination » + provenance cohérente.

Deux niveaux, sans réseau :

  1. UNITAIRE — `align_spans` (`pipeline.claims.span`) : ancrage exact, repli tolérant
     aux espaces, rejet d'une portion absente, offsets distincts pour des répétitions.
  2. INTÉGRATION — sur l'extraction CACHÉE d'un dataset (`claims.json`), vérifie que
     100% des claims sont des sous-chaînes EXACTES de leur avis : pour chaque claim
     ancré, `avis_text[start:end] == claim.text` ET `claim.text in avis_text`. C'est
     la garantie « zéro hallucination » mesurée sur le vrai corpus.

Usage :
    uv run python -m backend.scripts.selftest_extractive            # unitaire seul
    uv run python -m backend.scripts.selftest_extractive --dataset tiktok   # + intégration
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
    specs = [
        {"parts": ["TikTok est dangereux pour les jeunes"],   # exact, + target verbatim
         "target": "les jeunes"},
        {"parts": ["Il faut une régulation stricte"],          # tolérant aux espaces (saut de ligne)
         "target": "régulation"},
        {"parts": ["une régulation stricte"], "target": None},  # 1ʳᵉ occurrence
        {"parts": ["une régulation stricte"], "target": None},  # 2ᵉ occurrence → offset distinct
        {"parts": ["ASBESTOS interdit partout"], "target": None},  # absent → rejeté
    ]
    claims = align_spans(avis, specs)

    assert all(c.is_verbatim(avis) for c in claims), "span non verbatim"
    assert all("ASBESTOS" not in c.text for c in claims), "portion inventée acceptée"
    assert len(claims) == 4, f"claim non ancré non rejeté ({len(claims)})"
    # Cibles ancrées comme sous-chaînes exactes (ou None).
    for c in claims:
        if c.target is not None:
            ts, te = c.target
            assert avis[ts:te] in avis, "target non verbatim"
    assert claims[0].target is not None and avis[slice(*claims[0].target)] == "les jeunes"
    # Les deux occurrences de 'une régulation stricte' ont des offsets différents.
    occ = [c for c in claims if c.text == "une régulation stricte"]
    assert len(occ) == 2 and occ[0].start != occ[1].start, "répétitions non distinguées"

    # MULTI-SPANS : deux portions non-contiguës → UN claim, texte joint, 2 spans verbatim.
    multi = align_spans(avis, [{"parts": ["Je trouve que TikTok est dangereux pour les jeunes",
                                          "Il faut\n  une régulation stricte"],
                                "target": "TikTok"}])
    assert len(multi) == 1 and len(multi[0].spans) == 2, "multi-spans non regroupé"
    assert multi[0].is_verbatim(avis), "multi-spans non verbatim"

    # Repli + normalisation cache (dict) round-trip (mono-span ET multi-span + target).
    w = whole_avis_claim(avis)
    assert w.is_verbatim(avis) and as_claim(w.to_dict()) == w
    assert as_claim(multi[0].to_dict()) == multi[0], "round-trip cache multi-spans"
    print("✓ unitaire align_spans : exact / espaces / rejet / répétitions / "
          "multi-spans / target / repli")


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

    n_claims = anchored = exact = whole = multi = with_target = 0
    bad: list[str] = []
    for aid, claims in by_id.items():
        text = text_by_id.get(aid)
        if text is None:
            continue
        for c in claims:
            n_claims += 1
            if len(c.spans) > 1:
                multi += 1
            if c.target is not None:
                with_target += 1
            if len(c.spans) == 1 and c.start == 0 and c.end == len(text) and c.text == text:
                whole += 1                      # claim couvrant l'avis entier (repli OU
                #                                 sélection légitime d'un avis court entier)
            if not c.anchored:
                continue
            anchored += 1
            # is_verbatim valide CHAQUE span (texte joint) ET la target si présente.
            if c.is_verbatim(text):
                exact += 1
            else:
                bad.append(f"{aid}:{[tuple(s) for s in c.spans]}")

    assert not bad, f"{len(bad)} claims NON verbatim (ex: {bad[:3]})"
    pct = 100.0 * exact / anchored if anchored else 100.0
    print(f"✓ provenance {dataset} : {n_claims} claims ({whole} couvrent l'avis entier, "
          f"{multi} multi-spans, {with_target} avec cible) · "
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
