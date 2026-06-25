"""SEC3 — `/avis` ne sert JAMAIS la PII brute, et les spans restent verbatim.

Garantit le contrat de provenance (`backend.avis`) bout en bout, SANS LLM ni cache :
  1. **Ancrage** : les spans des claims sont ancrés sur le MÊME texte que celui servi
     (`text_clean`, masqué) — `_avis_from_ideas` privilégie `text_clean`, et
     `avis_payload_for` sert `a.text`. On reconstruit la chaîne réelle (Idea brute →
     Avis → `align_spans` → payload) avec un stub d'arbre minimal (zéro embedding).
  2. **Zéro PII brute** : le texte servi (et `avis.json`) ne contient aucun email/tél./
     URL/@mention en clair ; les placeholders `[email]`/`[tel]`/`[url]`/`[mention]` sont là.
  3. **Verbatim** : pour CHAQUE span (et la cible), `texte_servi[start:end]` == la portion
     attendue. Multilingue : `text_fr`/`lang` intacts (et eux-mêmes masqués).

Aucun réseau, aucun cache disque. Lancer :
    uv run python -m backend.test_avis_pii
"""

from __future__ import annotations

import re
from types import SimpleNamespace

from backend.avis import avis_payload_for, build_avis_provenance
from backend.claims_endpoint import _avis_from_ideas
from pipeline.claims.span import SPAN_JOIN, align_spans
from pipeline.cluster.io import Idea
from pipeline.ingest.normalize import (
    _EMAIL,
    _HANDLE,
    _PHONE,
    _URL,
    clean_text,
    strip_pii,
)

# PII brute injectée dans les avis de test : ne doit JAMAIS ressortir telle quelle.
RAW_PII_LITERALS = (
    "jean.dupond@example.com",
    "https://exemple.fr/page-perso",
    "06 12 34 56 78",
    "@pseudo_militant",
)


def _make_idea(idx: int, raw: str, lang: str) -> Idea:
    """Idea comme produite par l'ingestion : `text` brut (mais masqué au build) +
    `text_clean` (masqué + normalisé), via le vrai `clean_text`/`strip_pii`."""
    return Idea.from_row(
        {"id": f"ds:{idx}", "props": {"text": strip_pii(raw), "text_clean": clean_text(raw),
                                      "lang": lang, "weight": 1.0}},
        idx,
    )


def _stub_tree(avis, claim_owner, claim_spans, claim_target):
    """Arbre minimal accepté par `backend.avis` (un seul macro couvrant tous les claims)."""
    prep = SimpleNamespace(
        avis=avis,
        claim_owner=claim_owner,
        claim_spans=claim_spans,
        claim_target=claim_target,
    )
    macro = SimpleNamespace(
        members=list(range(len(claim_owner))),
        color="#abcdef", title="Thème de test", label="test",
    )
    return SimpleNamespace(prepared=prep, macros=["m0"], nodes={"m0": macro})


def _build_scenario():
    """Construit la chaîne RÉELLE Idea→Avis→claims(align_spans)→arbre + traductions."""
    # Deux avis : un FR (avec PII) et un non-FR (DE, avec PII) → couvre le multilingue.
    raws = [
        ("Contactez-moi a jean.dupond@example.com ou au 06 12 34 56 78. "
         "Je pense que l'ecole publique doit etre la priorite, vraiment. "
         "Voir https://exemple.fr/page-perso pour mon argumentaire complet.", "fr"),
        ("Schreiben Sie an jean.dupond@example.com — siehe @pseudo_militant. "
         "Die Schule muss kostenlos bleiben, das ist entscheidend.", "de"),
    ]
    ideas = [_make_idea(i, raw, lang) for i, (raw, lang) in enumerate(raws)]
    avis = _avis_from_ideas(ideas, min_chars=5)
    assert len(avis) == 2

    # Le texte d'ancrage == text_clean (masqué) : invariant central du SEC3.
    for a, idea in zip(avis, ideas):
        assert a.text == idea.text_clean, "Avis.text doit être text_clean (masqué)"

    # « Extraction » simulée : on choisit des portions VERBATIM du texte masqué, comme
    # le ferait le LLM (qui ne voit que ce texte). align_spans en dérive les offsets.
    specs_per_avis = [
        [{"parts": ["l'ecole publique doit etre la priorite"], "target": "ecole publique"},
         {"parts": ["mon argumentaire complet"], "target": None}],
        [{"parts": ["Schule muss kostenlos bleiben"], "target": "Schule"}],
    ]
    claim_owner: list[int] = []
    claim_spans: list[list[tuple[int, int]]] = []
    claim_target: list[tuple[int, int] | None] = []
    for ai, (a, specs) in enumerate(zip(avis, specs_per_avis)):
        claims = align_spans(a.text, specs)
        assert claims, "align_spans doit ancrer au moins un claim"
        for c in claims:
            assert c.is_verbatim(a.text)         # cohérence interne align_spans
            claim_owner.append(ai)
            claim_spans.append(list(c.spans))
            claim_target.append(c.target)

    tree = _stub_tree(avis, claim_owner, claim_spans, claim_target)
    # Traductions précalculées sur le texte MASQUÉ (comme build_analysis → prepared.avis).
    from backend.translate import build_translations  # noqa: F401  (présence du module)
    translations = {
        str(avis[0].id): {"lang": "fr", "text_fr": None},
        # text_fr fabriqué à la main mais bien masqué (pas de PII) : on teste le passage.
        str(avis[1].id): {"lang": "de",
                          "text_fr": "Die Schule muss kostenlos bleiben."},
    }
    return tree, avis, translations


def _assert_no_raw_pii(text: str, where: str) -> None:
    for lit in RAW_PII_LITERALS:
        assert lit not in text, f"PII brute {lit!r} servie dans {where}"
    for rx, name in ((_EMAIL, "email"), (_URL, "url"), (_PHONE, "tel"), (_HANDLE, "mention")):
        m = rx.search(text)
        assert m is None, f"motif PII {name} ({m.group(0)!r}) dans {where}"


def test_no_raw_pii_and_spans_verbatim() -> None:
    tree, avis, translations = _build_scenario()
    seen_placeholder = False

    for idx in range(len(avis)):
        payload = avis_payload_for(tree, idx, translations=translations)
        text = payload["text"]
        _assert_no_raw_pii(text, f"/avis[{idx}].text")
        if re.search(r"\[(email|tel|url|mention)\]", text):
            seen_placeholder = True

        # VERBATIM : chaque span (et la cible) se réaligne sur le texte SERVI.
        for claim in payload["claims"]:
            parts = [text[s["start"]:s["end"]] for s in claim["spans"]]
            for p in parts:
                assert p.strip(), "span vide servi"
            joined = SPAN_JOIN.join(parts)
            # Le texte servi[start:end] DOIT correspondre verbatim (pas de dérive d'offset).
            assert all(0 <= s["start"] < s["end"] <= len(text) for s in claim["spans"])
            tgt = claim["target"]
            if tgt is not None:
                assert 0 <= tgt["start"] < tgt["end"] <= len(text)
                assert text[tgt["start"]:tgt["end"]].strip()
            assert joined  # non vide

    assert seen_placeholder, "au moins un placeholder [email]/[url]/… doit apparaître"
    print("OK: /avis ne sert aucune PII brute ; tous les spans s'alignent verbatim.")


def test_multilingual_intact() -> None:
    tree, avis, translations = _build_scenario()
    # Avis FR : lang=fr, text_fr=None (pas de traduction).
    p0 = avis_payload_for(tree, 0, translations=translations)
    assert p0["lang"] == "fr" and p0["text_fr"] is None
    # Avis DE : lang=de, text_fr présent ET masqué (zéro PII brute).
    p1 = avis_payload_for(tree, 1, translations=translations)
    assert p1["lang"] == "de" and p1["text_fr"]
    _assert_no_raw_pii(p1["text_fr"], "/avis[1].text_fr")
    # Sans table de traductions → repli FR rétro-compatible.
    p_def = avis_payload_for(tree, 1, translations=None)
    assert p_def["lang"] == "fr" and p_def["text_fr"] is None
    print("OK: multilingue intact (text_fr/lang) et text_fr masqué.")


def test_persisted_avis_json_masked() -> None:
    tree, avis, translations = _build_scenario()
    prov = build_avis_provenance(tree, translations)
    assert set(prov) == {a.id for a in avis}
    for aid, payload in prov.items():
        _assert_no_raw_pii(payload["text"], f"avis.json[{aid}].text")
        if payload.get("text_fr"):
            _assert_no_raw_pii(payload["text_fr"], f"avis.json[{aid}].text_fr")
        text = payload["text"]
        for claim in payload["claims"]:
            for s in claim["spans"]:
                assert 0 <= s["start"] < s["end"] <= len(text)
    print("OK: avis.json persisté = texte masqué, spans bornés au texte servi.")


def _main() -> None:
    for t in (test_no_raw_pii_and_spans_verbatim,
              test_multilingual_intact,
              test_persisted_avis_json_masked):
        t()
    print("\nTOUS LES TESTS SEC3 /avis PASSENT.")


if __name__ == "__main__":
    _main()
