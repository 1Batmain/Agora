"""Groupe 2 — INVARIANTS /avis (le cœur) sur les VRAIS caches construits.

Pour quelques avis réels (découverts via `/citations`), on fige bout en bout :
  * **Verbatim** : pour CHAQUE span (et la cible), `text[start:end]` est valide et non
    vide — les offsets servis s'alignent sur le texte servi (zéro dérive).
  * **Zéro PII brute** : aucun email/URL en clair dans `text` (ni dans `text_fr`).
  * **Multilingue** : `lang` présent, clé `text_fr` présente ; cas allemand (xstance) →
    `text_fr` non nul.

Gardé par `require_ready` : skip propre si l'analyse n'est pas précalculée (jamais de
build déclenché). Le contrat de provenance est aussi prouvé hors-cache, sans réseau,
par `backend.test_avis_pii` (chaîne Idea→Avis→spans).
"""

from __future__ import annotations

import pytest

from pipeline.ingest.normalize import _EMAIL, _URL
from ._helpers import EXPECTED_DATASETS, require_ready

READY_DATASETS = EXPECTED_DATASETS


def _assert_no_raw_pii(text: str, where: str) -> None:
    assert _EMAIL.search(text) is None, f"email brut servi dans {where}"
    assert _URL.search(text) is None, f"URL brute servie dans {where}"


def _collect_avis_ids(client, dataset: str, limit: int = 10) -> list[str]:
    """Découvre des avis_id réels via les citations des thèmes (instantané, cache)."""
    themes = client.post("/analysis", json={"dataset": dataset}).json()["themes"]
    ids: list[str] = []
    seen: set[str] = set()
    for t in themes:
        r = client.get("/citations", params={"dataset": dataset, "theme_id": t["id"]})
        if r.status_code != 200:
            continue
        for entry in r.json():
            aid = entry.get("avis_id")
            if aid and str(aid) not in seen:
                seen.add(str(aid))
                ids.append(str(aid))
            if len(ids) >= limit:
                return ids
    return ids


def _assert_spans_valid(claim: dict, text: str) -> None:
    spans = claim["spans"]
    assert spans, "un claim doit porter ≥1 span"
    for s in spans:
        assert 0 <= s["start"] < s["end"] <= len(text), f"span hors-bornes: {s}"
        assert text[s["start"]:s["end"]].strip(), "span servi vide"
    tgt = claim.get("target")
    if tgt is not None:
        assert 0 <= tgt["start"] < tgt["end"] <= len(text), f"cible hors-bornes: {tgt}"
        assert text[tgt["start"]:tgt["end"]].strip(), "cible servie vide"


@pytest.mark.parametrize("dataset", READY_DATASETS)
def test_avis_spans_verbatim_and_no_pii(client, dataset):
    require_ready(client, dataset)
    ids = _collect_avis_ids(client, dataset)
    if not ids:
        pytest.skip(f"aucun avis_id exposé via /citations pour {dataset!r}")

    checked = 0
    for aid in ids:
        r = client.get(f"/avis/{aid}", params={"dataset": dataset})
        assert r.status_code == 200, r.text
        payload = r.json()
        text = payload["text"]
        assert isinstance(text, str) and text.strip(), "avis sans texte"
        # Zéro PII brute dans le texte servi (et sa traduction si présente).
        _assert_no_raw_pii(text, f"/avis/{aid}.text")
        if payload.get("text_fr"):
            _assert_no_raw_pii(payload["text_fr"], f"/avis/{aid}.text_fr")
        # Présence du marquage de langue.
        assert payload.get("lang"), "lang manquant"
        assert "text_fr" in payload, "clé text_fr manquante"
        # Verbatim : chaque span/cible s'aligne sur le texte servi.
        for claim in payload["claims"]:
            _assert_spans_valid(claim, text)
        checked += 1
    assert checked, "au moins un avis vérifié"


def test_xstance_german_has_text_fr(client):
    """xstance est multilingue : un avis non-FR doit porter une traduction `text_fr`."""
    dataset = "xstance"
    require_ready(client, dataset)
    ids = _collect_avis_ids(client, dataset, limit=40)
    if not ids:
        pytest.skip("aucun avis_id exposé via /citations pour xstance")

    found_non_fr = False
    for aid in ids:
        payload = client.get(f"/avis/{aid}", params={"dataset": dataset}).json()
        if payload.get("lang") and payload["lang"] != "fr":
            found_non_fr = True
            assert payload.get("text_fr"), f"avis {aid} ({payload['lang']}) sans text_fr"
            _assert_no_raw_pii(payload["text_fr"], f"/avis/{aid}.text_fr")
    if not found_non_fr:
        pytest.skip("aucun avis non-FR dans l'échantillon xstance")
