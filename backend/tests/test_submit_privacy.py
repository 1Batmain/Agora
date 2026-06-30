"""Groupe — /submit VIE PRIVÉE : masquage PII + agrégat sans verbatim d'autrui.

Audit privacy #1 (CRITIQUE) : `/submit` ne doit (a) jamais PERSISTER le texte brut
(PII), ni (b) RENVOYER le verbatim d'un autre citoyen. La réponse est un AGRÉGAT
non-PII : `n_similar` + `pct_panel`.

On NEUTRALISE l'embedder (pas de torch en test) et le store (capture en mémoire) : on
teste la grille de confidentialité, pas l'embedding nomic.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend import submissions

CID = "ameliorer-agora"  # consultation OUVERTE (cf. list_open_consultations)


@pytest.fixture
def stub_store(monkeypatch):
    """Embedder + store en mémoire ; une contribution voisine existe déjà (verbatim PII)."""
    captured: dict = {}

    monkeypatch.setattr(submissions, "embed_text", lambda t: np.array([1.0, 0.0, 0.0], np.float32))
    # Un voisin déjà présent, dont le texte contient une PII verbatim qui NE doit JAMAIS sortir.
    existing = [{"text": "Contactez Jean au 0612345678", "vec": [1.0, 0.0, 0.0]}]
    monkeypatch.setattr(submissions, "load_submissions", lambda cid: existing)

    def _append(cid, text, vec, ts):
        captured["cid"], captured["text"] = cid, text

    monkeypatch.setattr(submissions, "append_submission", _append)
    return captured


def test_response_is_aggregate_no_verbatim(client, stub_store):
    """La réponse porte n_similar + pct_panel, et AUCUN verbatim d'autrui."""
    r = client.post("/submit", json={"consultation_id": CID, "text": "je propose une idee"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_similar"] == 1
    assert body["pct_panel"] == 100          # 1 voisin sur 1 contribution du panel
    assert "nearest_excerpt" not in body
    # Le verbatim (et la PII) de l'autre citoyen n'apparaît nulle part dans la réponse.
    assert "Jean" not in r.text and "0612345678" not in r.text


def test_submitted_text_is_masked_before_storage(client, stub_store):
    """Le texte soumis est MASQUÉ (PII) avant stockage : pas de brut persisté."""
    r = client.post("/submit", json={
        "consultation_id": CID,
        "text": "mon mail est test@example.com et tel 0612345678",
    })
    assert r.status_code == 200, r.text
    stored = stub_store["text"]
    assert "test@example.com" not in stored
    assert "0612345678" not in stored
    assert "[email]" in stored and "[tel]" in stored
