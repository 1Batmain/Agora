"""Groupe 6 — /sandbox + /explain (recluster live, AUCUN LLM) via la couche HTTP.

Gardé par `has_claims` : sans `claims.json` caché, il n'y a rien à recluster → skip
(le cœur algorithmique est par ailleurs couvert par `backend.test_sandbox`). Ici on
fige le CONTRAT HTTP : `/sandbox` → clusters + trace{pairs,nodes} ; `/explain` cluster
et pair → critères chiffrés ; erreurs 422/404.

Endpoints PROTÉGÉS mais ouverts en test (pas de `AGORA_API_TOKEN`) ; la fenêtre de
rate-limit est vidée entre tests par l'autouse `_reset_rate_limit`.
"""

from __future__ import annotations

import pytest

from ._helpers import has_claims

DATASET = "tiktok"


@pytest.fixture
def _need_claims():
    if not has_claims(DATASET):
        pytest.skip(f"claims.json absent pour {DATASET!r} — `POST /build` pour activer")


def test_sandbox_payload(client, _need_claims):
    r = client.post("/sandbox", json={"dataset": DATASET, "alpha": 0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) >= {"params", "n_claims", "ms", "clusters", "trace"}
    assert body["n_claims"] > 0
    assert body["clusters"], "au moins un cluster"
    for c in body["clusters"]:
        assert set(c) >= {"id", "parent_id", "n_claims", "n_avis", "keywords",
                          "sample_claims", "cohesion"}
        assert 0.0 <= c["cohesion"] <= 1.0
    # decision-trace présente et cohérente.
    trace = body["trace"]
    assert set(trace) == {"pairs", "nodes"}
    assert len(trace["nodes"]) == len(body["clusters"])
    for pr in trace["pairs"]:
        assert set(pr) >= {"a", "b", "sim", "threshold", "cohesion_min", "merged"}


def test_explain_cluster_and_pair(client, _need_claims):
    r = client.post("/sandbox", json={"dataset": DATASET, "alpha": 0.5})
    assert r.status_code == 200, r.text
    cid = r.json()["clusters"][0]["id"]

    rc = client.get("/explain", params={"dataset": DATASET, "cluster": cid})
    assert rc.status_code == 200, rc.text
    ec = rc.json()
    assert ec["cluster"] == cid
    assert "criteria" in ec and ec["neighbors"], "voisinage chiffré attendu"

    nb = ec["neighbors"][0]["id"]
    rp = client.get("/explain", params={"dataset": DATASET, "pair": f"{cid},{nb}"})
    assert rp.status_code == 200, rp.text
    ep = rp.json()
    assert ep["pair"] == [cid, nb]
    assert {"sim", "threshold", "cohesion_a", "cohesion_b", "would_merge"} <= set(ep)


def test_explain_errors(client, _need_claims):
    # On peuple d'abord l'état /explain.
    assert client.post("/sandbox", json={"dataset": DATASET}).status_code == 200
    # pair mal formée → 422 ; cluster inconnu → 404.
    assert client.get("/explain", params={"dataset": DATASET, "pair": "n1"}).status_code == 422
    assert client.get("/explain", params={"dataset": DATASET}).status_code == 422
    assert client.get("/explain",
                      params={"dataset": DATASET, "cluster": "n999999"}).status_code == 404
