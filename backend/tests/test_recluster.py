"""Endpoint `POST /recluster` — re-clustering LIVE piloté par le seuil k-NN.

Vérifie le CONTRAT de forme `{themes, points, indices, meta}` sur le dataset par
défaut, l'ALIGNEMENT des points sur les ideas, la forme M5 des indices
(`{key, value, detail}`), et la MONOTONIE SOUPLE (un seuil plus haut ⇒ au moins
autant de thèmes — graphe plus clairsemé, communautés plus fines).

Zéro LLM, lecture des seuls vecteurs cachés. Les points UMAP réutilisent la projection
`umap2d.npy` : si `umap-learn` est absent ET le cache absent, `points` est vide — on
SKIPPE alors la seule assertion qui en dépend (jamais d'échec parasite), à l'image de
`test_density`. La whitelist sécurité (404 path-traversal) est testée inconditionnellement.
"""

from __future__ import annotations

from backend.recluster import DEFAULT_DATASET


def test_recluster_unknown_dataset_404(client):
    """Path-traversal / id hors whitelist → 404, AUCUN calcul (garde `_resolve`)."""
    r = client.post("/recluster", json={"dataset": "../etc"})
    assert r.status_code == 404, r.text


def test_recluster_shape(client):
    """`/recluster` sur le défaut → 200 + contrat `{themes, points, indices, meta}`."""
    r = client.post("/recluster", json={"dataset": DEFAULT_DATASET})
    assert r.status_code == 200, r.text
    body = r.json()

    # themes : non vide, shape /analysis (id/label/color/parent_id/has_children…).
    themes = body["themes"]
    assert themes, "themes ne doit pas être vide"
    t0 = themes[0]
    for key in ("id", "label", "color", "parent_id", "has_children", "n_avis", "n_claims"):
        assert key in t0, f"clé manquante dans theme: {key}"

    # indices : forme M5 {key, value, detail}, value ∈ [0..1].
    indices = body["indices"]
    assert indices, "indices ne doit pas être vide"
    for ind in indices:
        assert set(("key", "value", "detail")) <= set(ind), ind
        assert 0.0 <= ind["value"] <= 1.0

    # meta : seuil, défaut dérivé, comptes, coût.
    meta = body["meta"]
    assert meta["n_themes"] == len(themes)
    assert meta["knn_threshold"] is not None
    assert meta["knn_threshold_default"] is not None
    # Seuil au repos = défaut DÉRIVÉ (la Console démarre comme /analysis).
    assert meta["knn_threshold"] == meta["knn_threshold_default"]
    assert meta["took_ms"] >= 0

    # points : un par idée, aligné à l'ordre des ideas (ou vide si UMAP indisponible).
    points = body["points"]
    if points:
        assert len(points) == meta["n_ideas"]
        ids = {t["id"] for t in themes}
        p0 = points[0]
        for key in ("x", "z", "cluster_id", "color"):
            assert key in p0, f"clé manquante dans point: {key}"
        # cluster_id pointe une FEUILLE existante de l'arbre servi (alignement).
        assert all(p["cluster_id"] in ids for p in points)


def test_recluster_threshold_soft_monotonic(client):
    """Seuil plus haut ⇒ ≥ autant de thèmes (graphe plus clairsemé). Monotonie tolérante."""
    lo = client.post("/recluster", json={"dataset": DEFAULT_DATASET, "knn_threshold": 0.45})
    hi = client.post("/recluster", json={"dataset": DEFAULT_DATASET, "knn_threshold": 0.75})
    assert lo.status_code == 200 and hi.status_code == 200
    n_lo = lo.json()["meta"]["n_themes"]
    n_hi = hi.json()["meta"]["n_themes"]
    assert n_hi >= n_lo, f"monotonie violée: hi(0.75)={n_hi} < lo(0.45)={n_lo}"
