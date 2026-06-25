"""Groupe 1 — LECTURE / SHAPE : fige le contrat de forme des endpoints de lecture.

`/health`, `/datasets`, `/build_status`, `/flags` ne dépendent d'AUCUN précalcul →
toujours actifs. `/analysis` sur les 4 datasets est gardé par `require_ready` (skip
si l'analyse n'est pas construite, jamais d'échec ni de build déclenché).
"""

from __future__ import annotations

import pytest

from backend.recluster import DEFAULT_DATASET
from ._helpers import (
    EXPECTED_DATASETS,
    available_datasets,
    require_ready,
)


def test_health_shape(client):
    r = client.get("/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["default_dataset"] == DEFAULT_DATASET
    # `datasets` : un descripteur léger par id découvert (n_cached + loaded).
    assert isinstance(body["datasets"], dict) and body["datasets"]
    for ds_id, info in body["datasets"].items():
        assert set(info) >= {"n_cached", "loaded"}
        assert isinstance(info["loaded"], bool)
    # Les ids découverts == ceux servis par /datasets (cohérence interne).
    assert set(body["datasets"]) == available_datasets()


def test_datasets_listing(client):
    r = client.get("/datasets")
    assert r.status_code == 200, r.text
    items = r.json()
    assert isinstance(items, list) and items
    ids = {d["id"] for d in items}
    # Les 4 datasets attendus sont présents (intersection : on n'exige pas plus).
    assert set(EXPECTED_DATASETS) <= ids, f"datasets manquants: {set(EXPECTED_DATASETS) - ids}"
    for d in items:
        assert set(d) >= {"id", "label", "status", "n_nodes", "languages", "namings"}
        assert d["status"] in ("open", "closed")
        assert isinstance(d["n_nodes"], int) and d["n_nodes"] >= 0
        assert isinstance(d["languages"], list)


def test_build_status_shape(client):
    for ds in sorted(available_datasets()):
        r = client.get("/build_status", params={"dataset": ds})
        assert r.status_code == 200, r.text
        prog = r.json()
        assert prog["dataset"] == ds
        assert "status" in prog
        assert isinstance(prog["building"], bool)


def test_flags_shape(client):
    """`/flags` renvoie toujours un dict {dataset, flags} (vide si aucun feedback)."""
    r = client.get("/flags", params={"dataset": DEFAULT_DATASET})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dataset"] == DEFAULT_DATASET
    assert isinstance(body["flags"], (list, dict))


@pytest.mark.parametrize("dataset", EXPECTED_DATASETS)
def test_analysis_shape(client, dataset):
    """Sur chaque dataset PRÊT : status=ready, themes non vide, champs de thème figés."""
    require_ready(client, dataset)
    r = client.post("/analysis", json={"dataset": dataset})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ready"
    themes = body["themes"]
    assert isinstance(themes, list) and themes, "themes non vide"
    for t in themes:
        assert isinstance(t["id"], str) and t["id"]
        assert isinstance(t["keywords"], list)
        assert isinstance(t["n_avis"], int) and t["n_avis"] >= 0
        assert isinstance(t["has_children"], bool)
    # Au moins un macro (thème racine, parent_id None).
    assert any(t.get("parent_id") is None for t in themes), "au moins un thème racine"
