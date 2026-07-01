"""`/todo` — feuille de route COLLABORATIVE servie depuis `todo.json` (racine du repo).

Lecture pure (`GET`) toujours active, et ÉCRITURE collaborative (`POST` ajoute,
`PATCH` réclame/avance) sur un `todo.json` ISOLÉ en tmp (monkeypatch de `TODO_PATH`),
pour ne jamais toucher le fichier seedé du repo. Fige la FORME du contrat
`{items, updated_at?}` + `TodoItem{id, title, lane, status, assignee?}`.
"""

from __future__ import annotations

import json

import pytest

from backend import todo_store


def test_todo_shape(client):
    r = client.get("/todo")
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body, dict)
    assert "items" in body and isinstance(body["items"], list)
    # Le seed committé n'est jamais vide.
    assert body["items"], "todo.json doit être seedé"
    for it in body["items"]:
        assert set(it) >= {"id", "title", "lane", "status"}
        assert isinstance(it["id"], str) and it["id"]
        assert isinstance(it["title"], str) and it["title"]
        assert isinstance(it["lane"], str) and it["lane"]
        assert it["status"] in ("todo", "wip", "done")
        if "pr" in it:
            assert isinstance(it["pr"], (str, int))
        if "note" in it:
            assert isinstance(it["note"], str)
        if "assignee" in it:
            assert isinstance(it["assignee"], str)


@pytest.fixture
def todo_tmp(monkeypatch, tmp_path):
    """Isole l'écriture : `todo.json` dans un tmp seedé d'une tâche, jamais le repo."""
    path = tmp_path / "todo.json"
    path.write_text(
        json.dumps(
            {
                "updated_at": "2026-01-01",
                "items": [
                    {"id": "seed", "title": "Tâche seed", "lane": "backend", "status": "todo"}
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(todo_store, "TODO_PATH", path)
    return path


def test_post_todo_creates(client, todo_tmp):
    r = client.post("/todo", json={"title": "Brancher le lien in-app", "lane": "frontend"})
    assert r.status_code == 200, r.text
    item = r.json()["item"]
    assert item["status"] == "todo"
    assert item["lane"] == "frontend"
    assert item["title"] == "Brancher le lien in-app"
    assert item["id"]
    # Persisté : relire le fichier doit montrer 2 items (seed + nouveau).
    saved = json.loads(todo_tmp.read_text(encoding="utf-8"))
    assert len(saved["items"]) == 2
    assert any(it["id"] == item["id"] for it in saved["items"])


def test_post_todo_rejects_empty_title(client, todo_tmp):
    r = client.post("/todo", json={"title": "   ", "lane": "frontend"})
    assert r.status_code == 422


def test_post_todo_rejects_unknown_lane(client, todo_tmp):
    r = client.post("/todo", json={"title": "Tâche", "lane": "n-importe-quoi"})
    assert r.status_code == 422


def test_patch_todo_claims_and_advances(client, todo_tmp):
    r = client.patch("/todo/seed", json={"assignee": "bat", "status": "wip"})
    assert r.status_code == 200, r.text
    item = r.json()["item"]
    assert item["assignee"] == "bat"
    assert item["status"] == "wip"
    # Persisté.
    saved = json.loads(todo_tmp.read_text(encoding="utf-8"))
    seed = next(it for it in saved["items"] if it["id"] == "seed")
    assert seed["assignee"] == "bat" and seed["status"] == "wip"


def test_patch_todo_unknown_id_404(client, todo_tmp):
    r = client.patch("/todo/inexistant", json={"status": "done"})
    assert r.status_code == 404


def test_patch_todo_rejects_bad_status(client, todo_tmp):
    r = client.patch("/todo/seed", json={"status": "en-vrai-non"})
    assert r.status_code == 422
