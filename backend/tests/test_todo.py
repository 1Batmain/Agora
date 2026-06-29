"""`/todo` — feuille de route collaborative servie depuis `todo.json` (racine du repo).

Lecture pure, indépendante de tout précalcul → toujours active. Fige la FORME du
contrat `{items, updated_at?}` + `TodoItem{id, title, lane, status}`.
"""

from __future__ import annotations


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
