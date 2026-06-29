"""Lecture de la feuille de route collaborative (`todo.json` à la RACINE du repo).

Source UNIQUE et partagée, éditée À LA MAIN au merge des PR (pas de calcul, pas de
dataset). Le endpoint `/todo` ne fait que LIRE ce fichier instantanément.

Schéma d'un item (figé aussi côté front dans `contract.ts`) :

    TodoItem { id, title, lane, status: 'todo'|'wip'|'done', pr?, note? }

Le fichier porte `{ "items": TodoItem[], "updated_at"?: string }`. Lecture tolérante :
fichier absent / illisible → `{items: [], updated_at: None}` (jamais d'exception au SERVE).
"""

from __future__ import annotations

import json
from pathlib import Path

# Racine du repo : backend/todo_store.py → parent (backend) → parent (racine).
REPO_ROOT = Path(__file__).resolve().parent.parent
TODO_PATH = REPO_ROOT / "todo.json"

_STATUSES = {"todo", "wip", "done"}


def read_todo() -> dict:
    """Renvoie `{items, updated_at}` depuis `todo.json` (forme garantie, jamais d'exception)."""
    try:
        raw = json.loads(TODO_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"items": [], "updated_at": None}
    if not isinstance(raw, dict):
        return {"items": [], "updated_at": None}

    items = []
    for it in raw.get("items", []) if isinstance(raw.get("items"), list) else []:
        if not isinstance(it, dict):
            continue
        status = it.get("status")
        item = {
            "id": str(it.get("id", "")),
            "title": str(it.get("title", "")),
            "lane": str(it.get("lane", "")),
            "status": status if status in _STATUSES else "todo",
        }
        if it.get("pr") is not None:
            item["pr"] = it["pr"]
        if it.get("note"):
            item["note"] = str(it["note"])
        items.append(item)

    return {"items": items, "updated_at": raw.get("updated_at")}
