"""Feuille de route COLLABORATIVE (`todo.json` à la RACINE du repo).

Outil de coordination du hackathon : une source UNIQUE et partagée, versionnée (donc
récupérable), que les collaborateurs LISENT (`GET /todo`) et ÉCRIVENT en direct —
AJOUTER une tâche (`add_todo`), la RÉCLAMER / changer son statut (`patch_todo`). Pas
de dataset, pas de calcul : read-modify-write atomique du JSON.

Schéma d'un item (figé aussi côté front dans `contract.ts`) :

    TodoItem { id, title, lane, status: 'todo'|'wip'|'done', pr?, note?, assignee? }

Le fichier porte `{ "items": TodoItem[], "updated_at"?: string }`. Lecture tolérante :
fichier absent / illisible → `{items: [], updated_at: None}` (jamais d'exception au SERVE).
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from backend.analysis_store import write_json

# Racine du repo : backend/todo_store.py → parent (backend) → parent (racine).
REPO_ROOT = Path(__file__).resolve().parent.parent
TODO_PATH = REPO_ROOT / "todo.json"

_STATUSES = {"todo", "wip", "done"}

# Lanes connues de base (l'outil reste générique : toute lane DÉJÀ présente dans
# `todo.json` est également acceptée — cf. `known_lanes`).
_BASE_LANES = {"backend", "frontend", "pipeline", "research", "cross-lane"}


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
        if it.get("assignee"):
            item["assignee"] = str(it["assignee"])
        items.append(item)

    return {"items": items, "updated_at": raw.get("updated_at")}


def known_lanes() -> list[str]:
    """Lanes acceptées : le set de base UNION les lanes déjà présentes (triées)."""
    lanes = set(_BASE_LANES)
    for it in read_todo()["items"]:
        if it.get("lane"):
            lanes.add(it["lane"])
    return sorted(lanes)


def _slugify(text: str) -> str:
    """Slug court ASCII d'un titre, pour fabriquer un id stable et lisible."""
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:40] or "tache"


def _write(items: list[dict]) -> None:
    """Persiste atomiquement la feuille de route en bumpant `updated_at` (date du jour)."""
    write_json(TODO_PATH, {"updated_at": date.today().isoformat(), "items": items})


def add_todo(title: str, lane: str, note: str | None = None) -> dict:
    """Crée une tâche (`status='todo'`, id dérivé du titre) et persiste. Lève `ValueError`
    si le titre est vide ou la lane inconnue."""
    title = (title or "").strip()
    if not title:
        raise ValueError("Le titre est obligatoire.")
    lane = (lane or "").strip()
    if lane not in set(known_lanes()):
        raise ValueError(f"Lane inconnue : {lane!r}.")

    items = read_todo()["items"]
    existing = {it["id"] for it in items}
    base = _slugify(title)
    new_id, n = base, 2
    while new_id in existing:
        new_id = f"{base}-{n}"
        n += 1

    item = {"id": new_id, "title": title, "lane": lane, "status": "todo"}
    if note and note.strip():
        item["note"] = note.strip()
    items.append(item)
    _write(items)
    return item


def patch_todo(
    item_id: str,
    *,
    status: str | None = None,
    assignee: str | None = None,
) -> dict | None:
    """Réclame / réassigne (`assignee`) et/ou change le statut d'une tâche, puis persiste.

    Renvoie l'item modifié, ou `None` si l'id est inconnu. Lève `ValueError` sur un
    statut hors `{todo, wip, done}`. Un `assignee` vide DÉSASSIGNE (retire le champ).
    """
    items = read_todo()["items"]
    target = next((it for it in items if it["id"] == item_id), None)
    if target is None:
        return None

    if status is not None:
        if status not in _STATUSES:
            raise ValueError(f"Statut inconnu : {status!r}.")
        target["status"] = status
    if assignee is not None:
        who = assignee.strip()
        if who:
            target["assignee"] = who
        else:
            target.pop("assignee", None)

    _write(items)
    return target
