"""Persistance des FLAGS de feedback — sur un AVIS *ou* sur la SYNTHÈSE d'un thème.

Bob veut signaler qu'un artefact est mauvais — un avis mal découpé / mal ciblé /
mal extrait, OU une synthèse de thème hallucinée / mal résumée / mal clusterisée —
avec une **catégorie** et un **commentaire libre**, pour affiner le traitement
ensuite. C'est un artefact LÉGER, indépendant de l'analyse précalculée (pas de
calcul lourd) :

    backend/cache/<dataset>/flags.json
        →  {"<type>:<id>": {target_type, target_id, layer, category, text, …}}

UPSERT par couple `(target_type, target_id)` (crée OU met à jour), horodaté (UTC
ISO-8601). Écriture ATOMIQUE (temp → rename) pour qu'un GET concurrent ne lise
jamais un JSON à moitié écrit. Aucune valeur de corpus en dur : le `dataset` route
vers son propre fichier.

RÉTRO-COMPAT — l'ancien format était `{avis_id: {avis_id, text, …}}` (avis only,
sans `target_type`). `_load` migre ces entrées À LA VOLÉE en `target_type="avis"`,
clé `"avis:<id>"`, en conservant `avis_id` (== target_id) pour le front avis existant.
"""

from __future__ import annotations

from datetime import datetime, timezone

from backend.analysis_store import _read_json, write_json
from backend.recluster import dataset_dir

FLAGS_NAME = "flags.json"
AVIS = "avis"


def flags_path(dataset: str):
    return dataset_dir(dataset) / FLAGS_NAME


def _key(target_type: str, target_id: str) -> str:
    return f"{target_type}:{target_id}"


def _migrate(entry: dict) -> dict:
    """Normalise une entrée — l'ancien format avis (sans `target_type`) → modèle complet."""
    if entry.get("target_type"):
        return entry
    # Ancien flag avis : la clé portait l'avis_id, le dict ne sait que {avis_id, text, …}.
    aid = str(entry.get("avis_id", "") or entry.get("target_id", ""))
    return {
        "target_type": AVIS,
        "target_id": aid,
        "avis_id": aid,  # conservé pour le front avis existant (lit flag.avis_id)
        "layer": None,
        "category": None,
        "text": entry.get("text", ""),
        "created_at": entry.get("created_at", ""),
        "updated_at": entry.get("updated_at", ""),
    }


def _load(dataset: str) -> dict:
    """Charge le store en NORMALISANT les clés et entrées (migration douce de l'existant)."""
    data = _read_json(flags_path(dataset))
    if not isinstance(data, dict):
        return {}
    out: dict = {}
    for raw in data.values():
        if not isinstance(raw, dict):
            continue
        flag = _migrate(raw)
        out[_key(flag["target_type"], flag["target_id"])] = flag
    return out


def list_flags(dataset: str) -> list[dict]:
    """Tous les flags d'un dataset (tous types), du plus récemment modifié au plus ancien."""
    flags = list(_load(dataset).values())
    flags.sort(key=lambda f: f.get("updated_at", ""), reverse=True)
    return flags


def get_flag(dataset: str, target_type: str, target_id: str) -> dict | None:
    entry = _load(dataset).get(_key(target_type, str(target_id)))
    return entry if isinstance(entry, dict) else None


def upsert_flag(
    dataset: str,
    target_type: str,
    target_id: str,
    text: str,
    *,
    layer: int | None = None,
    category: str | None = None,
) -> dict:
    """Crée OU met à jour le flag d'une cible (horodaté). Renvoie le flag persisté."""
    target_id = str(target_id)
    key = _key(target_type, target_id)
    flags = _load(dataset)
    now = datetime.now(timezone.utc).isoformat()
    prev = flags.get(key) if isinstance(flags.get(key), dict) else {}
    flag = {
        "target_type": target_type,
        "target_id": target_id,
        "layer": layer,
        "category": category,
        "text": text,
        "created_at": prev.get("created_at") or now,
        "updated_at": now,
    }
    if target_type == AVIS:
        flag["avis_id"] = target_id  # rétro-compat front avis
    flags[key] = flag
    write_json(flags_path(dataset), flags)
    return flag


def delete_flag(dataset: str, target_type: str, target_id: str) -> bool:
    """Retire le flag d'une cible. Renvoie True s'il existait, False sinon."""
    key = _key(target_type, str(target_id))
    flags = _load(dataset)
    if key not in flags:
        return False
    del flags[key]
    write_json(flags_path(dataset), flags)
    return True
