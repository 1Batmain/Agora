"""Persistance des FLAGS de feedback sur l'extraction d'un avis.

Bob veut signaler qu'un avis est mal découpé / mal ciblé / mal extrait, avec un
**commentaire libre**, pour affiner le traitement ensuite. C'est un artefact LÉGER,
indépendant de l'analyse précalculée (pas de calcul lourd) :

    backend/cache/<dataset>/flags.json   →  {avis_id: {avis_id, text, updated_at}}

UPSERT par `avis_id` (crée OU met à jour), horodaté (UTC ISO-8601). Écriture
ATOMIQUE (temp → rename) pour qu'un GET concurrent ne lise jamais un JSON à moitié
écrit. Aucune valeur de corpus en dur : le `dataset` route vers son propre fichier.
"""

from __future__ import annotations

from datetime import datetime, timezone

from backend.analysis_store import _read_json, write_json
from backend.recluster import dataset_dir

FLAGS_NAME = "flags.json"


def flags_path(dataset: str):
    return dataset_dir(dataset) / FLAGS_NAME


def _load(dataset: str) -> dict:
    data = _read_json(flags_path(dataset))
    return data if isinstance(data, dict) else {}


def list_flags(dataset: str) -> list[dict]:
    """Tous les flags d'un dataset, du plus récemment modifié au plus ancien."""
    flags = list(_load(dataset).values())
    flags.sort(key=lambda f: f.get("updated_at", ""), reverse=True)
    return flags


def get_flag(dataset: str, avis_id: str) -> dict | None:
    entry = _load(dataset).get(str(avis_id))
    return entry if isinstance(entry, dict) else None


def upsert_flag(dataset: str, avis_id: str, text: str) -> dict:
    """Crée OU met à jour le flag d'un avis (horodaté). Renvoie le flag persisté."""
    avis_id = str(avis_id)
    flags = _load(dataset)
    now = datetime.now(timezone.utc).isoformat()
    existing = flags.get(avis_id) if isinstance(flags.get(avis_id), dict) else {}
    flag = {
        "avis_id": avis_id,
        "text": text,
        "created_at": existing.get("created_at", now),
        "updated_at": now,
    }
    flags[avis_id] = flag
    write_json(flags_path(dataset), flags)
    return flag


def delete_flag(dataset: str, avis_id: str) -> bool:
    """Retire le flag d'un avis. Renvoie True s'il existait, False sinon."""
    avis_id = str(avis_id)
    flags = _load(dataset)
    if avis_id not in flags:
        return False
    del flags[avis_id]
    write_json(flags_path(dataset), flags)
    return True
