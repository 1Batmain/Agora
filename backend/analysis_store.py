"""Persistance de l'analyse PRÉ-CALCULÉE d'un dataset (couche BUILD/SERVE).

Sépare proprement le **BUILD** (précalcul lourd, cf. `backend.build_analysis`) du
**SERVE** (les endpoints `/analysis`, `/insights`, `/citations` qui ne LISENT que
ces fichiers, jamais de calcul lourd à la requête).

Disposition sur disque, par dataset :

    backend/cache/<dataset>/analysis/
        status.json                 état du build (absent|building|ready|error) + progression
        analysis.json               payload /analysis complet (themes x,y + edges + params)
        citations/<theme_id>.json   claims triées centroïde, par nœud
        insights/global.json        synthèse LLM globale
        insights/<theme_id>.json    synthèse LLM par thème

Aucune valeur de corpus en dur. Écritures ATOMIQUES (fichier temp → rename) pour
qu'un SERVE concurrent ne lise jamais un JSON à moitié écrit.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from backend.recluster import dataset_dir

ANALYSIS_DIRNAME = "analysis"
STATUS_NAME = "status.json"
ANALYSIS_NAME = "analysis.json"
AVIS_NAME = "avis.json"
CITATIONS_DIRNAME = "citations"
INSIGHTS_DIRNAME = "insights"

# États possibles d'un build (la valeur de `status.json["status"]`).
ABSENT = "absent"        # rien de persisté (jamais construit)
BUILDING = "building"    # un build est en cours
READY = "ready"          # analyse complète disponible (servable instantanément)
ERROR = "error"          # le dernier build a échoué


# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #
def analysis_dir(dataset: str) -> Path:
    return dataset_dir(dataset) / ANALYSIS_DIRNAME


def status_path(dataset: str) -> Path:
    return analysis_dir(dataset) / STATUS_NAME


def analysis_path(dataset: str) -> Path:
    return analysis_dir(dataset) / ANALYSIS_NAME


def avis_path(dataset: str) -> Path:
    return analysis_dir(dataset) / AVIS_NAME


def _safe(name: str) -> str:
    """Nom de fichier sûr (les ids de nœuds sont `n0`, `n1`… ; on durcit quand même)."""
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name)) or "_"


def citations_path(dataset: str, theme_id: str) -> Path:
    return analysis_dir(dataset) / CITATIONS_DIRNAME / f"{_safe(theme_id)}.json"


def insights_path(dataset: str, level: str, theme_id: str | None) -> Path:
    name = "global" if level == "global" else _safe(theme_id or "")
    return analysis_dir(dataset) / INSIGHTS_DIRNAME / f"{name}.json"


# --------------------------------------------------------------------------- #
# I/O atomique
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_json(path: Path, data: Any) -> None:
    """Écrit `data` en JSON de façon atomique (temp + rename dans le même dossier)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# --------------------------------------------------------------------------- #
# Statut
# --------------------------------------------------------------------------- #
def read_status(dataset: str) -> dict | None:
    return _read_json(status_path(dataset))


def write_status(dataset: str, status: str, **fields: Any) -> dict:
    """Met à jour `status.json` (fusionne les champs ; conserve le reste)."""
    cur = read_status(dataset) or {}
    cur.update({"dataset": dataset, "status": status, **fields})
    write_json(status_path(dataset), cur)
    return cur


def state(dataset: str) -> str:
    """État effectif d'un dataset pour le SERVE.

    `ready` UNIQUEMENT si `status.json` dit ready ET `analysis.json` existe (sinon on
    n'a pas de quoi servir). Sinon renvoie l'état du status, ou `absent` si rien.
    """
    st = read_status(dataset)
    if st is None:
        return ABSENT
    s = st.get("status", ABSENT)
    if s == READY and not analysis_path(dataset).exists():
        return ABSENT
    return s


def progress(dataset: str) -> dict:
    """Bloc de progression à renvoyer au front quand l'analyse n'est pas prête."""
    st = read_status(dataset) or {}
    return {
        "status": st.get("status", ABSENT),
        "phase": st.get("phase"),
        "done": st.get("done"),
        "total": st.get("total"),
        "detail": st.get("detail"),
        "error": st.get("error"),
    }


# --------------------------------------------------------------------------- #
# Lecture des artefacts (SERVE)
# --------------------------------------------------------------------------- #
def read_analysis(dataset: str) -> dict | None:
    return _read_json(analysis_path(dataset))


def read_citations(dataset: str, theme_id: str) -> list | None:
    data = _read_json(citations_path(dataset, theme_id))
    return data if isinstance(data, list) else None


# Provenance avis : un seul fichier `{avis_id: {id,text,spans}}` par dataset, mis en
# cache mémoire (clé = mtime) pour ne pas relire le JSON à chaque requête /avis.
_AVIS_CACHE: dict[str, tuple[float, dict]] = {}


def read_avis_all(dataset: str) -> dict | None:
    """Provenance de TOUS les avis `{avis_id: {id,text,claims}}` (caché par mtime).

    Source unique pour `/avis` (un avis) ET `/avis_list` (liste/recherche) : on ne
    relit le gros JSON qu'au changement de `mtime`, puis on sert depuis la RAM.
    """
    path = avis_path(dataset)
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    cached = _AVIS_CACHE.get(dataset)
    if cached is None or cached[0] != mtime:
        data = _read_json(path)
        if not isinstance(data, dict):
            return None
        _AVIS_CACHE[dataset] = (mtime, data)
        cached = _AVIS_CACHE[dataset]
    return cached[1]


def read_avis(dataset: str, avis_id: str) -> dict | None:
    """Provenance d'UN avis `{id,text,claims}` depuis `avis.json` (caché par mtime)."""
    data = read_avis_all(dataset)
    if data is None:
        return None
    entry = data.get(str(avis_id))
    return entry if isinstance(entry, dict) else None


def read_insights(dataset: str, level: str, theme_id: str | None) -> dict | None:
    data = _read_json(insights_path(dataset, level, theme_id))
    return data if isinstance(data, dict) else None


# --------------------------------------------------------------------------- #
# Écriture des artefacts (BUILD)
# --------------------------------------------------------------------------- #
def write_analysis(dataset: str, payload: dict) -> None:
    write_json(analysis_path(dataset), payload)


def write_avis(dataset: str, provenance: dict) -> None:
    """Persiste la provenance de TOUS les avis (`{avis_id: {id,text,spans}}`)."""
    write_json(avis_path(dataset), provenance)
    _AVIS_CACHE.pop(dataset, None)


def write_citations(dataset: str, theme_id: str, citations: list) -> None:
    write_json(citations_path(dataset, theme_id), citations)


def write_insights(dataset: str, level: str, theme_id: str | None, payload: dict) -> None:
    write_json(insights_path(dataset, level, theme_id), payload)


def clear(dataset: str) -> None:
    """Supprime toute l'analyse persistée d'un dataset (pour un rebuild propre)."""
    import shutil

    d = analysis_dir(dataset)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
