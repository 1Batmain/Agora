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
OPINION_NAME = "opinion.json"
CLAIM_STANCE_NAME = "claim_stance.json"
ARGUMENTS_NAME = "arguments.json"
DEMOGRAPHICS_NAME = "demographics.json"
DUCKDB_NAME = "analysis.duckdb"
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


def opinion_path(dataset: str) -> Path:
    """Répartition d'opinion par thème feuille (objet de clivage T2 + stance agrégée).

    Artefact À PART, baké par `backend.build_opinion` — indépendant du build d'analyse
    (il ne touche jamais `analysis.json` ni les caches existants). Absent tant qu'on n'a
    pas baké l'opinion : les endpoints/serve dégradent gracieusement.
    """
    return analysis_dir(dataset) / OPINION_NAME


def claim_stance_path(dataset: str) -> Path:
    """Stance PAR CLAIM (`{claim_id: {stance, justif, proposition, theme_id}}`).

    Artefact À PART baké par `backend.build_opinion`, à côté d'`opinion.json` : la clé
    est l'id de claim servi par `/avis` (`f"{avis_id}#{index}"`). N'est émis que sur les
    thèmes assez purs ; les endpoints joignent gracieusement (absent → pas de stance).
    """
    return analysis_dir(dataset) / CLAIM_STANCE_NAME


def arguments_path(dataset: str) -> Path:
    """Arguments minés par thème (synthèse LLM sourcée sur contributions réelles).

    Artefact À PART et OPTIONNEL, baké par `backend.build_arguments` — jamais requis :
    les datasets déjà analysés n'en ont pas et tout dégrade gracieusement (contrat de
    rétro-compat). Ne touche à aucun artefact existant.
    """
    return analysis_dir(dataset) / ARGUMENTS_NAME


def demographics_path(dataset: str) -> Path:
    """Profil démographique du panel (global + majorités par thème).

    Artefact À PART et OPTIONNEL, baké par `backend.build_demographics` (pure
    jointure CSV enrichi ↔ avis, zéro LLM). Absent sur les datasets sans données
    démographiques : tout dégrade gracieusement.
    """
    return analysis_dir(dataset) / DEMOGRAPHICS_NAME


def _safe(name: str) -> str:
    """Nom de fichier sûr (les ids de nœuds sont `n0`, `n1`… ; on durcit quand même)."""
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name)) or "_"


def citations_path(dataset: str, theme_id: str) -> Path:
    return analysis_dir(dataset) / CITATIONS_DIRNAME / f"{_safe(theme_id)}.json"


def insights_path(dataset: str, level: str, theme_id: str | None) -> Path:
    # Cache BAKÉ (précalculé au build) : <dataset>/analysis/insights/<global|theme_id>.json,
    # nom SÉMANTIQUE. Le repli LIVE (`backend.insights._disk_path`) réutilise le même DIRNAME
    # mais sous <dataset>/insights/ et nommé par HASH — deux étages distincts, un seul littéral.
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


def read_opinion(dataset: str) -> dict | None:
    """Répartition d'opinion bakée (`{dataset, model, themes:[…]}`) ou None si absente."""
    data = _read_json(opinion_path(dataset))
    return data if isinstance(data, dict) else None


def read_arguments(dataset: str) -> dict | None:
    """Arguments minés bakés (`{dataset, model, themes:[…]}`) ou None si absents."""
    data = _read_json(arguments_path(dataset))
    return data if isinstance(data, dict) else None


def read_demographics(dataset: str) -> dict | None:
    """Profil démographique baké (`{dataset, axes, global, themes:[…]}`) ou None."""
    data = _read_json(demographics_path(dataset))
    return data if isinstance(data, dict) else None


# Stance par claim : un seul fichier `{claim_id: {…}}` par dataset, caché par mtime
# (joint à chaque requête `/avis`/`/avis_list`, comme la provenance avis).
_CLAIM_STANCE_CACHE: dict[str, tuple[float, dict]] = {}


def read_claim_stance(dataset: str) -> dict | None:
    """Stance par claim `{claim_id: {stance, justif, proposition, theme_id}}` (caché mtime).

    `None` si l'opinion n'a pas (encore) été bakée → les endpoints `/avis` dégradent
    gracieusement (claims servis sans stance).
    """
    path = claim_stance_path(dataset)
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    cached = _CLAIM_STANCE_CACHE.get(dataset)
    if cached is None or cached[0] != mtime:
        data = _read_json(path)
        if not isinstance(data, dict):
            return None
        _CLAIM_STANCE_CACHE[dataset] = (mtime, data)
        cached = _CLAIM_STANCE_CACHE[dataset]
    return cached[1]


# --------------------------------------------------------------------------- #
# Index DuckDB de LECTURE (hot path `/avis_list`) — cache dérivé, optionnel
# --------------------------------------------------------------------------- #
# Une connexion read-only par dataset, cachée par mtime (comme `avis.json`). C'est un
# CACHE dérivé : absent ou plus vieux que ses sources (`avis.json`/`claim_stance.json`)
# → on renvoie None et le serve retombe sur le chemin Python (aucun cache existant cassé).
_DUCKDB_CACHE: dict[str, tuple[float, Any]] = {}


def duckdb_path(dataset: str) -> Path:
    return analysis_dir(dataset) / DUCKDB_NAME


def avis_sources_sig(dataset: str) -> list[tuple[str, int]]:
    """Signature des sources de l'index : (nom, taille octets) pour avis.json + claim_stance.

    Taille = contenu (déterministe au `git checkout`), contrairement au mtime. Stockée dans
    la table `meta` du `.duckdb` au bake, recomparée au serve : l'index est VALIDE ssi les
    tailles courantes égalent celles du bake (self-heal si une source change ; robuste pour
    un `.duckdb` COMMITÉ servi en prod après `reset --hard`). `-1` = fichier absent (distingue
    « claim_stance jamais baké » de « claim_stance présent »).
    """
    out: list[tuple[str, int]] = []
    for name, path in ((AVIS_NAME, avis_path(dataset)),
                       (CLAIM_STANCE_NAME, claim_stance_path(dataset))):
        out.append((name, path.stat().st_size if path.exists() else -1))
    return out


def _duckdb_valid(con, dataset: str) -> bool:
    """L'index reflète-t-il les sources courantes ? (signature `meta` == tailles actuelles)."""
    try:
        rows = con.execute("SELECT source, n_bytes FROM meta").fetchall()
    except Exception:
        return False
    return dict(rows) == dict(avis_sources_sig(dataset))


def avis_duckdb_con(dataset: str):
    """Curseur DuckDB read-only pour `/avis_list`, ou None si l'index n'est pas servable.

    None dans tous les cas de repli : index absent, périmé (signature `meta` ≠ sources),
    `duckdb` non installé, ou ouverture en échec. Le caller (`server.get_avis_list`) retombe
    alors sur `avis.avis_list`. La connexion est cachée par mtime du `.duckdb` (un rebake en
    dev change le mtime → réouverture ; en prod le process redémarre) ; on renvoie un
    `.cursor()` thread-local (les endpoints sync tournent dans un threadpool).
    """
    p = duckdb_path(dataset)
    if not p.exists():
        return None
    m = p.stat().st_mtime
    try:
        import duckdb
    except ImportError:               # extra `collect`/`serve` non installé → fallback
        return None
    cached = _DUCKDB_CACHE.get(dataset)
    if cached is None or cached[0] != m:
        if cached is not None and cached[1] is not None:   # ferme l'ancienne (fichier remplacé)
            try:
                cached[1].close()
            except Exception:
                pass
        try:
            con = duckdb.connect(str(p), read_only=True)
        except Exception:
            _DUCKDB_CACHE[dataset] = (m, None)
            return None
        if not _duckdb_valid(con, dataset):    # index périmé → on mémorise le verdict négatif
            try:
                con.close()
            except Exception:
                pass
            con = None
        _DUCKDB_CACHE[dataset] = (m, con)
        cached = _DUCKDB_CACHE[dataset]
    if cached[1] is None:
        return None
    try:
        return cached[1].cursor()
    except Exception:
        return None


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


def write_opinion(dataset: str, payload: dict) -> None:
    """Persiste la répartition d'opinion (fichier À PART, n'efface aucun cache d'analyse)."""
    write_json(opinion_path(dataset), payload)


def write_arguments(dataset: str, payload: dict) -> None:
    """Persiste les arguments minés (fichier À PART, n'efface aucun cache d'analyse)."""
    write_json(arguments_path(dataset), payload)


def write_demographics(dataset: str, payload: dict) -> None:
    """Persiste le profil démographique (fichier À PART, n'efface aucun cache)."""
    write_json(demographics_path(dataset), payload)


def write_claim_stance(dataset: str, claim_stance: dict) -> None:
    """Persiste la stance par claim (`{claim_id: {…}}`, fichier À PART, voisin d'opinion.json)."""
    write_json(claim_stance_path(dataset), claim_stance)
    _CLAIM_STANCE_CACHE.pop(dataset, None)


def clear(dataset: str) -> None:
    """Supprime toute l'analyse persistée d'un dataset (pour un rebuild propre)."""
    import shutil

    d = analysis_dir(dataset)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
