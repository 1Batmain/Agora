"""Gestionnaire de BUILDS EN PROCESS SÉPARÉ (non bloquant) pour l'analyse des datasets.

Au démarrage du serveur — et à la première requête sur un dataset froid — on lance
ICI un **sous-process** (`python -m backend.build_analysis …`) qui exécute le pipeline
lourd SANS partager le GIL/RAM du process SERVE et SANS bloquer la boucle d'événements
FastAPI. Les endpoints SERVE n'attendent jamais : ils lisent le cache si prêt, sinon
renvoient `building` (le build avance dans son propre process).

Pourquoi un PROCESS et plus un thread daemon : le build est CPU/LLM lourd ; dans un
thread il partageait le GIL/RAM du serve et empêchait le multi-worker (cf. note de
déploiement en bas de ce module). En process séparé, le serve reste léger et répondant,
et peut tourner en multi-worker derrière un proxy.

Source de vérité de l'état = `status.json` (persisté, écrit par le SOUS-PROCESS lui-même
via `build_analysis`, survit aux redémarrages). Le SERVE continue de LIRE ce status, le
mécanisme est inchangé. Le registre en mémoire `_procs` (handle `Popen` par dataset)
empêche seulement de lancer DEUX builds du même dataset à la fois DANS CE PROCESS.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

from backend import analysis_store as store

# Racine du dépôt (parent du paquet `backend`) : le sous-process est lancé en
# `python -m backend.build_analysis`, donc depuis la racine où le paquet est importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent

# datasets dont un build tourne MAINTENANT dans ce process (anti double-lancement).
# Valeur = handle du sous-process ; on le sonde pour savoir s'il est toujours vivant.
_procs: dict[str, subprocess.Popen] = {}
_lock = threading.Lock()


def _reap_locked() -> None:
    """Retire du registre les builds dont le sous-process s'est terminé. À appeler sous `_lock`."""
    for ds_id in [d for d, p in _procs.items() if p.poll() is not None]:
        _procs.pop(ds_id, None)


def is_building(dataset: str) -> bool:
    with _lock:
        _reap_locked()
        return dataset in _procs


def active() -> list[str]:
    with _lock:
        _reap_locked()
        return sorted(_procs)


def _build_argv(dataset: str, build_kwargs: dict) -> list[str]:
    """Construit la ligne de commande du sous-process de build.

    `python -m backend.build_analysis --dataset <id>` + les options passées dans
    `build_kwargs` (backend/model/enrich_model/embedder/resolution/seed). Les clés à
    `None` sont omises (le CLI applique alors ses propres défauts). Aucune valeur de
    corpus codée en dur. `on_progress` n'a pas de sens entre process : ignoré.
    """
    argv = [sys.executable, "-m", "backend.build_analysis", "--dataset", dataset]
    flag_for = {
        "backend": "--backend",
        "model": "--model",
        "enrich_model": "--enrich-model",
        "embedder": "--embedder",
        "resolution": "--resolution",
        "seed": "--seed",
    }
    for key, flag in flag_for.items():
        val = build_kwargs.get(key)
        if val is not None:
            argv += [flag, str(val)]
    return argv


def ensure_build(ds, **build_kwargs) -> str:
    """Garantit qu'une analyse existe ou se construit pour `ds`. Renvoie l'état courant.

    - déjà `ready` → ne fait rien, renvoie ``ready`` ;
    - build déjà en cours (sous-process vivant) → renvoie ``building`` ;
    - sinon → lance un SOUS-PROCESS de build et renvoie ``building``.

    `ds` porte `.id` (le sous-process recharge le dataset léger depuis le cache via
    `build_analysis.load_dataset`, donc seul l'`id` traverse la frontière de process).
    `build_kwargs` (backend/model/enrich_model/embedder/resolution/seed) est transmis en
    arguments CLI. Ne lève jamais : un échec de build est capturé et persisté en
    `status=error` par le sous-process lui-même.
    """
    dataset = ds.id
    with _lock:
        _reap_locked()
        if store.state(dataset) == store.READY:
            return store.READY
        if dataset in _procs:
            return store.BUILDING

        # Marque l'état AVANT de lancer (la 1re requête voit déjà `building`) — le
        # sous-process ré-écrira ensuite la progression dans le même `status.json`.
        store.write_status(dataset, store.BUILDING, phase="queued",
                           detail="build en file d'attente")

        argv = _build_argv(dataset, build_kwargs)
        try:
            # Process SÉPARÉ : hérite de l'environnement (clés API…) et du cwd racine.
            # Non bloquant ; on ne lit pas ses flux (il écrit status.json + logue).
            proc = subprocess.Popen(argv, cwd=str(_REPO_ROOT))
        except Exception as exc:  # noqa: BLE001 — échec de spawn : persiste l'erreur
            store.write_status(dataset, store.ERROR, phase="error",
                               detail="échec du lancement du build", error=str(exc))
            return store.ERROR
        _procs[dataset] = proc

    return store.BUILDING


def ensure_all(datasets, **build_kwargs) -> dict[str, str]:
    """Lance (si besoin) un build pour chaque dataset sans analyse prête. → {id: état}."""
    return {ds.id: ensure_build(ds, **build_kwargs) for ds in datasets}


# --------------------------------------------------------------------------- #
# Déploiement — SERVE multi-worker (le build tourne HORS-serve)
# --------------------------------------------------------------------------- #
# Le build étant désormais lancé dans un PROCESS SÉPARÉ (et non plus un thread du
# process SERVE), le serveur FastAPI ne porte plus de charge CPU/LLM lourde : il ne
# fait que LIRE le cache persisté. On peut donc le servir en MULTI-WORKER derrière un
# proxy, par ex. :
#
#     gunicorn -k uvicorn.workers.UvicornWorker --workers N backend.server:app
#
# Notes :
#   - Chaque worker garde son propre registre `_procs` en mémoire : l'anti-double-build
#     est garanti DANS un worker. En multi-worker, faites tourner le build hors-serve
#     (CLI `python -m backend.build_analysis --dataset <id>`, ou un seul worker dédié /
#     AGORA_AUTOBUILD=0 sur les workers SERVE) pour éviter deux builds concurrents du
#     même dataset depuis des workers différents. La source de vérité reste `status.json`.
#   - Désactiver l'autobuild au démarrage des workers SERVE : `AGORA_AUTOBUILD=0`.
#   - Le sous-process de build survit à un redémarrage du serve (process indépendant) et
#     termine d'écrire `status.json` ; au reboot, le serve relit l'état persisté.
