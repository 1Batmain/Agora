"""Gestionnaire de BUILDS EN TÂCHE DE FOND (non bloquant) pour l'analyse des datasets.

Au démarrage du serveur — et à la première requête sur un dataset froid — on lance
ICI un thread qui exécute `backend.build_analysis.build_analysis` SANS bloquer la
boucle d'événements FastAPI. Les endpoints SERVE n'attendent jamais : ils lisent le
cache si prêt, sinon renvoient `building` (le build avance en fond).

Source de vérité de l'état = `status.json` (persisté, survit aux redémarrages). Le set
en mémoire `_active` empêche seulement de lancer DEUX builds du même dataset à la fois.
Threads `daemon` : ils n'empêchent pas l'arrêt du process.
"""

from __future__ import annotations

import threading

from backend import analysis_store as store
from backend.build_analysis import build_analysis

# datasets dont un build tourne MAINTENANT dans ce process (anti double-lancement).
_active: set[str] = set()
_lock = threading.Lock()


def is_building(dataset: str) -> bool:
    with _lock:
        return dataset in _active


def active() -> list[str]:
    with _lock:
        return sorted(_active)


def ensure_build(ds, **build_kwargs) -> str:
    """Garantit qu'une analyse existe ou se construit pour `ds`. Renvoie l'état courant.

    - déjà `ready` → ne fait rien, renvoie ``ready`` ;
    - build déjà en cours → renvoie ``building`` ;
    - sinon → lance un thread de build et renvoie ``building``.

    `ds` porte `.id` et `.ideas`. `build_kwargs` est transmis à `build_analysis`
    (backend/model/resolution…). Ne lève jamais : un échec de build est capturé et
    persisté en `status=error` par `build_analysis`.
    """
    dataset = ds.id
    with _lock:
        if store.state(dataset) == store.READY:
            return store.READY
        if dataset in _active:
            return store.BUILDING
        _active.add(dataset)

    # Marque l'état AVANT de lancer le thread (la 1re requête voit déjà `building`).
    store.write_status(dataset, store.BUILDING, phase="queued",
                       detail="build en file d'attente")

    def _run() -> None:
        try:
            build_analysis(ds, **build_kwargs)
        except Exception:  # noqa: BLE001 — déjà persisté en status=error par build_analysis
            pass
        finally:
            with _lock:
                _active.discard(dataset)

    threading.Thread(target=_run, name=f"build-{dataset}", daemon=True).start()
    return store.BUILDING


def ensure_all(datasets, **build_kwargs) -> dict[str, str]:
    """Lance (si besoin) un build pour chaque dataset sans analyse prête. → {id: état}."""
    return {ds.id: ensure_build(ds, **build_kwargs) for ds in datasets}
