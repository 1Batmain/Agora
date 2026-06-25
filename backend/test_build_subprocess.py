"""P2-DECOUPLE — `build_manager` lance le build dans un PROCESS SÉPARÉ (subprocess).

Verrouille le contrat de découplage BUILD/SERVE bout en bout, SANS réseau ni LLM :

  1. **Ligne de commande** : `ensure_build` spawne `python -m backend.build_analysis
     --dataset <id>` (+ flags passés en kwargs), via `sys.executable`, depuis la racine.
  2. **Non bloquant + état** : l'appel rend la main tout de suite (`building`), marque
     `status.json` en `building` AVANT de spawner, et `is_building` reflète le process vif.
  3. **Anti-double-build** : un 2ᵉ `ensure_build` pendant qu'un build tourne ne spawne PAS
     un second process.
  4. **Reaping** : quand le sous-process se termine, `is_building` repasse à False et un
     nouvel `ensure_build` peut relancer.
  5. **Court-circuit `ready`** : si l'analyse est déjà prête, aucun process n'est lancé.

On stube `subprocess.Popen` (faux handle au `poll()` contrôlable) et l'I/O `status.json`
de `analysis_store` → zéro process réel, zéro disque, déterministe. Lancer :
    uv run python -m backend.test_build_subprocess
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

from backend import analysis_store as store
from backend import build_manager as bm


class _FakeProc:
    """Faux `Popen` : `poll()` rend None tant que vivant, 0 une fois « terminé »."""

    def __init__(self, argv):
        self.argv = argv
        self._alive = True

    def finish(self):
        self._alive = False

    def poll(self):
        return None if self._alive else 0


class _Recorder:
    """Capture les Popen + simule l'état `analysis_store` en mémoire (pas de disque)."""

    def __init__(self, ready: bool = False):
        self.spawned: list[list[str]] = []
        self.procs: list[_FakeProc] = []
        self.status: dict | None = None
        self._ready = ready

    # --- stubs subprocess ---
    def popen(self, argv, **kwargs):
        self.spawned.append(argv)
        p = _FakeProc(argv)
        self.procs.append(p)
        return p

    # --- stubs analysis_store ---
    def state(self, dataset):
        return store.READY if self._ready else store.ABSENT

    def write_status(self, dataset, status, **fields):
        self.status = {"dataset": dataset, "status": status, **fields}
        return self.status


def _install(monkey: _Recorder) -> None:
    bm._procs.clear()
    bm.subprocess.Popen = monkey.popen          # type: ignore[assignment]
    store_state, store_write = store.state, store.write_status
    bm.store.state = monkey.state               # type: ignore[assignment]
    bm.store.write_status = monkey.write_status  # type: ignore[assignment]
    monkey._restore = (store_state, store_write)  # type: ignore[attr-defined]


def _restore(monkey: _Recorder) -> None:
    import subprocess as _sp

    bm.subprocess.Popen = _sp.Popen             # type: ignore[assignment]
    bm.store.state, bm.store.write_status = monkey._restore  # type: ignore[attr-defined]
    bm._procs.clear()


def _ds(ident: str = "tiktok"):
    return SimpleNamespace(id=ident)


def test_argv_is_separate_process_invocation() -> None:
    argv = bm._build_argv("tiktok", {"enrich_model": "mistral-small-latest", "seed": 42,
                                      "backend": None, "resolution": 1.5})
    assert argv[0] == sys.executable, "doit lancer l'interpréteur courant (process séparé)"
    assert argv[1:3] == ["-m", "backend.build_analysis"], "module CLI de build"
    assert argv[3:5] == ["--dataset", "tiktok"]
    # kwargs → flags ; None omis ; valeurs castées en str.
    assert "--enrich-model" in argv and "mistral-small-latest" in argv
    assert "--seed" in argv and "42" in argv
    assert "--resolution" in argv and "1.5" in argv
    assert "--backend" not in argv, "une clé None ne doit pas produire de flag"
    assert "--model" not in argv, "une clé absente ne doit pas produire de flag"
    print("OK: ensure_build spawnerait `python -m backend.build_analysis --dataset …`.")


def test_spawns_subprocess_non_blocking() -> None:
    rec = _Recorder()
    _install(rec)
    try:
        out = bm.ensure_build(_ds("tiktok"), enrich_model="x")
        assert out == store.BUILDING
        assert len(rec.spawned) == 1, "exactement UN sous-process lancé"
        assert rec.spawned[0][1:3] == ["-m", "backend.build_analysis"]
        # status.json marqué `building` AVANT le spawn (1re requête voit déjà building).
        assert rec.status and rec.status["status"] == store.BUILDING
        assert bm.is_building("tiktok") is True
        print("OK: spawn non bloquant, status=building, is_building=True.")
    finally:
        _restore(rec)


def test_no_double_build() -> None:
    rec = _Recorder()
    _install(rec)
    try:
        bm.ensure_build(_ds("tiktok"))
        bm.ensure_build(_ds("tiktok"))  # process toujours vif → pas de 2e spawn
        assert len(rec.spawned) == 1, "anti-double-build : un seul process pour le dataset"
        print("OK: pas de double-build concurrent du même dataset.")
    finally:
        _restore(rec)


def test_reaping_allows_relaunch() -> None:
    rec = _Recorder()
    _install(rec)
    try:
        bm.ensure_build(_ds("tiktok"))
        assert bm.is_building("tiktok") is True
        rec.procs[0].finish()  # le sous-process se termine
        assert bm.is_building("tiktok") is False, "process terminé → reaped"
        bm.ensure_build(_ds("tiktok"))  # peut relancer un build neuf
        assert len(rec.spawned) == 2
        print("OK: process terminé reaped, relance possible.")
    finally:
        _restore(rec)


def test_ready_short_circuits() -> None:
    rec = _Recorder(ready=True)
    _install(rec)
    try:
        out = bm.ensure_build(_ds("tiktok"))
        assert out == store.READY
        assert rec.spawned == [], "déjà ready → aucun process lancé"
        assert bm.is_building("tiktok") is False
        print("OK: analyse déjà prête → aucun build relancé.")
    finally:
        _restore(rec)


def _main() -> None:
    for t in (test_argv_is_separate_process_invocation,
              test_spawns_subprocess_non_blocking,
              test_no_double_build,
              test_reaping_allows_relaunch,
              test_ready_short_circuits):
        t()
    print("\nTOUS LES TESTS P2-DECOUPLE (build en process séparé) PASSENT.")


if __name__ == "__main__":
    _main()
