"""Verdict : le moteur DuckDB de `/avis_list` bat-il le scan Python (audit code #1) ?

Mesure, sur le plus gros dataset READY dont le `.duckdb` a été baké, la latence de
`avis.avis_list` (fallback, scan O(N) + fold Unicode par requête) vs `avis.avis_list_duckdb`
(SQL indexé). Vérifie D'ABORD la PARITÉ (mêmes `total`/ids sur 5 requêtes réelles), puis
chronomètre les deux chemins.

    uv run --extra collect python research/bench_duckdb_avis.py [dataset]
"""
from __future__ import annotations

import statistics
import sys
import time

from backend import analysis_store, avis


def _themes(dataset: str) -> list[dict]:
    return (analysis_store.read_analysis(dataset) or {}).get("themes", [])


def _pick_theme(themes: list[dict]) -> str | None:
    """Un macro (parent_id None) avec des enfants — filtre représentatif d'un sous-arbre."""
    parents = {t.get("parent_id") for t in themes}
    for t in themes:
        if t.get("parent_id") is None and t["id"] in parents:
            return t["id"]
    return themes[0]["id"] if themes else None


def _time(fn, n: int = 50) -> float:
    """Médiane (ms) de `n` exécutions, après un tour de chauffe."""
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main(dataset: str) -> int:
    avis_data = analysis_store.read_avis_all(dataset)
    if avis_data is None:
        print(f"{dataset}: pas de avis.json — bake d'abord l'analyse.")
        return 1
    themes = _themes(dataset)
    stance = analysis_store.read_claim_stance(dataset)
    con = analysis_store.avis_duckdb_con(dataset)
    if con is None:
        print(f"{dataset}: pas de analysis.duckdb frais — lance `python -m backend.bake_duckdb {dataset}`.")
        return 1

    theme_id = _pick_theme(themes)
    # 5 requêtes réalistes : recherche courante, recherche rare, thème, thème+recherche, stance.
    queries = [
        {"q": "internet"},
        {"q": "reglementation"},
        {"theme_id": theme_id},
        {"theme_id": theme_id, "q": "enfants"},
        {"stance": "favorable"},
    ]

    print(f"# {dataset} — {len(avis_data)} avis, {len(themes)} thèmes\n")
    print(f"{'requête':<38} {'total':>6}  {'fallback':>10}  {'duckdb':>9}  {'gain':>6}")
    print("-" * 78)
    gains = []
    ok = True
    for params in queries:
        py = avis.avis_list(avis_data, themes, claim_stance=stance, limit=15, **params)
        db = avis.avis_list_duckdb(con, themes, claim_stance=stance, limit=15, **params)
        same = (py["total"] == db["total"]
                and [i["avis_id"] for i in py["items"]] == [i["avis_id"] for i in db["items"]])
        ok = ok and same
        t_py = _time(lambda p=params: avis.avis_list(avis_data, themes, claim_stance=stance,
                                                      limit=15, **p))
        t_db = _time(lambda p=params: avis.avis_list_duckdb(con, themes, claim_stance=stance,
                                                            limit=15, **p))
        gain = t_py / t_db if t_db else float("inf")
        gains.append(gain)
        flag = "" if same else "  ✗ PARITÉ"
        label = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"{label:<38} {py['total']:>6}  {t_py:>8.2f}ms  {t_db:>7.2f}ms  {gain:>5.1f}×{flag}")
    print("-" * 78)
    print(f"parité: {'OK (5/5)' if ok else 'ÉCHEC'} · gain médian ×{statistics.median(gains):.1f}")
    return 0 if ok else 2


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "republique-numerique"
    raise SystemExit(main(ds))
