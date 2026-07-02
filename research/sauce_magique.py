"""HARNESS sauce_magique — évalue la re-coupe sur un `analysis.json` existant.

La logique (fonction objectif v1 + descente gloutonne `best_cut` + `recut_tree`)
vit dans **`backend/recut.py`** (intégrée au pipeline de build) ; ici on ne garde
que le banc d'essai : charger un `analysis.json`, comparer la coupe racine
(façade actuelle) à la coupe optimisée, lister la façade proposée. Sert aussi de
point de départ pour la CALIBRATION des poids contre les golds (étape suivante,
cf. `research/sauce_magique_note.md`).

Usage :
    uv run python research/sauce_magique.py [chemin/vers/analysis.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.recut import best_cut, sauce_magique  # noqa: E402

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "backend/cache/granddebat/analysis/analysis.json"
    themes = json.load(open(path))["themes"]
    roots = [t for t in themes if not t.get("parent_id")]
    s0, d0 = sauce_magique(roots)
    print(f"COUPE RACINE (macros actuels)  : {d0}")
    cut, d1 = best_cut(themes)
    print(f"COUPE OPTIMISÉE (sauce_magique): {d1}")
    print("\nFaçade optimisée (voix, titre) :")
    for n in sorted(cut, key=lambda x: -x["n_avis"])[:25]:
        print(f"  {n['n_avis']:6}  {(n.get('title') or n.get('label'))[:70]}")
    rest = len(cut) - 25
    if rest > 0:
        print(f"  … + {rest} autres clusters")
