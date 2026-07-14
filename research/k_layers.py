"""Mesure la chaîne d'emboîtement sur nos corpus — le harnais qui a produit le verdict.

L'ALGORITHME lui-même vit dans `pipeline/cluster/layers.py` (importé ici, pas recopié) et
est câblé dans le build derrière `AGORA_LAYERS=1`. Ce script n'est que le pilote de mesure :
il charge les claims servis du cache, recentre, balaie k, et écrit les chaînes.

RÉSULTATS (2026-07-14, espace recentré, claims du cache, seed 42) :
  tiktok                16 → 9 (0.70) → 4 (0.82)          cascade
  xstance               24 → 14 (0.77) → 7 (0.78)         cascade — 14 ≈ les 12 topics gold
  république-numérique  31 → 17 (0.69) → 9 (0.79) → 5 (0.65)   cascade
  tiktok+xstance        21 → 2 (0.94)                     TÉMOIN : 2 domaines collés exprès

LECTURE : aucun corpus RÉEL n'a de frontière macro nette (tout tient entre 0.65 et 0.82),
alors que le témoin artificiel monte à 0.94. La couche grossière est donc une COMMODITÉ DE
NAVIGATION, pas une structure que le corpus imposerait. La propreté sert de JAUGE CONTINUE
pour l'afficher honnêtement (facettes + confiance quand elle est basse), jamais de verdict
binaire — tracer une coupe « plat / feuilleté » exigerait un magic number, on s'y refuse.

Usage :
    uv run --extra embed-contender --extra faiss python -u research/k_layers.py [dataset ...]
    # dataset spécial "mix" = tiktok+xstance (témoin 2 domaines)
    # Lancer dans tmux : le balayage est long et un gros corpus tient la mémoire.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from backend import analysis as A
from backend.build_analysis import load_dataset
from pipeline.cluster.layers import CLEAN_FLOOR, centre, chain

OUT = Path("var/k_layers.json")   # écrit après CHAQUE corpus : une coupure ne perd rien


def _claim_vecs(ds: str) -> np.ndarray:
    """Claims SERVIS (cache) → vecteurs. Le modèle est lu du cache : jamais de ré-extraction."""
    model = json.loads(Path(f"backend/cache/{ds}/claims.json").read_text())["model"]
    tree = A.build_theme_tree(load_dataset(ds), model=model, embedder="nomic-v2", seed=42)
    return tree.prepared.claim_vecs.astype(np.float64)


def main(datasets: list[str]) -> None:
    done: dict[str, list] = json.loads(OUT.read_text()) if OUT.exists() else {}
    for ds in datasets:
        if ds == "mix":
            vecs = np.vstack([_claim_vecs("tiktok"), _claim_vecs("xstance")])
            name = "tiktok+xstance (témoin 2 domaines)"
        else:
            vecs = _claim_vecs(ds)
            name = ds

        skipped: list[int] = []
        levels = chain(centre(vecs), on_skip=lambda k, n: skipped.append(k))

        desc = " → ".join(f"{lv.n_clusters}" + (f"({lv.cleanliness:.2f})" if lv.cleanliness < 1 else "")
                          for lv in levels)
        print(f"{name:<34} {desc}", flush=True)
        print(f"{'':<34} propres (≥{CLEAN_FLOOR}) : "
              f"{[lv.n_clusters for lv in levels if lv.cleanliness >= CLEAN_FLOOR]}", flush=True)
        if skipped:   # troncature JAMAIS muette
            print(f"{'':<34} ⚠ paliers sautés (mémoire, n·k > plafond) : k={skipped}", flush=True)

        done[ds] = {
            "chain": [{"k": lv.k, "n_clusters": lv.n_clusters,
                       "cleanliness": round(lv.cleanliness, 4)} for lv in levels],
            "skipped_k": skipped,
        }
        OUT.write_text(json.dumps(done, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:] or ["tiktok", "mix"])
