"""Banc SBM emboîté vs chaîne — ÉTAPE 3/3 : COMPARAISON (env uv).

Confronte, sur chaque corpus, notre chaîne d'emboîtement et le SBM emboîté MDL :
  - STRUCTURE : combien de niveaux chacun dégage, et de quelles tailles.
  - GOLD (là où il existe) : ARI du MEILLEUR niveau de chaque méthode contre la vérité
    terrain (x-stance : 12 topics / 191 questions ; mélange : les 2 domaines).
Le témoin `mix` est le juge de paix : on SAIT qu'il y a 2 macros réels — la méthode qui les
retrouve nettement gagne le droit de dire « ce niveau est réel ».

    uv run python research/sbm_compare.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import adjusted_rand_score as ARI

OUT = Path("var/sbm")


def _load(path: Path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _best(parts: dict, gold: np.ndarray):
    """(niveau, ARI) du niveau le mieux aligné sur le gold."""
    return max(((k, ARI(gold, v)) for k, v in parts.items()), key=lambda t: t[1])


def main() -> None:
    results: dict = {}
    for name in ("tiktok", "xstance", "mix"):
        chain = _load(OUT / f"{name}.chain.npz")
        sbm = _load(OUT / f"{name}.sbm.npz")
        gold = _load(OUT / f"{name}.gold.npz")
        meta = json.loads((OUT / f"{name}.meta.json").read_text())
        sbm_meta = json.loads((OUT / f"{name}.sbm.json").read_text())

        print(f"\n=== {name}  (n={meta['n']}, graphe kNN k={meta['k_ref']}) ===")
        print(f"  chaîne : {[l[1] for l in meta['chain']]}  propretés {[l[2] for l in meta['chain']]}")
        print(f"  SBM    : {sbm_meta['levels_nblocks']}  (DL={sbm_meta['dl']:.0f})")

        row = {"n": meta["n"], "chain": meta["chain"],
               "sbm_levels": sbm_meta["levels_nblocks"], "sbm_dl": sbm_meta["dl"]}
        for gname, g in gold.items():
            ch = _best(chain, g)
            sb = _best(sbm, g)
            gagnant = "SBM" if sb[1] > ch[1] else "chaîne"
            print(f"  gold {gname:<11}: chaîne ARI={ch[1]:.3f} [{ch[0]}]  |  "
                  f"SBM ARI={sb[1]:.3f} [{sb[0]}]  → {gagnant}")
            row[f"ari_{gname}"] = {"chain": round(float(ch[1]), 3), "chain_level": ch[0],
                                   "sbm": round(float(sb[1]), 3), "sbm_level": sb[0]}
        results[name] = row

    Path("research/sbm_vs_chain_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2))
    print("\n→ research/sbm_vs_chain_results.json")


if __name__ == "__main__":
    main()
