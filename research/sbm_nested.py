"""Banc SBM emboîté vs chaîne — ÉTAPE 2/3 : le SBM emboîté MDL (graph-tool).

Tourne avec le PYTHON SYSTÈME (`/usr/bin/python3`), le seul qui porte `graph_tool`
(paquet Debian `python3-graph-tool`) — PAS l'env uv du projet. Charge chaque graphe kNN
exporté (`var/sbm/<name>.graph.npz`), infère la hiérarchie de blocs par minimisation de la
LONGUEUR DE DESCRIPTION (nested Stochastic Block Model, Peixoto : le nombre de niveaux et de
blocs n'est PAS choisi, il est celui qui comprime le mieux le graphe), et sauve les
partitions node→bloc par niveau (`var/sbm/<name>.sbm.npz`) + l'entropie (DL).

    /usr/bin/python3 research/sbm_nested.py [name ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import graph_tool.all as gt

OUT = Path("var/sbm")


def run(name: str) -> None:
    z = np.load(OUT / f"{name}.graph.npz")
    n, edges = int(z["n"]), z["edges"]

    g = gt.Graph(directed=False)
    g.add_vertex(n)
    g.add_edge_list(edges)

    gt.seed_rng(42)                                  # inférence stochastique → graine fixe
    np.random.seed(42)
    state = gt.minimize_nested_blockmodel_dl(g)
    dl = float(state.entropy())                      # longueur de description (bits/nats)

    levels: dict = {}
    nblocks: list = []
    for lv in range(len(state.get_levels())):
        arr = np.asarray(state.project_level(lv).get_blocks().a, dtype=np.int64)
        nb = len(set(arr.tolist()))
        levels[f"L{lv}_n{nb}"] = arr
        nblocks.append(nb)
        if nb == 1:                                  # sommet de la hiérarchie atteint
            break

    np.savez(OUT / f"{name}.sbm.npz", **levels)
    (OUT / f"{name}.sbm.json").write_text(json.dumps(
        {"dl": dl, "levels_nblocks": nblocks}, indent=2))
    print(f"{name:<8} DL={dl:.1f} niveaux(blocs)={nblocks}", flush=True)


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["tiktok", "xstance", "mix"]):
        run(nm)
