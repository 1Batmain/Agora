"""Chaîne d'emboîtement : k comme robinet de zoom, en remplacement de `derive_k(N)`.

`derive_k(N) = 3.8·log₁₀N` devine le nombre de thèmes à partir de la TAILLE du corpus —
une formule qui ne regarde jamais le contenu. Ici on MESURE : on balaie le rayon de
voisinage k (le zoom), on clusterise à chaque palier, et on lit la hiérarchie via
l'emboîtement des partitions les unes dans les autres.

  1. Balayage k (fin → très grossier) → une partition Leiden par k.
  2. Chaîne : depuis le niveau fin, on descend en gardant à chaque étage
     (~granularité/`STEP_RATIO`) le k dont la partition s'emboîte le MIEUX dans le niveau
     courant.
  3. PROPRETÉ d'un saut = emboîtement normalisé : 0 = hasard (étiquettes grossières
     mélangées), 1 = emboîtement parfait. Elle ne dépend d'AUCUN seuil arbitraire.

La propreté est une JAUGE CONTINUE, jamais un verdict binaire : mesurée sur nos corpus,
elle vaut 0.65–0.82 partout (cascade) alors qu'un mélange artificiel de deux domaines
étrangers monte à 0.94. Aucun corpus réel n'a de frontière macro nette — la couche
grossière est une commodité de navigation, et la propreté sert à l'afficher honnêtement
(nommage par facettes + confiance quand elle est basse), pas à la supprimer.

Verdict : `.agent/notes/HIERARCHY_LAYERS.md`. Harnais de mesure : `research/k_layers.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.cluster.adaptive import derive_defaults
from pipeline.cluster.knn import build_knn_graph, knn_search
from pipeline.cluster.leiden_cluster import run_leiden

# Balayage log-espacé du rayon de voisinage. C'est LE levier : on ne calcule pas k, on le
# parcourt. Étendu aux très grands k pour atteindre les niveaux à 2-3 clusters.
K_GRID = (6, 8, 10, 13, 16, 20, 26, 34, 45, 60, 80, 110, 150, 200, 270, 360,
          480, 640, 850, 1100, 1400, 1800)
STEP_RATIO = 1.7   # granularité ~divisée par ce facteur à chaque étage de la chaîne
N_NULL = 3         # tirages du modèle nul (mélange d'étiquettes) pour normaliser

# Plafond mémoire du balayage, en arêtes du graphe kNN (n·k). Ce n'est PAS un paramètre de
# méthode : c'est la limite de la machine. Les paliers trop coûteux sont sautés et SIGNALÉS.
MAX_EDGES = 12_000_000

# Repère de LECTURE de la propreté, pas un seuil de décision : mesurée, elle vaut 0.65–0.82
# sur tous nos corpus réels contre 0.94 sur un mélange artificiel de deux domaines. Le code
# ne s'en sert JAMAIS pour trancher « plat / feuilleté » — ce serait le magic number qu'on
# refuse. Il n'existe que pour annoter les sorties de mesure.
CLEAN_FLOOR = 0.5


def centre(vecs: np.ndarray) -> np.ndarray:
    """Recentre puis re-normalise : `v ← (v−μ)/‖v−μ‖`, μ = centroïde du corpus.

    Corrige l'ANISOTROPIE du modèle d'embedding (les vecteurs nomic vivent dans un cône
    étroit : cos entre deux claims au hasard = 0.59, norme du centroïde = 0.77). Zéro
    paramètre. Mesuré : +19 % d'ARI sur le gold, hubness 3.5 → 0.9.
    Verdict : `.agent/notes/EMBEDDING_SPACE.md`.
    """
    v = vecs.astype(np.float64)
    v -= v.mean(axis=0)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return np.ascontiguousarray(v)


@dataclass(frozen=True)
class Level:
    """Un étage de la chaîne."""
    k: int                       # rayon de voisinage qui produit cet étage
    n_clusters: int
    membership: np.ndarray       # un id de cluster par claim
    cleanliness: float           # propreté du saut depuis l'étage plus fin (1.0 = base)


def _nestedness(fine: np.ndarray, coarse: np.ndarray) -> float:
    """Part des claims dont le cluster fin tombe dans UN seul cluster grossier."""
    total = 0.0
    for f in np.unique(fine):
        total += np.bincount(coarse[fine == f]).max()
    return total / len(fine)


def chain(vecs: np.ndarray, *, resolution: float = 1.0, seed: int = 42,
          on_skip=None) -> list[Level]:
    """Chaîne des layers, du plus fin au plus grossier. `vecs` DOIT être déjà recentré."""
    v64 = np.ascontiguousarray(vecs.astype(np.float64))
    v32 = v64.astype(np.float32)
    n = len(v64)
    rng = np.random.default_rng(0)

    parts: dict[int, np.ndarray] = {}
    for k in K_GRID:
        if k >= n:
            continue
        # Le graphe kNN pèse n·k arêtes : sur un gros corpus les k les plus grossiers font
        # exploser la mémoire (36 k claims × k=1800 ≈ 65 M d'arêtes → OOM). On saute ces
        # paliers plutôt que de mourir — mais JAMAIS en silence : `on_skip` le remonte, car
        # une troncature muette se lirait comme « on a balayé jusqu'au bout ».
        if n * k > MAX_EDGES:
            if on_skip:
                on_skip(k, n)
            continue
        neighbors = knn_search(v32, k)
        dd = derive_defaults(v32, k=k, neighbors=neighbors)
        graph = build_knn_graph(v64, k=dd.k, threshold=dd.threshold, neighbors=neighbors)
        parts[k] = np.asarray(run_leiden(graph, resolution=resolution, seed=seed).membership)
        del neighbors, graph
    if not parts:
        return []

    # Un k représentatif par taille de partition : le plus fin qui l'atteint.
    by_size: dict[int, int] = {}
    for k in sorted(parts):                          # k croissant → n_clusters décroissant
        by_size.setdefault(len(set(parts[k])), k)
    levels = sorted(by_size.items(), reverse=True)   # (n_clusters, k), fin → grossier

    def cleanliness(k_fine: int, k_coarse: int) -> float:
        observed = _nestedness(parts[k_fine], parts[k_coarse])
        null = float(np.mean([_nestedness(parts[k_fine], rng.permutation(parts[k_coarse]))
                              for _ in range(N_NULL)]))
        return (observed - null) / max(1 - null, 1e-9)

    n_cur, k_cur = levels[0]
    out = [Level(k_cur, n_cur, parts[k_cur], 1.0)]
    while True:
        cands = [(nn, kk) for nn, kk in levels if nn <= n_cur / STEP_RATIO]
        if not cands:
            break
        n_next, k_next = max(cands, key=lambda t: cleanliness(k_cur, t[1]))
        out.append(Level(k_next, n_next, parts[k_next], cleanliness(k_cur, k_next)))
        n_cur, k_cur = n_next, k_next
    return out
