"""Partitionnement de clusters : `flat_partition` (SERVI) + `chain` (harnais R&D).

Ce module porte DEUX choses, à ne pas confondre :

- **`flat_partition` (chemin SERVI)** — partition à résolution γ sur UN graphe kNN fixe.
  C'est la couche FEUILLE du pipeline de prod (`backend/analysis.py`, `abstraction.py`) :
  on fixe le graphe et on clusterise à γ (le bouton direct de granularité), au lieu de
  balayer k (qui changeait le graphe et dégénérait en pure densification). Voir aussi
  `centre` (recentrage anti-anisotropie), également servi.

- **`chain` + K_GRID/STEP_RATIO/N_NULL/CLEAN_FLOOR/`_nestedness`/`Level` (HARNAIS R&D)** —
  la « chaîne d'emboîtement » : balayage de k, hiérarchie lue via l'emboîtement des
  partitions, propreté = emboîtement normalisé (0 hasard → 1 parfait). N'est PLUS dans le
  chemin servi (importé seulement par `research/`) — c'était l'exploration qui a MENÉ au
  verdict « γ, pas k ». Conservé comme instrument de mesure, pas comme moteur de prod.

Verdict : `.agent/notes/HIERARCHY_LAYERS.md`. Harnais de mesure : `research/k_layers.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.cluster.adaptive import derive_defaults
from pipeline.cluster.knn import build_knn_graph, knn_search, slice_neighbors
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

# --- Partition PLATE au pic de modularité (remplace le k-sweep) --------------------------- #
# k était un PROXY indirect de la résolution : faire varier k changeait le GRAPHE (densité
# 120× entre k=6 et k=2000, seuil adaptatif tombant à 0 → pure densification). Le bouton
# direct de granularité de la modularité est γ. On construit UN graphe fixe (un seul
# knn_search) et on balaie γ ; le pic de modularité donne le grain « naturel » du corpus
# (mesuré : xstance pic à 14 thèmes ≈ 12 topics gold). Cf. `research/gamma_sweep.py`.
K_GRAPH = 30                                            # voisinage du graphe FIXE
GAMMA_GRID = (0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0)  # résolutions balayées (recherche du pic)
# Résolution de la couche FEUILLE servie. On sert plus FIN que le pic de modularité : la couche
# fine porte le DÉTAIL (thèmes précis mais redondants), et le moteur d'abstraction remonte la
# STRUCTURE au-dessus (macros). Mesuré (Grand Débat 160k avis) : servir fin+abstraction retrouve
# les 4 domaines (ARI 0.60) mieux que servir le pic de modularité (14 thèmes, ARI 0.42).
# `FINE_GAMMA` est un paramètre de FORME (résolution = le bon bouton, cf. verdict k→γ), pas un
# seuil calé sur un corpus. Cf. `research/gd_nightly_results.json`.
FINE_GAMMA = 3.0


def flat_partition(vecs: np.ndarray, *, gamma: float | None = None,
                   seed: int = 42) -> tuple[np.ndarray, dict]:
    """Partition à résolution γ (Leiden) sur UN graphe kNN fixe. `vecs` DOIT être recentré.

    `gamma` donné → partition à cette résolution (un seul Leiden). `gamma=None` → on balaie
    `GAMMA_GRID` et on garde le PIC de modularité (le grain naturel). Renvoie `(membership,
    meta)`, `meta.curve` peuplée seulement en mode balayage.
    """
    v64 = np.ascontiguousarray(vecs.astype(np.float64))
    v32 = v64.astype(np.float32)
    n = len(v64)
    k = min(K_GRAPH, n - 1)
    nb = knn_search(v32, k)
    dd = derive_defaults(v32, k=k, neighbors=nb)
    graph = build_knn_graph(v64, k=dd.k, threshold=dd.threshold, neighbors=nb)

    if gamma is not None:
        r = run_leiden(graph, resolution=gamma, seed=seed)
        membership = np.asarray(r.membership)
        meta = {"gamma": gamma, "modularity": round(float(r.modularity), 4),
                "n_clusters": len(set(membership.tolist())),
                "k_graph": k, "threshold": round(dd.threshold, 4), "curve": []}
        return membership, meta

    best = None
    curve = []
    for gm in GAMMA_GRID:
        r = run_leiden(graph, resolution=gm, seed=seed)
        mod, nc = float(r.modularity), len(set(r.membership))
        curve.append({"gamma": gm, "n_clusters": nc, "modularity": round(mod, 4)})
        if best is None or mod > best[0]:
            best = (mod, gm, np.asarray(r.membership))
    mod, gm, membership = best
    meta = {"gamma": gm, "modularity": round(mod, 4),
            "n_clusters": len(set(membership.tolist())),
            "k_graph": k, "threshold": round(dd.threshold, 4), "curve": curve}
    return membership, meta


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

    # Le graphe kNN pèse n·k arêtes : sur un gros corpus les k les plus grossiers font
    # exploser la mémoire (36 k claims × k=1800 ≈ 65 M d'arêtes → OOM). On saute ces paliers
    # plutôt que de mourir — mais JAMAIS en silence : `on_skip` le remonte, car une troncature
    # muette se lirait comme « on a balayé jusqu'au bout ».
    usable = [k for k in K_GRID if k < n and n * k <= MAX_EDGES]
    for k in K_GRID:
        if k < n and n * k > MAX_EDGES and on_skip:
            on_skip(k, n)
    if not usable:
        return []

    # UN SEUL knn_search, au k le plus grand : le top-(k+1) d'un k plus fin est le PRÉFIXE
    # du top-(k_max+1), donc chaque palier se dérive par slice — identique au bit près à un
    # knn_search par k, mais O(n²d) payé une fois au lieu de ~15 (cf. `knn.slice_neighbors`).
    nb_max = knn_search(v32, max(usable))
    parts: dict[int, np.ndarray] = {}
    for k in usable:
        nb = slice_neighbors(nb_max, k)
        dd = derive_defaults(v32, k=k, neighbors=nb)
        graph = build_knn_graph(v64, k=dd.k, threshold=dd.threshold, neighbors=nb)
        parts[k] = np.asarray(run_leiden(graph, resolution=resolution, seed=seed).membership)
        del nb, graph

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
