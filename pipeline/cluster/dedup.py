"""T-N2.5 · Déduplication near-dup (avant le graphe k-NN).

Sur des avis citoyens réels, les gens **répètent** (mêmes mots, mêmes phrases).
Garder chaque copie gonfle artificiellement un thème et alourdit le rendu. On
collapse donc les quasi-doublons : toute paire de vecteurs de cosinus > `threshold`
est fusionnée (union-find). Le **représentant** d'un groupe = l'index minimal
(déterministe) ; son `weight` cumule le poids social de tout le groupe — la voix
n'est pas perdue, elle est *pondérée*.

Vecteurs supposés L2-normalisés (cosine = produit scalaire). Déterministe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DedupResult:
    keep: list[int]            # indices conservés (représentants), triés
    weights: np.ndarray        # poids cumulés, aligné sur `keep`
    n_in: int
    n_out: int
    groups: dict[int, list[int]]  # représentant -> indices absorbés (incl. lui-même)

    @property
    def n_collapsed(self) -> int:
        return self.n_in - self.n_out


def _find(parent: list[int], x: int) -> int:
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:  # path compression
        parent[x], x = root, parent[x]
    return root


def dedup_near(
    vecs: np.ndarray,
    weights: np.ndarray,
    threshold: float = 0.95,
    block: int = 512,
) -> DedupResult:
    """Fusionne les near-dups (cosine > threshold) en gardant 1 représentant.

    Calcul par blocs de lignes pour rester sobre en RAM sur ~1.5k vecteurs.
    """
    n = vecs.shape[0]
    parent = list(range(n))
    if n > 1:
        for start in range(0, n, block):
            stop = min(start + block, n)
            sims = vecs[start:stop] @ vecs.T          # (b, n)
            for r in range(stop - start):
                i = start + r
                # seules les colonnes j > i comptent (paires non ordonnées)
                js = np.nonzero(sims[r, i + 1:] > threshold)[0]
                for j in js + (i + 1):
                    a, b = _find(parent, i), _find(parent, int(j))
                    if a != b:
                        parent[min(a, b)] = max(a, b)  # racine = index max (stable)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(_find(parent, i), []).append(i)

    # représentant = index minimal du groupe (déterministe, lisible)
    rep_groups: dict[int, list[int]] = {}
    for members in groups.values():
        rep = min(members)
        rep_groups[rep] = sorted(members)

    keep = sorted(rep_groups.keys())
    new_w = np.array(
        [float(weights[rep_groups[rep]].sum()) for rep in keep], dtype=np.float32
    )
    return DedupResult(
        keep=keep,
        weights=new_w,
        n_in=n,
        n_out=len(keep),
        groups=rep_groups,
    )
