"""Palette catégorielle GÉNÉRATIVE — couleur = cluster_id Leiden.

Aucune liste de couleurs figée : sur des centaines de consultations, le nombre
de communautés varie et peut dépasser n'importe quel jeu codé en dur (collisions
au-delà de la longueur de la liste). On génère donc N couleurs distinctes par
échantillonnage **équiréparti de la teinte HSV** (saturation/valeur fixes pour
rester lisible sur fond sombre), puis conversion en hexadécimal. Déterministe,
sans dépendance lourde.
"""

from __future__ import annotations

import colorsys
from functools import lru_cache

NOISE_COLOR = "#555555"  # cluster_id == -1 (bruit HDBSCAN)

# Saturation / valeur fixes : assez vives pour se détacher d'un fond sombre,
# assez douces pour ne pas éblouir. La teinte seule porte la distinction.
_SAT = 0.62
_VAL = 0.85


def _hsv_hex(hue: float, sat: float = _SAT, val: float = _VAL) -> str:
    r, g, b = colorsys.hsv_to_rgb(hue % 1.0, sat, val)
    return "#{:02x}{:02x}{:02x}".format(round(r * 255), round(g * 255), round(b * 255))


@lru_cache(maxsize=None)
def palette(n: int) -> tuple[str, ...]:
    """`n` couleurs distinctes, teintes équiréparties sur le cercle chromatique."""
    if n <= 0:
        return ()
    return tuple(_hsv_hex(i / n) for i in range(n))


def color_for(cluster_id: int, n: int | None = None) -> str:
    """Couleur stable pour un `cluster_id` (−1 = bruit).

    `n` = nombre total de thèmes (recommandé) : la teinte est alors équirépartie
    sur exactement N couleurs, sans collision. Sans `n`, on dérive une teinte
    déterministe du nombre d'or (suite à faible discrépance) — distincte pour des
    id proches et stable, mais sans garantie d'équirépartition globale.
    """
    if cluster_id < 0:
        return NOISE_COLOR
    if n is not None and n > 0:
        return palette(n)[cluster_id % n]
    # Repli sans N connu : nombre d'or → teintes bien espacées pour id voisins.
    return _hsv_hex(cluster_id * 0.61803398875)
