"""Trois segmenteurs sémantiques sur fenêtre glissante (unité = MOT).

Tous reçoivent une matrice de vecteurs-mots `U` [n, dim] L2-normalisée (sortie de
`embeddings.embed_word_units`) et renvoient un ensemble de frontières internes
`b ⊂ {1..n-1}` (coupe entre l'unité `i-1` et `i`).

Seuils DÉRIVÉS de la distribution — et CALIBRÉS GLOBALEMENT (pool de tous les avis),
pas par-document : un seuil purement relatif à UN avis ne peut jamais s'abstenir sur
un mono cohérent (il coupe toujours au point « le moins pire »). En calibrant sur la
distribution GLOBALE des similarités intra-avis, un mono d'une cohérence typique ne
déclenche aucune coupe ; seules les ruptures atypiques (vrais virages de thème) le font.
Aucun magic-number absolu : tout sort de `GlobalStats` (μ/σ poolés).

- TextTiling : minima de cos(bloc-gauche, bloc-droite) sous `μ_bloc − c·σ_bloc`.
- Centroïde live : coupe quand `cos(mot, centroïde courant)` < `μ_adj − α·σ_adj`.
- Change-point : `ruptures` PELT/rbf, pénalité balayée (déjà une échelle globale).

`min_seg` borne la longueur mini d'un segment (anti-sur-découpe → faux positifs).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MIN_SEG = 3  # longueur mini d'un segment (mots) — anti micro-segments parasites


def _normalize_rows(m: np.ndarray) -> np.ndarray:
    nr = np.linalg.norm(m, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    return m / nr


def _smooth(U: np.ndarray, win: int) -> np.ndarray:
    """Moyenne glissante centrée (rayon win), re-normalisée. win<=1 → identité."""
    if win <= 1 or U.shape[0] <= 2:
        return U
    n = U.shape[0]
    out = np.empty_like(U)
    half = win // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = U[lo:hi].mean(axis=0)
    return _normalize_rows(out)


def block_sims(U: np.ndarray, W: int) -> np.ndarray:
    """cos(bloc-gauche W, bloc-droite W) à chaque jointure p ∈ [1, n-1] → [n-1]."""
    n = U.shape[0]
    sim = np.ones(max(0, n - 1), dtype=np.float64)
    for p in range(1, n):
        left = U[max(0, p - W):p].mean(axis=0)
        right = U[p:min(n, p + W)].mean(axis=0)
        ln, rn = np.linalg.norm(left), np.linalg.norm(right)
        sim[p - 1] = float(left @ right / (ln * rn)) if ln and rn else 1.0
    return sim


def adjacent_sims(U: np.ndarray, win: int) -> np.ndarray:
    """cos(mot_i, mot_{i-1}) sur U lissé (rayon `win`) → [n-1]."""
    Us = _smooth(U, win)
    n = Us.shape[0]
    return np.array([float(Us[i] @ Us[i - 1]) for i in range(1, n)], dtype=np.float64)


def running_novelties(U: np.ndarray, win: int) -> np.ndarray:
    """cos(mot_i, centroïde des mots 0..i-1), SANS coupe, sur U lissé → [n-1].

    Distribution propre du signal de `centroid_live` (≠ celle des cos adjacents :
    un centroïde moyenné a une norme plus faible et une similarité systématiquement
    plus basse). C'est CELLE-CI qu'il faut calibrer pour que centroid_live s'abstienne
    sur un mono cohérent.
    """
    Us = _smooth(U, win)
    n = Us.shape[0]
    out = np.empty(max(0, n - 1), dtype=np.float64)
    csum = Us[0].copy()
    for i in range(1, n):
        centroid = csum / i
        cn = np.linalg.norm(centroid)
        out[i - 1] = float(Us[i] @ centroid / cn) if cn else 1.0
        csum += Us[i]
    return out


# --------------------------------------------------------------------------- #
# Statistiques GLOBALES (calibration inter-avis) — calculées une fois.
# --------------------------------------------------------------------------- #
@dataclass
class GlobalStats:
    """μ/σ poolés sur TOUS les avis, par fenêtre W (calibration des seuils)."""
    per_W: dict[int, dict[str, float]] = field(default_factory=dict)

    def blk(self, W: int) -> tuple[float, float]:
        d = self.per_W[W]
        return d["blk_mu"], d["blk_sd"]

    def nov(self, W: int) -> tuple[float, float]:
        d = self.per_W[W]
        return d["nov_mu"], d["nov_sd"]


def compute_global_stats(Us: list[np.ndarray], W_grid: list[int]) -> GlobalStats:
    gs = GlobalStats()
    for W in W_grid:
        blk, nov = [], []
        for U in Us:
            if U.shape[0] >= 2:
                blk.append(block_sims(U, W))
                nov.append(running_novelties(U, max(1, W // 2)))
        blk_all = np.concatenate(blk) if blk else np.array([1.0])
        nov_all = np.concatenate(nov) if nov else np.array([1.0])
        gs.per_W[W] = {
            "blk_mu": float(blk_all.mean()), "blk_sd": float(blk_all.std() or 1e-6),
            "nov_mu": float(nov_all.mean()), "nov_sd": float(nov_all.std() or 1e-6),
        }
    return gs


def _enforce_min_seg(scored: list[tuple[int, float]], n: int, min_seg: int) -> set[int]:
    """Sélectionne des frontières par score décroissant en respectant `min_seg`."""
    chosen: list[int] = []
    for pos, _ in sorted(scored, key=lambda x: -x[1]):
        if pos < min_seg or pos > n - min_seg:
            continue
        if all(abs(pos - c) >= min_seg for c in chosen):
            chosen.append(pos)
    return set(chosen)


# --------------------------------------------------------------------------- #
# 1. TextTiling — cosine inter-blocs, seuil calibré globalement
# --------------------------------------------------------------------------- #
def texttiling(U: np.ndarray, *, W: int, c: float, gstats: GlobalStats,
               min_seg: int = MIN_SEG) -> set[int]:
    n = U.shape[0]
    if n < 2 * min_seg:
        return set()
    sim = block_sims(U, W)
    mu, sd = gstats.blk(W)
    cutoff = mu - c * sd  # coupe si les deux blocs sont anormalement DISsemblables
    m = len(sim)
    cand = []
    for i in range(m):
        left_ok = i == 0 or sim[i] <= sim[i - 1]
        right_ok = i == m - 1 or sim[i] <= sim[i + 1]
        if left_ok and right_ok and sim[i] < cutoff:
            cand.append((i + 1, mu - sim[i]))  # profondeur globale comme score de tri
    return _enforce_min_seg(cand, n, min_seg)


# --------------------------------------------------------------------------- #
# 2. Centroïde live — nouveauté vs centroïde du segment courant (seuil global)
# --------------------------------------------------------------------------- #
def centroid_live(U: np.ndarray, *, W: int, alpha: float, gstats: GlobalStats,
                  min_seg: int = MIN_SEG) -> set[int]:
    n = U.shape[0]
    if n < 2 * min_seg:
        return set()
    Us = _smooth(U, max(1, W // 2))
    mu, sd = gstats.nov(W)
    thresh = mu - alpha * sd

    boundaries: set[int] = set()
    seg_start = 0
    csum = Us[0].copy()
    ccount = 1
    for i in range(1, n):
        centroid = csum / ccount
        cn = np.linalg.norm(centroid)
        sim = float(Us[i] @ centroid / cn) if cn else 1.0
        if sim < thresh and (i - seg_start) >= min_seg and (n - i) >= min_seg:
            boundaries.add(i)
            seg_start = i
            csum = Us[i].copy()
            ccount = 1
        else:
            csum += Us[i]
            ccount += 1
    return boundaries


# --------------------------------------------------------------------------- #
# 3. Change-point — ruptures (PELT, coût rbf) sur la séquence d'embeddings
# --------------------------------------------------------------------------- #
def _ruptures_available() -> bool:
    try:
        import ruptures  # noqa: F401
        return True
    except ImportError:
        return False


def changepoint(U: np.ndarray, *, W: int, pen: float, gstats: GlobalStats | None = None,
                min_seg: int = MIN_SEG) -> set[int]:
    import ruptures as rpt

    n = U.shape[0]
    if n < 2 * min_seg:
        return set()
    signal = np.ascontiguousarray(_smooth(U, max(1, W // 2)), dtype=np.float64)
    algo = rpt.Pelt(model="rbf", min_size=min_seg, jump=1).fit(signal)
    bkps = algo.predict(pen=pen)
    return {b for b in bkps if 0 < b < n}


# --------------------------------------------------------------------------- #
# Registre + grille de balayage (segmenteur × W × seuil)
# --------------------------------------------------------------------------- #
SEGMENTERS = {
    "texttiling": texttiling,
    "centroid_live": centroid_live,
    "changepoint": changepoint,
}

# Nom du 3e paramètre (seuil) par méthode + valeurs balayées. Coefficients sans
# dimension (relatifs à μ/σ globaux) ou pénalité ruptures — aucun seuil absolu.
THRESH_NAME = {"texttiling": "c", "centroid_live": "alpha", "changepoint": "pen"}
THRESH_GRID = {
    "texttiling": [0.5, 1.0, 1.5, 2.0],
    "centroid_live": [0.5, 1.0, 1.5, 2.0],
    "changepoint": [1.0, 2.0, 3.0, 5.0, 8.0],
}
W_GRID = [3, 5, 8, 12]


def segment(method: str, U: np.ndarray, W: int, thr: float, gstats: GlobalStats,
            min_seg: int = MIN_SEG) -> set[int]:
    """Dispatch unifié : renvoie les frontières internes pour une config."""
    fn = SEGMENTERS[method]
    key = THRESH_NAME[method]
    return fn(U, W=W, gstats=gstats, min_seg=min_seg, **{key: thr})


def iter_configs(include_changepoint: bool = True):
    """Itère (method, W, thr) sur toute la grille."""
    for method in SEGMENTERS:
        if method == "changepoint" and not include_changepoint:
            continue
        for W in W_GRID:
            for thr in THRESH_GRID[method]:
                yield method, W, thr
