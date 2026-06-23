"""Métriques de segmentation, sur une séquence d'UNITÉS-MOTS (masse 1 / unité).

Conventions :
- Une segmentation = un ensemble de frontières internes `b ⊂ {1..n-1}` : `b` contient
  `i` si une coupe tombe ENTRE l'unité `i-1` et l'unité `i`. (0 et n exclus.)
- `seg_ids(n, b)` : étiquette de segment par unité (0,0,1,1,1,2…) — cumul des frontières.

Pk et WindowDiff (↓ = mieux) implémentés à la main (pas de dépendance nltk/segeval ;
définitions standard, Beeferman 1999 / Pevzner & Hearst 2002). `k` = moitié de la
longueur moyenne des segments de RÉFÉRENCE (convention Pk), borné [1, n-1].
"""

from __future__ import annotations

from dataclasses import dataclass


def seg_ids(n: int, boundaries: set[int]) -> list[int]:
    """Étiquette de segment par unité, à partir des frontières internes."""
    ids, cur = [], 0
    for i in range(n):
        if i in boundaries:
            cur += 1
        ids.append(cur)
    return ids


def pk_window(n: int, ref: set[int]) -> int:
    """k de Pk = moitié de la longueur moyenne des segments de référence."""
    n_seg = len(ref) + 1
    avg_len = n / n_seg
    return max(1, min(n - 1, round(avg_len / 2)))


def pk(n: int, ref: set[int], hyp: set[int], k: int | None = None) -> float:
    """Pk : proba qu'une fenêtre de `k` unités soit mal jugée (même seg. ou non)."""
    if n < 2:
        return 0.0
    if k is None:
        k = pk_window(n, ref)
    r = seg_ids(n, ref)
    h = seg_ids(n, hyp)
    err = total = 0
    for i in range(n - k):
        same_ref = r[i] == r[i + k]
        same_hyp = h[i] == h[i + k]
        err += same_ref != same_hyp
        total += 1
    return err / total if total else 0.0


def windowdiff(n: int, ref: set[int], hyp: set[int], k: int | None = None) -> float:
    """WindowDiff : compare le NOMBRE de frontières dans chaque fenêtre glissante."""
    if n < 2:
        return 0.0
    if k is None:
        k = pk_window(n, ref)
    r = seg_ids(n, ref)
    h = seg_ids(n, hyp)
    err = total = 0
    for i in range(n - k):
        rb = r[i + k] - r[i]  # nb de frontières ref dans (i, i+k]
        hb = h[i + k] - h[i]
        err += rb != hb
        total += 1
    return err / total if total else 0.0


@dataclass
class BoundaryCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, o: "BoundaryCounts") -> "BoundaryCounts":
        return BoundaryCounts(self.tp + o.tp, self.fp + o.fp, self.fn + o.fn)

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 1.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def boundary_counts(ref: set[int], hyp: set[int], tol: int = 1) -> BoundaryCounts:
    """TP/FP/FN des frontières avec tolérance ±`tol` unités (appariement glouton).

    Chaque frontière gold s'apparie à AU PLUS une frontière prédite (et vice-versa)
    dans la fenêtre ±tol, en privilégiant l'écart le plus faible.
    """
    ref_sorted = sorted(ref)
    hyp_sorted = sorted(hyp)
    used = [False] * len(hyp_sorted)
    tp = 0
    for rb in ref_sorted:
        best_j, best_d = -1, tol + 1
        for j, hb in enumerate(hyp_sorted):
            if used[j]:
                continue
            d = abs(hb - rb)
            if d <= tol and d < best_d:
                best_j, best_d = j, d
        if best_j >= 0:
            used[best_j] = True
            tp += 1
    fp = used.count(False)
    fn = len(ref_sorted) - tp
    return BoundaryCounts(tp=tp, fp=fp, fn=fn)
