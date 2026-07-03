"""Classification GÉNÉRIQUE des colonnes : stats en streaming, zéro sémantique.

Aucun libellé de question n'est interprété — seuls comptent les indicateurs
statistiques (longueur moyenne, diversité, motifs date/numérique). Les seuils
vivent dans `config.py` ; la table `questions` expose toutes les stats pour
rendre les erreurs de classification visibles en SQL.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterator

from . import config

_DATE_PATTERNS = (
    re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$"),
    re.compile(r"^\d{2}/\d{2}/\d{4}( \d{2}:\d{2}(:\d{2})?)?$"),
)
_NUMERIC = re.compile(r"^-?\d+([.,]\d+)?$")


@dataclass
class QuestionStats:
    question_index: int
    question: str
    n_answers: int = 0
    n_distinct: int = 0
    distinct_ratio: float | None = None
    avg_len: float | None = None
    max_len: int = 0
    kind: str = "empty"


class _Accumulator:
    __slots__ = ("distinct", "sum_len", "max_len", "n", "n_date", "n_numeric")

    def __init__(self) -> None:
        self.distinct: set[str] | None = set()
        self.sum_len = 0
        self.max_len = 0
        self.n = 0
        self.n_date = 0
        self.n_numeric = 0

    def update(self, value: str) -> None:
        self.n += 1
        self.sum_len += len(value)
        self.max_len = max(self.max_len, len(value))
        if self.distinct is not None:
            self.distinct.add(value)
            if len(self.distinct) >= config.DISTINCT_CAP:
                # Cap mémoire : au-delà, la diversité est déjà saturée.
                self.distinct = None
        if any(p.match(value) for p in _DATE_PATTERNS):
            self.n_date += 1
        if _NUMERIC.match(value):
            self.n_numeric += 1

    @property
    def n_distinct(self) -> int:
        return config.DISTINCT_CAP if self.distinct is None else len(self.distinct)


def _kind(acc: _Accumulator) -> str:
    if acc.n == 0:
        return "empty"
    if acc.n_date / acc.n >= config.DATE_SHARE_MIN:
        return "date"
    if acc.n_numeric / acc.n >= config.NUMERIC_SHARE_MIN:
        return "numeric"
    avg_len = acc.sum_len / acc.n
    distinct_ratio = acc.n_distinct / acc.n
    if acc.n >= config.OPEN_MIN_ANSWERS and (
        (avg_len >= config.OPEN_AVG_LEN_STRONG
         and distinct_ratio >= config.OPEN_DISTINCT_RATIO_FLOOR)
        or (avg_len >= config.OPEN_AVG_LEN_WEAK
            and distinct_ratio >= config.OPEN_DISTINCT_RATIO_MIN)
    ):
        return "open_text"
    return "closed"


def profile_columns(header: list[str],
                    rows: Callable[[], Iterator[list]]) -> list[QuestionStats]:
    """Passe 1 : profile chaque colonne (cellules vides/blanches = absentes)."""
    accs = [_Accumulator() for _ in header]
    for row in rows():
        for i, value in enumerate(row[: len(header)]):
            if value is None:
                continue
            value = value.strip()
            if value:
                accs[i].update(value)
    out = []
    for i, (question, acc) in enumerate(zip(header, accs)):
        out.append(QuestionStats(
            question_index=i,
            question=question,
            n_answers=acc.n,
            n_distinct=acc.n_distinct,
            distinct_ratio=(acc.n_distinct / acc.n) if acc.n else None,
            avg_len=(acc.sum_len / acc.n) if acc.n else None,
            max_len=acc.max_len,
            kind=_kind(acc),
        ))
    return out
