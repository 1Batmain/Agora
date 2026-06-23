"""Chargement de x-stance AVEC ses labels (vérité terrain).

⚠️ On lit le zip brut `data/raw/xstance-data-v1.0.zip` directement — PAS le
JSONL canonique `data/processed/ideas.jsonl` qui a droppé les labels.

Chaque commentaire x-stance : `{question_id, question, comment, label∈{FAVOR,
AGAINST}, language, topic}`. On filtre FR par défaut, on groupe par question, et
on ne garde que les questions assez fournies et bi-classes (matériau de scoring).
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

# Racine repo = deux niveaux au-dessus de eval/data.py.
REPO_ROOT = Path(__file__).resolve().parents[1]
XSTANCE_ZIP = REPO_ROOT / "data" / "raw" / "xstance-data-v1.0.zip"
COMMENT_FILES = ("train.jsonl", "valid.jsonl", "test.jsonl")

LABELS = ("FAVOR", "AGAINST")


@dataclass
class Question:
    """Un sous-ensemble : tous les commentaires d'une même question politique."""

    question_id: int
    question: str
    lang: str
    comments: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)  # "FAVOR" | "AGAINST" aligné

    @property
    def n(self) -> int:
        return len(self.comments)

    @property
    def n_favor(self) -> int:
        return sum(1 for x in self.labels if x == "FAVOR")

    @property
    def n_against(self) -> int:
        return sum(1 for x in self.labels if x == "AGAINST")

    def label_ids(self) -> list[int]:
        """Labels en entiers 0/1 (pour les métriques sklearn)."""
        return [LABELS.index(x) for x in self.labels]


def _iter_xstance(zip_path: Path, lang: str | None):
    if not zip_path.exists():
        raise FileNotFoundError(
            f"x-stance introuvable : {zip_path}\n"
            "Télécharge-le : uv run python -m pipeline.ingest.download --only xstance"
        )
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        for fname in COMMENT_FILES:
            if fname not in names:
                continue
            with z.open(fname) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if lang is not None and d.get("language") != lang:
                        continue
                    label = d.get("label")
                    comment = (d.get("comment") or "").strip()
                    if label not in LABELS or not comment:
                        continue
                    yield d, comment, label


def load_questions(
    lang: str | None = "fr",
    min_comments: int = 40,
    min_per_class: int = 5,
    zip_path: Path = XSTANCE_ZIP,
) -> list[Question]:
    """Charge x-stance, groupe par question, filtre les questions exploitables.

    Garde les questions avec ≥ `min_comments` commentaires ET au moins
    `min_per_class` exemples dans CHACUNE des deux classes (sinon le clustering
    n'a rien à séparer / les métriques sont dégénérées).
    """
    by_q: dict[int, Question] = {}
    for d, comment, label in _iter_xstance(zip_path, lang):
        qid = int(d["question_id"])
        q = by_q.get(qid)
        if q is None:
            q = Question(
                question_id=qid,
                question=(d.get("question") or "").strip(),
                lang=d.get("language", lang or "?"),
            )
            by_q[qid] = q
        q.comments.append(comment)
        q.labels.append(label)

    kept = [
        q
        for q in by_q.values()
        if q.n >= min_comments
        and q.n_favor >= min_per_class
        and q.n_against >= min_per_class
    ]
    # Tri déterministe (par question_id) — l'échantillonnage seedé se fait en aval.
    kept.sort(key=lambda q: q.question_id)
    return kept
