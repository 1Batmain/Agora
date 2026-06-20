"""Chargement des avis (Idea) et résolution de la source d'entrée.

Ordre de résolution (cf. brief) :
  1. `data/processed/ideas.jsonl`           (produit par la lane data)
  2. `pipeline/ingest/fixtures/ideas.sample.jsonl`  (fixture lane data)
  3. `pipeline/cluster/fixtures/ideas.sample.jsonl` (notre fixture de dev)
On ne bloque jamais sur la lane data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Racine repo = …/pipeline/cluster/io.py -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]

INPUT_CANDIDATES = [
    REPO_ROOT / "data" / "processed" / "ideas.jsonl",
    REPO_ROOT / "pipeline" / "ingest" / "fixtures" / "ideas.sample.jsonl",
    REPO_ROOT / "pipeline" / "cluster" / "fixtures" / "ideas.sample.jsonl",
]


@dataclass
class Idea:
    id: str
    text: str
    text_clean: str = ""
    ts: str | None = None
    lang: str = "fr"
    author_hash: str | None = None
    source: str = "unknown"
    weight: float = 1.0
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict, idx: int) -> "Idea":
        known = {
            "id", "text", "text_clean", "ts", "lang",
            "author_hash", "source", "weight",
        }
        text = row.get("text") or row.get("text_clean") or ""
        return cls(
            id=str(row.get("id") or row.get("idea_id") or f"idea-{idx}"),
            text=text,
            text_clean=row.get("text_clean") or text,
            ts=row.get("ts"),
            lang=row.get("lang", "fr"),
            author_hash=row.get("author_hash"),
            source=row.get("source", "unknown"),
            weight=float(row.get("weight", 1.0) or 1.0),
            extra={k: v for k, v in row.items() if k not in known},
        )


def resolve_input(explicit: str | None = None) -> Path:
    """Retourne le 1er chemin d'entrée existant (ou lève si fixture introuvable)."""
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Entrée explicite introuvable : {p}")
        return p
    for cand in INPUT_CANDIDATES:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "Aucune source d'avis trouvée (ni data/processed, ni fixtures)."
    )


def load_ideas(path: str | Path | None = None) -> list[Idea]:
    src = resolve_input(str(path) if path else None)
    ideas: list[Idea] = []
    with open(src, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            idea = Idea.from_row(row, i)
            if idea.text.strip():
                ideas.append(idea)
    return ideas
