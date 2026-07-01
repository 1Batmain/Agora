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
            "id", "type", "label", "props", "text", "text_clean", "ts", "lang",
            "author_hash", "source", "weight",
        }
        # Le JSONL canonique de la lane data niche les champs sous `props`
        # (cf. queue/cross-lane.md). Le fixture de dev cluster est plat. On
        # supporte les deux : `props` a la priorité, repli sur le top-level.
        props = row.get("props") or {}

        def get(key, default=None):
            if key in props:
                return props[key]
            return row.get(key, default)

        text = get("text") or get("text_clean") or ""
        return cls(
            id=str(row.get("id") or row.get("idea_id") or f"idea-{idx}"),
            text=text,
            text_clean=get("text_clean") or text,
            ts=get("ts"),
            lang=get("lang", "fr"),
            author_hash=get("author_hash"),
            source=get("source", "unknown"),
            weight=float(get("weight", 1.0) or 1.0),
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
