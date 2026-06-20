"""Échantillon TRILINGUE équilibré de x-stance pour le banc qualité.

x-stance est multilingue (DE/FR/IT) et chaque commentaire porte un **topic**
(12 thèmes politiques) — la même question existe dans les 3 langues. C'est le
terrain idéal du test multilingue : un bon modèle regroupe par **topic** (thème),
pas par **langue**.

On construit un corpus **équilibré par (topic × langue)** : pour les `n_topics`
thèmes les plus fournis, on tire le même nombre de commentaires dans CHAQUE
langue. Résultat : langues parfaitement équilibrées (entropie max → NMI(cluster,
langue) interprétable) et thèmes équilibrés (NMI(cluster, topic) interprétable).

On lit le zip brut `data/raw/xstance-data-v1.0.zip` (mêmes labels que eval.data),
mais ici on garde `language` et `topic`, pas FAVOR/AGAINST.
"""

from __future__ import annotations

import json
import random
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
XSTANCE_ZIP = REPO_ROOT / "data" / "raw" / "xstance-data-v1.0.zip"
COMMENT_FILES = ("train.jsonl", "valid.jsonl", "test.jsonl")
LANGS = ("de", "fr", "it")


@dataclass
class MultiCorpus:
    """Corpus trilingue équilibré, prêt pour embed → cluster → métriques."""

    texts: list[str]
    langs: list[str]   # "de" | "fr" | "it" aligné sur texts
    topics: list[str]  # thème (vérité terrain) aligné sur texts

    @property
    def n(self) -> int:
        return len(self.texts)

    @property
    def lang_counts(self) -> dict[str, int]:
        return dict(Counter(self.langs))

    @property
    def topic_counts(self) -> dict[str, int]:
        return dict(Counter(self.topics))

    def topic_ids(self) -> list[int]:
        order = sorted(set(self.topics))
        idx = {t: i for i, t in enumerate(order)}
        return [idx[t] for t in self.topics]

    def lang_ids(self) -> list[int]:
        idx = {lg: i for i, lg in enumerate(sorted(set(self.langs)))}
        return [idx[lg] for lg in self.langs]


def _iter_rows(zip_path: Path):
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
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def load_balanced(
    n_topics: int = 6,
    per_cell: int | None = None,
    max_per_cell: int = 130,
    min_chars: int = 15,
    seed: int = 42,
    langs: tuple[str, ...] = LANGS,
    zip_path: Path = XSTANCE_ZIP,
) -> MultiCorpus:
    """Construit un corpus équilibré (topic × langue).

    - `n_topics` : nb de thèmes (les plus fournis, garantis dans toutes les langues).
    - `per_cell` : commentaires par (topic, langue). Si None : min(`max_per_cell`,
      plus petite cellule disponible) → corpus parfaitement équilibré.
    - `min_chars` : filtre les commentaires trop courts (bruit).
    """
    # 1) Indexe (topic, langue) -> commentaires (dédupliqués, filtrés).
    cells: dict[tuple[str, str], list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    topic_total: Counter[str] = Counter()
    for d in _iter_rows(zip_path):
        lg = d.get("language")
        if lg not in langs:
            continue
        topic = d.get("topic")
        comment = (d.get("comment") or "").strip()
        if not topic or len(comment) < min_chars:
            continue
        key = (lg, comment)
        if key in seen:  # dédup exact (les gens recopient)
            continue
        seen.add(key)
        cells[(topic, lg)].append(comment)
        topic_total[topic] += 1

    # 2) Choisit les n_topics thèmes les plus fournis présents dans TOUTES les langues.
    eligible = [
        t for t, _ in topic_total.most_common()
        if all((t, lg) in cells for lg in langs)
    ]
    chosen = eligible[:n_topics]
    if not chosen:
        raise SystemExit("Aucun topic présent dans toutes les langues demandées.")

    # 3) per_cell = plus petite cellule (parmi topics×langues retenus), plafonné.
    smallest = min(len(cells[(t, lg)]) for t in chosen for lg in langs)
    cell = min(max_per_cell, smallest) if per_cell is None else per_cell

    # 4) Échantillonnage seedé, équilibré.
    rng = random.Random(seed)
    texts: list[str] = []
    out_langs: list[str] = []
    out_topics: list[str] = []
    for t in chosen:
        for lg in langs:
            pool = list(cells[(t, lg)])
            rng.shuffle(pool)
            take = pool[: min(cell, len(pool))]
            texts.extend(take)
            out_langs.extend([lg] * len(take))
            out_topics.extend([t] * len(take))

    # Mélange global (évite tout ordre topic/langue résiduel).
    order = list(range(len(texts)))
    rng.shuffle(order)
    return MultiCorpus(
        texts=[texts[i] for i in order],
        langs=[out_langs[i] for i in order],
        topics=[out_topics[i] for i in order],
    )
