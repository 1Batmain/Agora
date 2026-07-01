"""Génère le fixture committé `fixtures/ideas.sample.jsonl`.

100 % synthétique (thème mobilité urbaine) et anonyme : sûr à versionner.
Passe par le même mapping `to_idea` que le pipeline réel -> schéma identique,
de quoi démarrer les lanes nlp et eval sans données réelles.

Usage : uv run --with langdetect python -m pipeline.ingest.make_fixture [n]
"""
from __future__ import annotations

import json
import sys

from . import config, synthetic
from .build import to_idea


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    n = int(argv[0]) if argv else 60  # ~50 lignes après filtrage du bruit
    config.FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for rec in synthetic.generate(n=n, seed=7):
        idea = to_idea(rec)
        if idea is not None:
            lines.append(json.dumps(idea, ensure_ascii=False))
    config.FIXTURE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines)} lignes écrites dans {config.FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
