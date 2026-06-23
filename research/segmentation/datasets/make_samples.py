"""Construit les échantillons committés (déterministe) depuis les .jsonl bruts gitignorés.

    uv run --with numpy python -m eval.segmentation.datasets.make_samples

- `mabsa.sample.jsonl` : ~40 phrases variées (fr/de/en/es, domaines mêlés, dont des
  phrases multi-aspect ≥2 catégories).
- `wikisection.sample.jsonl` : ~20 docs (en/de), frontières + labels.
Aucune randomisation : sélection par tri + pas régulier → reproductible.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.segmentation.datasets.loaders import MABSA_DIR, WIKISECTION_DIR

HERE = Path(__file__).resolve().parent


def _raw(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def sample_mabsa() -> list[dict]:
    out: list[dict] = []
    for lang in ("fr", "de", "en", "es"):
        rows = _raw(MABSA_DIR / f"{lang}.jsonl")
        multi = [r for r in rows if r["multi_aspect"]]
        single = [r for r in rows if not r["multi_aspect"]]
        # 6 multi-aspect (domaines variés via pas régulier) + 4 single par langue = 40
        out += [multi[i] for i in range(0, len(multi), max(1, len(multi) // 6))][:6]
        out += [single[i] for i in range(0, len(single), max(1, len(single) // 4))][:4]
    return out


def sample_wikisection() -> list[dict]:
    out: list[dict] = []
    for lang in ("en", "de"):
        rows = _raw(WIKISECTION_DIR / f"{lang}.jsonl")
        # 5 disease + 5 city par langue = 20, pas régulier pour varier les articles
        for domain in ("disease", "city"):
            dom = [r for r in rows if r["domain"] == domain]
            out += [dom[i] for i in range(0, len(dom), max(1, len(dom) // 5))][:5]
    return out


def write(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{path.name}: {len(rows)} lignes")


def main() -> None:
    write(sample_mabsa(), HERE / "mabsa.sample.jsonl")
    write(sample_wikisection(), HERE / "wikisection.sample.jsonl")


if __name__ == "__main__":
    main()
