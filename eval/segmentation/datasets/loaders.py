"""Chargement des jeux réels convertis (M-ABSA, WikiSection) pour le segmenteur appris.

    uv run python -m eval.segmentation.datasets.loaders        # imprime les stats

- `load_mabsa(lang)`  → list[MabsaItem]  (signal A : multi-label aspect par phrase courte)
- `load_wikisection(lang)` → list[GoldItem]  (signal B : frontières de section, format gold)

Les `.jsonl` bruts vivent sous `data/datasets/` (gitignoré) ; régénère-les avec
`fetch_mabsa.py` / `fetch_wikisection.py`. `load_wikisection` renvoie directement des
`GoldItem` (cf. `seg_bench`) → réutilisables tels quels par le banc / l'entraînement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from eval.segmentation.seg_bench import GoldItem

ROOT = Path(__file__).resolve().parents[3]
MABSA_DIR = ROOT / "data" / "datasets" / "mabsa"
WIKISECTION_DIR = ROOT / "data" / "datasets" / "wikisection"

MABSA_LANGS = ["ar", "da", "de", "en", "es", "fr", "hi", "hr", "id", "ja", "ko",
               "nl", "pt", "ru", "sk", "sv", "sw", "th", "tr", "vi", "zh"]
WIKISECTION_LANGS = ["en", "de"]


@dataclass
class MabsaItem:
    """Phrase d'opinion courte + ses catégories d'aspect DISTINCTES (cible multi-label)."""
    id: str
    lang: str
    text: str
    aspect_categories: list[str]
    multi_aspect: bool                       # >=2 catégories distinctes
    domain: str = ""
    split: str = ""
    extra: dict = field(default_factory=dict)


def _read_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} absent — régénère avec le fetch correspondant "
            f"(voir eval/segmentation/datasets/ACQUIS_NOTE.md).")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_mabsa(lang: str, split: str | None = None) -> list[MabsaItem]:
    """Charge M-ABSA pour une langue. `split` ∈ {train,dev,test} pour filtrer (None = tout)."""
    items = []
    for r in _read_jsonl(MABSA_DIR / f"{lang}.jsonl"):
        if split is not None and r.get("split") != split:
            continue
        items.append(MabsaItem(
            id=r["id"], lang=r["lang"], text=r["text"],
            aspect_categories=r["aspect_categories"],
            multi_aspect=r["multi_aspect"],
            domain=r.get("domain", ""), split=r.get("split", ""),
        ))
    return items


def load_wikisection(lang: str, split: str | None = None) -> list[GoldItem]:
    """Charge WikiSection pour une langue → GoldItem (type multi, frontières + labels)."""
    items = []
    for r in _read_jsonl(WIKISECTION_DIR / f"{lang}.jsonl"):
        if split is not None and r.get("split") != split:
            continue
        items.append(GoldItem(
            id=r["id"], type="multi", text=r["text"],
            boundaries_char=list(r["boundaries_char"]),
            seg_themes=r.get("segment_labels", []),
        ))
    return items


def _mabsa_stats(lang: str) -> dict | None:
    try:
        items = load_mabsa(lang)
    except FileNotFoundError:
        return None
    n = len(items)
    n_multi = sum(1 for it in items if it.multi_aspect)
    cats = {c for it in items for c in it.aspect_categories}
    return {"n": n, "pct_multi": round(100 * n_multi / n, 1) if n else 0.0,
            "n_categories": len(cats)}


def _wikisection_stats(lang: str) -> dict | None:
    try:
        items = load_wikisection(lang)
    except FileNotFoundError:
        return None
    n = len(items)
    nb = sum(len(it.boundaries_char) for it in items)
    return {"n_docs": n, "n_boundaries": nb,
            "boundaries_per_doc": round(nb / n, 2) if n else 0.0}


def main() -> None:
    print("=== M-ABSA (signal A : multi-aspect par phrase) ===")
    for lang in ("fr", "de", "it", "en", "es"):
        s = _mabsa_stats(lang)
        if s is None:
            print(f"  {lang}: ABSENT (M-ABSA ne couvre pas cette langue)"
                  if lang == "it" else f"  {lang}: ABSENT (lancer fetch_mabsa)")
        else:
            print(f"  {lang}: {s['n']:>6} phrases, {s['pct_multi']}% multi-aspect, "
                  f"{s['n_categories']} catégories d'aspect")

    print("\n=== WikiSection (signal B : frontières de section) ===")
    for lang in ("en", "de"):
        s = _wikisection_stats(lang)
        if s is None:
            print(f"  {lang}: ABSENT (lancer fetch_wikisection)")
        else:
            print(f"  {lang}: {s['n_docs']:>6} docs, {s['n_boundaries']} frontières "
                  f"(~{s['boundaries_per_doc']}/doc)")


if __name__ == "__main__":
    main()
