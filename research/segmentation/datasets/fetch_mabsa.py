"""Télécharge M-ABSA (HF `Multilingual-NLP/M-ABSA`, Apache-2.0) et le convertit en
multi-label par phrase, une langue par fichier.

    uv run --with datasets --with huggingface_hub \
        python -m eval.segmentation.datasets.fetch_mabsa [--langs fr de en es ...]

Source : 7 domaines (coursera, food, hotel, laptop, phone, restaurant, sight) × 21 langues
× 3 splits (train/dev/test). Chaque ligne brute = `texte####[[terme, catégorie, polarité], ...]`
(triplets en syntaxe littérale Python). On extrait les **catégories d'aspect DISTINCTES**
de la phrase → cible multi-label (langue-agnostique, zéro lexique).

Sortie (gitignorée) : `data/datasets/mabsa/<lang>.jsonl`, une ligne JSON par phrase :
    {id, lang, domain, split, text, aspect_categories:[...], n_aspects, multi_aspect}
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

REPO = "Multilingual-NLP/M-ABSA"
DOMAINS = ["coursera", "food", "hotel", "laptop", "phone", "restaurant", "sight"]
ALL_LANGS = ["ar", "da", "de", "en", "es", "fr", "hi", "hr", "id", "ja", "ko",
             "nl", "pt", "ru", "sk", "sv", "sw", "th", "tr", "vi", "zh"]
SPLITS = {"train": "train", "dev": "dev", "test": "test"}

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "data" / "datasets" / "mabsa"


def parse_line(line: str) -> tuple[str, list[str]] | None:
    """`texte####[[term, cat, pol], ...]` → (texte nettoyé, [catégories distinctes])."""
    line = line.rstrip("\n")
    if "####" not in line:
        return None
    text, _, triplets_raw = line.partition("####")
    text = text.replace("‎", "").replace("‏", "").strip()
    if not text:
        return None
    try:
        triplets = ast.literal_eval(triplets_raw.strip() or "[]")
    except (ValueError, SyntaxError):
        return None
    cats: list[str] = []
    for t in triplets:
        if len(t) >= 2 and t[1]:
            c = str(t[1]).strip()
            if c and c not in cats:
                cats.append(c)
    return text, cats


def fetch(langs: list[str]) -> dict[str, dict]:
    from huggingface_hub import hf_hub_download

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict] = {}
    for lang in langs:
        rows = []
        n_multi = 0
        cat_set: set[str] = set()
        for domain in DOMAINS:
            for split, fname in SPLITS.items():
                path = f"{domain}/{lang}/{fname}.txt"
                try:
                    local = hf_hub_download(REPO, path, repo_type="dataset")
                except Exception as e:  # fichier absent pour ce (domaine,langue)
                    print(f"  skip {path}: {type(e).__name__}")
                    continue
                with open(local, encoding="utf-8") as fh:
                    for i, line in enumerate(fh):
                        parsed = parse_line(line)
                        if parsed is None:
                            continue
                        text, cats = parsed
                        multi = len(cats) >= 2
                        n_multi += int(multi)
                        cat_set.update(cats)
                        rows.append({
                            "id": f"mabsa-{lang}-{domain}-{split}-{i}",
                            "lang": lang,
                            "domain": domain,
                            "split": split,
                            "text": text,
                            "aspect_categories": cats,
                            "n_aspects": len(cats),
                            "multi_aspect": multi,
                        })
        out = OUT_DIR / f"{lang}.jsonl"
        with open(out, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[lang] = {
            "n": len(rows),
            "n_multi_aspect": n_multi,
            "pct_multi_aspect": round(100 * n_multi / len(rows), 1) if rows else 0.0,
            "n_distinct_categories": len(cat_set),
        }
        print(f"{lang}: {len(rows):>6} phrases, "
              f"{stats[lang]['pct_multi_aspect']}% multi-aspect, "
              f"{len(cat_set)} catégories → {out}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Télécharge + convertit M-ABSA.")
    ap.add_argument("--langs", nargs="*", default=ALL_LANGS,
                    help="codes langue (défaut : les 21).")
    args = ap.parse_args()
    stats = fetch(args.langs)
    print("\nTOTAL phrases:", sum(s["n"] for s in stats.values()))


if __name__ == "__main__":
    main()
