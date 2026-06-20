"""Construit le JSONL canonique `data/processed/ideas.jsonl`.

Pipeline : sources brutes -> nettoyage (T-D2) -> anonymisation + langue (T-D4)
-> objet Idea canonique (cf. queue/cross-lane.md). Régénère tout from scratch et
imprime des compteurs (total, par source, par langue, % vides retirés).

Si aucune source réelle n'est présente dans `data/raw/`, on tente un téléchargement
(T-D1) puis, en dernier recours, on génère un échantillon synthétique FR pour que
la lane nlp puisse démarrer.

Usage :
    uv run --with langdetect python -m pipeline.ingest.build
    uv run --with langdetect python -m pipeline.ingest.build --max-per-source 5000
    uv run python -m pipeline.ingest.build --synthetic   # force le synthétique
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

from . import config, download, lang, synthetic
from .anonymize import author_hash
from .normalize import clean_text, is_empty_like, make_label

csv.field_size_limit(10_000_000)  # certains témoignages libres sont longs


# ---------------------------------------------------------------------------
# Lecture des sources brutes -> enregistrements {raw_id, text, author, source, ts}
# ---------------------------------------------------------------------------
def read_xstance() -> Iterator[dict]:
    if not config.XSTANCE_ZIP.exists():
        return
    with zipfile.ZipFile(config.XSTANCE_ZIP) as z:
        names = set(z.namelist())
        for fname in config.XSTANCE_COMMENT_FILES:
            if fname not in names:
                continue
            with z.open(fname) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if config.XSTANCE_FR_ONLY and d.get("language") != "fr":
                        continue
                    yield {
                        "raw_id": str(d.get("id", "")),
                        "text": d.get("comment", "") or "",
                        "author": str(d.get("author", "")),
                        "source": "xstance",
                        "ts": "",  # x-stance ne fournit pas d'horodatage
                    }


def read_tiktok() -> Iterator[dict]:
    if not config.TIKTOK_CSV.exists():
        return
    with open(config.TIKTOK_CSV, encoding=config.TIKTOK_ENCODING, newline="") as f:
        rd = csv.reader(f, delimiter=";")
        try:
            next(rd)  # en-tête
        except StopIteration:
            return
        ncols = max(config.TIKTOK_ID_COL, config.TIKTOK_TS_COL, config.TIKTOK_TEXT_COL)
        for row in rd:
            if len(row) <= ncols:
                continue
            text = row[config.TIKTOK_TEXT_COL]
            if not text.strip():  # on ne garde que les réponses à la question ouverte
                continue
            yield {
                "raw_id": row[config.TIKTOK_ID_COL],
                "text": text,
                "author": row[config.TIKTOK_ID_COL],  # 1 réponse = 1 répondant
                "source": "tiktok",
                "ts": row[config.TIKTOK_TS_COL],
            }


def read_synthetic(n: int = 300) -> Iterator[dict]:
    yield from synthetic.generate(n=n)


# ---------------------------------------------------------------------------
# Transformation -> Idea canonique
# ---------------------------------------------------------------------------
def to_idea(rec: dict) -> dict | None:
    """Mappe un enregistrement brut vers un nœud Idea, ou None si vide/quasi-vide."""
    raw_text = rec.get("text", "") or ""
    text_clean = clean_text(raw_text)
    if is_empty_like(text_clean):
        return None
    source = rec["source"]
    idea_id = f"{source}:{rec['raw_id']}"
    return {
        "id": idea_id,
        "type": "idea",
        "label": make_label(text_clean, config.LABEL_MAXLEN),
        "props": {
            "text": raw_text.strip(),
            "text_clean": text_clean,
            "ts": (rec.get("ts") or "").strip(),
            "lang": lang.detect_lang(text_clean),
            "author_hash": author_hash(rec.get("author", ""), source),
            "source": source,
            "weight": 1.0,
        },
    }


def assert_no_pii(idea: dict) -> None:
    """Garde-fou : author_hash opaque (hex, longueur fixe), pas l'auteur d'origine."""
    h = idea["props"]["author_hash"]
    assert len(h) == config.AUTHOR_HASH_LEN and all(c in "0123456789abcdef" for c in h), (
        f"author_hash non conforme: {h!r}"
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def iter_sources(use_synthetic: bool, max_per_source: int | None) -> Iterator[dict]:
    readers: list[tuple[str, Iterable[dict]]]
    if use_synthetic:
        readers = [("synthetic", read_synthetic())]
    else:
        readers = [("xstance", read_xstance()), ("tiktok", read_tiktok())]
    for name, it in readers:
        count = 0
        for rec in it:
            yield rec
            count += 1
            if max_per_source and count >= max_per_source:
                print(f"  [cap ] {name}: limité à {max_per_source}")
                break


def build(out: Path, use_synthetic: bool, max_per_source: int | None) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    by_source = Counter()
    by_lang = Counter()
    seen_raw = 0
    kept = 0

    tmp = out.with_suffix(out.suffix + ".part")
    with open(tmp, "w", encoding="utf-8") as fout:
        for rec in iter_sources(use_synthetic, max_per_source):
            seen_raw += 1
            idea = to_idea(rec)
            if idea is None:
                continue
            assert_no_pii(idea)
            by_source[idea["props"]["source"]] += 1
            by_lang[idea["props"]["lang"]] += 1
            fout.write(json.dumps(idea, ensure_ascii=False) + "\n")
            kept += 1
    tmp.replace(out)

    return {
        "seen_raw": seen_raw,
        "kept": kept,
        "dropped_empty": seen_raw - kept,
        "by_source": dict(by_source),
        "by_lang": dict(by_lang),
    }


def print_report(out: Path, stats: dict, used_synthetic: bool) -> None:
    pct_empty = (100 * stats["dropped_empty"] / stats["seen_raw"]) if stats["seen_raw"] else 0
    print("\n" + "=" * 56)
    print(f"  ideas.jsonl régénéré : {out}")
    if used_synthetic:
        print("  ⚠ SOURCE = échantillon SYNTHÉTIQUE (fallback, thème mobilité urbaine)")
    print("-" * 56)
    print(f"  total avis (lignes)     : {stats['kept']}")
    print(f"  lus en entrée (bruts)   : {stats['seen_raw']}")
    print(f"  vides/quasi-vides retirés: {stats['dropped_empty']} ({pct_empty:.1f} %)")
    print(f"  par source              : {stats['by_source']}")
    print(f"  par langue              : {stats['by_lang']}")
    print(f"  détection langue        : {'langdetect' if lang.has_langdetect() else 'heuristique (repli)'}")
    print("=" * 56)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Construit data/processed/ideas.jsonl")
    ap.add_argument("--synthetic", action="store_true",
                    help="forcer l'échantillon synthétique (ignore data/raw)")
    ap.add_argument("--max-per-source", type=int, default=None,
                    help="plafonner le nb d'avis par source (box sobre)")
    ap.add_argument("--out", type=Path, default=config.IDEAS_JSONL)
    args = ap.parse_args(argv)

    use_synthetic = args.synthetic
    if not use_synthetic:
        have = config.XSTANCE_ZIP.exists() or config.TIKTOK_CSV.exists()
        if not have:
            print("Aucune source dans data/raw/ — tentative de téléchargement…")
            download.main([])
            have = config.XSTANCE_ZIP.exists() or config.TIKTOK_CSV.exists()
        if not have:
            print("Téléchargement indisponible — repli sur l'échantillon synthétique.",
                  file=sys.stderr)
            use_synthetic = True

    stats = build(args.out, use_synthetic, args.max_per_source)
    print_report(args.out, stats, use_synthetic)

    if stats["kept"] == 0:
        print("ERREUR : aucun avis produit.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
