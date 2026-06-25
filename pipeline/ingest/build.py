"""Construit le JSONL canonique `data/processed/ideas.jsonl`.

Pipeline : sources brutes -> nettoyage (T-D2) -> anonymisation + langue (T-D4)
-> objet Idea canonique (cf. queue/cross-lane.md). Régénère tout from scratch et
imprime des compteurs (total, par source, par langue, % vides retirés).

Les sources sont **déclaratives** : chaque consultation est décrite par un
descripteur JSON (`descriptors/*.json`) lu par un seul `read_generic`
(`sources.py`). Ajouter une consultation = ajouter un descripteur, **pas de
code** (audit #1). Voir `pipeline/ingest/README.md`.

Si aucune source réelle n'est présente dans `data/raw/`, on tente un
téléchargement (T-D1, via les `url` des descripteurs) puis, en dernier recours,
on génère un échantillon synthétique FR pour que la lane nlp puisse démarrer.

Usage :
    uv run --with langdetect python -m pipeline.ingest.build
    uv run --with langdetect python -m pipeline.ingest.build --max-per-source 5000
    uv run python -m pipeline.ingest.build --descriptor path/to/desc.json --out x.jsonl
    uv run python -m pipeline.ingest.build --synthetic   # force le synthétique
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

from . import config, download, lang, synthetic
from .anonymize import author_hash
from .normalize import clean_text, is_empty_like, make_label, strip_pii
from .sources import SourceDescriptor, load_descriptors, read_generic


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
    # Langue fournie par la source si présente, sinon (re)détection. Défaut 'und'
    # (jamais 'fr') pour ne pas mal-étiqueter un corpus importé sans langue (#13).
    src_lang = (rec.get("lang") or "").strip()
    lang_code = src_lang if src_lang else lang.detect_lang(text_clean)
    # Poids social : fourni par la source si mappé, sinon 1.0.
    try:
        weight = float(rec.get("weight", 1.0))
    except (TypeError, ValueError):
        weight = 1.0
    return {
        "id": idea_id,
        "type": "idea",
        "label": make_label(text_clean, config.LABEL_MAXLEN),
        "props": {
            # `text` = original PRESQUE intact (casse/ponctuation conservées pour le
            # toggle « voir l'original ») MAIS PII évidentes masquées : on ne PERSISTE
            # jamais d'email/tél./URL/@mention en clair (SEC3). `text_clean` masque les
            # mêmes PII puis normalise davantage (espaces/typographie) — c'est lui qui
            # sert de texte canonique et d'ancrage des spans des claims.
            "text": strip_pii(raw_text.strip()),
            "text_clean": text_clean,
            "ts": (rec.get("ts") or "").strip(),
            "lang": lang_code,
            "author_hash": author_hash(rec.get("author", ""), source),
            "source": source,
            "weight": weight,
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
def iter_sources(
    readers: list[tuple[str, Iterable[dict]]], max_per_source: int | None
) -> Iterator[dict]:
    for name, it in readers:
        count = 0
        for rec in it:
            yield rec
            count += 1
            if max_per_source and count >= max_per_source:
                print(f"  [cap ] {name}: limité à {max_per_source}")
                break


def build_readers(
    descriptors: list[SourceDescriptor], use_synthetic: bool
) -> list[tuple[str, Iterable[dict]]]:
    if use_synthetic:
        return [("synthetic", read_synthetic())]
    return [(d.name, read_generic(d)) for d in descriptors]


def build(
    out: Path,
    readers: list[tuple[str, Iterable[dict]]],
    max_per_source: int | None,
) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    by_source = Counter()
    by_lang = Counter()
    seen_raw = 0
    kept = 0

    tmp = out.with_suffix(out.suffix + ".part")
    with open(tmp, "w", encoding="utf-8") as fout:
        for rec in iter_sources(readers, max_per_source):
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
                    help="forcer l'échantillon synthétique (ignore les descripteurs)")
    ap.add_argument("--descriptor", type=Path, action="append", default=None,
                    help="descripteur(s) JSON explicite(s) (répétable). "
                         "Par défaut : tous ceux de descriptors/.")
    ap.add_argument("--max-per-source", type=int, default=None,
                    help="plafonner le nb d'avis par source (box sobre)")
    ap.add_argument("--out", type=Path, default=config.IDEAS_JSONL)
    args = ap.parse_args(argv)

    use_synthetic = args.synthetic
    descriptors: list[SourceDescriptor] = []
    if not use_synthetic:
        descriptors = load_descriptors(args.descriptor)
        have = any(d.resolved_path().exists() for d in descriptors)
        if not have and args.descriptor is None:
            print("Aucune source dans data/raw/ — tentative de téléchargement…")
            download.main([])
            have = any(d.resolved_path().exists() for d in descriptors)
        if not have:
            print("Source(s) indisponible(s) — repli sur l'échantillon synthétique.",
                  file=sys.stderr)
            use_synthetic = True

    readers = build_readers(descriptors, use_synthetic)
    stats = build(args.out, readers, args.max_per_source)
    print_report(args.out, stats, use_synthetic)

    if stats["kept"] == 0:
        print("ERREUR : aucun avis produit.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
