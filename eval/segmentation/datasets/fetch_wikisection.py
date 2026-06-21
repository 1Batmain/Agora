"""Convertit WikiSection (CC-BY-SA-3.0, EN/DE) au format gold du harness (frontières).

    # 1. cloner + décompresser (gitignoré) :
    git clone https://github.com/sebastianarnold/WikiSection data/datasets/WikiSection
    (cd data/datasets/WikiSection && tar xzf wikisection_dataset_json.tar.gz)
    # 2. convertir :
    uv run python -m eval.segmentation.datasets.fetch_wikisection [--langs en de]

Chaque doc brut = `text` + `annotations[]` (begin, length, sectionHeading, sectionLabel).
Les sections WikiSection sont longues (~docs de 14k car) vs nos avis (~300 car). On :
  1. fusionne les annotations consécutives de MÊME `sectionLabel` (une frontière = vrai
     changement de thème, pas une sous-section) ;
  2. tronque chaque segment fusionné à `--max-seg-chars` en coupant à une **frontière de
     phrase** (langue-agnostique : `.!?` ou saut de ligne) ;
  3. fenêtre les segments en docs de `--segs-per-doc` segments (≥2 → ≥1 frontière interne),
     proches en taille de nos avis.

Sortie (gitignorée) : `data/datasets/wikisection/<lang>.jsonl`, format gold (gold_large) :
    {id, lang, domain, split, type:"multi", text, boundaries_char:[...],
     segment_labels:[sectionLabel,...], segment_headings:[...]}
Les frontières tombent AVANT l'espace de jointure (convention `seg_bench.load_gold`).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "data" / "datasets" / "WikiSection"
OUT_DIR = ROOT / "data" / "datasets" / "wikisection"
DOMAINS = ["disease", "city"]
SPLITS = ["train", "validation", "test"]

# fin de phrase : ponctuation .!?… (+ guillemets/parenthèses fermants) suivie d'espace, ou
# saut de ligne. Langue-agnostique (pas de liste d'abréviations — acceptable pour tronquer).
_SENT_END = re.compile(r"(?<=[.!?…])[\"'»)\]]?\s|\n+")


def truncate_to_sentence(text: str, max_chars: int) -> str:
    """Tronque `text` à <= max_chars en finissant sur une frontière de phrase si possible."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = max_chars
    # cherche la dernière fin de phrase avant max_chars
    best = -1
    for m in _SENT_END.finditer(text[:max_chars + 1]):
        best = m.end()
    if best >= max_chars * 0.4:  # garde-fou : pas une troncature ridiculement courte
        cut = best
    return text[:cut].strip()


def merge_same_label(annotations: list[dict]) -> list[dict]:
    """Fusionne les annotations adjacentes de même sectionLabel → segments thématiques."""
    merged: list[dict] = []
    for a in sorted(annotations, key=lambda x: x["begin"]):
        label = a.get("sectionLabel") or a.get("sectionHeading") or "?"
        if merged and merged[-1]["label"] == label:
            merged[-1]["end"] = a["begin"] + a["length"]
        else:
            merged.append({"label": label,
                           "heading": a.get("sectionHeading", ""),
                           "begin": a["begin"],
                           "end": a["begin"] + a["length"]})
    return merged


def doc_to_items(doc: dict, lang: str, domain: str, split: str,
                 max_seg_chars: int, segs_per_doc: int) -> list[dict]:
    text = doc["text"]
    segs = merge_same_label(doc["annotations"])
    # texte de chaque segment thématique, tronqué à taille d'avis
    units = []
    for s in segs:
        raw = text[s["begin"]:s["end"]]
        t = truncate_to_sentence(raw, max_seg_chars)
        if t:
            units.append((t, s["label"], s["heading"]))
    if len(units) < 2:
        return []
    items = []
    base = doc["id"].rsplit("/", 1)[-1]
    for w, i in enumerate(range(0, len(units), segs_per_doc)):
        window = units[i:i + segs_per_doc]
        if len(window) < 2:  # dernière fenêtre orpheline → ignorée (pas de frontière)
            continue
        parts = [u[0] for u in window]
        full = " ".join(parts)
        boundaries, off = [], 0
        for p in parts[:-1]:
            off += len(p)
            boundaries.append(off)   # frontière AVANT l'espace de jointure
            off += 1
        items.append({
            "id": f"wikisection-{lang}-{domain}-{base}-{w}",
            "lang": lang,
            "domain": domain,
            "split": split,
            "type": "multi",
            "text": full,
            "boundaries_char": boundaries,
            "segment_labels": [u[1] for u in window],
            "segment_headings": [u[2] for u in window],
        })
    return items


def convert(langs: list[str], max_seg_chars: int, segs_per_doc: int) -> dict[str, dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict] = {}
    for lang in langs:
        rows, n_boundaries = [], 0
        for domain in DOMAINS:
            for split in SPLITS:
                f = SRC_DIR / f"wikisection_{lang}_{domain}_{split}.json"
                if not f.exists():
                    print(f"  skip (absent): {f.name}")
                    continue
                docs = json.loads(f.read_text(encoding="utf-8"))
                for doc in docs:
                    for it in doc_to_items(doc, lang, domain, split,
                                           max_seg_chars, segs_per_doc):
                        rows.append(it)
                        n_boundaries += len(it["boundaries_char"])
        out = OUT_DIR / f"{lang}.jsonl"
        with open(out, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        avg_len = sum(len(r["text"]) for r in rows) / len(rows) if rows else 0
        stats[lang] = {
            "n_docs": len(rows),
            "n_boundaries": n_boundaries,
            "boundaries_per_doc": round(n_boundaries / len(rows), 2) if rows else 0,
            "avg_text_chars": round(avg_len),
        }
        print(f"{lang}: {len(rows):>6} docs, {n_boundaries} frontières "
              f"(~{stats[lang]['boundaries_per_doc']}/doc), "
              f"texte moyen {stats[lang]['avg_text_chars']} car → {out}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Convertit WikiSection au format gold.")
    ap.add_argument("--langs", nargs="*", default=["en", "de"])
    ap.add_argument("--max-seg-chars", type=int, default=350,
                    help="taille max d'un segment (tronqué à la phrase).")
    ap.add_argument("--segs-per-doc", type=int, default=3,
                    help="segments par doc fenêtré (>=2 → >=1 frontière).")
    args = ap.parse_args()
    if not SRC_DIR.exists():
        raise SystemExit(f"Source absente : {SRC_DIR}. Clone + décompresse d'abord "
                         "(voir docstring).")
    stats = convert(args.langs, args.max_seg_chars, args.segs_per_doc)
    print("\nTOTAL docs:", sum(s["n_docs"] for s in stats.values()))


if __name__ == "__main__":
    main()
