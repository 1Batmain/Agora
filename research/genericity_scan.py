"""Scan de généricité — repère mécaniquement les hardcodings/magic-numbers
corpus-spécifiques qui casseraient le pipeline sur une AUTRE consultation.

LECTURE SEULE : n'écrit rien, n'importe aucun code de prod. Sert à reproduire /
rafraîchir le rapport `eval/genericity_audit.md` après des modifs.

Usage :
    uv run python -m eval.genericity_scan
    uv run python -m eval.genericity_scan --json   # sortie machine

Le scan est volontairement simple (regex + heuristiques) : il SIGNALE des
candidats, il ne JUGE pas. Le rapport rédigé tranche le contexte (défaut exposé
en knob = OK vs constante enfouie non dérivée = à corriger).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Périmètre : code de prod + éval. On ignore data/, venv, node_modules, et les
# artefacts d'analyse (.md, .json de résultats).
SCAN_DIRS = ["pipeline", "backend", "frontend/src", "eval"]
IGNORE_PARTS = {".venv", "node_modules", "__pycache__", "fixtures", "dist"}
CODE_SUFFIXES = {".py", ".ts", ".tsx"}

# --- Catégories de patterns -------------------------------------------------
# 1. Littéraux de domaine codés en dur (corpus TikTok).
DOMAIN_LITERALS = re.compile(
    r"\b(tiktok|tik[\s_\-]?tok|harc[eè]l|mal-?[eê]tre|t[eé]moignage)\b", re.IGNORECASE
)

# 2. Magic numbers : flottants en [0,1) (seuils cosine) ou petits entiers
#    assignés à des knobs connus.
KNOB_ASSIGN = re.compile(
    r"\b(threshold|resolution(?:_macro|_sub)?|min_sub_size|min_chars|dedup|"
    r"\bk\b|dup_threshold|max_df|min_df|top_k|label_k|n_neighbors|"
    r"min_cluster_size)\s*[:=]\s*([0-9]+\.?[0-9]*)",
    re.IGNORECASE,
)

# 3. Hypothèses de langue.
LANG_ASSUMPTION = re.compile(
    r"(FRENCH_STOPWORDS|STOPWORDS\b|stop_words|french|fran[cç]ais|"
    r"lang\s*==\s*['\"]\w+['\"]|default\s*[:=]\s*['\"]fr['\"]|"
    r"\[a-z[^\]]*[àâäéèêëîïôöùûüç])",
    re.IGNORECASE,
)

# 4. Hypothèses de source/format.
SOURCE_ASSUMPTION = re.compile(
    r"(source\s*[:=]\s*['\"]\w+['\"]|encoding\s*[:=]\s*['\"][\w\-]+['\"]|"
    r"delimiter\s*=\s*['\"].['\"]|_COL\b|cp1252|csv\.reader)",
    re.IGNORECASE,
)

CATEGORIES = [
    ("domain_literal", DOMAIN_LITERALS),
    ("magic_number", KNOB_ASSIGN),
    ("lang_assumption", LANG_ASSUMPTION),
    ("source_assumption", SOURCE_ASSUMPTION),
]


def iter_code_files() -> list[Path]:
    files: list[Path] = []
    for d in SCAN_DIRS:
        base = REPO_ROOT / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.suffix not in CODE_SUFFIXES:
                continue
            if IGNORE_PARTS & set(p.parts):
                continue
            files.append(p)
    return sorted(files)


def scan() -> list[dict]:
    hits: list[dict] = []
    for path in iter_code_files():
        rel = path.relative_to(REPO_ROOT)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for n, line in enumerate(lines, 1):
            for cat, rx in CATEGORIES:
                if rx.search(line):
                    hits.append({
                        "category": cat,
                        "file": str(rel),
                        "line": n,
                        "text": line.strip()[:160],
                    })
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan de généricité (read-only).")
    ap.add_argument("--json", action="store_true", help="sortie JSON")
    args = ap.parse_args()

    hits = scan()
    if args.json:
        print(json.dumps(hits, ensure_ascii=False, indent=2))
        return

    by_cat: dict[str, list[dict]] = {}
    for h in hits:
        by_cat.setdefault(h["category"], []).append(h)

    order = ["domain_literal", "lang_assumption", "source_assumption", "magic_number"]
    for cat in order:
        items = by_cat.get(cat, [])
        print(f"\n=== {cat}  ({len(items)} hits) ===")
        for h in items:
            print(f"  {h['file']}:{h['line']}  {h['text']}")
    print(f"\nTOTAL : {len(hits)} candidats (juger le contexte — cf. genericity_audit.md)")


if __name__ == "__main__":
    main()
