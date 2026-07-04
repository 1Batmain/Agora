#!/usr/bin/env python3
"""Ajoute sexe/âge à une copie du CSV brut d'une consultation — pas de duckdb.

Variante de `characteristics.py` sans aucune dépendance à `consultations.duckdb` :
tout se joue sur le CSV publié dans `data/collect/raw/<slug>/`. Les colonnes
"fermées" candidates (sexe/âge) sont recalculées directement depuis le CSV avec
la même heuristique statistique que le pipeline (`pipeline.collect.classify`,
zéro sémantique), puis :

1. Extraction RÉELLE (`_detect_and_extract`, identique à `characteristics.py`) :
   une colonne fermée n'est retenue comme "sexe" ou "âge" que si le vocabulaire
   réel de ses réponses correspond aux motifs attendus. Aucune valeur devinée :
   une cellule qui ne matche pas reste vide. Les tranches d'âge gardent leur
   libellé source (pas de réalignement vers un référentiel fixe).

2. Complétion SYNTHÉTIQUE du sexe pour toute ligne encore vide après l'étape 1
   (Homme 48% / Femme 48% / Autre 2% / Aucun 2%, seed fixe) — usage test
   uniquement, aucune de ces valeurs n'est une réponse d'un citoyen.

Sortie : une copie du CSV source, deux colonnes `sexe`/`age` ajoutées en fin de
ligne, écrite dans `data/<nom-du-fichier>-brute.csv` (un fichier par CSV brut
trouvé sous `data/collect/raw/`).

Usage : uv run python scripts/characteristics_csv.py
"""
from __future__ import annotations

import csv
import random
import re
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # script lancé hors -m : pipeline/ pas sur sys.path

from pipeline.collect.classify import profile_columns  # noqa: E402

RAW_DIR = REPO_ROOT / "data" / "collect" / "raw"
OUT_DIR = REPO_ROOT / "data"

SYNTHETIC_SEED = 20260703
SYNTHETIC_SEXE_WEIGHTS = [("Homme", 0.48), ("Femme", 0.48), ("Autre", 0.02), ("Aucun", 0.02)]

# Cardinalité au-delà de laquelle une colonne "closed" ne peut plus être un axe
# sexe/âge (ce sont par construction de petits ensembles fermés de catégories).
MAX_AXIS_DISTINCT = 10
# Part (pondérée par le nombre de réponses) des valeurs devant matcher le
# vocabulaire attendu pour retenir la colonne — tolère un peu de bruit source
# sans accepter une colonne hors sujet.
MATCH_SHARE_MIN = 0.90

SEXE_ALIASES = {
    "homme": "Homme", "masculin": "Homme", "h": "Homme", "m": "Homme",
    "femme": "Femme", "feminin": "Femme", "f": "Femme",
    "autre": "Autre", "non binaire": "Autre",
    "aucun": "Aucun", "ne se prononce pas": "Aucun", "non precise": "Aucun",
    "non renseigne": "Aucun", "ne souhaite pas repondre": "Aucun",
    "prefere ne pas repondre": "Aucun",
}

_AGE_PATTERNS = (
    re.compile(r"^moins de\s*\d{1,2}\s*(ans)?$"),
    re.compile(r"^plus de\s*\d{1,2}\s*(ans)?$"),
    re.compile(r"^\d{1,2}\s*(a|à)?\s*\d{1,2}\s*(ans)?$"),          # "18 29", "18 a 29 ans"
    re.compile(r"^\d{1,2}\s*(ans)?\s*(ou|et)\s*plus$"),             # "65 ou plus"
)


def _normalize(value: str) -> str:
    """Casse/accents/ponctuation neutralisés — comparaison de vocabulaire, pas de libellé."""
    s = unicodedata.normalize("NFKD", value)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r"[\-_/]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _is_age_bracket(norm: str) -> bool:
    return any(p.match(norm) for p in _AGE_PATTERNS)


# --- lecture CSV autonome (sniff encodage/délimiteur, pas de dépendance) -----

def _sniff(path: Path) -> tuple[str, str]:
    head = path.open("rb").read(1 << 16)
    try:
        text = head.decode("utf-8")
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        text = head.decode("cp1252", errors="replace")
        encoding = "cp1252"
    first_line = text.splitlines()[0] if text else ""
    delimiter = max((";", ",", "\t"), key=first_line.count)
    return encoding, delimiter


def _read_csv_rows(path: Path) -> tuple[list[str], list[list[str]], str, str]:
    encoding, delimiter = _sniff(path)
    with path.open(encoding=encoding, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader)
        width = len(header)
        rows = [row[:width] + [""] * (width - len(row)) for row in reader]
    return header, rows, delimiter, encoding


def _closed_candidates(header: list[str], rows: list[list[str]]) -> list[int]:
    """Colonnes 'fermées' recalculées depuis le CSV — même heuristique que le
    pipeline (`pipeline.collect.classify`), sans passer par duckdb."""
    stats = profile_columns(header, lambda: iter(rows))
    return [s.question_index for s in stats if s.kind == "closed"]


def _detect_and_extract(rows: list[list[str]],
                        candidates: list[int]) -> tuple[dict[int, str], dict[int, str]]:
    """Retourne (row_num -> sexe, row_num -> age) réels pour ce fichier."""
    sexe_by_row: dict[int, str] = {}
    age_by_row: dict[int, str] = {}
    sexe_col = age_col = None

    for col in candidates:
        counted = sexe_matched = age_matched = 0
        norms: set[str] = set()
        for row in rows:
            raw = row[col].strip() if col < len(row) else ""
            if not raw:
                continue
            counted += 1
            norm = _normalize(raw)
            norms.add(norm)
            if norm in SEXE_ALIASES:
                sexe_matched += 1
            if _is_age_bracket(norm):
                age_matched += 1
        if counted == 0 or len(norms) > MAX_AXIS_DISTINCT:
            continue

        if sexe_col is None and sexe_matched / counted >= MATCH_SHARE_MIN \
                and ({"homme", "femme"} & norms):
            sexe_col = col
        elif age_col is None and age_matched / counted >= MATCH_SHARE_MIN:
            age_col = col

    if sexe_col is not None:
        for row_num, row in enumerate(rows):
            raw = row[sexe_col].strip() if sexe_col < len(row) else ""
            if raw:
                canon = SEXE_ALIASES.get(_normalize(raw))
                if canon:
                    sexe_by_row[row_num] = canon
    if age_col is not None:
        for row_num, row in enumerate(rows):
            raw = row[age_col].strip() if age_col < len(row) else ""
            if raw and _is_age_bracket(_normalize(raw)):
                age_by_row[row_num] = raw  # libellé réel, tel que publié

    return sexe_by_row, age_by_row


def _fill_synthetic_sexe(sexe_by_row: dict[int, str], n_rows: int) -> dict[int, str]:
    """Tire un sexe synthétique pour toute ligne encore vide après l'extraction réelle."""
    rng = random.Random(SYNTHETIC_SEED)
    labels = [w[0] for w in SYNTHETIC_SEXE_WEIGHTS]
    weights = [w[1] for w in SYNTHETIC_SEXE_WEIGHTS]
    filled = dict(sexe_by_row)
    n_synthetic = 0
    for row_num in range(n_rows):
        if row_num not in filled:
            filled[row_num] = rng.choices(labels, weights)[0]
            n_synthetic += 1
    print(f"  complétion synthétique — {n_synthetic} lignes")
    return filled


def _process_file(path: Path) -> None:
    header, rows, delimiter, encoding = _read_csv_rows(path)
    candidates = _closed_candidates(header, rows)
    sexe_by_row, age_by_row = _detect_and_extract(rows, candidates)
    if sexe_by_row or age_by_row:
        detail = []
        if sexe_by_row:
            detail.append(f"sexe: {len(sexe_by_row)} lignes")
        if age_by_row:
            detail.append(f"age: {len(age_by_row)} lignes ({sorted(set(age_by_row.values()))})")
        print(f"  [ ok ] {path.name} : {', '.join(detail)}")
    else:
        print(f"  [none] {path.name} : aucune colonne sexe/âge reconnue")

    sexe_by_row = _fill_synthetic_sexe(sexe_by_row, len(rows))

    out_path = OUT_DIR / f"{path.stem}-brute.csv"
    with out_path.open("w", newline="", encoding=encoding) as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(header + ["sexe", "age"])
        for row_num, row in enumerate(rows):
            writer.writerow(row + [sexe_by_row.get(row_num, ""), age_by_row.get(row_num, "")])
    print(f"  -> {out_path.relative_to(REPO_ROOT)}")


def main() -> None:
    csv_files = sorted(RAW_DIR.glob("*/*.csv"))
    if not csv_files:
        raise SystemExit(f"aucun CSV brut trouvé sous {RAW_DIR}")
    for path in csv_files:
        _process_file(path)


if __name__ == "__main__":
    main()
