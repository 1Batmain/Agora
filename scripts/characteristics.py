#!/usr/bin/env python3
"""Renseigne sexe/âge dans `consultations.duckdb` — script autonome (colonnes
`sexe`/`age` de la table `responses`).

Ne dépend d'aucun module `pipeline/collect`. Deux passes, dans cet ordre :

1. Extraction RÉELLE (`_extract_real`) : relit les CSV bruts déjà téléchargés
   dans `data/collect/raw/` et reconnaît une colonne "closed" (déjà cataloguée
   par le pipeline principal dans `questions`) comme "sexe" ou "âge" seulement
   si le vocabulaire réel de ses réponses correspond aux motifs attendus.
   Aucune valeur n'est devinée : une cellule qui ne matche pas reste NULL.
   Les tranches d'âge gardent leur libellé source (le portail utilise des
   bornes différentes selon les consultations, ex. 18-29/30-49/50-65/65+ —
   pas de réalignement forcé vers un référentiel fixe, ce serait fabriquer).

2. Complétion SYNTHÉTIQUE (`_fill_synthetic_sexe`) : pour tout répondant
   encore sans sexe après l'étape 1, tire une valeur au hasard (Homme 48% /
   Femme 48% / Autre 2% / Aucun 2%, seed fixe) — usage test uniquement,
   aucune de ces valeurs n'est une réponse d'un citoyen.

Les deux passes écrivent directement sur `responses` (sexe/age dupliqués sur
toutes les lignes question/réponse d'un même répondant, identifié par
(consultation_slug, source_file, row_num)) plutôt que dans une table à part :
un simple filtre SQL sur `responses`/`contributions` suffit, pas de jointure.

Usage : uv run python scripts/characteristics.py
"""
from __future__ import annotations

import csv
import random
import re
import unicodedata
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "data" / "collect" / "consultations.duckdb"
RAW_DIR = REPO_ROOT / "data" / "collect" / "raw"

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


def _read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    encoding, delimiter = _sniff(path)
    with path.open(encoding=encoding, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader)
        width = len(header)
        rows = [row[:width] + [""] * (width - len(row)) for row in reader]
    return header, rows


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


def _extract_real(con: duckdb.DuckDBPyConnection) -> None:
    files = con.execute("""
        SELECT f.consultation_slug, f.filename
        FROM files f
        WHERE f.status = 'ok' AND f.format = 'csv'
    """).fetchall()

    n_written = 0
    for slug, filename in files:
        path = RAW_DIR / slug / filename
        if not path.exists():
            print(f"  [skip] {slug}/{filename} : fichier brut absent en local")
            continue

        candidates = con.execute("""
            SELECT question_index FROM questions
            WHERE consultation_slug = ? AND source_file = ? AND kind = 'closed'
            ORDER BY question_index
        """, [slug, filename]).fetchall()
        candidates = [c[0] for c in candidates]
        if not candidates:
            continue

        _, rows = _read_csv_rows(path)
        sexe_by_row, age_by_row = _detect_and_extract(rows, candidates)
        if not sexe_by_row and not age_by_row:
            print(f"  [none] {slug}/{filename} : aucune colonne sexe/âge reconnue")
            continue

        row_nums = sorted(set(sexe_by_row) | set(age_by_row))
        con.executemany(
            "UPDATE responses SET sexe = ?, age = ? "
            "WHERE consultation_slug = ? AND source_file = ? AND row_num = ?",
            [(sexe_by_row.get(rn), age_by_row.get(rn), slug, filename, rn) for rn in row_nums],
        )
        n_written += len(row_nums)
        detail = []
        if sexe_by_row:
            detail.append(f"sexe: {len(sexe_by_row)} lignes")
        if age_by_row:
            detail.append(f"age: {len(age_by_row)} lignes ({sorted(set(age_by_row.values()))})")
        print(f"  [ ok ] {slug}/{filename} : {', '.join(detail)}")

    print(f"extraction réelle — {n_written} répondants mis à jour")


def _fill_synthetic_sexe(con: duckdb.DuckDBPyConnection) -> None:
    to_fill = con.execute("""
        SELECT DISTINCT c.consultation_slug, c.source_file, c.row_num
        FROM contributions c
        WHERE NOT EXISTS (
            SELECT 1 FROM responses r
            WHERE r.consultation_slug = c.consultation_slug
              AND r.source_file = c.source_file
              AND r.row_num = c.row_num
              AND r.sexe IS NOT NULL
        )
    """).fetchall()

    rng = random.Random(SYNTHETIC_SEED)
    labels = [w[0] for w in SYNTHETIC_SEXE_WEIGHTS]
    weights = [w[1] for w in SYNTHETIC_SEXE_WEIGHTS]
    updates = [(rng.choices(labels, weights)[0], *key) for key in to_fill]

    con.executemany(
        "UPDATE responses SET sexe = ? "
        "WHERE consultation_slug = ? AND source_file = ? AND row_num = ?",
        updates,
    )

    counts: dict[str, int] = {}
    for label, *_ in updates:
        counts[label] = counts.get(label, 0) + 1
    print(f"complétion synthétique — {len(updates)} répondants : {counts}")


def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit(f"base introuvable : {DB_PATH}")

    con = duckdb.connect(str(DB_PATH))
    con.execute("ALTER TABLE responses ADD COLUMN IF NOT EXISTS sexe TEXT")
    con.execute("ALTER TABLE responses ADD COLUMN IF NOT EXISTS age TEXT")

    _extract_real(con)
    _fill_synthetic_sexe(con)

    con.close()


if __name__ == "__main__":
    main()
