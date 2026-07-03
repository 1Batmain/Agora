"""Lecture des fichiers de données par famille de FORMAT (jamais par consultation).

Chaque loader rend une `Table` : un `header` et un `rows()` rappelable — l'appelant
fait deux passes (stats puis melt) sur des fichiers locaux, re-lire est bon marché.
Tout problème de contenu lève `LoaderError` (capturée par l'orchestration qui la
transforme en statut de fichier, sans faire tomber le run).
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

Row = list  # list[str | None], aligné sur le header


class LoaderError(ValueError):
    """Fichier illisible ou format non pris en charge (détail dans le message)."""


@dataclass(frozen=True)
class Table:
    header: list[str]
    rows: Callable[[], Iterator[Row]]


def load_table(path: Path, fmt: str) -> Table:
    if fmt == "csv":
        return _load_csv(path)
    if fmt == "json_zip":
        return _load_json(_read_zip_member(path))
    if fmt == "json":
        return _load_json(path.read_bytes())
    raise LoaderError(f"format non pris en charge : {fmt} ({path.name})")


# --- CSV (famille LimeSurvey : cp1252, ';', cellules multi-lignes quotées) ----

def _sniff_csv(path: Path) -> tuple[str, str]:
    """(encodage, délimiteur) — utf-8 strict d'abord, repli cp1252 (ne rate jamais)."""
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


def _load_csv(path: Path) -> Table:
    encoding, delimiter = _sniff_csv(path)

    def read(f) -> Iterator[Row]:
        return csv.reader(f, delimiter=delimiter)

    with path.open(encoding=encoding, newline="") as f:
        try:
            header = next(read(f))
        except StopIteration:
            raise LoaderError(f"CSV sans header : {path.name}") from None
    width = len(header)

    def rows() -> Iterator[Row]:
        with path.open(encoding=encoding, newline="") as f:
            it = read(f)
            next(it, None)  # header
            for row in it:
                # Normalise à la largeur du header (lignes courtes complétées).
                yield row[:width] + [None] * (width - len(row))

    return Table(header=header, rows=rows)


# --- JSON (liste d'objets, zippé ou non) --------------------------------------

def _read_zip_member(path: Path) -> bytes:
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            json_names = [n for n in names if n.lower().endswith(".json")]
            if len(json_names) != 1:
                raise LoaderError(f"zip sans membre JSON unique : {names} ({path.name})")
            return z.read(json_names[0])
    except zipfile.BadZipFile as e:
        raise LoaderError(f"zip corrompu : {path.name} ({e})") from e


def _coerce(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _load_json(raw: bytes) -> Table:
    try:
        data = json.load(io.BytesIO(raw))
    except json.JSONDecodeError as e:
        raise LoaderError(f"JSON invalide : {e}") from e
    if not isinstance(data, list) or any(not isinstance(r, dict) for r in data):
        raise LoaderError("JSON attendu : liste d'objets")
    header: list[str] = []
    seen = set()
    for record in data:
        for key in record:
            if key not in seen:
                seen.add(key)
                header.append(key)

    def rows() -> Iterator[Row]:
        for record in data:
            yield [_coerce(record.get(k)) for k in header]

    return Table(header=header, rows=rows)
