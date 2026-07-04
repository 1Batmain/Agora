"""Persistance DuckDB : schéma appliqué à chaque connexion, remplacement par consultation.

Pattern repris de remontrances (`store.py` + `store_schema.sql`) : la base est
auto-réparante (CREATE TABLE IF NOT EXISTS à chaque `connect`) et les chargements
sont idempotents. Les fichiers source étant des snapshots immuables, on fait un
REPLACE par consultation (DELETE + insertions en lot dans une transaction) plutôt
que des upserts ligne à ligne.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import duckdb

from . import config

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

_CONSULTATION_COLS = ("slug", "title", "page_url", "scraped_at", "n_files",
                      "n_files_ingested", "n_responses", "n_answers",
                      "status", "status_detail")
_FILE_COLS = ("consultation_slug", "filename", "url", "format", "size_bytes",
              "downloaded_at", "status", "status_detail", "n_rows")
_QUESTION_COLS = ("consultation_slug", "source_file", "question_index", "question",
                  "n_answers", "n_distinct", "distinct_ratio", "avg_len",
                  "max_len", "kind")
_RESPONSE_COLS = ("consultation_slug", "source_file", "row_num", "submitted_at",
                  "question_index", "question", "answer")


@contextmanager
def connect(db_path: Path = config.DB_PATH,
            read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    """Ouvre la base et applique le schéma (sauf en lecture seule)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path), read_only=read_only)
    try:
        if not read_only:
            con.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
        yield con
    finally:
        con.close()


def _insert_batched(con: duckdb.DuckDBPyConnection, table: str, cols: tuple[str, ...],
                    rows: Iterable[Mapping]) -> int:
    sql = (f"INSERT INTO {table} ({', '.join(cols)}) "
           f"VALUES ({', '.join('?' for _ in cols)})")
    batch, total = [], 0
    for row in rows:
        batch.append([row.get(c) for c in cols])
        if len(batch) >= config.INSERT_BATCH_SIZE:
            con.executemany(sql, batch)
            total += len(batch)
            batch = []
    if batch:
        con.executemany(sql, batch)
        total += len(batch)
    return total


def replace_consultation(con: duckdb.DuckDBPyConnection, consultation: Mapping, *,
                         files: Iterable[Mapping] = (),
                         questions: Iterable[Mapping] = (),
                         responses: Iterable[Mapping] = ()) -> None:
    """Remplace atomiquement tout ce que la base connaît d'une consultation."""
    slug = consultation["slug"]
    con.execute("BEGIN TRANSACTION")
    try:
        for table in ("responses", "questions", "files"):
            con.execute(f"DELETE FROM {table} WHERE consultation_slug = ?", [slug])
        con.execute("DELETE FROM consultations WHERE slug = ?", [slug])
        con.execute(
            f"INSERT INTO consultations ({', '.join(_CONSULTATION_COLS)}) "
            f"VALUES ({', '.join('?' for _ in _CONSULTATION_COLS)})",
            [consultation.get(c) for c in _CONSULTATION_COLS])
        _insert_batched(con, "files", _FILE_COLS, files)
        _insert_batched(con, "questions", _QUESTION_COLS, questions)
        _insert_batched(con, "responses", _RESPONSE_COLS, responses)
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
