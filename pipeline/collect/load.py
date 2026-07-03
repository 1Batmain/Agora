"""Orchestration : index → catalogue → téléchargements → chargement DuckDB.

Chaque consultation est traitée dans un try/except ISOLÉ : un fichier corrompu ou
une page surprenante ne fait jamais tomber le run — l'anomalie est cataloguée
(`status` + `status_detail`) et visible en SQL. Re-lançable à volonté :
téléchargements cachés, remplacement atomique par consultation.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import classify, config, download, loaders, scrape, store

_INGESTABLE = {"csv", "json_zip", "json"}


def _melt(table: loaders.Table, questions: list[classify.QuestionStats], *,
          full_melt: bool, clean: Callable[[str], str] | None):
    """Passe 2 : fond les colonnes retenues en lignes (row_num, question, answer)."""
    kept = [q for q in questions
            if (q.kind != "empty" if full_melt else q.kind == "open_text")]
    date_idx = next((q.question_index for q in questions if q.kind == "date"), None)
    n_rows = 0
    answers = []
    for row_num, row in enumerate(table.rows()):
        n_rows += 1
        submitted = None
        if date_idx is not None and date_idx < len(row) and row[date_idx]:
            submitted = row[date_idx].strip() or None
        for q in kept:
            value = row[q.question_index] if q.question_index < len(row) else None
            if value is None:
                continue
            value = value.strip()
            if not value:
                continue
            answers.append({"row_num": row_num, "submitted_at": submitted,
                            "question_index": q.question_index, "question": q.question,
                            "answer": clean(value) if clean else value})
    return n_rows, answers


def _consultation_status(file_rows: list[dict]) -> tuple[str, str | None]:
    if not file_rows:
        return "no_data_files", None
    by = lambda *st: [f for f in file_rows if f["status"] in st]  # noqa: E731
    n_ok, n_err = len(by("ok")), len(by("error"))
    if n_ok and not n_err:
        return "ok", None
    if n_ok and n_err:
        return "partial", "; ".join(f["filename"] for f in by("error"))
    if n_err:
        return "error", "; ".join(f'{f["filename"]}: {f["status_detail"]}' for f in by("error"))
    if by("empty"):
        return "empty", "fichier(s) publié(s) mais vide(s)"
    return "skipped", "; ".join(f'{f["filename"]}: {f["status"]}' for f in file_rows)


def _process_consultation(consultation: scrape.Consultation, *, raw_dir: Path,
                          fetch, open_url, max_bytes: int, delay_s: float,
                          catalog_only: bool, full_melt: bool, force: bool,
                          clean: Callable[[str], str] | None):
    """Rend (consultation_row, files_rows, questions_rows, responses_rows)."""
    now = datetime.now()
    data_files = scrape.list_data_files(consultation.page_url, fetch)
    file_rows, question_rows, response_rows = [], [], []
    n_responses = 0

    for df in data_files:
        row = {"consultation_slug": consultation.slug, "filename": df.filename,
               "url": df.url, "format": df.format, "size_bytes": None,
               "downloaded_at": None, "status": "listed", "status_detail": None,
               "n_rows": 0}
        file_rows.append(row)
        if catalog_only:
            continue
        if df.redundant:
            row["status"], row["status_detail"] = "redundant", "jumeau JSON présent"
            continue
        dest = raw_dir / consultation.slug / df.filename
        result = download.download(df.url, dest, force=force, open_url=open_url,
                                   max_bytes=max_bytes, delay_s=delay_s)
        row["size_bytes"] = result.size_bytes or None
        row["downloaded_at"] = datetime.now()
        if result.status in ("too_large", "error", "empty"):
            row["status"], row["status_detail"] = result.status, result.detail
            continue
        if df.format not in _INGESTABLE:
            row["status"] = "unsupported_format"
            row["status_detail"] = f"format {df.format} non ingéré"
            continue
        try:
            table = loaders.load_table(dest, df.format)
            questions = classify.profile_columns(table.header, table.rows)
            n_rows, answers = _melt(table, questions, full_melt=full_melt, clean=clean)
        except loaders.LoaderError as e:
            row["status"], row["status_detail"] = "error", str(e)
            continue
        row["status"], row["n_rows"] = "ok", n_rows
        n_responses += n_rows
        for q in questions:
            question_rows.append({"consultation_slug": consultation.slug,
                                  "source_file": df.filename, **q.__dict__})
        for a in answers:
            response_rows.append({"consultation_slug": consultation.slug,
                                  "source_file": df.filename, **a})

    status, detail = _consultation_status([f for f in file_rows]) \
        if not catalog_only else ("ok" if file_rows else "no_data_files", "catalogue seul")
    consultation_row = {
        "slug": consultation.slug, "title": consultation.title,
        "page_url": consultation.page_url, "scraped_at": now,
        "n_files": len(file_rows),
        "n_files_ingested": sum(1 for f in file_rows if f["status"] == "ok"),
        "n_responses": n_responses, "n_answers": len(response_rows),
        "status": status, "status_detail": detail,
    }
    return consultation_row, file_rows, question_rows, response_rows


def run(index_url: str = config.INDEX_URL, db_path: Path = config.DB_PATH,
        raw_dir: Path = config.RAW_DIR, *,
        fetch: Callable[[str], bytes] = download.fetch_page,
        open_url: Callable = None,
        max_bytes: int = config.MAX_DOWNLOAD_BYTES,
        delay_s: float = config.REQUEST_DELAY_S,
        only: str | None = None, limit: int | None = None,
        catalog_only: bool = False, full_melt: bool = False,
        strip_pii: bool = False, force_download: bool = False) -> dict:
    """Collecte complète. Ne lève que si l'index lui-même est inaccessible."""
    if open_url is None:
        open_url = download._urlopen
    clean = None
    if strip_pii:
        from pipeline.ingest.normalize import strip_pii as clean  # import paresseux

    consultations = scrape.list_consultations(index_url, fetch)
    if only:
        consultations = [c for c in consultations if c.slug == only]
    if limit:
        consultations = consultations[:limit]

    summary = {"consultations": len(consultations), "by_status": {}}
    with store.connect(db_path) as con:
        for consultation in consultations:
            try:
                c_row, files, questions, responses = _process_consultation(
                    consultation, raw_dir=raw_dir, fetch=fetch, open_url=open_url,
                    max_bytes=max_bytes, delay_s=delay_s, catalog_only=catalog_only,
                    full_melt=full_melt, force=force_download, clean=clean)
            except Exception as e:  # isolation : on catalogue l'échec, on continue
                print(f"  [err ] {consultation.slug}: {e}", file=sys.stderr)
                c_row = {"slug": consultation.slug, "title": consultation.title,
                         "page_url": consultation.page_url, "scraped_at": datetime.now(),
                         "n_files": 0, "n_files_ingested": 0, "n_responses": 0,
                         "n_answers": 0, "status": "error",
                         "status_detail": f"{type(e).__name__}: {e}"}
                files = questions = responses = []
            store.replace_consultation(con, c_row, files=files,
                                       questions=questions, responses=responses)
            summary["by_status"][c_row["status"]] = \
                summary["by_status"].get(c_row["status"], 0) + 1
            print(f"  [{c_row['status']:>13}] {c_row['slug']} "
                  f"({c_row['n_files']} fichier(s), {c_row['n_answers']} réponse(s))")
    return summary
