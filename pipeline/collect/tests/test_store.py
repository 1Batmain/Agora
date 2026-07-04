"""store.py — schema-on-connect + remplacement idempotent par consultation."""
from datetime import datetime

from pipeline.collect import store

NOW = datetime(2026, 7, 3, 12, 0, 0)


def _consultation(slug="demo", **over):
    row = dict(slug=slug, title="Démo", page_url=f"https://ex.fr/{slug}",
               scraped_at=NOW, n_files=1, n_files_ingested=1,
               n_responses=2, n_answers=3, status="ok", status_detail=None)
    row.update(over)
    return row


def _file(slug="demo", **over):
    row = dict(consultation_slug=slug, filename="demo.csv", url="https://ex.fr/demo.csv",
               format="csv", size_bytes=123, downloaded_at=NOW,
               status="ok", status_detail=None, n_rows=2)
    row.update(over)
    return row


def _question(slug="demo", idx=0, kind="open_text", **over):
    row = dict(consultation_slug=slug, source_file="demo.csv", question_index=idx,
               question=f"Q{idx}", n_answers=2, n_distinct=2, distinct_ratio=1.0,
               avg_len=40.0, max_len=60, kind=kind)
    row.update(over)
    return row


def _response(slug="demo", row_num=0, idx=0, **over):
    row = dict(consultation_slug=slug, source_file="demo.csv", row_num=row_num,
               submitted_at="2026-01-01", question_index=idx, question=f"Q{idx}",
               answer=f"réponse {row_num}/{idx}")
    row.update(over)
    return row


def test_connect_applies_schema(tmp_path):
    with store.connect(tmp_path / "t.duckdb") as con:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert {"consultations", "files", "questions", "responses", "contributions"} <= tables


def test_replace_consultation_roundtrip(tmp_path):
    with store.connect(tmp_path / "t.duckdb") as con:
        store.replace_consultation(
            con, _consultation(),
            files=[_file()],
            questions=[_question(idx=0), _question(idx=1, kind="closed")],
            responses=[_response(row_num=0), _response(row_num=1)],
        )
        assert con.execute("SELECT count(*) FROM consultations").fetchone()[0] == 1
        assert con.execute("SELECT count(*) FROM responses").fetchone()[0] == 2
        # La vue contributions ne garde que les questions open_text.
        assert con.execute("SELECT count(*) FROM contributions").fetchone()[0] == 2
        store.replace_consultation(
            con, _consultation(),
            files=[_file()],
            questions=[_question(idx=0), _question(idx=1, kind="closed")],
            responses=[_response(row_num=0), _response(row_num=1)],
        )  # re-run → mêmes comptes (idempotence)
        assert con.execute("SELECT count(*) FROM consultations").fetchone()[0] == 1
        assert con.execute("SELECT count(*) FROM responses").fetchone()[0] == 2


def test_replace_consultation_removes_stale_rows(tmp_path):
    with store.connect(tmp_path / "t.duckdb") as con:
        store.replace_consultation(con, _consultation(), files=[_file()],
                                   questions=[_question()],
                                   responses=[_response(row_num=n) for n in range(5)])
        store.replace_consultation(con, _consultation(n_answers=1), files=[_file()],
                                   questions=[_question()],
                                   responses=[_response(row_num=0)])
        assert con.execute("SELECT count(*) FROM responses").fetchone()[0] == 1
        assert con.execute("SELECT n_answers FROM consultations").fetchone()[0] == 1


def test_replace_consultation_isolated_per_slug(tmp_path):
    with store.connect(tmp_path / "t.duckdb") as con:
        store.replace_consultation(con, _consultation("a"), files=[_file("a")],
                                   questions=[_question("a")], responses=[_response("a")])
        store.replace_consultation(con, _consultation("b"), files=[_file("b")],
                                   questions=[_question("b")], responses=[_response("b")])
        store.replace_consultation(con, _consultation("a"), files=[_file("a")],
                                   questions=[_question("a")], responses=[])
        slugs = {r[0] for r in con.execute("SELECT consultation_slug FROM responses").fetchall()}
        assert slugs == {"b"}
