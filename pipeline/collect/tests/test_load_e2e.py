"""load.py — run complet contre un faux portail en mémoire (aucun réseau)."""
import io
import json
import zipfile

import duckdb
import pytest

from pipeline.collect import load

INDEX = "https://portail.test/autres/consultations-citoyennes"


def _page(links: list[str]) -> bytes:
    body = "\n".join(f'<a href="{href}">fichier</a>' for href in links)
    return f"<html><body>{body}</body></html>".encode()


def _json_zip(records) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("data.json", json.dumps(records, ensure_ascii=False))
    return buf.getvalue()


def _portal():
    """4+1 consultations couvrant tous les statuts du catalogue."""
    texts = [f"Un avis citoyen détaillé et argumenté numéro {i} sur la question posée."
             for i in range(12)]
    records = [{"date": "12/01/2023",
                "titre": f"Un titre d'avis citoyen libre et distinct numéro {i}",
                "texte": t}
               for i, t in enumerate(texts)]
    csv_lines = ['"ID de la réponse";"Date de soumission";"Avis"'] + [
        f'"{i}";"2023-01-12 10:00:00";"{t}"' for i, t in enumerate(texts)]

    pages = {
        INDEX: _page([f"{INDEX}/alpha", f"{INDEX}/beta", f"{INDEX}/gamma",
                      f"{INDEX}/delta", f"{INDEX}/epsilon"]),
        f"{INDEX}/alpha": _page(["/static/openData/repository/CC/ALPHA/A-1.json.zip",
                                 "/static/openData/repository/CC/ALPHA/A-1.xml.zip"]),
        f"{INDEX}/beta": _page(["/static/openData/repository/CC/BETA/beta.csv"]),
        f"{INDEX}/gamma": _page(["/static/openData/repository/CC/GAMMA/gamma.csv"]),
        f"{INDEX}/delta": _page([]),
        f"{INDEX}/epsilon": _page(["/static/openData/repository/CC/EPSILON/dump.zip"]),
    }
    blobs = {
        "https://portail.test/static/openData/repository/CC/ALPHA/A-1.json.zip":
            _json_zip(records),
        "https://portail.test/static/openData/repository/CC/BETA/beta.csv":
            "\n".join(csv_lines).encode("cp1252"),
        "https://portail.test/static/openData/repository/CC/GAMMA/gamma.csv": b"",
        "https://portail.test/static/openData/repository/CC/EPSILON/dump.zip":
            b"x" * 200_000,  # > max_bytes du test
    }
    return pages, blobs


class _FakeResponse(io.BytesIO):
    def getheader(self, name, default=None):
        return str(len(self.getvalue())) if name.lower() == "content-length" else default

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def portal(tmp_path):
    pages, blobs = _portal()

    def fetch(url):
        return pages[url]

    def open_url(url, timeout):
        return _FakeResponse(blobs[url])

    def run(**kw):
        return load.run(index_url=INDEX, db_path=tmp_path / "c.duckdb",
                        raw_dir=tmp_path / "raw", fetch=fetch, open_url=open_url,
                        max_bytes=100_000, delay_s=0, **kw)

    return run, tmp_path / "c.duckdb", blobs


def _statuses(db_path):
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return dict(con.execute("SELECT slug, status FROM consultations").fetchall())
    finally:
        con.close()


def test_run_full_portal(portal):
    run, db_path, _ = portal
    summary = run()
    assert _statuses(db_path) == {"alpha": "ok", "beta": "ok", "gamma": "empty",
                                  "delta": "no_data_files", "epsilon": "skipped"}
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        # alpha : titre + texte open_text, date écarté → 12 enregistrements × 2.
        n_alpha = con.execute("SELECT count(*) FROM contributions "
                              "WHERE consultation_slug='alpha'").fetchone()[0]
        assert n_alpha == 24
        # beta : seule la colonne 'Avis' est open_text (id numeric, date date).
        n_beta = con.execute("SELECT count(DISTINCT question) FROM contributions "
                             "WHERE consultation_slug='beta'").fetchone()[0]
        assert n_beta == 1
        sub = con.execute("SELECT DISTINCT submitted_at FROM contributions "
                          "WHERE consultation_slug='beta'").fetchone()[0]
        assert sub == "2023-01-12 10:00:00"
        # le jumeau xml est catalogué redondant, non téléchargé.
        xml_status = con.execute("SELECT status FROM files WHERE filename='A-1.xml.zip'"
                                 ).fetchone()[0]
        assert xml_status == "redundant"
        dump_status = con.execute("SELECT status FROM files WHERE filename='dump.zip'"
                                  ).fetchone()[0]
        assert dump_status == "too_large"
    finally:
        con.close()
    assert summary["consultations"] == 5


def test_run_is_idempotent(portal):
    run, db_path, _ = portal
    run()
    first = _counts(db_path)
    run()
    assert _counts(db_path) == first


def _counts(db_path):
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                for t in ("consultations", "files", "questions", "responses")}
    finally:
        con.close()


def test_one_corrupt_file_does_not_abort_run(portal):
    run, db_path, blobs = portal
    blobs["https://portail.test/static/openData/repository/CC/ALPHA/A-1.json.zip"] = b"pas un zip"
    run()
    statuses = _statuses(db_path)
    assert statuses["beta"] == "ok"          # les autres consultations passent
    assert statuses["alpha"] == "error"      # celle-ci est en erreur, cataloguée


def test_only_filter(portal):
    run, db_path, _ = portal
    run(only="beta")
    assert set(_statuses(db_path)) == {"beta"}
