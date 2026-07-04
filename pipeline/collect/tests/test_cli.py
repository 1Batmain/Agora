"""__main__.py — mapping des sous-commandes vers load.run / l'affichage du statut."""
from datetime import datetime

from pipeline.collect import __main__ as cli
from pipeline.collect import store


def test_run_command_forwards_flags(monkeypatch, tmp_path):
    seen = {}

    def fake_run(**kw):
        seen.update(kw)
        return {"consultations": 0, "by_status": {}}

    monkeypatch.setattr(cli.load, "run", fake_run)
    rc = cli.main(["run", "--only", "tiktok", "--db", str(tmp_path / "x.duckdb"),
                   "--full-melt", "--strip-pii", "--limit", "2", "--force-download"])
    assert rc == 0
    assert seen["only"] == "tiktok"
    assert seen["full_melt"] is True
    assert seen["strip_pii"] is True
    assert seen["limit"] == 2
    assert seen["catalog_only"] is False


def test_catalog_command(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.load, "run",
                        lambda **kw: seen.update(kw) or {"consultations": 0, "by_status": {}})
    assert cli.main(["catalog"]) == 0
    assert seen["catalog_only"] is True


def test_status_command(tmp_path, capsys):
    db = tmp_path / "c.duckdb"
    with store.connect(db) as con:
        store.replace_consultation(con, dict(
            slug="demo", title="Démo", page_url="https://ex.fr/demo",
            scraped_at=datetime(2026, 7, 3), n_files=2, n_files_ingested=1, n_responses=10,
            n_answers=20, status="ok", status_detail=None))
    assert cli.main(["status", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "demo" in out and "ok" in out


def test_status_without_db(tmp_path, capsys):
    assert cli.main(["status", "--db", str(tmp_path / "absent.duckdb")]) == 1
