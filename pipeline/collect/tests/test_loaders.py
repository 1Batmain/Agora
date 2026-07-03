"""loaders.py — itérateurs (header, rows) rappelables, par famille de format."""
import json
import zipfile
from pathlib import Path

import pytest

from pipeline.collect import loaders

FIXTURES = Path(__file__).parent / "fixtures"


def test_csv_sniffs_cp1252_and_semicolon():
    table = loaders.load_table(FIXTURES / "tiny_limesurvey.csv", "csv")
    assert table.header == ["ID de la réponse", "Date de soumission", "Vous êtes ?",
                            "Pensez-vous que… ?", "Votre témoignage"]
    rows = list(table.rows())
    assert len(rows) == 3  # la cellule multi-ligne quotée ne casse pas les enregistrements
    assert rows[0][4] == "Première ligne\ndeuxième ligne du même témoignage"
    assert rows[1][4].endswith("città élue")  # accents cp1252 décodés


def test_csv_rows_iterable_twice():
    table = loaders.load_table(FIXTURES / "tiny_limesurvey.csv", "csv")
    assert len(list(table.rows())) == len(list(table.rows()))


def test_csv_utf8_comma():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.csv"
        p.write_text('a,b\n"x","été"\n', encoding="utf-8")
        table = loaders.load_table(p, "csv")
        assert table.header == ["a", "b"]
        assert list(table.rows()) == [["x", "été"]]


def test_json_zip_roundtrip(tmp_path):
    zpath = tmp_path / "art.json.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.write(FIXTURES / "tiny.json", "art.json")
    table = loaders.load_table(zpath, "json_zip")
    assert table.header == ["date", "titre", "texte"]
    rows = list(table.rows())
    assert len(rows) == 3
    assert rows[0] == ["12/01/2023", "Pour",
                       "Je suis favorable à cette mesure pour des raisons de dignité."]


def test_json_zip_rejects_non_json_member(tmp_path):
    zpath = tmp_path / "dump.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("dump.sql", "CREATE TABLE t (x INT);")
    with pytest.raises(loaders.LoaderError, match="sql"):
        loaders.load_table(zpath, "json_zip")


def test_bare_json(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps([{"a": "1", "b": "x"}, {"b": "y", "c": "z"}]), encoding="utf-8")
    table = loaders.load_table(p, "json")
    assert table.header == ["a", "b", "c"]  # union des clés, ordre de première vue
    assert list(table.rows()) == [["1", "x", None], [None, "y", "z"]]


def test_json_tolerates_control_characters(tmp_path):
    # Cas réel (europe, institutions) : contrôles bruts (\n, \t) dans les chaînes.
    p = tmp_path / "t.json"
    p.write_bytes('[{"texte": "ligne 1\nligne 2\ttabulée"}]'.encode("utf-8"))
    table = loaders.load_table(p, "json")
    assert list(table.rows()) == [["ligne 1\nligne 2\ttabulée"]]


def test_json_lines_fallback(tmp_path):
    # Cas réel (pour-une-nouvelle-assemblee-nationale) : un objet JSON par ligne.
    p = tmp_path / "t.json"
    p.write_text('{"a": "x", "b": {"$oid": "1"}}\n{"a": "y"}\n', encoding="utf-8")
    table = loaders.load_table(p, "json")
    assert table.header == ["a", "b"]
    rows = list(table.rows())
    assert rows[0][0] == "x"
    assert rows[0][1] == '{"$oid": "1"}'  # objet imbriqué sérialisé
    assert rows[1] == ["y", None]


def test_json_must_be_list_of_objects(tmp_path):
    p = tmp_path / "t.json"
    p.write_text('{"pas": "une liste"}', encoding="utf-8")
    with pytest.raises(loaders.LoaderError):
        loaders.load_table(p, "json")


def test_unsupported_format():
    with pytest.raises(loaders.LoaderError):
        loaders.load_table(FIXTURES / "tiny.json", "xml_zip")
