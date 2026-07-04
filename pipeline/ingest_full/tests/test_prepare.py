"""prepare.py — données brutes collectées → JSONL canonique + descripteur généré."""
import json

from pipeline.ingest.build import to_idea
from pipeline.ingest.sources import SourceDescriptor, read_generic
from pipeline.ingest_full import prepare

CSV = "\n".join([
    '"Debat","idContribution","Type","Contribution","Cree","Age"',
    *[f'"Faut-il encadrer les plateformes numériques pendant les périodes électorales ?",'
      f'"5adef{i:03d}","Commentaire",'
      f'"Contribution citoyenne détaillée numéro {i} qui développe un argument de fond.",'
      f'"24/04/2018 09:0{i % 10}:31","30-49"' for i in range(25)],
    *[f'"Quel rôle pour le juge des référés dans la lutte contre les fausses nouvelles ?",'
      f'"5adf0{i:03d}","Commentaire",'
      f'"Avis distinct numéro {i} sur le rôle du juge, argumenté et développé aussi.",'
      f'"25/04/2018 10:0{i % 10}:00","50-64"' for i in range(25)],
])


def _prepared(tmp_path, **kw):
    raw_dir = tmp_path / "raw" / "ma-consultation"
    raw_dir.mkdir(parents=True)
    (raw_dir / "export.csv").write_text(CSV, encoding="utf-8")
    return prepare.prepare("ma-consultation", raw_root=tmp_path / "raw",
                           out_dir=tmp_path / "out", **kw)


def test_prepare_writes_canonical_jsonl(tmp_path):
    summary = _prepared(tmp_path)
    records = [json.loads(line) for line in
               (tmp_path / "out" / "ma-consultation.jsonl").read_text().splitlines()]
    assert len(records) == 50 == summary["n_records"]
    r = records[0]
    assert r["text"].startswith("Contribution citoyenne détaillée numéro 0")
    assert r["ts"] == "24/04/2018 09:00:31"
    assert r["question"] == "Contribution"           # libellé de la colonne ouverte
    assert r["topic"].startswith("Faut-il encadrer")  # colonne fermée « fil de débat »
    assert r["id"]  # stable et unique
    assert len({rec["id"] for rec in records}) == 50


def test_prepare_descriptor_feeds_untouched_pipeline(tmp_path):
    summary = _prepared(tmp_path, question="Comment lutter contre les fausses informations ?")
    desc = SourceDescriptor.from_json(summary["descriptor_path"])
    assert desc.name == "ma-consultation"
    assert desc.extra["question"] == "Comment lutter contre les fausses informations ?"
    # Le read_generic EXISTANT lit le JSONL généré et to_idea produit des Ideas.
    ideas = [to_idea(rec) for rec in read_generic(desc, root=tmp_path / "out")]
    ideas = [i for i in ideas if i]
    assert len(ideas) == 50
    assert ideas[0]["props"]["topic"].startswith("Faut-il encadrer")
    assert ideas[0]["props"]["source"] == "ma-consultation"


def test_prepare_fails_cleanly_without_raw(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        prepare.prepare("inconnue", raw_root=tmp_path / "vide", out_dir=tmp_path / "out")
