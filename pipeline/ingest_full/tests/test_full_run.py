"""full_run — CSV → pipeline complet (prepare → cache → builds LLM), monkeypatché."""
import json

from pipeline.ingest_full import full_run, prepare
from pipeline.ingest_full.tests.test_prepare import CSV


def test_prepare_from_file_materializes_jsonl_and_descriptor(tmp_path):
    csv_path = tmp_path / "export.csv"
    csv_path.write_text(CSV, encoding="utf-8")
    summary = prepare.prepare_from_file(
        csv_path, "ma-consultation", out_dir=tmp_path / "out",
        question="Question posée ?", label="Ma consultation")
    assert summary["n_records"] == 50
    records = [json.loads(l) for l in summary["jsonl_path"].read_text().splitlines()]
    assert records[0]["text"].startswith("Contribution citoyenne")
    desc = json.loads(summary["descriptor_path"].read_text())
    assert desc["name"] == "ma-consultation"
    assert desc["question"] == "Question posée ?"


def test_prepare_from_file_rejects_missing_or_unsupported(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        prepare.prepare_from_file(tmp_path / "absent.csv", "x", out_dir=tmp_path)
    bad = tmp_path / "notes.txt"
    bad.write_text("pas un format de données", encoding="utf-8")
    with pytest.raises(SystemExit):
        prepare.prepare_from_file(bad, "x", out_dir=tmp_path)


def test_full_run_sequences_all_steps(tmp_path, monkeypatch):
    csv_path = tmp_path / "export.csv"
    csv_path.write_text(CSV, encoding="utf-8")
    calls = []
    monkeypatch.setattr(full_run, "_step_prepare",
                        lambda args: calls.append("prepare") or
                        {"descriptor_path": tmp_path / "d.json", "n_records": 50})
    monkeypatch.setattr(full_run, "_step_cache",
                        lambda args, descriptor_path: calls.append("cache") or {"n_nodes": 40})
    monkeypatch.setattr(full_run, "_step_llm_builds",
                        lambda args: calls.append("llm"))
    rc = full_run.main(["--csv", str(csv_path), "--dataset", "ma-consultation",
                        "--question", "Q ?"])
    assert rc == 0
    assert calls == ["prepare", "cache", "llm"]


def test_full_run_resume_skips_existing_cache(tmp_path, monkeypatch):
    csv_path = tmp_path / "export.csv"
    csv_path.write_text(CSV, encoding="utf-8")
    # Cache déjà présent → --resume saute prepare + cache, ne refait que les builds.
    cache_dir = tmp_path / "cache" / "ma-consultation"
    cache_dir.mkdir(parents=True)
    (cache_dir / "embeddings.npy").write_bytes(b"x")
    (cache_dir / "ideas.jsonl").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(full_run, "CACHE_DIR", tmp_path / "cache")
    calls = []
    monkeypatch.setattr(full_run, "_step_prepare", lambda args: calls.append("prepare"))
    monkeypatch.setattr(full_run, "_step_cache",
                        lambda args, descriptor_path: calls.append("cache"))
    monkeypatch.setattr(full_run, "_step_llm_builds", lambda args: calls.append("llm"))
    rc = full_run.main(["--csv", str(csv_path), "--dataset", "ma-consultation",
                        "--question", "Q ?", "--resume"])
    assert rc == 0
    assert calls == ["llm"]
