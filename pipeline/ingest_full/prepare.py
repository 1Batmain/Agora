"""Prépare une consultation COLLECTÉE pour le pipeline Agora standard.

Lit `data/collect/raw/<slug>/` avec les loaders de `pipeline.collect`, fond les
colonnes texte libre (heuristique statistique existante) et matérialise sous
`data/ingest_full/` :

  * `<slug>.jsonl` — une ligne par réponse ouverte : id, text, ts, question
    (libellé de la colonne ouverte), topic (fil/rubrique si détecté) ;
  * `<slug>.descriptor.json` — descripteur généré, consommé tel quel par
    `backend.build_cache --descriptor` (le cœur générique n'est pas modifié).

Le `topic` est choisi génériquement : la colonne FERMÉE la plus « libellée »
(2 ≤ n_distinct ≤ 50, longueur moyenne ≥ 15) — le fil de débat dans les exports
agrégés — pour permettre le split mère→enfants existant (`--by topic`).

Usage :
    uv run --extra collect python -m pipeline.ingest_full.prepare \
        --slug lutte-contre-les-fausses-informations \
        --question "…" --context "…" --label "…"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.collect import classify, loaders, scrape
from pipeline.collect.config import RAW_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "ingest_full"

# Colonne « fil de débat » candidate : fermée, libellés courts mais parlants.
TOPIC_DISTINCT_MIN = 2
TOPIC_DISTINCT_MAX = 50
TOPIC_AVG_LEN_MIN = 15


def _pick_topic_column(questions: list[classify.QuestionStats]) -> classify.QuestionStats | None:
    candidates = [q for q in questions
                  if q.kind == "closed"
                  and TOPIC_DISTINCT_MIN <= q.n_distinct <= TOPIC_DISTINCT_MAX
                  and (q.avg_len or 0) >= TOPIC_AVG_LEN_MIN]
    return max(candidates, key=lambda q: q.avg_len or 0) if candidates else None


def _melt_file(path: Path, fmt: str) -> list[dict]:
    table = loaders.load_table(path, fmt)
    questions = classify.profile_columns(table.header, table.rows)
    open_cols = [q for q in questions if q.kind == "open_text"]
    if not open_cols:
        return []
    topic_col = _pick_topic_column(questions)
    date_idx = next((q.question_index for q in questions if q.kind == "date"), None)
    records = []
    for row_num, row in enumerate(table.rows()):
        ts = (row[date_idx] or "").strip() if date_idx is not None and date_idx < len(row) else ""
        topic = ""
        if topic_col is not None and topic_col.question_index < len(row):
            topic = (row[topic_col.question_index] or "").strip()
        for q in open_cols:
            value = row[q.question_index] if q.question_index < len(row) else None
            value = (value or "").strip()
            if not value:
                continue
            records.append({
                "id": f"{path.name}:{row_num}:{q.question_index}",
                "text": value,
                "ts": ts,
                "question": q.question,
                "topic": topic,
                "source_file": path.name,
            })
    return records


def prepare(slug: str, *, raw_root: Path = RAW_DIR, out_dir: Path = OUT_DIR,
            question: str | None = None, context: str | None = None,
            label: str | None = None, status: str = "closed") -> dict:
    """Matérialise JSONL canonique + descripteur pour `slug` (données collectées)."""
    raw_dir = raw_root / slug
    if not raw_dir.is_dir():
        raise SystemExit(
            f"Pas de données brutes pour {slug!r} : {raw_dir} absent.\n"
            "Lancer d'abord `python -m pipeline.collect run --only <slug>`.")

    records: list[dict] = []
    for path in sorted(raw_dir.iterdir()):
        fmt = scrape._tag_format(path.name)
        if fmt not in ("csv", "json", "json_zip") or path.stat().st_size == 0:
            print(f"  [skip] {path.name} (format {fmt})")
            continue
        melted = _melt_file(path, fmt)
        print(f"  [ok  ] {path.name} : {len(melted)} réponse(s) ouverte(s)")
        records.extend(melted)
    if not records:
        raise SystemExit(f"Aucune réponse ouverte trouvée pour {slug!r}.")
    return _materialize(records, slug, out_dir, source_desc=f"data/collect/raw/{slug}/",
                        question=question, context=context, label=label, status=status)


def prepare_from_file(path: Path, dataset_id: str, *, out_dir: Path = OUT_DIR,
                      question: str | None = None, context: str | None = None,
                      label: str | None = None, status: str = "closed") -> dict:
    """Matérialise JSONL canonique + descripteur depuis UN fichier arbitraire (csv/json).

    Point d'entrée du pipeline « fichier → analyse complète » (full_run) : mêmes
    heuristiques génériques que pour les consultations collectées.
    """
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"Fichier introuvable : {path}")
    fmt = scrape._tag_format(path.name)
    if fmt not in ("csv", "json", "json_zip"):
        raise SystemExit(f"Format non pris en charge : {path.name} "
                         "(attendu .csv, .json ou .json.zip)")
    records = _melt_file(path, fmt)
    print(f"  [ok  ] {path.name} : {len(records)} réponse(s) ouverte(s)")
    if not records:
        raise SystemExit(f"Aucune réponse ouverte trouvée dans {path.name}.")
    return _materialize(records, dataset_id, out_dir, source_desc=str(path),
                        question=question, context=context, label=label, status=status)


def _materialize(records: list[dict], slug: str, out_dir: Path, *, source_desc: str,
                 question: str | None, context: str | None,
                 label: str | None, status: str) -> dict:
    """Écrit `<slug>.jsonl` + `<slug>.descriptor.json` sous `out_dir`. Rend un résumé."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{slug}.jsonl"
    tmp = jsonl_path.with_suffix(".jsonl.part")
    with open(tmp, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(jsonl_path)

    descriptor = {
        "name": slug,
        "status": status,
        "format": "jsonl",
        "path": str(jsonl_path.resolve()),
        "columns": {"id": "id", "text": "text", "ts": "ts"},
        "props": {"question": "question", "topic": "topic", "source_file": "source_file"},
        "_note": f"Descripteur GÉNÉRÉ par pipeline.ingest_full depuis "
                 f"{source_desc} — ne pas éditer, régénérer.",
    }
    if label:
        descriptor["label"] = label
    if question:
        descriptor["question"] = question
    if context:
        descriptor["context"] = context
    descriptor_path = out_dir / f"{slug}.descriptor.json"
    descriptor_path.write_text(json.dumps(descriptor, ensure_ascii=False, indent=2),
                               encoding="utf-8")

    topics = {r["topic"] for r in records if r["topic"]}
    return {"n_records": len(records), "jsonl_path": jsonl_path,
            "descriptor_path": descriptor_path, "n_topics": len(topics)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Prépare une consultation collectée pour le pipeline Agora.")
    ap.add_argument("--slug", required=True, help="slug de la consultation collectée")
    ap.add_argument("--question", default=None, help="question posée (cadre l'extraction)")
    ap.add_argument("--context", default=None, help="contexte de la consultation")
    ap.add_argument("--label", default=None, help="libellé d'affichage (UI)")
    args = ap.parse_args(argv)

    summary = prepare(args.slug, question=args.question,
                      context=args.context, label=args.label)
    print(f"\n✓ {summary['jsonl_path']}  ({summary['n_records']} réponses, "
          f"{summary['n_topics']} fils)")
    print(f"✓ {summary['descriptor_path']}")
    print("\nÉtape suivante (pipeline standard, inchangé) :")
    print(f"  uv run python -m backend.build_cache --dataset {args.slug} \\")
    print(f"      --descriptor {summary['descriptor_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
