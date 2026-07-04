"""CLI du collecteur — `python -m pipeline.collect {run,catalog,status}`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config, load, store


def _add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--db", type=Path, default=config.DB_PATH,
                    help=f"chemin de la base DuckDB (défaut : {config.DB_PATH})")


def _print_summary(summary: dict) -> None:
    by = ", ".join(f"{k}={v}" for k, v in sorted(summary["by_status"].items()))
    print(f"\n{summary['consultations']} consultation(s) — {by or 'aucune'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m pipeline.collect",
        description="Collecte open data des consultations citoyennes AN → DuckDB.")
    sub = ap.add_subparsers(dest="command", required=True)

    run_ap = sub.add_parser("run", help="scrape + télécharge + charge dans DuckDB")
    _add_common(run_ap)
    run_ap.add_argument("--only", help="ne traiter qu'un slug de consultation")
    run_ap.add_argument("--limit", type=int, help="ne traiter que N consultations")
    run_ap.add_argument("--force-download", action="store_true",
                        help="re-télécharger même si présent dans data/collect/raw/")
    run_ap.add_argument("--full-melt", action="store_true",
                        help="fondre TOUTES les colonnes non vides (défaut : texte libre)")
    run_ap.add_argument("--strip-pii", action="store_true",
                        help="masquer emails/téléphones/URLs dans les réponses")

    cat_ap = sub.add_parser("catalog", help="scrape seul : écrit consultations+files")
    _add_common(cat_ap)
    cat_ap.add_argument("--limit", type=int)

    st_ap = sub.add_parser("status", help="affiche le catalogue depuis la base")
    _add_common(st_ap)

    args = ap.parse_args(argv)

    if args.command in ("run", "catalog"):
        summary = load.run(
            db_path=args.db,
            only=getattr(args, "only", None),
            limit=args.limit,
            force_download=getattr(args, "force_download", False),
            full_melt=getattr(args, "full_melt", False),
            strip_pii=getattr(args, "strip_pii", False),
            catalog_only=(args.command == "catalog"),
        )
        _print_summary(summary)
        return 0

    # status
    if not args.db.exists():
        print(f"pas de base : {args.db} (lancer `run` d'abord)", file=sys.stderr)
        return 1
    with store.connect(args.db, read_only=True) as con:
        rows = con.execute(
            "SELECT slug, status, n_files, n_files_ingested, n_answers, "
            "coalesce(status_detail, '') FROM consultations ORDER BY slug").fetchall()
    for slug, status, n_files, n_ing, n_answers, detail in rows:
        print(f"{status:>15}  {slug:<55} {n_ing}/{n_files} fichier(s)  "
              f"{n_answers:>8} réponse(s)  {detail}")
    print(f"\n{len(rows)} consultation(s) au catalogue — base : {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
