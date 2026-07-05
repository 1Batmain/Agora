"""BUILD : compile la provenance d'un dataset en un `analysis.duckdb` de LECTURE.

Le hot path `/avis_list` scanne aujourd'hui `avis.json` en Python et replie l'Unicode
(NFD + casefold) de CHAQUE avis à CHAQUE requête (audit code #1 : O(N) + fold par requête).
Ce script bake une base DuckDB *dérivée* (un index de LECTURE, jamais une source de vérité)
pour que le filtrage thème/stance/recherche + la pagination se fassent en SQL indexé.

Disposition (à côté des autres artefacts SERVE) :

    backend/cache/<dataset>/analysis/analysis.duckdb

Tables :
  * `avis`   (rank, key, avis_id, text, text_fold, lang, payload) — `payload` = l'entrée
             `avis.json` VERBATIM (JSON), pour reconstruire l'item servi À L'IDENTIQUE ;
             `text_fold` = `avis._fold(text)` précalculé (recherche sous-chaîne insensible
             casse/accents en SQL, MÊME sémantique que le fallback, cf. `contains()`).
  * `claims` (avis_rank, claim_id, cluster_id, leaf_id, filter_theme, stance) — une ligne
             par claim ; `filter_theme` = `leaf_id or cluster_id` (la clé du filtre par
             sous-arbre), `stance` jointe depuis `claim_stance.json` au BUILD.
  * `themes` (id, parent_id, title, color) — hiérarchie (self-contained : bake autonome).

La base est un CACHE : elle se reconstruit et le serve dégrade gracieusement en son absence
(ou si elle est plus vieille que `avis.json`/`claim_stance.json`, cf. `analysis_store`).

Usage :
    uv run --extra collect --extra serve python -m backend.bake_duckdb <dataset> [<dataset>…]
    uv run … python -m backend.bake_duckdb --all       # tous les datasets `ready`
"""
from __future__ import annotations

import json
import os
import sys

import duckdb

from backend import analysis_store
from backend.analysis_store import duckdb_path      # SSOT du chemin `<ds>/analysis/analysis.duckdb`
from backend.avis import _fold

# Colonnes insérées (ordre = ordre des `?` dans les INSERT).
_AVIS_COLS = ("rank", "key", "avis_id", "text", "text_fold", "lang", "payload")
_CLAIM_COLS = ("avis_rank", "claim_id", "cluster_id", "leaf_id", "filter_theme", "stance")
_THEME_COLS = ("id", "parent_id", "title", "color")


def _schema(con: duckdb.DuckDBPyConnection) -> None:
    """(Re)crée un schéma vierge — bake idempotent (DROP puis CREATE)."""
    con.execute("DROP TABLE IF EXISTS claims")
    con.execute("DROP TABLE IF EXISTS avis")
    con.execute("DROP TABLE IF EXISTS themes")
    con.execute("DROP TABLE IF EXISTS meta")
    con.execute(
        "CREATE TABLE avis ("
        "  rank      INTEGER,"
        "  key       TEXT,"
        "  avis_id   TEXT,"
        "  text      TEXT,"
        "  text_fold TEXT,"
        "  lang      TEXT,"
        "  payload   TEXT"          # entrée avis.json verbatim (json.dumps)
        ")")
    con.execute(
        "CREATE TABLE claims ("
        "  avis_rank    INTEGER,"
        "  claim_id     TEXT,"
        "  cluster_id   TEXT,"
        "  leaf_id      TEXT,"
        "  filter_theme TEXT,"
        "  stance       TEXT"
        ")")
    con.execute(
        "CREATE TABLE themes ("
        "  id TEXT, parent_id TEXT, title TEXT, color TEXT)")
    # Signature des sources (taille en octets par fichier) : l'index est VALIDE tant que
    # `avis.json`/`claim_stance.json` ont la même taille qu'au bake. La taille est un contenu
    # (déterministe au `git checkout`), là où le mtime ne l'est PAS — indispensable pour un
    # `.duckdb` COMMITÉ servi en prod après `reset --hard` (cf. analysis_store.avis_duckdb_con).
    con.execute("CREATE TABLE meta (source TEXT, n_bytes BIGINT)")


def write_meta(con: duckdb.DuckDBPyConnection, signature: list[tuple[str, int]]) -> None:
    """Enregistre la signature (source, taille) des fichiers dont dérive l'index."""
    con.execute("DELETE FROM meta")
    if signature:
        con.executemany("INSERT INTO meta (source, n_bytes) VALUES (?, ?)",
                        [list(row) for row in signature])


def _fill(con: duckdb.DuckDBPyConnection, avis_data: dict, claim_stance: dict | None,
          themes: list[dict]) -> tuple[int, int]:
    """Charge avis + claims + themes. Renvoie (n_avis, n_claims)."""
    stance = claim_stance or {}
    avis_rows: list[list] = []
    claim_rows: list[list] = []
    rank = 0
    for key, entry in avis_data.items():
        if not isinstance(entry, dict):        # même garde que le fallback (avis.avis_list)
            continue
        text = entry.get("text") or ""
        avis_rows.append([
            rank, key, entry.get("id", key), text, _fold(text),
            entry.get("lang", "fr"), json.dumps(entry, ensure_ascii=False)])
        for c in entry.get("claims") or []:
            filter_theme = c.get("leaf_id") or c.get("cluster_id")
            cid = c.get("id")
            claim_rows.append([
                rank, cid, c.get("cluster_id"), c.get("leaf_id"), filter_theme,
                (stance.get(cid) or {}).get("stance")])
        rank += 1

    if avis_rows:
        con.executemany(
            f"INSERT INTO avis ({', '.join(_AVIS_COLS)}) "
            f"VALUES ({', '.join('?' for _ in _AVIS_COLS)})", avis_rows)
    if claim_rows:
        con.executemany(
            f"INSERT INTO claims ({', '.join(_CLAIM_COLS)}) "
            f"VALUES ({', '.join('?' for _ in _CLAIM_COLS)})", claim_rows)
    theme_rows = [[t.get("id"), t.get("parent_id"), t.get("title"), t.get("color")]
                  for t in (themes or []) if t.get("id")]
    if theme_rows:
        con.executemany(
            f"INSERT INTO themes ({', '.join(_THEME_COLS)}) "
            f"VALUES ({', '.join('?' for _ in _THEME_COLS)})", theme_rows)
    return len(avis_rows), len(claim_rows)


def _index(con: duckdb.DuckDBPyConnection) -> None:
    """Index de filtrage (thème/stance) + pagination."""
    con.execute("CREATE INDEX ix_avis_rank ON avis(rank)")
    con.execute("CREATE INDEX ix_claims_rank ON claims(avis_rank)")
    con.execute("CREATE INDEX ix_claims_theme ON claims(filter_theme)")
    con.execute("CREATE INDEX ix_claims_stance ON claims(stance)")


def _try_fts(con: duckdb.DuckDBPyConnection) -> bool:
    """Best-effort : index plein-texte BM25 sur `text_fold` (extension `fts`).

    Demandé au cahier des charges, mais NON utilisé sur le hot path : la recherche `q`
    reste une sous-chaîne foldée (`contains`) pour une PARITÉ exacte avec le fallback (BM25
    est tokenisé → sémantique différente). L'index est donc bâti comme capacité disponible
    (recherche classée future) quand l'extension charge ; ignoré silencieusement sinon
    (offline/CI sans réseau) — la base reste pleinement fonctionnelle.
    """
    try:
        con.execute("INSTALL fts")
        con.execute("LOAD fts")
        con.execute("PRAGMA create_fts_index('avis', 'rank', 'text_fold', overwrite=1)")
        return True
    except Exception:
        return False


def build_db(con: duckdb.DuckDBPyConnection, avis_data: dict,
             claim_stance: dict | None = None,
             themes: list[dict] | None = None) -> dict:
    """Construit tout le schéma dans `con` (dataset-agnostique — testable en isolation)."""
    _schema(con)
    n_avis, n_claims = _fill(con, avis_data, claim_stance, themes or [])
    _index(con)
    fts = _try_fts(con)
    return {"n_avis": n_avis, "n_claims": n_claims, "fts": fts}


def bake(dataset: str) -> dict:
    """Lit les artefacts SERVE d'un dataset et (re)génère son `analysis.duckdb`.

    Écriture ATOMIQUE : on bake dans un fichier temporaire voisin puis `os.replace`, pour
    qu'un SERVE concurrent ne lise jamais une base à moitié écrite.
    """
    avis_data = analysis_store.read_avis_all(dataset)
    if avis_data is None:
        raise SystemExit(f"{dataset}: pas de avis.json (analyse non bâtie) — rien à baker.")
    claim_stance = analysis_store.read_claim_stance(dataset)
    themes = (analysis_store.read_analysis(dataset) or {}).get("themes", [])

    dst = duckdb_path(dataset)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".duckdb.tmp")
    if tmp.exists():
        tmp.unlink()
    con = duckdb.connect(str(tmp))
    try:
        stats = build_db(con, avis_data, claim_stance, themes)
        write_meta(con, analysis_store.avis_sources_sig(dataset))
    finally:
        con.close()
    os.replace(tmp, dst)
    stats["path"] = str(dst)
    return stats


def _main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "--all":
        from backend.recluster import list_datasets
        targets = [d for d in list_datasets()
                   if analysis_store.state(d) == analysis_store.READY]
    else:
        targets = argv
    for ds in targets:
        stats = bake(ds)
        print(f"[bake] {ds}: {stats['n_avis']} avis, {stats['n_claims']} claims, "
              f"fts={'oui' if stats['fts'] else 'non'} → {stats['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
