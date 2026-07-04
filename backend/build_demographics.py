"""BUILD DEMOGRAPHICS — profil du panel (global + groupe MAJORITAIRE par thème),
persisté dans `analysis/demographics.json`. Pure JOINTURE : zéro LLM, zéro torch.

Source : un CSV « enrichi » portant des colonnes démographiques nommées (par défaut
`sexe`/`age`, cf. `scripts/characteristics_csv.py` qui les ajoute au CSV publié —
extraction réelle + complétion synthétique du sexe à usage de test). La jointure se
fait PAR LIGNE : les ids d'avis produits par `pipeline.ingest_full.prepare` embarquent
l'ordinal de ligne du fichier source (`…:<row>:<col>`), stable puisque le CSV enrichi
est une copie ligne-à-ligne du CSV brut.

Agrégats :
  - `global` : distribution de chaque axe sur les avis ANALYSÉS (jointure réussie) ;
  - `themes` : pour CHAQUE nœud de l'arbre (feuilles ET parents), la MAJORITÉ par axe
    `{label, n, share}` (share = part parmi les avis du thème AYANT RENSEIGNÉ l'axe).
    L'appartenance avis↔thème vient des fichiers `citations/<theme>.json` (qui listent
    TOUS les claims d'un nœud avec leur avis_id, parents inclus) — un avis est compté
    UNE fois par thème.

Artefact À PART et OPTIONNEL : ne touche à rien d'existant ; absent → les endpoints
et le front dégradent gracieusement (contrat de rétro-compat).

Usage :
    uv run python -m backend.build_demographics --dataset <id> \
        --csv data/<fichier>-brute.csv [--axes sexe,age]
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

from backend import analysis_store as store

DEFAULT_AXES = ("sexe", "age")

# `…:<row>:<col>` en fin d'id d'avis (convention pipeline.ingest_full.prepare).
_ROW_IN_ID = re.compile(r":(\d+):(\d+)$")


def _log(msg: str) -> None:
    print(f"[build_demographics] {msg}", flush=True)


def _row_of(avis_id: str) -> int | None:
    m = _ROW_IN_ID.search(avis_id)
    return int(m.group(1)) if m else None


def _sniff(path: Path) -> tuple[str, str]:
    head = path.open("rb").read(1 << 16)
    try:
        text = head.decode("utf-8")
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        text = head.decode("cp1252", errors="replace")
        encoding = "cp1252"
    first = text.splitlines()[0] if text else ""
    return encoding, max((";", ",", "\t"), key=first.count)


def _axis_values_by_row(csv_path: Path, axes: tuple[str, ...]) -> list[dict[str, str]]:
    """[{axe: valeur non vide}] par ligne de données (colonnes cherchées PAR NOM)."""
    encoding, delimiter = _sniff(csv_path)
    with csv_path.open(encoding=encoding, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader)
        idx = {axis: header.index(axis) for axis in axes if axis in header}
        missing = [a for a in axes if a not in idx]
        if missing:
            raise SystemExit(f"Colonne(s) absente(s) du CSV {csv_path.name} : {missing} "
                             f"(en-têtes : {header[-6:]})")
        out = []
        for row in reader:
            out.append({axis: row[i].strip() for axis, i in idx.items()
                        if i < len(row) and row[i].strip()})
    return out


def _majority(counts: Counter) -> dict | None:
    """{label, n, share} du groupe majoritaire — égalité tranchée par ordre alphabétique."""
    if not counts:
        return None
    total = sum(counts.values())
    label, n = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {"label": label, "n": n, "share": round(n / total, 3)}


def build_demographics(dataset: str, csv_path: Path,
                       axes: tuple[str, ...] = DEFAULT_AXES) -> dict:
    """Joint le CSV enrichi aux avis analysés, agrège, persiste. Rend le payload."""
    avis_all = store.read_avis_all(dataset)
    if not avis_all:
        raise SystemExit(f"Pas d'avis.json pour {dataset!r} — builder l'analyse d'abord.")
    analysis = store.read_analysis(dataset) or {}
    theme_ids = [t["id"] for t in analysis.get("themes", [])]

    by_row = _axis_values_by_row(Path(csv_path), axes)

    # Jointure avis → valeurs d'axes (par ordinal de ligne embarqué dans l'id).
    values_by_avis: dict[str, dict[str, str]] = {}
    for avis_id in avis_all:
        row = _row_of(avis_id)
        if row is not None and 0 <= row < len(by_row) and by_row[row]:
            values_by_avis[avis_id] = by_row[row]

    # Avis par thème (chaque nœud, parents inclus) via les fichiers citations —
    # ils listent TOUS les claims du nœud avec leur avis_id ; dédup par avis.
    avis_by_theme: dict[str, set[str]] = {}
    for tid in theme_ids:
        citations = store.read_citations(dataset, tid) or []
        members = {c.get("avis_id") for c in citations} & values_by_avis.keys()
        if members:
            avis_by_theme[tid] = members

    def _distributions(avis_ids) -> dict[str, Counter]:
        dist: dict[str, Counter] = {axis: Counter() for axis in axes}
        for aid in avis_ids:
            for axis, value in values_by_avis[aid].items():
                dist[axis][value] += 1
        return dist

    # GLOBAL = toutes les CONTRIBUTIONS du CSV (le panel décrit qui a répondu à la
    # consultation, doublons regroupés inclus) — pas seulement les avis analysés.
    global_dist: dict[str, Counter] = {axis: Counter() for axis in axes}
    for row_values in by_row:
        for axis, value in row_values.items():
            global_dist[axis][value] += 1
    themes = []
    for tid in sorted(avis_by_theme):
        dist = _distributions(avis_by_theme[tid])
        majority = {axis: m for axis in axes if (m := _majority(dist[axis]))}
        themes.append({
            "theme_id": tid,
            "n_avis": len(avis_by_theme[tid]),
            "majority": majority,
            "counts": {axis: dict(dist[axis].most_common()) for axis in axes},
        })

    payload = {
        "dataset": dataset,
        "axes": list(axes),
        "csv": Path(csv_path).name,
        "n_contributions": len(by_row),
        "n_avis_matched": len(values_by_avis),
        "n_avis_total": len(avis_all),
        "global": {axis: dict(global_dist[axis].most_common()) for axis in axes},
        "themes": themes,
        "_note": "Jointure par ligne avec le CSV enrichi (cf. scripts/characteristics_csv.py "
                 "— le sexe peut inclure une complétion synthétique à usage de test).",
    }
    store.write_demographics(dataset, payload)
    _log(f"{dataset} · ✓ demographics.json écrit · {len(values_by_avis)}/{len(avis_all)} "
         f"avis joints · {len(themes)} thèmes")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bake le profil démographique du panel (global + majorités par thème).")
    ap.add_argument("--dataset", required=True, help="id du dataset (sous backend/cache/)")
    ap.add_argument("--csv", required=True, type=Path,
                    help="CSV enrichi (colonnes démographiques nommées, ex. sexe/age)")
    ap.add_argument("--axes", default=",".join(DEFAULT_AXES),
                    help="axes = noms de colonnes du CSV (défaut : sexe,age)")
    args = ap.parse_args()
    axes = tuple(a.strip() for a in args.axes.split(",") if a.strip())
    build_demographics(args.dataset, args.csv, axes=axes)


if __name__ == "__main__":
    main()
