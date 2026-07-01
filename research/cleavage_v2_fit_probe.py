"""Sonde : le fit cosinus-vs-CENTROÏDE discrimine-t-il mal parce que le centroïde est
dominé par la facette la plus BRUYANTE ? On teste une alternative — cosinus
proposition↔TITRE (le sujet déclaré du thème) — sur les mêmes lignes que cleavage_v2.

Lecture seule (réutilise research/cleavage_v2_results.json). Sortie : tableau imprimé.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from backend.analysis import DEFAULT_EMBEDDER
from pipeline.claims.pipeline import embed_claim_texts

RES = Path(__file__).resolve().parent / "cleavage_v2_results.json"


def main() -> None:
    d = json.loads(RES.read_text(encoding="utf-8"))
    rows = d["rows"]
    titles = [r["title"] for r in rows]
    v1 = [r["v1_objet"] for r in rows]
    v2 = [r["v2_objet"] for r in rows]
    T = embed_claim_texts(titles, embedder=DEFAULT_EMBEDDER)
    A = embed_claim_texts(v1, embedder=DEFAULT_EMBEDDER)
    B = embed_claim_texts(v2, embedder=DEFAULT_EMBEDDER)

    print(f"{'id':4} {'cFitv1':>6} {'cFitv2':>6} | {'tAlnv1':>6} {'tAlnv2':>6}  title")
    c1b = c2b = t1b = t2b = 0
    for i, r in enumerate(rows):
        ca1, ca2 = r["fit_v1"], r["fit_v2"]                  # cos vs centroïde (déjà calculé)
        ta1 = float(np.dot(A[i], T[i]))                      # cos proposition v1 ↔ titre
        ta2 = float(np.dot(B[i], T[i]))                      # cos proposition v2 ↔ titre
        c2b += ca2 > ca1; t2b += ta2 > ta1
        print(f"{r['theme_id']:4} {ca1:>6.3f} {ca2:>6.3f} | {ta1:>6.3f} {ta2:>6.3f}  {r['title']}")
    n = len(rows)
    print(f"\nv2 meilleure (centroïde-fit) : {c2b}/{n}")
    print(f"v2 meilleure (titre-align)   : {t2b}/{n}")
    # n0 = cas de Bob : v2 doit gagner si la métrique capte la centralité.
    r0 = next(r for r in rows if r["theme_id"] == "n0")
    print(f"\nn0 (cas Bob) centroïde-fit : v1={r0['fit_v1']} v2={r0['fit_v2']}  "
          f"→ {'v2 mieux' if r0['fit_v2']>r0['fit_v1'] else 'v1 mieux (métrique TROMPE)'}")
    i0 = rows.index(r0)
    ta1_0, ta2_0 = float(np.dot(A[i0], T[i0])), float(np.dot(B[i0], T[i0]))
    print(f"n0 (cas Bob) titre-align   : v1={ta1_0:.3f} v2={ta2_0:.3f}  "
          f"→ {'v2 mieux' if ta2_0>ta1_0 else 'v1 mieux'}")


if __name__ == "__main__":
    main()
