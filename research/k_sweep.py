"""VERDICT — balayage de `k` (voisins k-NN) : modularité + alignement taxo officielle.

R&D pur (aucun fichier produit modifié). Pour CHAQUE dataset caché, on balaie le
nombre de voisins `k` du graphe k-NN et on RÉUTILISE telle quelle la brique de
production `backend.live_cluster.build_live_tree(ideas, vecs, weights, k=k)` — le
même chemin que le levier de la Console. Pour chaque `k` on mesure :

  - `n_macros`, `n_leaves`            : la GRANULARITÉ (gros k → moins de thèmes ?).
  - `modularity` (Leiden racine)      : la QUALITÉ INTRINSÈQUE du partitionnement
                                        (Q sur le graphe k-NN, via igraph `.modularity`).

Pour `republique-numerique` UNIQUEMENT — alignement à la TAXO OFFICIELLE (le plan
du projet de loi : Titre I/II/III = 3 axes). La vérité-terrain par contribution est
la colonne `Catégorie` du CSV source data.gouv.fr (code de section du projet de loi
sur les Proposition/Modification ; pour les Argument, la section est héritée du parent
via la colonne `Lié.à..`). On joint le cache au CSV par (Identifiant + meilleur match
texte) puis on compare le clustering au niveau MACRO aux axes via ARI / NMI / V-measure.

Question tranchée : un grand `k` gagne-t-il OBJECTIVEMENT (modularité ET alignement),
ou seulement un découpage plus GROSSIER (moins de thèmes, sans gain de qualité) ?

Lancer :
    uv run --extra contender --extra embed-contender --extra faiss --with fastapi \
        python research/k_sweep.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

# Racine du dépôt sur le path (lancement `python research/k_sweep.py` depuis la racine).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Réutilisation EXPLICITE des briques de production (cf. brief) — aucun re-embed, zéro LLM.
from backend.live_cluster import build_live_tree
from backend.recluster import load_cache
from pipeline.cluster.adaptive import derive_defaults, derive_k
from pipeline.cluster.knn import build_knn_graph, knn_search
from pipeline.cluster.leiden_cluster import DEFAULT_SEED, run_leiden

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
)

DATASETS = ["tiktok", "granddebat", "xstance", "republique-numerique"]
K_GRID = [8, 12, 16, 24, 36, 50, 80, 120, 200]
SEED = DEFAULT_SEED

# CSV source officiel (open data Etalab / data.gouv.fr) — porte la colonne `Catégorie`
# = code de section du projet de loi (notre gold). Caché en scratchpad (≈31 Mo) ; jamais
# committé. Cf. descripteur `pipeline/ingest/descriptors/republique-numerique.json`.
REPNUM_CSV_URL = (
    "https://static.data.gouv.fr/resources/"
    "consultation-sur-le-projet-de-loi-republique-numerique/"
    "20151218-191444/projet-de-loi-numerique-consultation-anonyme.csv"
)
_SCRATCH = Path("/tmp/agora-k-sweep")
REPNUM_CSV_CANDIDATES = [
    Path("/tmp/repnum.csv"),
    _SCRATCH / "repnum_consultation.csv",
    Path("data/raw/republique_numerique_consultation.csv"),
]


# --------------------------------------------------------------------------- #
# Modularité racine (qualité intrinsèque) — réplique EXACTE du graphe racine de
# `build_live_tree` : k → voisinage → seuil dérivé(k) → graphe → Leiden.
# --------------------------------------------------------------------------- #
def root_modularity(vecs: np.ndarray, k: int) -> tuple[float, int, float, float]:
    """Modularité Leiden au niveau RACINE pour un `k` donné (+ n_fine, seuil, degré moyen).

    Reproduit pas-à-pas les lignes RACINE de `build_live_tree` (knn_search →
    derive_defaults(k) → build_knn_graph(threshold dérivé) → run_leiden), donc le `Q`
    rapporté est EXACTEMENT celui de la partition fine que l'arbre coarsene ensuite.
    """
    n = vecs.shape[0]
    k = max(2, min(int(k), n - 1))
    v32 = np.ascontiguousarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(v32, axis=1, keepdims=True)
    v32 = v32 / np.where(norms > 0, norms, 1.0)
    vecs64 = v32.astype(np.float64)
    neighbors = knn_search(v32, k)
    derived = derive_defaults(v32, k=k, neighbors=neighbors)
    thr = float(derived.threshold)
    graph = build_knn_graph(vecs64, k=k, threshold=thr, neighbors=neighbors)
    res = run_leiden(graph, resolution=1.0, seed=SEED)
    return res.modularity, res.n_clusters, thr, graph.avg_degree


# --------------------------------------------------------------------------- #
# Granularité + assignation macro par idée — via `build_live_tree` (chemin Console).
# --------------------------------------------------------------------------- #
def tree_stats(ideas, vecs, weights, k: int):
    """Construit l'arbre live au `k` donné → (n_macros, n_leaves, macro_of[idée])."""
    tree = build_live_tree(ideas, vecs, weights, k=k, seed=SEED)
    n_leaves = sum(1 for nd in tree.nodes.values() if not nd.children)
    n = len(ideas)
    macro_of = [None] * n
    for mid in tree.macros:
        for i in tree.nodes[mid].members:
            macro_of[i] = mid
    return len(tree.macros), n_leaves, macro_of


# --------------------------------------------------------------------------- #
# Gold officiel repnum — axes du projet de loi (Titre I/II/III) par contribution.
# --------------------------------------------------------------------------- #
def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _resolve_csv() -> Path:
    for p in REPNUM_CSV_CANDIDATES:
        if p.exists():
            return p
    dest = REPNUM_CSV_CANDIDATES[1]
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [repnum] téléchargement du CSV officiel → {dest} …", flush=True)
    urllib.request.urlretrieve(REPNUM_CSV_URL, dest)
    return dest


def _titre(section: str | None) -> str | None:
    """Réduit un code de section (« TITRE Ier - Chapitre … ») à son AXE (le Titre)."""
    m = re.match(r"(TITRE\s+\w+)", section or "")
    return m.group(1) if m else None


def repnum_gold_axes(ideas) -> list[str | None]:
    """Étiquette d'AXE officiel (Titre I/II/III) par idée cachée, alignée à l'ordre `ideas`.

    Jointure cache→CSV : `id = source:Identifiant`. L'Identifiant est unique PAR type
    mais réutilisé entre types (une Proposition et un Argument peuvent le partager) →
    on lève l'ambiguïté par le MEILLEUR match texte (`difflib`) sur les candidats du
    même Identifiant. Section gold :
      - Proposition / Modification → leur `Catégorie` (code de section).
      - Argument                   → section HÉRITÉE du parent (« Lié.à.. »).
    Puis section → axe (Titre). Renvoie `None` si non résolvable (rare).
    """
    csv.field_size_limit(10 ** 7)
    path = _resolve_csv()
    rows: list[dict] = []
    with open(path, encoding="utf-8", newline="") as fh:
        rows.extend(csv.DictReader(fh))

    by_ident: dict[str, list[dict]] = defaultdict(list)
    sec_of: dict[tuple[str, str], str] = {}      # (type, ident) → section, pour les parents
    for r in rows:
        t = r["Type.de.contenu"]
        ident = r["Identifiant"].strip()
        by_ident[ident].append(r)
        if t in ("Proposition", "Modification") and r["Catégorie"].startswith("TITRE"):
            sec_of[(t, ident)] = r["Catégorie"].strip()

    def parent_section(r: dict) -> str | None:
        m = re.match(r'\s*(\w+)\s+"?(\d+)"?', r.get("Lié.à..", "") or "")
        return sec_of.get((m.group(1), m.group(2))) if m else None

    def row_section(r: dict) -> str | None:
        t = r["Type.de.contenu"]
        if t in ("Proposition", "Modification"):
            return r["Catégorie"].strip() if r["Catégorie"].startswith("TITRE") else None
        if t == "Argument":
            return parent_section(r)
        return None

    gold: list[str | None] = []
    for idea in ideas:
        ident = idea.id.split(":", 1)[1] if ":" in idea.id else idea.id
        text = _norm(getattr(idea, "text_clean", None) or idea.text)[:300]
        cands = by_ident.get(ident, [])
        if not cands:
            gold.append(None)
            continue
        best = cands[0] if len(cands) == 1 else max(
            cands, key=lambda r: SequenceMatcher(None, _norm(r["Contenu"])[:300], text).ratio()
        )
        gold.append(_titre(row_section(best)))
    return gold


def alignment(macro_of: list, gold: list[str | None]) -> dict | None:
    """ARI / NMI / V-measure entre l'assignation macro et l'axe officiel (paires gold∧macro)."""
    pairs = [(m, g) for m, g in zip(macro_of, gold) if m is not None and g is not None]
    if len(pairs) < 2:
        return None
    pred = [m for m, _ in pairs]
    true = [g for _, g in pairs]
    return {
        "n": len(pairs),
        "ari": round(adjusted_rand_score(true, pred), 4),
        "nmi": round(normalized_mutual_info_score(true, pred), 4),
        "v": round(v_measure_score(true, pred), 4),
    }


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
def sweep_dataset(dataset: str) -> dict:
    print(f"\n=== {dataset} ===", flush=True)
    ideas, vecs, weights = load_cache(dataset)
    n = len(ideas)
    k_default = derive_k(n)
    print(f"  N={n}  derive_k(N)={k_default}", flush=True)

    gold = None
    if dataset == "republique-numerique":
        gold = repnum_gold_axes(ideas)
        cov = Counter(g for g in gold if g is not None)
        print(f"  gold axes (Titre): {dict(cov)}  | sans-gold={sum(g is None for g in gold)}",
              flush=True)

    # `k` candidats : la grille + le défaut dérivé (dédupliqué, trié, ≤ n-1).
    ks = sorted({k for k in (K_GRID + [k_default]) if 2 <= k <= n - 1})
    rows = []
    for k in ks:
        mod, n_fine, thr, deg = root_modularity(vecs, k)
        n_macros, n_leaves, macro_of = tree_stats(ideas, vecs, weights, k)
        row = {
            "k": k,
            "is_default": k == k_default,
            "threshold": round(thr, 4),
            "avg_degree": round(deg, 2),
            "n_fine_root": n_fine,
            "n_macros": n_macros,
            "n_leaves": n_leaves,
            "modularity": round(mod, 4),
        }
        if gold is not None:
            row["align"] = alignment(macro_of, gold)
        tag = " *défaut*" if row["is_default"] else ""
        extra = ""
        if row.get("align"):
            a = row["align"]
            extra = f"  ARI={a['ari']:.3f} NMI={a['nmi']:.3f} V={a['v']:.3f}"
        print(f"  k={k:>3}{tag:9}  thr={thr:.3f}  deg={deg:5.1f}  "
              f"macros={n_macros:>3}  leaves={n_leaves:>4}  Q={mod:.4f}{extra}", flush=True)
        rows.append(row)
    return {"dataset": dataset, "n": n, "k_default": k_default, "rows": rows}


def main():
    results = [sweep_dataset(ds) for ds in DATASETS]
    out = _SCRATCH / "k_sweep_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRésultats JSON → {out}", flush=True)

    # Résumé : k argmax par métrique et par dataset.
    print("\n=== ARGMAX par métrique ===", flush=True)
    for r in results:
        rows = r["rows"]
        best_q = max(rows, key=lambda x: x["modularity"])
        line = (f"{r['dataset']:>22}: max Q @ k={best_q['k']} (Q={best_q['modularity']:.4f}, "
                f"macros={best_q['n_macros']})")
        if rows and rows[0].get("align"):
            best_ari = max(rows, key=lambda x: x["align"]["ari"])
            best_nmi = max(rows, key=lambda x: x["align"]["nmi"])
            best_v = max(rows, key=lambda x: x["align"]["v"])
            line += (f" | max ARI @ k={best_ari['k']} ({best_ari['align']['ari']:.3f})"
                     f" | max NMI @ k={best_nmi['k']} ({best_nmi['align']['nmi']:.3f})"
                     f" | max V @ k={best_v['k']} ({best_v['align']['v']:.3f})")
        print(line, flush=True)


if __name__ == "__main__":
    sys.exit(main())
