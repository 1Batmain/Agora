"""Run NOCTURNE — Grand Débat, VOLUME, les DEUX approches de macro-structure.

Compare, sur les 4 thèmes officiels (gold), au niveau AVIS :
  Approche A — couche PLATE au pic de modularité (le servi actuel).
  Approche B — couche FINE (γ élevé, redondante) + moteur d'abstraction B (profil ré-embeddé).

Question : laquelle retrouve le mieux les 4 domaines (ARI) ? B mérite-t-elle son coût ?
Résilient : écrit les résultats au fur et à mesure (`research/gd_nightly_results.json`), log
détaillé (`var/gd-nightly.log`). Chaque approche isolée en try/except.

    OMP_NUM_THREADS=4 MISTRAL_API_KEY=$(cat var/mistral.key) \
      uv run --extra embed-contender --extra faiss python -u research/gd_nightly.py [CAP_par_thème]
"""
from __future__ import annotations

import csv
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import adjusted_rand_score as ARI

from pipeline.cluster import abstraction as ab
from pipeline.cluster.adaptive import derive_defaults
from pipeline.cluster.knn import build_knn_graph, knn_search
from pipeline.cluster.layers import centre
from pipeline.cluster.leiden_cluster import run_leiden
from pipeline.cluster.mistral_client import chat
from pipeline.embed.embedder import embed

csv.field_size_limit(10_000_000)

THEMES = {
    "démocratie":   ("data/raw/gd_democratie.csv",   "renouer le lien"),
    "fiscalité":    ("data/raw/gd_fiscalite.csv",     "fiscalité plus juste"),
    "écologie":     ("data/raw/gd_ecologie.csv",      "apporter des réponses"),
    "organisation": ("data/raw/gd_organisation.csv",  "pensez-vous de l'organisation"),
}
K_GRAPH = 30
GAMMAS = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0]     # A = pic de modularité ; B = γ le plus fin
OUT = Path("research/gd_nightly_results.json")


def _log(msg):
    print(msg, flush=True)


def _sample(path, keyword, cap, min_chars=40):
    with open(path, encoding="utf-8") as fh:
        r = csv.reader(fh)
        header = next(r)
        col = next((i for i, h in enumerate(header) if keyword.lower() in h.lower()), None)
        out = []
        for row in r:
            if col is not None and col < len(row):
                t = row[col].strip()
                if len(t) >= min_chars:
                    out.append(t)
            if len(out) >= cap:
                break
    return out


def _composition(macro_of, gold):
    comp = []
    for m in sorted(set(macro_of.tolist())):
        mask = macro_of == m
        maj = np.bincount(gold[mask], minlength=len(THEMES))
        comp.append({"macro": int(m), "n": int(mask.sum()),
                     "dominante": list(THEMES)[int(maj.argmax())],
                     "purete": round(float(maj.max() / maj.sum()), 3)})
    return comp


def main(cap=40000):
    results = {}

    _log(f"[1] Échantillon (cap {cap}/thème)…")
    texts, gold = [], []
    for i, (theme, (path, kw)) in enumerate(THEMES.items()):
        s = _sample(path, kw, cap)
        texts += s
        gold += [i] * len(s)
        _log(f"    {theme:<13} {len(s)} avis")
    gold = np.array(gold)
    results["n_avis"] = len(texts)
    results["par_theme"] = {t: int((gold == i).sum()) for i, t in enumerate(THEMES)}
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    _log(f"[2] Embedding {len(texts)} avis (nomic, throttlé)…")
    V = centre(embed(texts, model_id="nomic-v2").astype(np.float64))
    V32 = V.astype(np.float32)

    _log("[3] Graphe fixe + balayage γ…")
    nb = knn_search(V32, K_GRAPH)
    dd = derive_defaults(V32, k=K_GRAPH, neighbors=nb)
    g = build_knn_graph(V, k=dd.k, threshold=dd.threshold, neighbors=nb)
    parts, curve = {}, []
    for gamma in GAMMAS:
        r = run_leiden(g, resolution=gamma, seed=42)
        parts[gamma] = np.asarray(r.membership)
        curve.append({"gamma": gamma, "n": len(set(r.membership)), "modularity": round(float(r.modularity), 4)})
        _log(f"    γ={gamma}  {curve[-1]['n']} thèmes  Q={curve[-1]['modularity']}")
    results["gamma_curve"] = curve
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    # ---- Approche A : pic de modularité ----
    try:
        peak = max(curve, key=lambda c: c["modularity"])["gamma"]
        pa = parts[peak]
        results["approche_A"] = {
            "description": "couche plate au pic de modularité",
            "gamma": peak, "n_themes": int(len(set(pa.tolist()))),
            "ARI_vs_4domaines": round(float(ARI(gold, pa)), 3),
            "composition": _composition(pa, gold),
        }
        _log(f"[A] pic γ={peak} → {results['approche_A']['n_themes']} thèmes · ARI={results['approche_A']['ARI_vs_4domaines']}")
    except Exception:
        results["approche_A"] = {"error": traceback.format_exc()}
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    # ---- Approche B : couche fine + abstraction ----
    try:
        fine_gamma = GAMMAS[-1]                       # le plus fin (redondance présente)
        pf = parts[fine_gamma]
        clusters = {}
        for idx, c in enumerate(pf.tolist()):
            clusters.setdefault(c, []).append(idx)
        cluster_ids = list(clusters.keys())
        cluster_texts = [[texts[i] for i in clusters[c][:20]] for c in cluster_ids]
        _log(f"[B] couche fine γ={fine_gamma} → {len(cluster_ids)} thèmes · abstraction (profils)…")
        res = ab.compute(cluster_texts, chat_fn=chat,
                         embed_fn=lambda t: embed(t, model_id="nomic-v2"), model="mistral-small-latest")
        if res is None:
            results["approche_B"] = {"description": "couche fine + abstraction", "resultat": "None (pas de macro)"}
        else:
            f2m = {cluster_ids[ti]: m for ti, m in enumerate(res["assign"])}
            macro_of = np.array([f2m[c] for c in pf.tolist()])
            results["approche_B"] = {
                "description": "couche fine + moteur d'abstraction B (profil ré-embeddé)",
                "fine_gamma": fine_gamma, "n_fine": len(cluster_ids),
                "n_macros": int(len(set(res["assign"]))),
                "ARI_macros_vs_4domaines": round(float(ARI(gold, macro_of)), 3),
                "ARI_fine_vs_4domaines": round(float(ARI(gold, pf)), 3),
                "composition": _composition(macro_of, gold),
                "profils_exemples": res["profiles"][:4],
            }
            _log(f"[B] {len(cluster_ids)} fins → {results['approche_B']['n_macros']} macros · "
                 f"ARI_macros={results['approche_B']['ARI_macros_vs_4domaines']} (fin={results['approche_B']['ARI_fine_vs_4domaines']})")
    except Exception:
        results["approche_B"] = {"error": traceback.format_exc()}
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    _log("=== FINI ===")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40000)
