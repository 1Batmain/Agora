"""Banc x-stance : gold à DEUX niveaux, pour valider la géométrie sans dépenser un centime.

x-stance (ZurichNLP) porte trois étiquettes humaines, produites hors du projet :
    topic       (12 valeurs)   → vérité terrain des MACROS
    question_id (191 valeurs)  → vérité terrain des SOUS-THÈMES
    label       (FAVOR/AGAINST)→ vérité terrain de la STANCE

Le pipeline clusterise des CLAIMS (extraction LLM, payante). Ce banc clusterise des AVIS
(embeddings locaux, gratuits). Il faut donc D'ABORD vérifier que la conclusion tirée au
niveau avis TIENT au niveau claims — sur les 3000 avis où l'on possède les deux.

Usage :
    python research/xstance_bench.py transfert          # avis vs claims, sur `xstance`
    python research/xstance_bench.py courbe [dataset]   # ARI vs N, brut vs centré
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # racine du dépôt

import numpy as np
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI

from pipeline.cluster.adaptive import derive_defaults, derive_k
from pipeline.cluster.knn import build_knn_graph, knn_search
from pipeline.cluster.leiden_cluster import run_leiden


def centre(V: np.ndarray) -> np.ndarray:
    X = V.astype(np.float64) - V.astype(np.float64).mean(axis=0)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return np.ascontiguousarray(X)


def cluster(V: np.ndarray, resolution: float = 1.0, seed: int = 42) -> np.ndarray:
    V = np.ascontiguousarray(V.astype(np.float64))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    V32 = V.astype(np.float32)
    k = derive_k(len(V))
    nb = knn_search(V32, k)
    dd = derive_defaults(V32, k=k, neighbors=nb)
    g = build_knn_graph(V, k=dd.k, threshold=dd.threshold, neighbors=nb)
    return np.array(run_leiden(g, resolution=resolution, seed=seed).membership)


def load_avis(ds: str):
    """Vecteurs d'AVIS + les deux golds, alignés."""
    V = np.load(f"backend/cache/{ds}/embeddings.npy")
    topics, questions = [], []
    for line in open(f"backend/cache/{ds}/ideas.jsonl"):
        p = json.loads(line)["props"]
        topics.append(p.get("topic"))
        questions.append(p.get("question_id"))
    return V, np.array(topics), np.array(questions)


def load_claims(ds: str):
    """Vecteurs de CLAIMS + golds hérités de leur avis (même ordre que `_flatten`)."""
    V = np.load(f"backend/cache/{ds}/claims_emb.npz")["vecs"]
    claims = json.loads(Path(f"backend/cache/{ds}/claims.json").read_text())["claims"]
    topics, questions = [], []
    for line in open(f"backend/cache/{ds}/ideas.jsonl"):
        d = json.loads(line)
        for _ in claims.get(d["id"], []):
            topics.append(d["props"].get("topic"))
            questions.append(d["props"].get("question_id"))
    return V, np.array(topics), np.array(questions)


def _row(nom, lab, topics, questions):
    # ⚠ ARI contre 191 questions avec ~18 communautés est PLAFONNÉ par construction.
    # Le NMI est l'indicateur lisible à ce niveau ; l'ARI n'y sert qu'en RELATIF.
    return (f"{nom:<22}{len(set(lab)):>7}{ARI(topics, lab):>9.3f}{NMI(topics, lab):>8.3f}"
            f"{ARI(questions, lab):>10.3f}{NMI(questions, lab):>8.3f}")


def transfert(ds: str = "xstance") -> None:
    """La conclusion « recentrer aide » tient-elle au niveau AVIS comme au niveau CLAIMS ?"""
    hdr = f"{'unité / espace':<22}{'comm.':>7}{'ARI_top':>9}{'NMI':>8}{'ARI_ques':>10}{'NMI':>8}"
    print(f"### transfert avis ↔ claims ({ds})\n")
    print(hdr)
    print("-" * len(hdr))
    for unite, loader in (("avis", load_avis), ("claims", load_claims)):
        V, top, ques = loader(ds)
        for espace, X in (("brut", V), ("centré", centre(V))):
            print(_row(f"{unite} · {espace}", cluster(X), top, ques))
    print("\nLecture : si le SENS du gain (brut → centré) est le même sur les deux unités,")
    print("le banc au niveau avis (gratuit) est un proxy valide du niveau claims (payant).")


def courbe(ds: str = "xstance-large") -> None:
    """ARI vs N — les points dont T-N9 a besoin pour dériver des LOIS au lieu de constantes."""
    V, top, ques = load_avis(ds)
    n = len(V)
    tailles = [t for t in (1000, 2000, 4000, 8000, 16000, n) if t <= n]
    rng = np.random.default_rng(42)
    hdr = (f"{'N':>7}{'espace':>9}{'k':>5}{'comm.':>7}{'ARI_top':>9}{'ARI_ques':>10}"
           f"{'NMI_top':>9}")
    print(f"### courbe N ({ds}, {n} avis, gold : {len(set(top))} topics / "
          f"{len(set(ques))} questions)\n")
    print(hdr)
    print("-" * len(hdr))
    for t in tailles:
        idx = rng.choice(n, t, replace=False) if t < n else np.arange(n)
        for espace, X in (("brut", V[idx]), ("centré", centre(V[idx]))):
            lab = cluster(X)
            print(f"{t:>7}{espace:>9}{derive_k(t):>5}{len(set(lab)):>7}"
                  f"{ARI(top[idx], lab):>9.3f}{ARI(ques[idx], lab):>10.3f}"
                  f"{NMI(top[idx], lab):>9.3f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "transfert"
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    (transfert(arg or "xstance") if mode == "transfert" else courbe(arg or "xstance-large"))
