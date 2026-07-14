"""NAMING CONTRASTIF — avant/après des TITRES LLM (lane nommage).

Contexte : `research/cluster_merge_note.md` (verdict NON à la fusion) attribue le
symptôme de « sous-consolidation » perçu par Bob à un ARTEFACT DE NOMMAGE : des
clusters DISTINCTS reçoivent des titres quasi-synonymes (n0 « Réseaux sociaux et
addiction » vs n265 « Boucle d'addiction aux applications »). Ce banc mesure le
remède : titrer chaque nœud par ce qui le DISTINGUE de son frère le plus proche.

Reconstruit l'arbre depuis les caches (zéro extraction, modèle épinglé) puis, pour
chaque nœud, génère le titre NON-CONTRASTIF (ancien schéma) et le titre CONTRASTIF
(nouveau), et affiche l'avant/après sur les paires proches + le coût LLM.

    MISTRAL_API_KEY=$(cat var/mistral.key) uv run --extra faiss \
        python research/naming_contrastif.py --dataset tiktok
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import titles  # noqa: E402
from backend.analysis import build_theme_tree  # noqa: E402
from backend.build_analysis import EXTRACT_MODEL, load_dataset  # noqa: E402
from pipeline.cluster import mistral_client  # noqa: E402


def _cache_model(dataset: str) -> str:
    from backend.recluster import dataset_dir
    rec = json.loads((dataset_dir(dataset) / "claims.json").read_text(encoding="utf-8"))
    return rec.get("model") or EXTRACT_MODEL


def _count_calls():
    """Wrappe mistral_client.chat pour COMPTER les appels LLM réels (cache miss)."""
    orig = mistral_client.chat
    state = {"n": 0}

    def counting(*a, **k):
        state["n"] += 1
        return orig(*a, **k)

    mistral_client.chat = counting
    return state, lambda: setattr(mistral_client, "chat", orig)


def closest_pairs(tree, neighbors, k=8):
    """Paires (a,b) frères mutuellement les plus proches, triées par sim décroissante."""
    seen, pairs = set(), []
    for a, b in neighbors.items():
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        ca = np.asarray(tree.nodes[a].centroid, dtype=np.float64)
        cb = np.asarray(tree.nodes[b].centroid, dtype=np.float64)
        pairs.append((float(ca @ cb), a, b))
    pairs.sort(reverse=True)
    return pairs[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tiktok")
    ap.add_argument("--pairs", type=int, default=8)
    args = ap.parse_args()

    model = _cache_model(args.dataset)
    ds = load_dataset(args.dataset)
    tree = build_theme_tree(ds, model=model)
    neighbors = titles.nearest_neighbor_map(tree)
    print(f"# dataset={args.dataset}  modèle={model}")
    print(f"# {len(tree.nodes)} nœuds, {len(tree.macros)} macros, "
          f"{len(neighbors)} avec voisin\n")

    state, restore = _count_calls()
    try:
        # AVANT : titre non-contrastif (ancien schéma). APRÈS : titre contrastif.
        before, after, nbid = {}, {}, {}
        for nid, node in tree.nodes.items():
            nb = tree.nodes.get(neighbors.get(nid))
            before[nid] = titles.title_for_node(args.dataset, node, model=model)
            after[nid] = titles.title_for_node(args.dataset, node, neighbor=nb, model=model)
            nbid[nid] = nb.id if nb else None
        n_calls = state["n"]
    finally:
        restore()

    pairs = closest_pairs(tree, neighbors, k=args.pairs)
    print(f"=== AVANT / APRÈS — {len(pairs)} paires frères les + proches ===\n")
    for sim, a, b in pairs:
        na, nb = tree.nodes[a], tree.nodes[b]
        print(f"paire {a} ⟷ {b}  (sim centroïde {sim:.3f})")
        for nid in (a, b):
            n = tree.nodes[nid]
            print(f"  {nid}  (voisin={nbid[nid]}, n_avis={n.n_avis})")
            print(f"     AVANT : «{before[nid]}»")
            print(f"     APRÈS : «{after[nid]}»")
            print(f"     mots-clés : {', '.join((n.keywords or [])[:6])}")
        print()

    print(f"=== COÛT ===")
    print(f"  {n_calls} appels LLM (modèle {model}) pour re-baker les titres "
          f"contrastifs de {len(tree.nodes)} nœuds")
    print(f"  (les titres AVANT étaient déjà cachés ou générés une fois ; "
          f"le delta net = titres contrastifs neufs, clé de cache incluant le voisin)")


if __name__ == "__main__":
    main()
