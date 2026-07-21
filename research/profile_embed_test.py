"""Tuner ce qu'on EMBEDDE pour la couche macro (moteur B) — étiquette courte vs PROFIL.

Bob : embedder un PROFIL de cluster (fidèle, ≤512 tokens), pas seulement une étiquette courte,
pour garder la précision jusqu'aux couches abstraites. Mesuré ici : le profil merge-t-il les
thèmes redondants (addiction, algo→tristesse) TOUT EN restant fidèle, vs l'étiquette canonique
courte (qui merge fort mais perd la précision) ?

    MISTRAL_API_KEY=$(cat var/mistral.key) uv run --extra embed-contender --extra faiss \
        python research/profile_embed_test.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from backend import analysis as A
from backend.build_analysis import load_dataset
from pipeline.cluster.layers import centre, flat_partition
from pipeline.cluster.mistral_client import chat
from pipeline.embed.embedder import embed

MODEL = "mistral-small-latest"


def short_label(claims):
    ex = "\n".join(f"- {c[:180]}" for c in claims[:15])
    m = [{"role": "system", "content": "Donne le SUJET en 3 à 6 mots, catégorie canonique générique. Juste la catégorie."},
         {"role": "user", "content": f"Témoignages :\n{ex}\n\nCatégorie :"}]
    return chat(m, model=MODEL, temperature=0.0, max_tokens=25).strip()


def profile(claims):
    ex = "\n".join(f"- {c[:200]}" for c in claims[:20])
    m = [{"role": "system", "content":
          "Rédige un PROFIL de ce thème en 3 à 5 phrases (≤ 300 mots) : de QUOI parlent ces "
          "témoignages et quelles POSITIONS ils portent. Précis et fidèle. Commence par une "
          "phrase qui nomme le SUJET de fond, puis précise les angles. Ce profil servira à "
          "regrouper ce thème avec ceux du même sujet."},
         {"role": "user", "content": f"Témoignages :\n{ex}\n\nProfil :"}]
    return chat(m, model=MODEL, temperature=0.0, max_tokens=450).strip()


def _cluster(texts, names, pops):
    V = centre(embed(texts, model_id="nomic-v2").astype(np.float64))
    part, meta = flat_partition(V, seed=42)
    groups = {}
    for i, c in enumerate(part.tolist()):
        groups.setdefault(c, []).append(i)
    print(f"  → {len(groups)} macros (γ={meta['gamma']}) :")
    for g in sorted(groups.values(), key=lambda idx: -sum(pops[i] for i in idx)):
        print(f"     ● {sum(pops[i] for i in g):>4} : {' + '.join(names[i] for i in g)}")
    S = V @ V.T
    pairs = sorted(((S[i, j], i, j) for i in range(len(texts)) for j in range(i + 1, len(texts))),
                   reverse=True)[:4]
    print("  paires les plus proches :", ", ".join(f"{names[i]}~{names[j]}({s:+.2f})" for s, i, j in pairs))


def main():
    model = json.loads(Path("backend/cache/tiktok/claims.json").read_text())["model"]
    tree = A.build_theme_tree(load_dataset("tiktok"), model=model, seed=42)
    themes = sorted(tree.macros, key=lambda m: -tree.nodes[m].n_avis)
    reps = {m: (tree.nodes[m].representative_claims
                or [tree.prepared.claim_texts[i] for i in tree.nodes[m].members[:20]]) for m in themes}
    names = [" ".join(tree.nodes[m].label.split()[:2]) for m in themes]
    pops = [tree.nodes[m].n_avis for m in themes]

    print("Génération étiquettes + profils…")
    labels = [short_label(reps[m]) for m in themes]
    profs = [profile(reps[m]) for m in themes]
    for nm, l, p in zip(names, labels, profs):
        print(f"\n[{nm}]\n  étiquette : {l}\n  profil    : {p[:200]}…")

    print("\n===== ÉTIQUETTE COURTE =====")
    _cluster(labels, names, pops)
    print("\n===== PROFIL =====")
    _cluster(profs, names, pops)


if __name__ == "__main__":
    main()
