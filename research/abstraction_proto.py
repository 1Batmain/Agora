"""Proto du MOTEUR D'ABSTRACTION — la couche macro par ré-embedding d'étiquettes canoniques.

Validé (`research/synthesis_embed_note.md`) : ré-embedder une étiquette CANONIQUE par thème
convertit la redondance sémantique en proximité géométrique. Ce proto enchaîne le pipeline :

  couche plate (γ, pic de modularité) → étiquette canonique LLM par thème → ré-embedding local
  → clustering des étiquettes (même moteur `flat_partition`) = couche MACRO.

But : voir les thèmes redondants (addiction…) fusionner en un macro, SANS souder des sujets
distincts. Mesure avant de câbler. Zéro modif pipeline.

    MISTRAL_API_KEY=$(cat var/mistral.key) uv run --extra embed-contender --extra faiss \
        python research/abstraction_proto.py [dataset]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from backend import analysis as A
from backend.build_analysis import load_dataset
from pipeline.cluster.layers import centre, flat_partition
from pipeline.cluster.mistral_client import chat
from pipeline.embed.embedder import embed

LABEL_MODEL = "mistral-small-latest"


def canonical_label(claims: list[str]) -> str:
    ex = "\n".join(f"- {c[:180]}" for c in claims[:15])
    msg = [
        {"role": "system", "content":
         "Donne le SUJET de ce groupe de témoignages en 3 à 6 mots, sous forme de CATÉGORIE "
         "canonique et générique (ex: « dépendance aux réseaux sociaux », « protection des "
         "mineurs »). Juste la catégorie, rien d'autre."},
        {"role": "user", "content": f"Témoignages :\n{ex}\n\nCatégorie :"},
    ]
    return chat(msg, model=LABEL_MODEL, temperature=0.0, max_tokens=25).strip()


def main(ds: str = "tiktok") -> None:
    model = json.loads(Path(f"backend/cache/{ds}/claims.json").read_text())["model"]
    tree = A.build_theme_tree(load_dataset(ds), model=model, seed=42)
    themes = sorted(tree.macros, key=lambda m: -tree.nodes[m].n_avis)
    reps = {m: (tree.nodes[m].representative_claims
                or [tree.prepared.claim_texts[i] for i in tree.nodes[m].members[:15]])
            for m in themes}

    print(f"{ds} : {len(themes)} thèmes plats (γ pic de modularité). Étiquettes canoniques…\n")
    labels = {m: canonical_label(reps[m]) for m in themes}
    for m in themes:
        print(f"   {tree.nodes[m].n_avis:>4}  {labels[m]:<38}  [{' '.join(tree.nodes[m].label.split()[:3])}]")

    # Ré-embedding LOCAL des étiquettes → clustering (même moteur) = macros.
    L = embed([labels[m] for m in themes], model_id="nomic-v2").astype(np.float64)
    macro_of, meta = flat_partition(centre(L), seed=42)

    groups: dict[int, list[int]] = {}
    for i, mc in enumerate(macro_of.tolist()):
        groups.setdefault(mc, []).append(i)

    print(f"\n=== {len(groups)} MACROS (clustering des étiquettes, γ={meta['gamma']}) ===")
    for mc, idxs in sorted(groups.items(), key=lambda kv: -sum(tree.nodes[themes[i]].n_avis for i in kv[1])):
        pop = sum(tree.nodes[themes[i]].n_avis for i in idxs)
        merged = " + ".join(labels[themes[i]] for i in idxs)
        print(f"  ● {pop:>4} avis · {len(idxs)} thème(s) : {merged}")

    # Transparence : matrice de cosinus des étiquettes recentrées.
    C = centre(L); S = C @ C.T
    print("\nCosinus des étiquettes (recentrées) — paires les plus proches :")
    pairs = sorted(((S[i, j], i, j) for i in range(len(themes)) for j in range(i + 1, len(themes))),
                   reverse=True)[:5]
    for s, i, j in pairs:
        print(f"  {s:+.2f}  {labels[themes[i]]:<34} ~ {labels[themes[j]]}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["tiktok"]))
