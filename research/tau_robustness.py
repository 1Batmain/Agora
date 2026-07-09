"""Harnais du verdict `.agent/notes/HIERARCHY_TAU.md` — freins de la subdivision.

Deux mesures, ZÉRO appel LLM (claims + embeddings servis du cache disque) :

  1. FORME    — la forme de l'arbre selon la règle de subdivision, sur N corpus.
  2. ROBUSTESSE — on retire au hasard 1 % des claims, 6 tirages, et on regarde de
     combien la forme bouge. C'est la mesure qui a condamné `tau` : l'arbre passait
     de 304 thèmes (prof. 4) à 17 (prof. 0) selon le tirage.

Ce script parle à l'ANCIENNE API (`_derive_tau`, `RES_LADDER`) pour pouvoir comparer
l'avant/après. Depuis le verdict, ces symboles n'existent plus dans `backend.analysis` :
les variantes A/A'/B/C sont donc HISTORIQUES et se skippent proprement. Seules F/G/H
(le pipeline servi) tournent encore. Gardé comme trace reproductible du verdict.

Usage :
    uv run --extra embed-contender --extra faiss python research/tau_robustness.py \
        [dataset ...]                       # défaut : tiktok
"""
from __future__ import annotations

import dataclasses
import json
import random
import sys
from pathlib import Path
from statistics import median

import numpy as np

from backend import analysis as A
from backend.build_analysis import load_dataset
from backend.claims_endpoint import prepare_claims
from pipeline.cluster import adaptive as AD

_MSS = AD.derive_min_sub_size
_HAS_TAU = hasattr(A, "_derive_tau")          # False depuis le verdict (attendu)


# --------------------------------------------------------------------------- #
# Mesures
# --------------------------------------------------------------------------- #
def shape(tree) -> dict:
    nodes = tree.nodes
    leaves = [n for n in nodes.values() if not n.children]
    av = sorted(n.n_avis for n in leaves)
    return {
        "macros": len(tree.macros),
        "themes": len(nodes),
        "feuilles": len(leaves),
        "prof": max(n.depth for n in nodes.values()) if nodes else 0,
        "struct": sum(1 for m in tree.macros if nodes[m].children) / max(len(tree.macros), 1),
        "p50av": median(av) if av else 0,
        "orphel": sum(1 for x in av if x < 5),       # feuille portée par <5 citoyens
    }


def drop_claims(prep, frac: float, seed: int):
    """Retire `frac` des claims au hasard, en gardant `prepared` cohérent."""
    rng = random.Random(seed)
    n = len(prep.claim_texts)
    keep = sorted(rng.sample(range(n), n - int(n * frac)))
    idx = np.asarray(keep)
    return dataclasses.replace(
        prep,
        claim_texts=[prep.claim_texts[i] for i in keep],
        claim_owner=[prep.claim_owner[i] for i in keep],
        claim_weight=prep.claim_weight[idx],
        claim_vecs=prep.claim_vecs[idx],
        claim_spans=[prep.claim_spans[i] for i in keep],
        claim_target=[prep.claim_target[i] for i in keep],
        target_vecs=prep.target_vecs[idx],
        target_mask=prep.target_mask[idx],
    )


def main(datasets: list[str]) -> None:
    if _HAS_TAU:
        print("⚠️  `_derive_tau` existe encore : le pré-filtre a été réintroduit ?\n"
              "    Relire .agent/notes/HIERARCHY_TAU.md avant d'aller plus loin.\n")

    for name in datasets:
        rec = json.loads(Path(f"backend/cache/{name}/claims.json").read_text())
        model = rec["model"]
        ds = load_dataset(name)
        full = prepare_claims(ds, backend="api", model=model)
        n = len(full.claim_texts)
        print(f"\n### {name}  ({n} claims, model={model}, "
              f"min_sub_size={_MSS(n)})")

        print(f"{'tirage':<10} {'macros':>6} {'themes':>7} {'feuil':>6} {'prof':>5} "
              f"{'struct':>7} {'p50av':>6} {'orphel':>7}")
        formes = []
        for s in range(6):
            prep = full if s == 0 else drop_claims(full, 0.01, s)
            m = shape(A.build_theme_tree(ds, prepared=prep, seed=42))
            formes.append((m["themes"], m["prof"]))
            tag = "complet" if s == 0 else f"-1% #{s}"
            print(f"{tag:<10} {m['macros']:>6} {m['themes']:>7} {m['feuilles']:>6} "
                  f"{m['prof']:>5} {m['struct']:>6.0%} {m['p50av']:>6} {m['orphel']:>7}")

        ths = [f[0] for f in formes]
        prs = [f[1] for f in formes]
        print(f"  → thèmes {min(ths)}–{max(ths)} (amplitude {max(ths) - min(ths)}), "
              f"profondeur {min(prs)}–{max(prs)}")
        print("  Repère du verdict (tiktok, ANCIENNE règle `tau`) : "
              "amplitude 287 thèmes, profondeur 0–4.")


if __name__ == "__main__":
    main(sys.argv[1:] or ["tiktok"])
