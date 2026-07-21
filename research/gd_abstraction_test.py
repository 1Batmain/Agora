"""Test DÉCISIF du moteur d'abstraction sur le Grand Débat MULTI-thèmes (niveau avis).

tiktok est mono-sujet → pas de vraie structure macro. Le Grand Débat a 4 thèmes officiels
DISTINCTS : c'est le juge du moteur B. Test structurel au niveau AVIS (pas de claims LLM
coûteux) : échantillon des 4 thèmes (une question ouverte majeure par thème) → embed → γ plat
→ moteur B (profil ré-embeddé) → les macros retrouvent-elles les 4 thèmes ? (ARI vs gold).

    MISTRAL_API_KEY=$(cat var/mistral.key) OMP_NUM_THREADS=3 \
      uv run --extra embed-contender --extra faiss python research/gd_abstraction_test.py [N_par_thème]
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import adjusted_rand_score as ARI

from pipeline.cluster import abstraction as ab
from pipeline.cluster.layers import centre, flat_partition
from pipeline.cluster.mistral_client import chat
from pipeline.embed.embedder import embed

csv.field_size_limit(10_000_000)

# thème → (fichier, mot-clé pour trouver la colonne de la question ouverte majeure)
THEMES = {
    "démocratie":   ("data/raw/gd_democratie.csv",   "renouer le lien"),
    "fiscalité":    ("data/raw/gd_fiscalite.csv",     "fiscalité plus juste"),
    "écologie":     ("data/raw/gd_ecologie.csv",      "apporter des réponses"),
    "organisation": ("data/raw/gd_organisation.csv",  "pensez-vous de l'organisation"),
}


def _sample(path: str, keyword: str, n: int, min_chars: int = 40) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        r = csv.reader(fh)
        header = next(r)
        col = next((i for i, h in enumerate(header) if keyword.lower() in h.lower()), None)
        if col is None:
            raise SystemExit(f"colonne introuvable ({keyword!r}) dans {path}")
        out = []
        for row in r:
            if col < len(row):
                t = row[col].strip()
                if len(t) >= min_chars:
                    out.append(t)
            if len(out) >= n:
                break
    return out


def main(n: int = 2000) -> None:
    texts, gold = [], []
    for i, (theme, (path, kw)) in enumerate(THEMES.items()):
        s = _sample(path, kw, n)
        texts += s
        gold += [i] * len(s)
        print(f"  {theme:<13} {len(s)} avis (col « {kw} »)")
    gold = np.array(gold)
    print(f"total {len(texts)} avis · embedding (nomic, throttlé)…", flush=True)

    V = centre(embed(texts, model_id="nomic-v2").astype(np.float64))

    # Couche plate γ
    part, meta = flat_partition(V, seed=42)
    n_fine = len(set(part.tolist()))
    print(f"couche plate : {n_fine} thèmes (γ={meta['gamma']}, modularité {meta['modularity']})", flush=True)

    # Moteur B : profil par thème → ré-embed → clustering = macros
    clusters = {}
    for idx, c in enumerate(part.tolist()):
        clusters.setdefault(c, []).append(idx)
    cluster_texts = [[texts[i] for i in mem[:20]] for mem in clusters.values()]
    res = ab.compute(cluster_texts, chat_fn=chat, embed_fn=lambda t: embed(t, model_id="nomic-v2"),
                     model="mistral-small-latest")
    if res is None:
        print("abstraction : None (trop peu de thèmes ou 1 seul macro)"); return

    # macro par AVIS = macro de son thème fin
    theme_ids = list(clusters.keys())
    fine_to_macro = {theme_ids[ti]: m for ti, m in enumerate(res["assign"])}
    macro_of_avis = np.array([fine_to_macro[c] for c in part.tolist()])
    n_macro = len(set(res["assign"]))

    print(f"\nMOTEUR B : {n_fine} thèmes → {n_macro} macros")
    print(f"  ARI(macros, 4 thèmes officiels) = {ARI(gold, macro_of_avis):.3f}")
    print(f"  ARI(couche plate, 4 thèmes)     = {ARI(gold, part):.3f}  (référence)")
    print("\nComposition des macros (thème officiel dominant par macro) :")
    for m in sorted(set(res["assign"])):
        mask = macro_of_avis == m
        maj = np.bincount(gold[mask], minlength=4)
        dom = list(THEMES)[maj.argmax()]
        print(f"  ● macro {m} : {mask.sum():>4} avis · dominante « {dom} » ({maj.max()/maj.sum():.0%})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
