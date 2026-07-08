"""BUILD FONDATION pour le proto d'ÉMERGENCE d'arguments (lane stance-argmining).

Construit (extraction claims LLM + embeddings + arbre de thèmes) un dataset plus fourni que
`lutte` et DUMP les essentiels sous `research/emerge_cache/<ds>/` pour que le proto d'émergence
itère SANS recharger torch ni ré-extraire :
  - `claim_vecs.npz`  : embeddings claims L2-normalisés (index global).
  - `claims.jsonl`    : {gi, avis_id, text, spans, target} par claim (verbatim).
  - `leaves.json`     : feuilles de l'arbre → {theme_id, title, keywords, member_gis}.

On NE build PAS opinion/stance/enrichissement (coût minimal) : le proto d'émergence teste
justement si les arguments — et l'axe de clivage — émergent des claims SANS cible pré-établie.

Isolé en DEV (clé Mistral). Rien de servi n'est touché.
    MISTRAL_API_KEY=$(cat var/mistral.key) \
    uv run python research/emerge_build.py --dataset republique-numerique
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backend.analysis import build_theme_tree
from backend.build_analysis import EXTRACT_MODEL, load_dataset
from pipeline.cluster import mistral_client

OUT = Path(__file__).resolve().parent / "emerge_cache"


def run(dataset: str) -> None:
    ds = load_dataset(dataset)
    print(f"[emerge-build] {dataset} · extraction claims + arbre (modèle {EXTRACT_MODEL})…")
    tree = build_theme_tree(ds, model=EXTRACT_MODEL)
    p = tree.prepared
    n = len(p.claim_texts)
    print(f"[emerge-build] {n} claims · {len(tree.nodes)} nœuds "
          f"· extraction {p.extracted} avis ré-extraits ({p.cold_seconds:.0f}s)")

    out = OUT / dataset
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "claim_vecs.npz", vecs=p.claim_vecs.astype(np.float32))
    with (out / "claims.jsonl").open("w") as f:
        for gi in range(n):
            avis_id = p.avis[p.claim_owner[gi]].id
            f.write(json.dumps({
                "gi": gi, "avis_id": avis_id, "text": p.claim_texts[gi],
                "spans": [list(s) for s in p.claim_spans[gi]],
                "target": list(p.claim_target[gi]) if p.claim_target[gi] else None,
            }, ensure_ascii=False) + "\n")
    leaves = [{"theme_id": nid, "title": tree.nodes[nid].title or tree.nodes[nid].label,
               "keywords": list(tree.nodes[nid].keywords or []),
               "member_gis": list(tree.nodes[nid].members)}
              for nid in tree.order if not tree.nodes[nid].children]
    (out / "leaves.json").write_text(json.dumps(leaves, ensure_ascii=False, indent=2))
    # avis text (pour l'audit verbatim en aval)
    avis_txt = {a.id: a.text for a in p.avis}
    (out / "avis.json").write_text(json.dumps(avis_txt, ensure_ascii=False))
    sizes = sorted((len(l["member_gis"]) for l in leaves), reverse=True)
    print(f"[emerge-build] {len(leaves)} feuilles · tailles (claims) top10 : {sizes[:10]}")
    print(f"[emerge-build] → {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build fondation émergence (R&D).")
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    if not mistral_client.available():
        raise SystemExit("Pas de clé Mistral. Abandon.")
    run(args.dataset)


if __name__ == "__main__":
    main()
