"""Re-bake TITRES tiktok — AVANT/APRÈS du titrage ancré (lane titles-ancrés).

AVANT  = titre SERVI (cache analysis.json) + repro « ancienne méthode » AU MODÈLE COURANT
         (isole l'effet ANCRAGE de l'effet changement de modèle mistral-large).
APRÈS  = titrage ANCRÉ (`backend.titles`), deux régimes :
           • repli  : prompt ancré (mots-clés = ancres) sur les représentatives ;
           • plein  : + sélection DISTINCTIVE sur TOUTES les claims du nœud (member_texts).
Zéro modif du pipeline : recharge le cache (extraction 100% cachée), recut = façade servie.

    MISTRAL_API_KEY=$(cat var/mistral.key) uv run --extra faiss \
        python research/titles_anchor_bake.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from backend import titles
from backend.analysis import build_theme_tree
from backend.build_analysis import load_dataset
from backend.recluster import dataset_dir
from backend.recut import recut_tree
from pipeline.cluster import mistral_client

DATASET = "tiktok"
PAIRS = [("n0", "n265"), ("n161", "n280"), ("n265", "n275"), ("n243", "n275")]
MODEL = mistral_client.NAMING_MODEL


def cache_model(ds: str) -> str:
    rec = json.loads((dataset_dir(ds) / "claims.json").read_text(encoding="utf-8"))
    return rec.get("model") or rec.get("meta", {}).get("model")


def served_titles(ds: str) -> dict[str, str]:
    p = dataset_dir(ds) / "analysis" / "analysis.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    return {t["id"]: t.get("title", "") for t in d.get("themes", [])}


def old_method_title(node) -> str:
    """Ancienne méthode AU MODÈLE COURANT : représentatives + mots-clés, prompt « résume »."""
    kw = ", ".join((node.keywords or [])[:titles.MAX_KEYWORDS])
    reps = "\n".join(f"- {r}" for r in (node.representative_claims or [])[:titles.REP_PER_THEME])
    system = (
        "Tu nommes des thèmes issus d'un regroupement automatique de contributions "
        "citoyennes. Tu produis un TITRE COURT, neutre et descriptif du SUJET du thème "
        "— pas une phrase complète, sans ponctuation finale, sans guillemets, sans "
        "préfixe (« Thème : »…). Tu n'inventes rien hors des éléments fournis."
    )
    user = (
        "Voici les éléments d'un thème : ses mots-clés distinctifs et quelques "
        "contributions représentatives. Donne UN SEUL titre de 3 à 7 mots qui résume "
        "le sujet du thème, dans la langue dominante des contributions. Réponds "
        "UNIQUEMENT par le titre, rien d'autre.\n\n"
        f"Mots-clés : {kw or '(aucun)'}\n\n"
        f"Contributions représentatives :\n{reps or '(aucune)'}\n"
    )
    try:
        raw = mistral_client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=MODEL, temperature=titles.TITLE_TEMPERATURE, max_tokens=titles.TITLE_MAX_TOKENS)
    except mistral_client.MistralError as e:
        return f"<LLM error: {e}>"
    return titles._clean_title(raw)


def main():
    ds = load_dataset(DATASET)
    tree = build_theme_tree(ds, model=cache_model(DATASET))
    assert tree.prepared.extracted == 0, "extraction aurait dû être 100% cachée"
    rc = recut_tree(tree)
    print(f"# recut : {rc['avant']['n_clusters']}→{rc['apres']['n_clusters']} macros" if rc
          else "# recut no-op")
    print(f"# {len(tree.macros)} macros ; n_claims={len(tree.prepared.claim_texts)} ; modèle={MODEL}\n")

    served = served_titles(DATASET)
    claim_texts = tree.prepared.claim_texts
    idf = tree.claim_idf or {}
    targets = sorted({n for p in PAIRS for n in p}, key=lambda x: int(x[1:]))

    rows = {}
    for tid in targets:
        node = tree.nodes[tid]
        member_texts = [claim_texts[i] for i in node.members]
        anc_full = titles._anchor_claims(node, member_texts, idf)
        rows[tid] = {
            "kw": ", ".join((node.keywords or [])[:6]),
            "n": len(node.members),
            "served": served.get(tid, "?"),
            "old_ctrl": old_method_title(node),
            "anchored_fallback": titles.title_for_node(DATASET, node, model=MODEL, refresh=True),
            "anchored_full": titles.title_for_node(
                DATASET, node, model=MODEL, refresh=True,
                member_texts=member_texts, idf=idf),
            "anchor_claims_full": anc_full,
        }

    for tid in targets:
        r = rows[tid]
        print(f"── {tid}  (n_claims={r['n']})  mots-clés: {r['kw']}")
        print(f"    AVANT servi         : {r['served']}")
        print(f"    AVANT (old @ courant): {r['old_ctrl']}")
        print(f"    APRÈS ancré (repli) : {r['anchored_fallback']}")
        print(f"    APRÈS ancré (plein) : {r['anchored_full']}")
        print(f"      claims d'ancrage (plein) : {r['anchor_claims_full'][:3]}")
        print()

    print("=== PAIRES PROCHES — divergence des titres ===")
    for a, b in PAIRS:
        ca, cb = tree.nodes[a].centroid, tree.nodes[b].centroid
        sim = float(np.dot(ca, cb) / (np.linalg.norm(ca) * np.linalg.norm(cb)))
        print(f"\nsim({a},{b})={sim:.3f}")
        for tid in (a, b):
            r = rows[tid]
            print(f"  {tid}  servi:«{r['served']}»  →  ancré-plein:«{r['anchored_full']}»")


if __name__ == "__main__":
    main()
