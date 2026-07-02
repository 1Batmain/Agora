"""ONE-SHOT — applique la re-coupe sauce_magique à un cache d'analyse EXISTANT, sans LLM.

Post-traite `backend/cache/<dataset>/analysis/` déjà construit (ex. granddebat corpus
complet 22k, façade effondrée à 99,9 %) avec la MÊME logique que le pipeline
(`backend.recut.recut_tree`, câblée au build depuis « feat(cluster): re-coupe
sauce_magique ») — SANS re-extraction ni AUCUN appel LLM d'enrichissement :

  1. reconstruit l'arbre depuis les caches claims/embeddings (déterministe, seed figé) ;
  2. SANITY CHECK : l'arbre reconstruit doit être IDENTIQUE à l'analysis.json persisté
     (mêmes ids, parent_id, n_claims, n_avis) — sinon abort (le code a divergé du cache) ;
  3. recopie les champs LLM existants (title/hook/description + label/keywords) par id
     de nœud — ils restent valides, les nœuds conservés sont inchangés ;
  4. re-coupe (`recut_tree`) puis réécrit `analysis.json` (payload complet : arêtes de
     co-occurrence recalculées sur la nouvelle façade, dataset_stats, params.recut) et
     `avis.json` (cluster_id/couleur des claims = NOUVEAUX macros) via les MÊMES
     fonctions que le build (`analysis_payload`, `build_avis_provenance`) ;
  5. met à jour `status.json` (n_themes/n_macros).

Les citations (par nœud, ids inchangés), insights (par nœud), opinion.json et
claim_stance.json (ids de claims inchangés — même aplatissement `prepared`) restent
valides tels quels. Les fichiers des nœuds dissous deviennent orphelins (jamais servis).

Usage :
    MISTRAL_API_KEY=$(cat var/mistral.key) uv run --extra contender \
        --extra embed-contender --extra faiss \
        python research/apply_recut.py --dataset granddebat
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import analysis_store as store          # noqa: E402
from backend.analysis import analysis_payload, build_theme_tree  # noqa: E402
from backend.avis import build_avis_provenance       # noqa: E402
from backend.build_analysis import EXTRACT_MODEL, load_dataset   # noqa: E402
from backend.claims_endpoint import (                 # noqa: E402
    CLAIMS_NAME, DEFAULT_MIN_CHARS, _avis_from_ideas, _load_claims_cache,
)
from backend.recluster import dataset_dir             # noqa: E402
from backend.recut import recut_tree                  # noqa: E402
from backend.translate import translations_path      # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-coupe sauce_magique d'un cache d'analyse existant (sans LLM).")
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    dataset = args.dataset

    old = store.read_analysis(dataset)
    if not old:
        sys.exit(f"pas d'analysis.json persisté pour {dataset!r} — rien à re-couper.")
    old_by_id = {t["id"]: t for t in old["themes"]}

    # 1) Arbre reconstruit depuis les caches (extraction+embed cachés → zéro LLM).
    #    FAIL-CLOSED : le cache claims est clé par MODÈLE — on passe explicitement le
    #    modèle d'EXTRACTION du build (sinon le défaut `ministral-3b` raterait le cache
    #    et relancerait une extraction complète), et on VÉRIFIE la couverture AVANT
    #    d'appeler le pipeline (aucun avis manquant ⇒ aucun appel LLM possible).
    ds = load_dataset(dataset)
    avis = _avis_from_ideas(ds.ideas, DEFAULT_MIN_CHARS)
    cached = _load_claims_cache(dataset_dir(dataset) / CLAIMS_NAME, EXTRACT_MODEL)
    missing = [a.id for a in avis if a.id not in cached]
    if missing:
        sys.exit(f"ABORT : {len(missing)} avis absents du cache claims ({EXTRACT_MODEL}) "
                 f"— une extraction serait déclenchée (ex.: {missing[:3]}).")
    tree = build_theme_tree(ds, model=EXTRACT_MODEL)
    assert tree.prepared.extracted == 0, "l'extraction aurait dû être 100 % cachée"

    # 2) SANITY : l'arbre reconstruit == l'arbre persisté (sinon le recut ne
    #    s'appliquerait pas au même objet que les artefacts existants → abort).
    rebuilt = {nid: (tree.nodes[nid].parent_id, tree.nodes[nid].n_claims, tree.nodes[nid].n_avis)
               for nid in tree.order}
    persisted = {t["id"]: (t["parent_id"], t["n_claims"], t["n_avis"]) for t in old["themes"]}
    if rebuilt != persisted:
        only_new = set(rebuilt) - set(persisted)
        only_old = set(persisted) - set(rebuilt)
        diff = [k for k in set(rebuilt) & set(persisted) if rebuilt[k] != persisted[k]]
        sys.exit("ABORT : l'arbre reconstruit diverge de l'analysis.json persisté "
                 f"(nouveaux={sorted(only_new)[:5]} disparus={sorted(only_old)[:5]} "
                 f"modifiés={sorted(diff)[:5]}…) — re-bake complet requis.")
    print(f"[apply_recut] {dataset} · arbre reproduit à l'identique "
          f"({len(tree.order)} nœuds, {len(tree.macros)} macros)")

    # 3) Champs LLM existants recopiés par id (nœuds inchangés → toujours valides).
    for nid, node in tree.nodes.items():
        t = old_by_id[nid]
        node.title = t.get("title") or node.label
        node.hook = t.get("hook", "")
        node.description = t.get("description", "")
        node.label = t.get("label") or node.label
        node.keywords = t.get("keywords") or node.keywords

    # 4) Re-coupe + réécriture analysis.json / avis.json (mêmes fonctions que le build).
    rc = recut_tree(tree)
    if rc is None:
        sys.exit("la coupe optimale est déjà la façade actuelle — rien à faire.")
    print(f"[apply_recut] re-coupe : {rc['avant']['n_clusters']}→{rc['apres']['n_clusters']} macros, "
          f"top1 {rc['avant']['top1']:.1%}→{rc['apres']['top1']:.1%}, "
          f"cohésion {rc['avant']['cohesion']}→{rc['apres']['cohesion']}, {rc['n_dissous']} nœud(s) dissous")

    payload = analysis_payload(tree)
    payload["status"] = store.READY
    store.write_analysis(dataset, payload)

    # Traductions FR : cache persisté relu tel quel (zéro appel), forme {aid:{lang,text_fr}}.
    tr_raw = {}
    tp = translations_path(dataset)
    if tp.exists():
        tr_raw = json.loads(tp.read_text(encoding="utf-8"))
    translations = {aid: {"lang": e.get("lang", "fr"), "text_fr": e.get("text_fr")}
                    for aid, e in tr_raw.items() if isinstance(e, dict)}
    store.write_avis(dataset, build_avis_provenance(tree, translations))

    # 5) status.json reflète la nouvelle façade.
    store.write_status(dataset, store.READY, n_themes=len(tree.order),
                       n_macros=len(tree.macros))

    macros = sorted((tree.nodes[m] for m in tree.macros), key=lambda n: -n.n_avis)
    total = sum(m.n_avis for m in macros)
    print(f"[apply_recut] ✓ {dataset} · {len(tree.macros)} macros · "
          f"top1 {macros[0].n_avis}/{total} = {macros[0].n_avis / total:.1%}")
    for m in macros[:15]:
        print(f"   {m.id:>6} {m.n_avis:>6}  {m.title[:70]}")


if __name__ == "__main__":
    main()
