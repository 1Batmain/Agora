"""RE-COUPE « sauce_magique » v1 — la COUPE optimale de l'arbre devient la façade macro.

Problème observé à l'échelle (granddebat 22k avis) : le coarsening racine fusionne
presque tout en UN macro géant (99,9 % des voix) — la façade s'effondre alors que la
STRUCTURE fine de l'arbre reste bonne. Plutôt que de retoucher le clustering, on
cherche a posteriori le « niveau de récursivité » le plus lisible : la COUPE de
l'arbre (antichaîne couvrant les feuilles) qui minimise une fonction objectif.

Fonction objectif (à MINIMISER), termes en tension :

    sauce_magique = α·(1 − cohésion)            # qualité sémantique (pondérée voix)
                  + β·|log(N_eff / N_cible)|    # lisibilité : nb EFFECTIF de clusters
                  + γ·poussière                 # voix dans les clusters minuscules
                  + δ·top1                      # dominance du plus gros (anti-géant)

N_eff = exp(entropie de Shannon des parts de voix) — un géant à 99 % → ~1 ; douze
clusters équilibrés → ~12. N_cible dérivé des données (~ln(voix)·1.4), aucune
constante de corpus. v1 : poids (1, 0.5, 1, 1) NON CALIBRÉS — à calibrer contre les
golds (témoin granddebat / xstance / repnum) par corrélation de rang. Versionné :
toute modification des termes/poids = nouvelle version notée ici. Recherche de la
coupe par descente GLOUTONNE (non exhaustive) : tant qu'éclater un nœud en ses
enfants améliore le score, on éclate. Harness d'expérimentation :
`research/sauce_magique.py` ; verdict : `research/sauce_magique_note.md`.

`recut_tree` applique la coupe à un `ThemeTree` construit : les nœuds de la coupe
sont RE-RACINÉS (parent_id=None, ce sont les nouveaux macros), leurs ancêtres
dissous sont retirés, les sous-arbres sous la coupe conservés tels quels.
INVARIANTS : ids de nœuds INCHANGÉS (citations/insights/avis/opinion les
référencent), `depth`/`has_children` recalculés, ordre stable (macros par poids
social décroissant, sous-arbres en parcours préfixe), couleurs macro réassignées
depuis la palette (source unique) et propagées à chaque sous-arbre, convergence
recalculée (son shrinkage dépend de la façade macro).
"""

from __future__ import annotations

import math

# Poids v1 de la fonction objectif — NON CALIBRÉS (cf. docstring module).
W = {"alpha": 1.0, "beta": 0.5, "gamma": 1.0, "delta": 1.0}
DUST_SHARE = 0.005   # cluster « poussière » : < 0,5 % des voix


def sauce_magique(cut: list[dict], *, weights: dict = W) -> tuple[float, dict]:
    """Score d'une coupe (liste de nœuds {n_avis, cohesion}) → (score, détail).

    Chaque nœud porte `n_avis` (voix) et `cohesion` (cohésion sémantique du nœud,
    alias `consensus` accepté — déjà calculée par l'arbre). Détail chiffré renvoyé
    pour la traçabilité (n_eff, n_cible, top1…).
    """
    voices = [max(1, n["n_avis"]) for n in cut]
    total = sum(voices)
    shares = [v / total for v in voices]
    # cohésion pondérée par les voix (cohésion sémantique du nœud)
    coh = sum(s * (n.get("cohesion") or n.get("consensus") or 0.0) for s, n in zip(shares, cut))
    # nombre effectif de clusters (entropie de Shannon)
    ent = -sum(s * math.log(s) for s in shares if s > 0)
    n_eff = math.exp(ent)
    n_cible = max(6.0, math.log(total) * 1.4)   # ~ln(voix)·1.4 → ~14 à 22k, ~11 à 3k
    dust = sum(s for s in shares if s < DUST_SHARE)
    top1 = max(shares)
    score = (weights["alpha"] * (1 - coh)
             + weights["beta"] * abs(math.log(n_eff / n_cible))
             + weights["gamma"] * dust
             + weights["delta"] * top1)
    return score, {"n_clusters": len(cut), "n_eff": round(n_eff, 1), "n_cible": round(n_cible, 1),
                   "cohesion": round(coh, 3), "dust": round(dust, 3), "top1": round(top1, 3),
                   "score": round(score, 4)}


def best_cut(themes: list[dict], *, weights: dict = W) -> tuple[list[dict], dict]:
    """Descente gloutonne : part des racines, éclate le nœud qui améliore le plus le score.

    `themes` = tous les nœuds de l'arbre ({id, parent_id, n_avis, cohesion}) ; renvoie
    (coupe = sous-liste de `themes`, détail du score). Gloutonne, non exhaustive : ne
    fait que DESCENDRE (jamais de fusion) — la coupe est toujours à-ou-sous les racines.
    """
    children: dict[str | None, list[dict]] = {}
    for t in themes:
        children.setdefault(t.get("parent_id"), []).append(t)
    cut = list(children.get(None, []))
    score, detail = sauce_magique(cut, weights=weights)
    improved = True
    while improved:
        improved = False
        best = (score, None)
        for i, node in enumerate(cut):
            kids = children.get(node["id"])
            if not kids:
                continue
            cand = cut[:i] + kids + cut[i + 1:]
            s, _ = sauce_magique(cand, weights=weights)
            if s < best[0] - 1e-9:
                best = (s, (i, kids))
        if best[1] is not None:
            i, kids = best[1]
            cut = cut[:i] + kids + cut[i + 1:]
            score, detail = sauce_magique(cut, weights=weights)
            improved = True
    return cut, detail


def recut_tree(tree, *, weights: dict = W) -> dict | None:
    """RE-RACINE `tree` (ThemeTree) sur sa coupe sauce_magique optimale, EN PLACE.

    Renvoie le détail `{avant, apres, weights, n_dissous}` (aussi posé sur
    `tree.recut`, exposé dans `params.recut` du payload), ou `None` si la coupe
    optimale est déjà la façade actuelle (arbre inchangé). Cf. INVARIANTS du
    docstring module — en particulier : aucun id de nœud ne change.
    """
    nodes = tree.nodes
    view = [{"id": n.id, "parent_id": n.parent_id, "n_avis": n.n_avis,
             "cohesion": n.consensus} for n in nodes.values()]
    roots = [v for v in view if v["parent_id"] is None]
    if not roots:
        return None
    _, before = sauce_magique(roots, weights=weights)
    cut, after = best_cut(view, weights=weights)
    cut_ids = [c["id"] for c in cut]
    if set(cut_ids) == {r["id"] for r in roots}:
        return None                      # coupe optimale = racines actuelles → no-op

    # Sous-arbres CONSERVÉS = fermeture descendante de la coupe ; le reste (les
    # ancêtres stricts des nœuds de coupe) est dissous.
    keep: set[str] = set()
    stack = list(cut_ids)
    while stack:
        nid = stack.pop()
        if nid in keep:
            continue
        keep.add(nid)
        stack.extend(nodes[nid].children)
    dissolved = [nid for nid in nodes if nid not in keep]
    for nid in dissolved:
        del nodes[nid]

    # Re-racinage : la coupe devient les macros (poids social décroissant, départage
    # par la position d'origine → ordre stable), profondeurs recalculées en préfixe.
    orig_pos = {nid: i for i, nid in enumerate(tree.order)}
    macros = sorted(cut_ids, key=lambda nid: (-nodes[nid].weight,
                                              orig_pos.get(nid, len(orig_pos))))
    order: list[str] = []

    def _walk(nid: str, depth: int) -> None:
        node = nodes[nid]
        node.depth = depth
        order.append(nid)
        for cid in node.children:
            _walk(cid, depth + 1)

    for mid in macros:
        nodes[mid].parent_id = None
        _walk(mid, 0)
    tree.order = order
    tree.macros = macros

    # Couleurs macro réassignées depuis la palette (source unique), propagées aux
    # sous-arbres ; convergence recalculée (le shrinkage k = population macro MÉDIANE
    # dépend de la façade). Import local : le harness research n'a pas besoin de numpy.
    from backend.analysis import _assign_colors, _assign_convergence
    _assign_colors(nodes, macros)
    _assign_convergence(nodes, macros)

    detail = {"avant": before, "apres": after,
              "weights": dict(weights), "n_dissous": len(dissolved)}
    tree.recut = detail
    return detail
