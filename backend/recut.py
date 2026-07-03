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

REGROUPEMENT DE LA POUSSIÈRE : après la re-coupe, les macros minuscules (part de
voix < `DUST_SHARE`) sont regroupés sous UN nœud synthétique unique `n_dust`
(« Contributions isolées », couleur grise neutre) dont ils deviennent les ENFANTS
— ids intacts, voix conservées, rien supprimé, tout reste navigable. Le nœud
poussière N'ENTRE PAS dans la fonction sauce_magique : le score `apres` rapporté
est celui de la COUPE (poussières comptées individuellement), calculé AVANT ce
regroupement de façade — `n_dust` est un post-traitement de présentation, jamais
un terme de l'objectif.
"""

from __future__ import annotations

import copy
import math

# Poids v1 de la fonction objectif — NON CALIBRÉS (cf. docstring module).
W = {"alpha": 1.0, "beta": 0.5, "gamma": 1.0, "delta": 1.0}
DUST_SHARE = 0.005   # cluster « poussière » : < 0,5 % des voix
DUST_ID = "n_dust"                    # id stable du nœud synthétique de regroupement
DUST_LABEL = "Contributions isolées"  # label/title du nœud poussière


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


def _subtree(nodes: dict, root_id: str) -> list[str]:
    """Ids du sous-arbre fermé sous `root_id` (racine incluse)."""
    out: list[str] = []
    stack = [root_id]
    while stack:
        nid = stack.pop()
        out.append(nid)
        stack.extend(getattr(nodes[nid], "children", None) or [])
    return out


def _regroup_dust(nodes: dict, macros: list[str], *, dust_share: float = DUST_SHARE):
    """Regroupe les macros < `dust_share` des voix sous un nœud synthétique unique.

    Les macros minuscules (poussière) deviennent les ENFANTS de `n_dust`
    (« Contributions isolées ») : ids INTACTS, voix conservées, rien supprimé, tout
    reste navigable (le nœud poussière a `has_children`). Le nœud synthétique porte
    les SOMMES (n_avis/n_claims/weight) et l'union des `members` (pour que le balayage
    claim→macro de `backend.avis` couvre bien ses claims) ; cohésion/dispersion non
    recalculables sans vecteurs → 0. Renvoie (nouvelle façade macro, nœud poussière) ;
    (façade inchangée, None) si moins de 2 poussières (rien à regrouper).
    """
    total = sum(max(0, nodes[m].n_avis) for m in macros)
    if total <= 0:
        return list(macros), None
    dust_ids = [m for m in macros if nodes[m].n_avis / total < dust_share]
    if len(dust_ids) < 2:                 # « regrouper » suppose une pluralité
        return list(macros), None

    dust = copy.copy(nodes[dust_ids[0]])  # même type (ThemeNode / mock) → champs valides
    dust.id = DUST_ID
    dust.parent_id = None
    dust.depth = 0
    dust.children = list(dust_ids)
    dust.n_avis = sum(nodes[m].n_avis for m in dust_ids)
    dust.n_claims = sum(nodes[m].n_claims for m in dust_ids)
    dust.weight = sum(nodes[m].weight for m in dust_ids)
    dust.label = DUST_LABEL
    dust.title = DUST_LABEL
    dust.hook = ""
    dust.description = ""
    dust.keywords = []
    dust.representative_claims = []
    dust.consensus = 0.0                  # cohésion : non recalculable ici → 0
    dust.dispersion = 0.0
    dust.convergence = 0.0
    members: list[int] = []
    for m in dust_ids:
        members.extend(getattr(nodes[m], "members", None) or [])
    dust.members = members

    nodes[DUST_ID] = dust
    for m in dust_ids:
        nodes[m].parent_id = DUST_ID
    dust_set = set(dust_ids)
    facade = [m for m in macros if m not in dust_set] + [DUST_ID]
    return facade, dust


def recut_tree(tree, *, weights: dict = W) -> dict | None:
    """RE-RACINE `tree` (ThemeTree) sur sa coupe sauce_magique optimale, EN PLACE.

    Renvoie le détail `{avant, apres, weights, n_dissous, poussiere}` (aussi posé sur
    `tree.recut`, exposé dans `params.recut` du payload), ou `None` si la coupe
    optimale est déjà la façade actuelle (arbre inchangé). `poussiere` = `None` ou
    `{id, n_macros, n_avis, share}` si des macros minuscules ont été regroupés sous
    `n_dust`. Cf. INVARIANTS du docstring module — en particulier : aucun id de nœud
    ne change, et `n_dust` n'entre pas dans le score sauce_magique rapporté.
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

    # REGROUPEMENT DE LA POUSSIÈRE : les macros de la coupe pesant < DUST_SHARE des
    # voix deviennent les ENFANTS d'un unique nœud synthétique `n_dust` (façade lisible,
    # poussières navigables). Fait AVANT le tri/la re-racinage pour que `n_dust` prenne
    # sa place dans l'ordre macro ; le score `apres` (déjà calculé sur la coupe) reste
    # celui des poussières individuelles → `n_dust` hors sauce_magique (cf. docstring).
    orig_pos = {nid: i for i, nid in enumerate(tree.order)}
    facade, dust = _regroup_dust(nodes, cut_ids, dust_share=DUST_SHARE)

    # Re-racinage : la coupe devient les macros (poids social décroissant, départage
    # par la position d'origine → ordre stable), profondeurs recalculées en préfixe.
    macros = sorted(facade, key=lambda nid: (-nodes[nid].weight,
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

    # Le sous-arbre poussière garde une couleur GRISE NEUTRE (palette, source unique),
    # réappliquée APRÈS _assign_colors (qui aurait sinon donné à `n_dust` une teinte de
    # macro comme les autres).
    if dust is not None:
        from pipeline.cluster.palette import NOISE_COLOR
        for nid in _subtree(nodes, DUST_ID):
            nodes[nid].color = NOISE_COLOR

    total_facade = sum(max(0, nodes[m].n_avis) for m in macros)
    detail = {"avant": before, "apres": after,
              "weights": dict(weights), "n_dissous": len(dissolved),
              "poussiere": None if dust is None else {
                  "id": DUST_ID,
                  "n_macros": len(dust.children),
                  "n_avis": dust.n_avis,
                  "share": round(dust.n_avis / total_facade, 4) if total_facade else 0.0,
              }}
    tree.recut = detail
    return detail
