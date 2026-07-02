"""Contrat de `backend.recut` — re-coupe sauce_magique d'un arbre de thèmes.

Verrouille, sur un arbre synthétique (macro GÉANT à ~99 % des voix + racine
minuscule), les invariants de `recut_tree` :

  * la coupe optimale ÉCLATE le géant : ses enfants deviennent des macros
    (`parent_id=None`), le géant (ancêtre dissous) est RETIRÉ ;
  * les ids des nœuds conservés sont INCHANGÉS (citations/insights/avis/opinion
    les référencent) ;
  * `depth` recalculé sur tout le sous-arbre, ordre stable (macros par poids
    social décroissant, préfixe) ;
  * couleurs macro réassignées (palette, source unique) et PROPAGÉES aux
    sous-arbres — distinctes entre macros, identiques dans un même sous-arbre ;
  * no-op (`None`) quand la façade actuelle est déjà la coupe optimale.

Aucun LLM, aucun disque : nœuds `SimpleNamespace` (mêmes attributs que ThemeNode).
"""

from __future__ import annotations

from types import SimpleNamespace

from backend.recut import best_cut, recut_tree, sauce_magique


def _node(nid, parent_id, n_avis, consensus, children=(), depth=0):
    return SimpleNamespace(
        id=nid, parent_id=parent_id, depth=depth, n_avis=n_avis, n_claims=n_avis,
        consensus=consensus, weight=float(n_avis), children=list(children),
        color="", convergence=0.0,
    )


def _giant_tree():
    """Racine géante r0 (999 voix, 99,9 %) à 3 enfants équilibrés + racine r1 (1 voix)."""
    nodes = {
        "r0": _node("r0", None, 999, 0.3, children=["c1", "c2", "c3"]),
        "c1": _node("c1", "r0", 333, 0.8, children=["g1", "g2"], depth=1),
        "c2": _node("c2", "r0", 333, 0.8, depth=1),
        "c3": _node("c3", "r0", 333, 0.8, depth=1),
        # Petits-enfants MOINS cohésifs que leur parent (et g2 poussière) : les éclater
        # dégrade le score → la descente gloutonne s'arrête au niveau des enfants.
        "g1": _node("g1", "c1", 331, 0.5, depth=2),
        "g2": _node("g2", "c1", 2, 0.5, depth=2),
        "r1": _node("r1", None, 1, 1.0),
    }
    return SimpleNamespace(
        nodes=nodes, order=["r0", "c1", "g1", "g2", "c2", "c3", "r1"],
        macros=["r0", "r1"], recut=None,
    )


def test_recut_eclate_le_geant_et_preserve_les_ids():
    tree = _giant_tree()
    detail = recut_tree(tree)
    assert detail is not None, "un macro à 99,9 % doit être éclaté"
    assert detail["apres"]["score"] < detail["avant"]["score"]
    assert detail["apres"]["top1"] < detail["avant"]["top1"]
    assert detail["n_dissous"] == 1                        # seul r0 est dissous
    assert tree.recut is detail                            # traçabilité (params.recut)

    # Ancêtre dissous retiré, ids conservés INCHANGÉS, aucun nouvel id.
    assert "r0" not in tree.nodes
    assert set(tree.nodes) == {"c1", "c2", "c3", "g1", "g2", "r1"}

    # Nouvelle façade : les enfants du géant + la racine minuscule, poids décroissant
    # (départage par position d'origine → c1 avant c2 avant c3, ordre STABLE).
    assert tree.macros == ["c1", "c2", "c3", "r1"]
    for mid in tree.macros:
        assert tree.nodes[mid].parent_id is None
        assert tree.nodes[mid].depth == 0

    # Sous-arbre conservé tel quel, profondeurs recalculées, ordre préfixe.
    assert tree.nodes["c1"].children == ["g1", "g2"]
    assert tree.nodes["g1"].parent_id == "c1" and tree.nodes["g1"].depth == 1
    assert tree.order == ["c1", "g1", "g2", "c2", "c3", "r1"]

    # Couleurs : réassignées, distinctes entre macros, propagées au sous-arbre.
    macro_colors = [tree.nodes[m].color for m in tree.macros]
    assert all(c.startswith("#") for c in macro_colors)
    assert len(set(macro_colors)) == len(macro_colors)
    assert tree.nodes["g1"].color == tree.nodes["c1"].color
    assert tree.nodes["g2"].color == tree.nodes["c1"].color


def test_recut_noop_quand_facade_deja_optimale():
    """Façade équilibrée sans enfants → la coupe optimale est la racine : no-op."""
    nodes = {f"m{i}": _node(f"m{i}", None, 100, 0.8) for i in range(8)}
    tree = SimpleNamespace(nodes=nodes, order=list(nodes), macros=list(nodes), recut=None)
    before_order = list(tree.order)
    assert recut_tree(tree) is None
    assert tree.order == before_order and set(tree.macros) == set(nodes)


def test_best_cut_et_score_coherents():
    """`best_cut` sur la vue dicts trouve la même coupe que `recut_tree`."""
    tree = _giant_tree()
    view = [{"id": n.id, "parent_id": n.parent_id, "n_avis": n.n_avis,
             "cohesion": n.consensus} for n in tree.nodes.values()]
    cut, detail = best_cut(view)
    assert {c["id"] for c in cut} == {"c1", "c2", "c3", "r1"}
    score, d = sauce_magique(cut)
    assert d == detail                                     # même détail chiffré
