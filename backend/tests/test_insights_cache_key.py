"""M1 — la clé de cache des insights doit intégrer les synthèses ENFANTS (bottom-up).

Sans ça, un re-bake d'un thème PARENT avec des synthèses enfants changées ferait un
cache HIT sur l'ancienne synthèse. Une feuille (sans enfants) garde une clé stable.
"""
from backend.insights import _cache_key


def test_leaf_key_is_stable():
    """Feuille / pas d'enfants → clé identique quel que soit `child_insights` vide."""
    k = _cache_key("ds", "theme", "n1", "m", 1.0)
    assert k == _cache_key("ds", "theme", "n1", "m", 1.0, None)
    assert k == _cache_key("ds", "theme", "n1", "m", 1.0, {})
    # pas de marqueur bottom-up pour une feuille
    assert "bottomup" not in k


def test_children_markdown_changes_key():
    """Un markdown enfant CHANGÉ → clé DIFFÉRENTE (pas de HIT sur l'ancienne synthèse)."""
    leaf = _cache_key("ds", "theme", "n0", "m", 1.0)
    ka = _cache_key("ds", "theme", "n0", "m", 1.0, {"n1": "A", "n2": "B"})
    kb = _cache_key("ds", "theme", "n0", "m", 1.0, {"n1": "A", "n2": "B-modifié"})
    assert ka != leaf          # avec enfants ≠ feuille
    assert ka != kb            # markdown enfant changé ≠
    assert "bottomup" in ka


def test_child_order_independent():
    """Clé déterministe : tri par id → l'ordre d'insertion des enfants n'importe pas."""
    k1 = _cache_key("ds", "theme", "n0", "m", 1.0, {"n1": "A", "n2": "B"})
    k2 = _cache_key("ds", "theme", "n0", "m", 1.0, {"n2": "B", "n1": "A"})
    assert k1 == k2


def test_different_children_set_changes_key():
    """Un enfant AJOUTÉ/RETIRÉ change la clé."""
    k1 = _cache_key("ds", "theme", "n0", "m", 1.0, {"n1": "A"})
    k2 = _cache_key("ds", "theme", "n0", "m", 1.0, {"n1": "A", "n2": "B"})
    assert k1 != k2
