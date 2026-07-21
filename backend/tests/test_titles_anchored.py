"""Titrage ANCRÉ — sélection des claims d'entrée + clé de cache versionnée par méthode.

Tests PURS (zéro LLM, zéro disque) des rouages déterministes de `backend.titles` :
`_anchor_claims` (sélection distinctive via `develop.select_distinctive_claims`, repli sur
les représentatives, dédoublonnage, bornage) et `_content_key` (la MÉTHODE de sélection
`ANCHOR_METHOD` fait partie du hash → un changement de méthode invalide le cache).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend import titles


@dataclass
class _Node:
    id: str = "n0"
    label: str = "a b c"
    keywords: list[str] = field(default_factory=list)
    representative_claims: list[str] = field(default_factory=list)


def test_anchor_claims_member_texts_picks_distinctive():
    """Avec `member_texts` + idf, on sélectionne les claims DENSES en vocab distinctif."""
    idf = {"commun": 0.1, "rare": 5.0, "specifique": 5.0}
    members = [
        "commun commun commun",     # générique
        "rare rare specifique",     # le plus distinctif
        "commun rare",
    ]
    out = titles._anchor_claims(_Node(), members, idf)
    assert out[0] == "rare rare specifique"          # le plus distinctif en tête
    assert set(out) == set(members)                  # tous présents (k ≥ n)


def test_anchor_claims_fallback_keeps_representative_order():
    """Sans `member_texts`, on garde les représentatives déjà re-rankées (pas de
    re-tri par distinctivité : idf local dégénéré sur un petit pool)."""
    reps = ["premier developpe", "second argument", "troisieme point"]
    node = _Node(representative_claims=reps)
    assert titles._anchor_claims(node, None, None) == reps


def test_anchor_claims_dedups_and_bounds_length():
    """Doublons littéraux écartés ; chaque claim bornée à CLAIM_MAX_CHARS."""
    long_claim = "mot " * 200
    node = _Node(representative_claims=["Doublon", "doublon", long_claim])
    out = titles._anchor_claims(node, None, None)
    lows = [c.lower() for c in out]
    assert len(lows) == len(set(lows))                          # aucun doublon (casse ignorée)
    assert all(len(c) <= titles.CLAIM_MAX_CHARS for c in out)   # borné


def test_anchor_claims_empty_never_crashes():
    """Nœud sans représentatives ni membres → liste vide, jamais d'exception."""
    assert titles._anchor_claims(_Node(), None, None) == []
    assert titles._anchor_claims(_Node(), [], None) == []


def test_content_key_depends_on_anchor_method(monkeypatch):
    """`ANCHOR_METHOD` est DANS la clé : changer la méthode invalide le cache titres."""
    node = _Node(keywords=["x", "y"])
    anchors = ["claim a", "claim b"]
    k1 = titles._content_key("ds", node, "model-z", anchors)
    monkeypatch.setattr(titles, "ANCHOR_METHOD", "ctfidf-distinctive-v2")
    k2 = titles._content_key("ds", node, "model-z", anchors)
    assert k1 != k2                                   # méthode différente ⇒ hash différent


def test_content_key_depends_on_anchors():
    """Des claims d'ancrage différentes ⇒ clé différente (re-génération ciblée)."""
    node = _Node(keywords=["x"])
    k1 = titles._content_key("ds", node, "m", ["claim a"])
    k2 = titles._content_key("ds", node, "m", ["claim b"])
    assert k1 != k2


def test_content_key_invariant_to_anchor_order():
    """Régression (incident 4/07) : c'est l'ENSEMBLE des ancres qui définit le titre,
    pas leur ordre. Une permutation NE DOIT PAS changer la clé — sinon le re-ranking
    « développement » ou le passage plein-ancrage↔repli déclenche une re-génération en
    avalanche (titres retombés en mots-clés au rebuild parallèle)."""
    node = _Node(keywords=["x", "y"])
    anchors = ["claim gamma", "claim alpha", "claim beta"]
    base = titles._content_key("ds", node, "m", anchors)
    assert titles._content_key("ds", node, "m", list(reversed(anchors))) == base
    assert titles._content_key("ds", node, "m", ["claim alpha", "claim beta",
                                                 "claim gamma"]) == base


# --- Titrage des MACROS (ombrelle sur les titres d'enfants) ------------------------------- #

def test_macro_messages_carry_child_titles():
    """Le prompt de macro montre les titres d'ENFANTS (pas les keywords diffus)."""
    kids = ["Contrôle parental des ados", "Interdiction des réseaux aux mineurs"]
    msgs = titles._macro_title_messages(kids)
    user = msgs[-1]["content"]
    assert all(k in user for k in kids)
    assert "OMBRELLE" in msgs[0]["content"] or "ombrelle" in msgs[0]["content"].lower()


def test_macro_key_invariant_to_child_order():
    """Régression : c'est l'ENSEMBLE des titres d'enfants qui définit l'ombrelle, pas
    leur ordre — une permutation ne doit pas flipper la clé (avalanche de re-génération)."""
    node = _Node()
    kids = ["Titre gamma", "Titre alpha", "Titre beta"]
    base = titles._macro_key("ds", node, "m", kids)
    assert titles._macro_key("ds", node, "m", list(reversed(kids))) == base


def test_macro_key_versioned_by_method():
    """`MACRO_METHOD` fait partie du hash → un changement de méthode invalide le cache."""
    node = _Node()
    kids = ["Titre a", "Titre b"]
    k1 = titles._macro_key("ds", node, "m", kids)
    # méthode différente ⇒ hash différent (distinct du titrage ancré `_content_key`)
    assert k1 != titles._content_key("ds", node, "m", kids)
