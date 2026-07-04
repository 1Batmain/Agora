"""Sélection DISTINCTIVE — `develop.cluster_term_weights` / `select_distinctive_claims`.

Helper PARTAGÉ (titrage ancré + lane stance) : surface les claims d'un nœud DENSES en
vocabulaire c-TF-IDF distinctif plutôt que le médoïde générique (cf.
`research/cluster_merge_note.md`). Tests PURS (zéro LLM, zéro cache) : poids c-TF-IDF,
ordre par densité, DÉTERMINISME strict, départage par index, restriction aux ancres,
repli idf autonome, et cas dégénérés.
"""

from __future__ import annotations

from pytest import approx

from backend.develop import cluster_term_weights, corpus_idf, select_distinctive_claims

# idf EXPLICITE (générique-agnostique) : « commun » écrasé, « rare »/« specifique »
# porteurs — reproduit ce que fait l'idf corpus sur un terme partagé vs distinctif.
_IDF = {"commun": 0.1, "rare": 5.0, "specifique": 5.0}
_TEXTS = [
    "commun commun commun",       # 0 : uniquement le terme commun (générique)
    "commun rare",                # 1 : contient le terme rare
    "commun commun",              # 2 : uniquement le terme commun (générique)
    "rare rare specifique",       # 3 : le plus dense en vocabulaire distinctif
]


def test_cluster_term_weights_favours_distinctive():
    """Poids = tf_cluster · idf : un terme distinctif pèse plus qu'un terme partout."""
    w = cluster_term_weights(_TEXTS, _IDF)
    # tf : commun=6, rare=3, specifique=1 → 6·0.1=0.6 ; 3·5=15 ; 1·5=5.
    assert w["rare"] == approx(15.0)
    assert w["specifique"] == approx(5.0)
    assert w["commun"] == approx(0.6)          # 6·0.1 (tolérance flottante)
    assert w["rare"] > w["commun"]                       # distinctif > générique
    assert cluster_term_weights([], _IDF) == {}          # cas vide


def test_select_orders_by_distinctive_density():
    """Ordre = densité c-TF-IDF moyenne (tokens DISTINCTS) décroissante ; départage index."""
    order = select_distinctive_claims(_TEXTS, _IDF, k=5)
    # densités (moyenne sur le VOCABULAIRE, pas les répétitions) :
    #   t3={rare,specifique}=(15+5)/2=10 > t1={commun,rare}=(0.6+15)/2=7.8 > t0=t2=0.6
    #   → t0 avant t2 (index).
    assert order == [3, 1, 0, 2]
    assert select_distinctive_claims(_TEXTS, _IDF, k=2) == [3, 1]


def test_select_is_deterministic():
    """Même entrée ⇒ même sortie, sans exception (aucun tie non résolu)."""
    a = select_distinctive_claims(_TEXTS, _IDF, k=3)
    b = select_distinctive_claims(_TEXTS, _IDF, k=3)
    assert a == b == [3, 1, 0]


def test_anchor_terms_restrict_vocabulary():
    """`anchor_terms` restreint le vocabulaire porteur → seul le claim qui les emploie surface."""
    # Seul « specifique » compte : uniquement le claim 3 en contient (densité > 0).
    order = select_distinctive_claims(_TEXTS, _IDF, k=5, anchor_terms=["specifique"])
    assert order[0] == 3                                 # le seul porteur passe en tête
    assert set(order) == {0, 1, 2, 3}                    # les autres suivent (densité 0)
    # Ancres en casse mixte : normalisées en lower comme le tokenizer. Sur « rare »,
    # t1={commun,rare} et t3={rare,specifique} portent la MÊME densité (15/2, moyenne
    # sur le vocabulaire) → départage par index : t1 en tête.
    assert select_distinctive_claims(_TEXTS, _IDF, k=1, anchor_terms=["RARE"]) == [1]


def test_idf_repli_when_empty_is_valid_and_deterministic():
    """idf vide ⇒ repli sur `corpus_idf(texts)` : permutation valide, stable, sans crash."""
    out = select_distinctive_claims(_TEXTS, {}, k=2)
    assert out == select_distinctive_claims(_TEXTS, {}, k=2)   # déterministe
    assert len(out) == 2 and len(set(out)) == 2
    assert all(0 <= i < len(_TEXTS) for i in out)
    # le repli calcule bien un idf non vide (sinon poids tous nuls) :
    assert corpus_idf(_TEXTS)                                   # dict non vide


def test_degenerate_cases():
    """Bornes : k≤0, texts vide, k > n → jamais d'exception, sortie cohérente."""
    assert select_distinctive_claims(_TEXTS, _IDF, k=0) == []
    assert select_distinctive_claims([], _IDF, k=5) == []
    assert select_distinctive_claims(_TEXTS, _IDF, k=99) == [3, 1, 0, 2]   # k > n borné
