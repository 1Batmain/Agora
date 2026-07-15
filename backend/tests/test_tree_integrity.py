"""Intégrité de l'arbre servi — régressions constatées sur le rebuild tiktok.

1. Le cache de claims est clé par MODÈLE : un appelant qui oublie `model=` récupère le
   défaut du backend et ré-extrait tout, écrasant le cache. Doit lever, pas détruire.
2. Un arbre entièrement plat (aucun macro subdivisé) est un ÉCHEC de build, pas un
   résultat : la hiérarchie est le produit.
3. Les CLIQUETS des verdicts : `tau`/`RES_LADDER` (pré-filtre) et `sauce_magique`
   (re-coupe) ont été RETIRÉS. Les réintroduire exige de ré-ouvrir le verdict.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from backend import analysis as A
from backend.build_analysis import FlatTreeError, _assert_tree_is_structured
from backend.claims_endpoint import (
    ClaimsCacheModelMismatch,
    _cached_claims_model,
)


# --------------------------------------------------------------------------- #
# 1. Cache de claims : divergence de modèle ⇒ échec explicite, jamais d'écrasement
# --------------------------------------------------------------------------- #
def test_cached_claims_model_lit_la_cle(tmp_path):
    p = tmp_path / "claims.json"
    p.write_text(json.dumps({"model": "mistral-large-latest", "claims": {}}), encoding="utf-8")
    assert _cached_claims_model(p) == "mistral-large-latest"


def test_cached_claims_model_absent_ou_illisible(tmp_path):
    assert _cached_claims_model(tmp_path / "nexiste-pas.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{pas du json", encoding="utf-8")
    assert _cached_claims_model(bad) is None


def test_prepare_claims_leve_sur_modele_divergent(tmp_path, monkeypatch):
    """Le scénario exact qui a détruit le cache mistral-large du rebuild tiktok."""
    import backend.claims_endpoint as CE

    ddir = tmp_path / "tiktok"
    ddir.mkdir()
    (ddir / "claims.json").write_text(
        json.dumps({"model": "mistral-large-latest", "claims": {"a0": []}}), encoding="utf-8")
    (ddir / "meta.json").write_text(json.dumps({"question": "q ?"}), encoding="utf-8")

    monkeypatch.setattr(CE, "dataset_dir", lambda _id: ddir)
    monkeypatch.setattr(CE, "ALLOW_REEXTRACT", False)

    ds = SimpleNamespace(id="tiktok", ideas=[
        SimpleNamespace(id="a0", text="un texte assez long pour passer min_chars",
                        text_clean="un texte assez long pour passer min_chars", weight=1.0)])

    # `model=None` → défaut du backend api (ministral-3b-latest) ≠ clé du cache.
    with pytest.raises(ClaimsCacheModelMismatch) as exc:
        CE.prepare_claims(ds, backend="api", model=None)
    msg = str(exc.value)
    assert "mistral-large-latest" in msg and "AGORA_ALLOW_REEXTRACT" in msg
    # Le cache n'a PAS été touché.
    assert _cached_claims_model(ddir / "claims.json") == "mistral-large-latest"


# --------------------------------------------------------------------------- #
# 3. Arbre plat ⇒ FlatTreeError (avant toute dépense LLM)
# --------------------------------------------------------------------------- #
def _fake_tree(n_macros: int, *, structured: bool):
    nodes = {}
    macros = []
    for i in range(n_macros):
        mid = f"n{i}"
        kids = [f"n{i}c"] if structured and i == 0 else []
        nodes[mid] = SimpleNamespace(id=mid, children=kids, n_claims=100 + i)
        macros.append(mid)
        for c in kids:
            nodes[c] = SimpleNamespace(id=c, children=[], n_claims=50)
    return SimpleNamespace(nodes=nodes, macros=macros)


def test_arbre_plat_leve():
    with pytest.raises(FlatTreeError) as exc:
        _assert_tree_is_structured(_fake_tree(15, structured=False))
    msg = str(exc.value)
    assert "arbre plat" in msg and "AUCUN avec sous-thèmes" in msg   # diagnostic actionnable


def test_arbre_structure_passe():
    _assert_tree_is_structured(_fake_tree(15, structured=True))


def test_corpus_mono_facette_passe():
    """<3 macros : la platitude est un résultat légitime, pas une pathologie."""
    _assert_tree_is_structured(_fake_tree(2, structured=False))


def test_arbre_plat_tolere_si_flag(monkeypatch):
    import backend.build_analysis as B
    monkeypatch.setattr(B, "ALLOW_FLAT_TREE", True)
    B._assert_tree_is_structured(_fake_tree(15, structured=False))   # ne lève pas


# --------------------------------------------------------------------------- #
# 4. Cliquets des verdicts : `tau`/`RES_LADDER`, `sauce_magique`, `_subdivide` retirés
# --------------------------------------------------------------------------- #
def test_tau_et_res_ladder_ont_disparu():
    """`tau` basculait sur 2 claims d'écart ; RES_LADDER ne montait jamais.

    Verdict `.agent/notes/HIERARCHY_TAU.md`. Ce test est le cliquet : si quelqu'un
    réintroduit un pré-filtre, il doit d'abord ré-ouvrir le verdict.
    """
    assert not hasattr(A, "_derive_tau")
    assert not hasattr(A, "RES_LADDER")
    assert "tau" not in A.ThemeTree.__dataclass_fields__


def test_sauce_magique_a_disparu():
    """La re-coupe FRAGMENTAIT des thèmes cohérents ; sa raison d'être (le macro géant de
    granddebat) était un artefact d'ANISOTROPIE, qui s'évapore une fois l'espace recentré
    (top1 : 0.999 → 0.185). La hiérarchie a une seule autorité : la chaîne d'emboîtement.

    Verdict `.agent/notes/HIERARCHY_LAYERS.md`. Cliquet : ne pas la ressusciter en silence.
    """
    import importlib

    import backend.build_analysis as B
    assert not hasattr(A, "recut_tree"), "analysis ne doit plus appliquer de re-coupe"
    assert not hasattr(B, "recut_tree")
    assert "recut" not in A.ThemeTree.__dataclass_fields__
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend.recut")


def test_subdivide_a_disparu():
    """La re-clusterisation Leiden des feuilles (`_subdivide`, pilotée par `derive_k`)
    contredisait la chaîne d'emboîtement et re-découpait des thèmes cohérents. Retirée :
    la hiérarchie a une seule autorité, la chaîne.

    Verdict `.agent/notes/HIERARCHY_LAYERS.md`. Cliquet : ne pas la réintroduire en silence.
    """
    assert not hasattr(A, "_subdivide")
    assert not hasattr(A, "MAX_DEPTH")          # le garde-fou de profondeur n'a plus d'objet


class _Lv:
    def __init__(self, membership, k=0, n=0, cleanliness=1.0):
        self.membership = np.asarray(membership)
        self.k, self.n_clusters, self.cleanliness = k, n, cleanliness


def test_chaine_multi_niveaux_emboitee():
    """L'arbre suit TOUTE la chaîne : on ne fixe pas le nombre de niveaux. Une chaîne à 3
    étages (fin → moyen → grossier) qui s'emboîtent produit un arbre à 3 profondeurs, membres
    ancrés dans les feuilles."""
    # 8 claims. Fin : 4 paires. Moyen : 2 groupes de 2 paires. Grossier : 1... on garde 2 macros.
    fine   = [0, 0, 1, 1, 2, 2, 3, 3]          # 4 clusters fins
    medium = [0, 0, 0, 0, 1, 1, 1, 1]          # 2 groupes (paires 0-1 / 2-3)
    chain = [_Lv(fine, n=4), _Lv(medium, n=2, cleanliness=0.8)]
    forest = A._chain_hierarchy(chain)
    # 2 racines, chacune avec 2 feuilles fines → profondeurs {0,1}
    assert len(forest) == 2
    for members, kids in forest:
        assert len(kids) == 2
        assert sorted(members) == sorted(m for km, _ in kids for m in km)   # membres = ∪ feuilles

    # 3 étages : fin(4) → moyen(2) → grossier(1 seul → déplié, pas de racine redondante)
    coarse = [0, 0, 0, 0, 0, 0, 0, 0]
    forest3 = A._chain_hierarchy([_Lv(fine, n=4), _Lv(medium, n=2), _Lv(coarse, n=1)])
    # la racine unique qui embrasse tout est dépliée → on garde les 2 nœuds du niveau moyen
    assert len(forest3) == 2

    # Construction complète via _build_macro_forest : la profondeur suit la chaîne.
    vecs = np.eye(8, dtype=np.float32)
    texts = ["alpha", "alpha", "beta", "beta", "gamma", "gamma", "delta", "delta"]
    built, _order, macros, _thr = A._build_macro_forest(
        [], vecs, np.ones(8), list(range(8)), texts, hierarchy=forest)
    assert len(macros) == 2                              # 2 racines
    assert max(n.depth for n in built.values()) == 1     # 2 niveaux (feuilles à la profondeur 1)
    assert sum(1 for n in built.values() if not n.children) == 4   # 4 feuilles fines
