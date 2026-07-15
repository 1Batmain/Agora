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
    la hiérarchie a une seule autorité, la chaîne (2 niveaux mesurés).

    Verdict `.agent/notes/HIERARCHY_LAYERS.md`. Cliquet : ne pas la réintroduire en silence.
    """
    assert not hasattr(A, "_subdivide")
    assert not hasattr(A, "MAX_DEPTH")          # le garde-fou de profondeur n'a plus d'objet


def test_arbre_de_la_chaine_a_deux_niveaux(monkeypatch):
    """`_build_subtree` n'attache QUE les `forced_children` (les thèmes fins de la chaîne) :
    un nœud sans enfants forcés est une feuille — plus de descente au-delà de 2 niveaux."""
    import numpy as np
    vecs = np.eye(4, dtype=np.float32)[[0, 1, 2, 3]]
    nodes, order = {}, []
    # macro avec 2 clusters fins forcés → profondeur 1 ; les fins n'ont pas d'enfants forcés.
    A._build_subtree([0, 1, 2, 3], None, 0, [0], nodes, order, vecs,
                     np.ones(4), [0, 1, 2, 3], forced_children=[[0, 1], [2, 3]])
    depths = sorted(n.depth for n in nodes.values())
    assert depths == [0, 1, 1]                  # 1 macro + 2 feuilles, jamais de niveau 2
    assert all(not nodes[c].children for c in nodes if nodes[c].depth == 1)
