"""Intégrité de l'arbre servi — trois régressions constatées sur le rebuild tiktok.

1. `build_theme_tree` applique la re-coupe sauce_magique LUI-MÊME. Sinon
   `build_opinion`/`build_arguments`, qui rappellent `build_theme_tree` de leur côté,
   travaillent sur la façade PRÉ-coupe et émettent des `theme_id` que `analysis.json`
   ne contient pas (thème fantôme servi).
2. Le cache de claims est clé par MODÈLE : un appelant qui oublie `model=` récupère le
   défaut du backend et ré-extrait tout, écrasant le cache. Doit lever, pas détruire.
3. Un arbre entièrement plat (aucun macro subdivisé) est un ÉCHEC de build, pas un
   résultat : la hiérarchie est le produit.
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
# 1. La re-coupe est appliquée DANS build_theme_tree (tous les builders, même arbre)
# --------------------------------------------------------------------------- #
def test_build_theme_tree_applique_la_recoupe(monkeypatch):
    """Le contrat : construire l'arbre ⇒ la façade est déjà la coupe optimale."""
    appels: list[object] = []

    def spy(tree, **kw):
        appels.append(tree)
        return None                       # no-op : façade déjà optimale

    monkeypatch.setattr(A, "recut_tree", spy)

    # Un `prepared` minimal suffit : 6 claims, 2 avis, vecteurs 4-D normalisés.
    import numpy as np
    vecs = np.array([[1.0, 0, 0, 0], [0.99, 0.14, 0, 0], [0.98, 0.2, 0, 0],
                     [0, 1.0, 0, 0], [0.14, 0.99, 0, 0], [0.2, 0.98, 0, 0]],
                    dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    textes = ["algorithme addictif scroll", "algorithme recommandation vidéo",
              "addiction scroll infini", "harcèlement commentaires haineux",
              "harcèlement scolaire réseau", "propos haineux modération"]
    prepared = SimpleNamespace(
        avis=[SimpleNamespace(id="a0", text="x", weight=1.0),
              SimpleNamespace(id="a1", text="y", weight=1.0)],
        claims_by_id={}, claim_texts=textes, claim_owner=[0, 0, 0, 1, 1, 1],
        claim_weight=np.ones(6), claim_vecs=vecs,
        claim_spans=[[(0, 1)]] * 6, claim_target=[None] * 6,
        target_vecs=np.zeros((6, 4), dtype=np.float32),
        target_mask=np.zeros(6, dtype=bool),
        backend=None, model="m", embedder="e", min_chars=1, extracted=0,
    )
    A.build_theme_tree(SimpleNamespace(id="t", ideas=[]), prepared=prepared, seed=42)
    assert len(appels) == 1, "build_theme_tree doit appliquer recut_tree exactement une fois"


def test_build_analysis_nappelle_plus_recut_lui_meme():
    """La re-coupe ne doit plus être câblée chez UN seul appelant (cause du thème fantôme)."""
    import backend.build_analysis as B
    assert not hasattr(B, "recut_tree"), (
        "build_analysis ne doit plus importer recut_tree : la re-coupe appartient "
        "à build_theme_tree, sinon opinion/arguments voient un autre arbre."
    )


# --------------------------------------------------------------------------- #
# 2. Cache de claims : divergence de modèle ⇒ échec explicite, jamais d'écrasement
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
def _fake_tree(n_macros: int, *, structured: bool, tau: float = 0.2145):
    nodes = {}
    macros = []
    for i in range(n_macros):
        mid = f"n{i}"
        kids = [f"n{i}c"] if structured and i == 0 else []
        nodes[mid] = SimpleNamespace(id=mid, children=kids, dispersion=0.18 + 0.001 * i)
        macros.append(mid)
        for c in kids:
            nodes[c] = SimpleNamespace(id=c, children=[], dispersion=0.1)
    return SimpleNamespace(nodes=nodes, macros=macros, tau=tau)


def test_arbre_plat_leve():
    with pytest.raises(FlatTreeError) as exc:
        _assert_tree_is_structured(_fake_tree(15, structured=False))
    msg = str(exc.value)
    assert "AUCUN subdivisé" in msg
    assert "tau" in msg and "0.2145" in msg      # le diagnostic doit être actionnable


def test_arbre_structure_passe():
    _assert_tree_is_structured(_fake_tree(15, structured=True))


def test_corpus_mono_facette_passe():
    """<3 macros : la platitude est un résultat légitime, pas une pathologie."""
    _assert_tree_is_structured(_fake_tree(2, structured=False))


def test_arbre_plat_tolere_si_flag(monkeypatch):
    import backend.build_analysis as B
    monkeypatch.setattr(B, "ALLOW_FLAT_TREE", True)
    B._assert_tree_is_structured(_fake_tree(15, structured=False))   # ne lève pas
