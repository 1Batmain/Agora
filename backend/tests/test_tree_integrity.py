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
# 3. Partition DÉGÉNÉRÉE (un thème avale tout) ⇒ FlatTreeError (avant dépense LLM)
# --------------------------------------------------------------------------- #
def _fake_tree(tailles: list[int]):
    """Arbre PLAT (chaque thème = un macro sans enfants) aux tailles données."""
    nodes, macros = {}, []
    for i, t in enumerate(tailles):
        mid = f"n{i}"
        nodes[mid] = SimpleNamespace(id=mid, children=[], n_claims=t)
        macros.append(mid)
    return SimpleNamespace(nodes=nodes, macros=macros)


def test_partition_saine_plate_passe():
    """Une partition plate ÉQUILIBRÉE est un résultat légitime (la hiérarchie vient après)."""
    _assert_tree_is_structured(_fake_tree([300, 250, 200, 180, 120, 90]))


def test_theme_geant_leve():
    with pytest.raises(FlatTreeError) as exc:
        _assert_tree_is_structured(_fake_tree([950, 20, 15, 10, 5]))   # un thème = 94 %
    assert "avale" in str(exc.value)


def test_mono_theme_passe():
    """<2 thèmes : mono-thème légitime."""
    _assert_tree_is_structured(_fake_tree([1000]))


def test_theme_geant_tolere_si_flag(monkeypatch):
    import backend.build_analysis as B
    monkeypatch.setattr(B, "ALLOW_FLAT_TREE", True)
    B._assert_tree_is_structured(_fake_tree([950, 20, 15, 10, 5]))     # ne lève pas


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


def test_flat_partition_pic_de_modularite():
    """`flat_partition` balaie γ sur UN graphe fixe et renvoie la partition au pic de
    modularité. Sur 2 blobs nets, elle doit dégager ≥2 thèmes propres."""
    from pipeline.cluster import layers
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (120, 32)) + 8 * np.eye(32)[0]
    b = rng.normal(0, 1, (120, 32)) - 8 * np.eye(32)[0]
    V = np.vstack([a, b])
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    membership, meta = layers.flat_partition(layers.centre(V), seed=42)
    assert meta["n_clusters"] >= 2
    assert 0.0 <= meta["modularity"] <= 1.0
    assert meta["gamma"] in layers.GAMMA_GRID
    assert len(membership) == len(V)


def test_arbre_plat_via_macro_forest():
    """Partition plate → `_build_macro_forest` construit un arbre à profondeur 0 (chaque
    cluster = un thème, aucun enfant)."""
    vecs = np.eye(6, dtype=np.float32)
    texts = ["addiction scroll compulsif", "addiction écran dépendance",
             "harcèlement commentaires haineux", "harcèlement scolaire propos",
             "comparaison corps physique", "corps féminin normes beauté"]
    hierarchy = [([0, 1], []), ([2, 3], []), ([4, 5], [])]     # 3 thèmes plats
    built, _order, macros, _thr = A._build_macro_forest(
        [], vecs, np.ones(6), list(range(6)), texts, hierarchy=hierarchy)
    assert len(macros) == 3
    assert max(n.depth for n in built.values()) == 0          # tout à plat
    assert all(not n.children for n in built.values())
