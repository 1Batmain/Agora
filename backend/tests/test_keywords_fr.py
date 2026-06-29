"""Tests unitaires — traduction FR des mots-clés au build (`backend.keywords_fr`).

Sans LLM : on monkeypatch `translate_batch` par un dictionnaire fixe. Vérifie le
no-op mono-FR, le remappage (ordre + dédup), la re-dérivation du label et le cache.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend import keywords_fr


def _avis(idx: int, text: str):
    return SimpleNamespace(id=f"a{idx}", text=text)


def _node(keywords, label):
    return SimpleNamespace(keywords=list(keywords), label=label)


def _tree(nodes: dict, avis: list):
    return SimpleNamespace(nodes=nodes, prepared=SimpleNamespace(avis=avis))


# Dictionnaire de traduction DE/IT → FR pour les tests (LLM simulé).
_FR = {
    "iniziativa": "initiative",
    "schweiz": "suisse",
    "lavoro": "travail",
    "frauen": "femmes",
    "volksinitiative": "initiative populaire",
}


@pytest.fixture
def fake_translate(monkeypatch):
    """Remplace `translate_batch` : renvoie la trad du dico, l'original sinon."""
    calls = {"terms": []}

    def _fake(texts, *, model, chat=None):
        calls["terms"].extend(texts)
        return [_FR.get(t, t) for t in texts]

    monkeypatch.setattr(keywords_fr, "translate_batch", _fake)
    return calls


def test_mono_fr_is_noop(tmp_path, monkeypatch, fake_translate):
    """Corpus 100 % FR (lang=fr) → aucun appel LLM, arbre inchangé."""
    monkeypatch.setattr(keywords_fr, "dataset_dir", lambda d: tmp_path)
    nodes = {"n0": _node(["retraites", "travail"], "retraites · travail")}
    tree = _tree(nodes, [_avis(0, "Texte français"), _avis(1, "Autre texte")])
    lang_of = {"a0": "fr", "a1": "fr"}

    out = keywords_fr.translate_tree_keywords("ds", tree, lang_of)

    assert out == {}
    assert fake_translate["terms"] == []  # zéro appel LLM
    assert nodes["n0"].keywords == ["retraites", "travail"]
    assert not (tmp_path / keywords_fr.KEYWORDS_FR_NAME).exists()


def test_multilingual_remap_order_and_label(tmp_path, monkeypatch, fake_translate):
    """Corpus multilingue → mots-clés traduits, ordre préservé, label re-dérivé."""
    monkeypatch.setattr(keywords_fr, "dataset_dir", lambda d: tmp_path)
    nodes = {
        "n0": _node(["iniziativa", "schweiz", "lavoro"], "iniziativa · schweiz · lavoro"),
        "n1": _node(["frauen"], "frauen"),
    }
    tree = _tree(nodes, [_avis(0, "Ein deutscher Text"), _avis(1, "Un texte français")])
    lang_of = {"a0": "de", "a1": "fr"}

    out = keywords_fr.translate_tree_keywords("ds", tree, lang_of)

    assert out["iniziativa"] == "initiative"
    assert nodes["n0"].keywords == ["initiative", "suisse", "travail"]  # ordre préservé
    assert nodes["n0"].label == "initiative · suisse · travail"          # label re-dérivé
    assert nodes["n1"].keywords == ["femmes"]
    assert (tmp_path / keywords_fr.KEYWORDS_FR_NAME).exists()


def test_dedup_on_collision(tmp_path, monkeypatch):
    """Deux termes qui traduisent vers le même FR → dédupliqués, ordre préservé."""
    monkeypatch.setattr(keywords_fr, "dataset_dir", lambda d: tmp_path)

    collide = {"iniziativa": "initiative", "volksinitiative": "initiative"}

    def _fake(texts, *, model, chat=None):
        return [collide.get(t, t) for t in texts]

    monkeypatch.setattr(keywords_fr, "translate_batch", _fake)
    nodes = {"n0": _node(["iniziativa", "volksinitiative", "schweiz"], "x")}
    tree = _tree(nodes, [_avis(0, "Deutscher Text")])
    lang_of = {"a0": "de"}

    keywords_fr.translate_tree_keywords("ds", tree, lang_of)

    assert nodes["n0"].keywords == ["initiative", "schweiz"]  # collision fusionnée


def test_cache_reused_on_second_call(tmp_path, monkeypatch, fake_translate):
    """Un 2ᵉ build ne re-traduit PAS les termes déjà cachés (idempotence)."""
    monkeypatch.setattr(keywords_fr, "dataset_dir", lambda d: tmp_path)
    mk_tree = lambda: _tree(
        {"n0": _node(["iniziativa", "schweiz"], "x")}, [_avis(0, "Deutscher Text")]
    )
    lang_of = {"a0": "de"}

    keywords_fr.translate_tree_keywords("ds", mk_tree(), lang_of)
    first = list(fake_translate["terms"])
    assert sorted(first) == ["iniziativa", "schweiz"]

    fake_translate["terms"].clear()
    tree2 = mk_tree()
    keywords_fr.translate_tree_keywords("ds", tree2, lang_of)
    assert fake_translate["terms"] == []  # tout vient du cache
    assert tree2.nodes["n0"].keywords == ["initiative", "suisse"]


def test_llm_failure_keeps_original(tmp_path, monkeypatch):
    """Échec LLM (None) → terme source conservé, rien de faux mis en cache."""
    monkeypatch.setattr(keywords_fr, "dataset_dir", lambda d: tmp_path)
    monkeypatch.setattr(
        keywords_fr, "translate_batch", lambda texts, *, model, chat=None: [None] * len(texts)
    )
    nodes = {"n0": _node(["iniziativa", "schweiz"], "x")}
    tree = _tree(nodes, [_avis(0, "Deutscher Text")])

    out = keywords_fr.translate_tree_keywords("ds", tree, {"a0": "de"})

    assert out == {}
    assert nodes["n0"].keywords == ["iniziativa", "schweiz"]  # inchangé (repli)
