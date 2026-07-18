"""Moteur d'abstraction — fonctions pures (chat_fn injecté, zéro dépendance LLM réelle)."""
from __future__ import annotations

import json
from pathlib import Path

from pipeline.cluster import abstraction as ab


def _fake_chat(labels_by_text):
    """chat_fn simulé : étiquette selon un mot-clé, groupe addiction+scroll ensemble."""
    def chat(messages, **_kw):
        sys = messages[0]["content"]
        user = messages[-1]["content"]
        if "groupes" in sys or "Regroupe" in sys:      # étape regroupement
            # 0,1 = addiction (même groupe) ; 2 = harcèlement ; 3 = corps
            return json.dumps({"groupes": [
                {"titre": "Addiction", "indices": [0, 1]},
                {"titre": "Harcèlement", "indices": [2]},
                {"titre": "Image du corps", "indices": [3]},
            ]})
        for kw, lab in labels_by_text.items():         # étape étiquette
            if kw in user:
                return lab
        return "autre"
    return chat


def test_compute_fusionne_les_redondants():
    chat = _fake_chat({"scroll": "addiction aux réseaux", "appli": "addiction aux réseaux",
                       "haine": "harcèlement en ligne", "corps": "image du corps"})
    clusters = [["scroll heures"], ["appli désinstaller"], ["haine insultes"], ["corps comparaison"]]
    r = ab.compute(clusters, chat_fn=chat, model="x")
    assert r is not None
    assert r["assign"][0] == r["assign"][1]              # les 2 addictions fusionnent
    assert len({r["assign"][0], r["assign"][2], r["assign"][3]}) == 3   # 3 macros distincts
    assert len(r["macros"]) == 3


def test_dedup_partition_stricte():
    """Un thème double-assigné par le LLM ne va que dans UN macro (premier qui le réclame)."""
    def chat(messages, **_kw):
        if "groupes" in messages[0]["content"]:
            return json.dumps({"groupes": [
                {"titre": "A", "indices": [0, 1]},
                {"titre": "B", "indices": [1, 2, 3]},      # 1 réclamé deux fois
            ]})
        return "label"
    r = ab.compute([["a"], ["b"], ["c"], ["d"]], chat_fn=chat, model="x")
    assert r["assign"][1] == 0                            # thème 1 → premier groupe (A)
    assert sum(1 for a in r["assign"] if a == r["assign"][1]) == 2   # A = {0,1}


def test_corpus_trop_petit_pas_dabstraction():
    assert ab.compute([["a"], ["b"]], chat_fn=lambda *a, **k: "x", model="m") is None


def test_cache_par_signature(tmp_path):
    clusters = [[0, 1], [2, 3], [4, 5]]
    result = {"labels": ["x"], "macros": ["M"], "assign": [0, 0, 1]}
    p = tmp_path / "abstraction.json"
    ab.save(p, clusters, result)
    assert ab.load(p, clusters) == result                # même partition → hit
    assert ab.load(p, [[0], [1, 2, 3, 4, 5]]) is None     # partition changée → miss
