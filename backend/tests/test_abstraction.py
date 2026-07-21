"""Moteur d'abstraction B — profil ré-embeddé (chat_fn / embed_fn injectés, zéro LLM réel)."""
from __future__ import annotations

import json

import numpy as np

from pipeline.cluster import abstraction as ab


def test_compute_fusionne_par_reembedding():
    """Profil par thème → ré-embedding → clustering : deux thèmes au profil proche fusionnent."""
    # 4 thèmes : 0 et 1 « addiction » (profils quasi identiques), 2 et 3 distincts.
    prof = {0: "addiction aux réseaux", 1: "addiction aux réseaux et écrans",
            2: "harcèlement en ligne", 3: "image du corps et comparaison"}

    def chat(messages, **_kw):
        u = messages[-1]["content"]
        if "addiction" in u or "scroll" in u or "appli" in u:
            return "Sujet : addiction aux réseaux sociaux. Les témoignages décrivent la dépendance."
        if "haine" in u:
            return "Sujet : harcèlement en ligne. Insultes et menaces."
        return "Sujet : image du corps. Comparaison et normes de beauté."

    def emb(texts):
        # embeddings jouets : par mot-clé du profil → addiction proche, autres orthogonaux
        def v(t):
            if "addiction" in t: return [1.0, 0.0, 0.0]
            if "harcèlement" in t: return [0.0, 1.0, 0.0]
            return [0.0, 0.0, 1.0]
        return np.array([v(t) for t in texts], dtype=float)

    clusters = [["scroll heures"], ["appli désinstaller"], ["haine insultes"], ["corps comparaison"]]
    r = ab.compute(clusters, chat_fn=chat, embed_fn=emb, model="x")
    assert r is not None
    assert r["assign"][0] == r["assign"][1]                 # les 2 addictions fusionnent
    assert len(set(r["assign"])) >= 2                       # ≥2 macros
    assert "profiles" in r and len(r["profiles"]) == 4


def test_corpus_trop_petit_pas_dabstraction():
    assert ab.compute([["a"], ["b"]], chat_fn=lambda *a, **k: "x",
                      embed_fn=lambda t: np.zeros((len(t), 3)), model="m") is None


def test_cache_par_signature(tmp_path):
    clusters = [[0, 1], [2, 3], [4, 5]]
    result = {"profiles": ["x"], "assign": [0, 0, 1]}
    p = tmp_path / "abstraction.json"
    ab.save(p, clusters, result, embedder="nomic-v2", chat_model="m")
    assert ab.load(p, clusters, embedder="nomic-v2", chat_model="m") == result   # même clé → hit
    assert ab.load(p, [[0], [1, 2, 3, 4, 5]], embedder="nomic-v2", chat_model="m") is None  # partition ≠ → miss


def test_cache_invalide_si_embedder_differe(tmp_path):
    """Un cache construit avec un embedder n'est JAMAIS re-servi pour un autre (licence +
    cohérence d'espace) : jina ≠ nomic-v2 même à partition identique."""
    clusters = [[0, 1], [2, 3], [4, 5]]
    result = {"profiles": ["x"], "assign": [0, 0, 1]}
    p = tmp_path / "abstraction.json"
    ab.save(p, clusters, result, embedder="tomaarsen/jina-embeddings-v3-hf", chat_model="m")
    assert ab.load(p, clusters, embedder="nomic-v2", chat_model="m") is None     # embedder ≠ → miss
    assert ab.load(p, clusters, embedder="tomaarsen/jina-embeddings-v3-hf",
                   chat_model="m") == result                                     # même embedder → hit
