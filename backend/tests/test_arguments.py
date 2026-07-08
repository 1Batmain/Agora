"""Argument mining — fonctions pures (back-match fail-closed, dédup/rollup, parsing),
persistance de l'artefact À PART `arguments.json` et endpoint `/arguments`.

Comme pour l'opinion : la logique est testée DIRECTEMENT (zéro réseau, vecteurs
synthétiques), l'endpoint en monkeypatchant `analysis_store.read_arguments` — le
chemin « artefact absent » est un contrat dur (les datasets déjà analysés ne seront
pas recalculés : rien ne doit casser sans `arguments.json`).
"""

from __future__ import annotations

from pathlib import Path

from backend import analysis_store
from backend.recluster import DEFAULT_DATASET
from pipeline.cluster import mistral_client


# --------------------------------------------------------------------------- #
# llm_cache : transmission de json_mode (les synthèses d'arguments l'exigent)
# --------------------------------------------------------------------------- #
def test_cached_llm_forwards_json_mode(tmp_path, monkeypatch):
    from backend.llm_cache import cached_llm

    seen = {}

    def fake_chat(messages, *, model, temperature, max_tokens, json_mode=False, timeout=None):
        seen.update(json_mode=json_mode, model=model)
        return '{"arguments": []}'

    monkeypatch.setattr(mistral_client, "available", lambda: True)
    monkeypatch.setattr(mistral_client, "chat", fake_chat)
    value, source = cached_llm(
        mem_cache={}, key="k", disk_path=tmp_path / "k.json",
        build_messages=lambda: [{"role": "user", "content": "x"}],
        fallback_fn=lambda reason, exc=None: "repli",
        model="m", max_tokens=64, temperature=0.2, json_mode=True,
    )
    assert seen["json_mode"] is True
    assert source == "generated"


# --------------------------------------------------------------------------- #
# Store : trio arguments_path / read_arguments / write_arguments
# --------------------------------------------------------------------------- #
def test_arguments_store_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis_store, "arguments_path",
                        lambda _ds: tmp_path / "arguments.json")
    assert analysis_store.read_arguments("ds") is None  # absent → None gracieux
    payload = {"dataset": "ds", "themes": [{"theme_id": "n1", "arguments": []}]}
    analysis_store.write_arguments("ds", payload)
    assert analysis_store.read_arguments("ds") == payload


def test_arguments_path_is_a_standalone_artifact():
    p = analysis_store.arguments_path("demo")
    assert p.name == "arguments.json"
    assert p.parent == analysis_store.analysis_dir("demo")


# --------------------------------------------------------------------------- #
# Back-matching fail-closed (vecteurs synthétiques, zéro LLM)
# --------------------------------------------------------------------------- #
def _unit(v):
    import numpy as np
    a = np.asarray(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def test_back_match_keeps_supported_and_drops_orphan():
    import numpy as np
    from backend.build_arguments import back_match

    a0 = _unit([1.0, 0.1, 0.0])   # argument soutenu
    a1 = _unit([0.0, 0.0, 1.0])   # argument orthogonal aux claims → droppé
    args = np.stack([a0, a1])
    group = np.stack([_unit([1.0, 0.0, 0.0]), _unit([1.0, 0.2, 0.0]),
                      _unit([0.9, 0.3, 0.0]), _unit([0.0, 1.0, 0.0])])  # dernier hors-sujet
    kept = back_match(args, group, sim_threshold=0.60, min_support=3)
    assert [m.arg_index for m in kept] == [0]
    m = kept[0]
    assert len(m.assigned) == 3          # le claim hors-sujet n'est pas compté
    sims = [s for _, s in zip(m.assigned, m.sims)]
    assert sims == sorted(m.sims, reverse=True)  # sources triées par similarité desc


def test_back_match_argmax_is_exclusive():
    """Un claim proche de DEUX arguments n'est compté qu'une fois (Σ support ≤ n)."""
    import numpy as np
    from backend.build_arguments import back_match

    a0 = _unit([1.0, 0.0])
    a1 = _unit([0.95, 0.31])      # très proche de a0
    group = np.stack([_unit([1.0, 0.05]) for _ in range(4)])
    kept = back_match(np.stack([a0, a1]), group, sim_threshold=0.60, min_support=1)
    total = sum(len(m.assigned) for m in kept)
    assert total <= len(group)


def test_back_match_min_support_boundary():
    import numpy as np
    from backend.build_arguments import back_match

    arg = _unit([1.0, 0.0])
    group = np.stack([_unit([1.0, 0.1]), _unit([1.0, 0.2])])  # 2 soutiens seulement
    assert back_match(arg[None, :], group, sim_threshold=0.60, min_support=3) == []
    kept = back_match(arg[None, :], group, sim_threshold=0.60, min_support=2)
    assert len(kept) == 1 and len(kept[0].assigned) == 2


def test_dedup_candidates_greedy():
    """Deux candidats quasi identiques → le second est absorbé (ordre LLM préservé)."""
    import numpy as np
    from backend.build_arguments import dedup_candidates

    v = _unit([1.0, 0.0, 0.0])
    w = _unit([0.99, 0.14, 0.0])   # cos ≈ 0.99
    x = _unit([0.0, 1.0, 0.0])
    assert dedup_candidates(np.stack([v, w, x]), threshold=0.85) == [0, 2]


# --------------------------------------------------------------------------- #
# Rollup parents — fusion par embeddings, sommes honnêtes, cap
# --------------------------------------------------------------------------- #
def _arg_entry(theme_id, text, n_support, vec, sources=None):
    return {
        "theme_id": theme_id, "stance": "pour", "argument": text,
        "n_support": n_support, "weight": float(n_support), "share": 0.5,
        "sources": sources or [{"avis_id": f"a{n_support}", "claim_id": f"a{n_support}#0",
                                "text": text, "similarity": 0.7}],
        "_vec": vec,
    }


def test_rollup_merges_near_duplicates_and_sums_support():
    from backend.build_arguments import merge_arguments

    v = _unit([1.0, 0.0])
    w = _unit([0.99, 0.14])
    x = _unit([0.0, 1.0])
    merged = merge_arguments(
        [_arg_entry("n2", "argument A", 5, v),
         _arg_entry("n3", "argument A bis", 3, w),   # fusionné dans A
         _arg_entry("n4", "argument B", 2, x)],
        dedup_threshold=0.85, parent_max=7, top_sources=5)
    assert len(merged) == 2
    top = merged[0]
    assert top["n_support"] == 8 and top["weight"] == 8.0
    assert sorted(top["merged_from"]) == ["n2", "n3"]
    assert len(top["sources"]) == 2  # sources des deux feuilles, re-triées
    assert merged[1]["n_support"] == 2


def test_rollup_respects_parent_max():
    from backend.build_arguments import merge_arguments

    entries = [_arg_entry(f"n{i}", f"argument {i}", 10 - i,
                          _unit([1.0 if j == i else 0.0 for j in range(5)]))
               for i in range(5)]
    merged = merge_arguments(entries, dedup_threshold=0.99, parent_max=3, top_sources=5)
    assert len(merged) == 3
    assert [m["n_support"] for m in merged] == [10, 9, 8]  # tri par support desc


# --------------------------------------------------------------------------- #
# Parsing durci de la SÉLECTION LLM (V-SELECT) — l'invariant verbatim tient PAR le fait
# que les indices renvoyés sont TOUJOURS bornés à [0, n) : un argument = group[indice] =
# un claim RÉEL (jamais une phrase inventée).
# --------------------------------------------------------------------------- #
def test_parse_selected_bounds_and_dedup():
    from backend.build_arguments import _parse_selected

    # bornés à [0, n), dédupliqués, ordre préservé (5 hors bornes écarté, 2 dédup)
    assert _parse_selected('{"selected": [2, 0, 2, 5]}', n=4, max_k=5) == [2, 0]
    # dict-form {"i":k} toléré (phrasé de mistral-large)
    assert _parse_selected('{"selected": [{"i": 1}, {"i": 3}]}', n=4, max_k=5) == [1, 3]
    # tronqué à max_k
    assert _parse_selected('{"selected": [0, 1, 2, 3, 4, 5]}', n=10, max_k=3) == [0, 1, 2]


def test_parse_selected_hardened():
    from backend.build_arguments import _parse_selected

    assert _parse_selected('{"selected": ["2", null, "x", 3]}', n=4, max_k=5) == [2, 3]
    assert _parse_selected("pas du json", n=4, max_k=5) == []
    assert _parse_selected('{"autre": 1}', n=4, max_k=5) == []
    assert _parse_selected('{"selected": "pas une liste"}', n=4, max_k=5) == []
    # tous les indices renvoyés sont dans les bornes → argument toujours un claim réel (verbatim)
    got = _parse_selected('{"selected": [-1, 0, 99, 3]}', n=4, max_k=5)
    assert all(0 <= i < 4 for i in got) and got == [0, 3]


# --------------------------------------------------------------------------- #
# Endpoint /arguments — dégradation gracieuse (contrat de rétro-compat)
# --------------------------------------------------------------------------- #
def test_arguments_absent_is_graceful(client, monkeypatch):
    """Artefact non baké (tous les datasets existants) → 200 + liste vide."""
    monkeypatch.setattr(analysis_store, "read_arguments", lambda _ds: None)
    r = client.get("/arguments", params={"dataset": DEFAULT_DATASET})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dataset"] == DEFAULT_DATASET
    assert body["themes"] == [] and body["status"] == "absent"


def test_arguments_served_shape(client, monkeypatch):
    fixture = {
        "dataset": DEFAULT_DATASET,
        "model": "mistral-small-latest",
        "themes": [
            {"theme_id": "n1", "mode": "pour_contre", "proposition": "instaurer le RIC",
             "arguments": [
                 {"id": "n1:pour:0", "theme_id": "n1", "stance": "pour",
                  "argument": "Renforce la participation directe des citoyens",
                  "n_support": 4, "weight": 4.0, "share": 0.4,
                  "sources": [{"avis_id": "a1", "claim_id": "a1#0",
                               "text": "le RIC redonnerait la parole au peuple",
                               "similarity": 0.71}]},
             ]},
        ],
    }
    monkeypatch.setattr(analysis_store, "read_arguments", lambda _ds: fixture)
    r = client.get("/arguments", params={"dataset": DEFAULT_DATASET})
    assert r.status_code == 200, r.text
    themes = r.json()["themes"]
    assert len(themes) == 1
    arg = themes[0]["arguments"][0]
    assert set(arg) >= {"id", "stance", "argument", "n_support", "sources"}
    src = arg["sources"][0]
    assert set(src) >= {"avis_id", "claim_id", "text", "similarity"}


def test_arguments_unknown_dataset(client):
    """Id non whitelisté → 404 (garde path-traversal partagée)."""
    r = client.get("/arguments", params={"dataset": "../etc"})
    assert r.status_code == 404, r.text
