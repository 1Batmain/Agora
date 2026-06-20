"""Relabel un graph.json existant — réutilise les clusters (PAS de ré-embed).

Recalcule uniquement les `label`/`keywords` des thèmes avec le nouveau nommage
c-TF-IDF + mots-vides corpus-dérivés. Reproductible (aucun aléa, naming = texte).

    uv run python -m pipeline.cluster.relabel_graph IN.json OUT.json
"""

from __future__ import annotations

import json
import sys

from pipeline.cluster.naming import derive_corpus_stopwords, name_clusters


def relabel(graph: dict) -> dict:
    text_by_id = {n["id"]: n.get("props", {}).get("text", "") for n in graph["nodes"]}
    themes = graph["themes"]

    # Corpus global = un avis unique = un document (déduplique les ids).
    all_ids = {i for t in themes for i in t["member_ids"]}
    corpus_stop, diag = derive_corpus_stopwords([text_by_id[i] for i in all_ids])

    by_level: dict[int, list[dict]] = {}
    for t in themes:
        by_level.setdefault(t["level"], []).append(t)

    new_names: dict[tuple[int, int], dict] = {}  # (level, cluster_id) -> name

    # Macros (level 0) : contrastés entre eux.
    macros = by_level.get(0, [])
    macro_docs = {t["cluster_id"]: [text_by_id[i] for i in t["member_ids"]] for t in macros}
    for cid, nm in name_clusters(macro_docs, corpus_stopwords=corpus_stop).items():
        new_names[(0, cid)] = nm

    # Sous-thèmes (level 1) : contrastés DANS leur macro parent.
    subs = by_level.get(1, [])
    subs_by_parent: dict[int, list[dict]] = {}
    for t in subs:
        subs_by_parent.setdefault(t["parent_id"], []).append(t)
    for _parent, children in subs_by_parent.items():
        sub_docs = {t["cluster_id"]: [text_by_id[i] for i in t["member_ids"]] for t in children}
        for cid, nm in name_clusters(sub_docs, corpus_stopwords=corpus_stop).items():
            new_names[(1, cid)] = nm

    for t in themes:
        nm = new_names.get((t["level"], t["cluster_id"]))
        if nm:
            t["label"] = nm["label"]
            t["keywords"] = nm["keywords"]

    graph.setdefault("meta", {})["naming"] = {
        "method": "c-TF-IDF + corpus-derived stopwords (document max_df cutoff + multilingual functional)",
        "functional_source": diag.get("functional_source"),
        "domain_cutoff_df": diag.get("domain_cutoff_df"),
        "n_domain_stopwords": diag.get("n_domain"),
        "domain_examples": diag.get("domain_examples"),
        "n_functional": diag.get("n_functional"),
        "n_stopwords": diag.get("n_stopwords"),
    }
    return graph


def main() -> None:
    in_path = sys.argv[1] if len(sys.argv) > 1 else "frontend/public/graph.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else in_path
    with open(in_path, encoding="utf-8") as f:
        graph = json.load(f)
    graph = relabel(graph)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False)
    print(f"relabeled → {out_path}")


if __name__ == "__main__":
    main()
