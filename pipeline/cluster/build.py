"""Pipeline batch NLP → GraphPayload coloré par thème.

  ideas.jsonl → embed (e5-small) → graphe k-NN cosine → Leiden →
  scoring + naming TF-IDF → graph.json {meta, nodes, links, themes}

Usage :
    uv run python -m pipeline.cluster.build [--input PATH] [--out PATH]
        [--k 10] [--threshold 0.80] [--resolution 1.0] [--seed 42]
        [--fixture]   # écrit aussi pipeline/cluster/fixtures/graph.sample.json

Le format de sortie est le GraphPayload du contrat cross-lane (aligné dummy) :
    nodes : GraphNode{ id, type, label, props{...}, cluster_id, color }
    links : GraphLink{ source, target, type, props{weight} }
    themes: Theme{ cluster_id, member_ids, size, weight_sum, diversity,
                   consensus, centroid, label, keywords, color }
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pipeline.cluster import io
from pipeline.cluster.adaptive import EDGE_SIGMA, derive_defaults
from pipeline.cluster.dedup import dedup_near
from pipeline.cluster.hierarchy import (
    DEFAULT_RESOLUTION_MACRO,
    DEFAULT_RESOLUTION_SUB,
    run_hierarchical,
)
from pipeline.cluster.knn import build_knn_graph
from pipeline.cluster.leiden_cluster import run_leiden
from pipeline.cluster.naming import derive_corpus_stopwords, name_clusters
from pipeline.cluster.palette import color_for
from pipeline.cluster.scoring import rank_clusters, score_cluster
from pipeline.embed.embedder import Embedder

REPO_ROOT = io.REPO_ROOT
DEFAULT_OUT = REPO_ROOT / "data" / "graph.json"
FIXTURE_OUT = REPO_ROOT / "pipeline" / "cluster" / "fixtures" / "graph.sample.json"


def _node_props(idea, weight: float) -> dict:
    return {
        "text": idea.text,
        "text_clean": idea.text_clean or idea.text,
        "ts": idea.ts,
        "lang": idea.lang,
        "author_hash": idea.author_hash,
        "source": idea.source,
        "weight": round(float(weight), 4),
    }


def _node_label(idea) -> str:
    return (idea.text[:80] + "…") if len(idea.text) > 80 else idea.text


def _theme_entry(cid, member_idxs, ideas, score, name, color, *,
                 level, parent_id, children) -> dict:
    """Une entrée `Theme` du contrat (commune aux deux niveaux)."""
    return {
        "cluster_id": cid,
        "level": level,
        "parent_id": parent_id,
        "children": children,
        "member_ids": [ideas[i].id for i in member_idxs],
        "size": score.size,
        "weight_sum": score.weight_sum,
        "diversity": score.diversity,
        "consensus": score.consensus,
        "centroid": score.centroid,
        "label": name["label"],
        "keywords": name["keywords"],
        "color": color,
    }


def _build_flat(ideas, vecs, weights, knn, *, resolution, seed, with_hdbscan,
                dup_threshold):
    """Mode PLAT (non-régression) : Leiden 1 niveau. level=0, sans hiérarchie."""
    leiden = run_leiden(knn, resolution=resolution, seed=seed)
    membership = leiden.membership

    hdbscan_meta = None
    if with_hdbscan:
        from pipeline.cluster import hdbscan_contender as hc

        if hc.available():
            res = hc.run_hdbscan(vecs, seed=seed)
            hdbscan_meta = {
                "n_clusters": res.n_clusters,
                "n_noise": res.n_noise,
                "params": res.params,
            }
        else:
            hdbscan_meta = {"available": False}

    members: dict[int, list[int]] = defaultdict(list)
    for idx, cid in enumerate(membership):
        members[cid].append(idx)

    scores = {cid: score_cluster(idxs, vecs, weights, dup_threshold=dup_threshold)
              for cid, idxs in members.items()}
    cluster_docs = {cid: [ideas[i].text for i in idxs] for cid, idxs in members.items()}
    # Mots-vides saturants dérivés du corpus GLOBAL (un avis = un document).
    corpus_stop, _ = derive_corpus_stopwords([idea.text for idea in ideas])
    names = name_clusters(cluster_docs, corpus_stopwords=corpus_stop)

    n_colors = leiden.n_clusters
    nodes = []
    for idx, idea in enumerate(ideas):
        cid = int(membership[idx])
        nodes.append({
            "id": idea.id,
            "type": "idea",
            "label": _node_label(idea),
            "props": _node_props(idea, weights[idx]),
            "cluster_id": cid,
            "color": color_for(cid, n_colors),
        })

    themes = []
    for cid in rank_clusters(scores):
        nm = names.get(cid, {"label": f"thème {cid}", "keywords": []})
        themes.append(_theme_entry(
            cid, members[cid], ideas, scores[cid], nm, color_for(cid, n_colors),
            level=0, parent_id=None, children=[],
        ))

    clustering_meta = {
        "mode": "flat",
        "primary": "leiden",
        "leiden": {
            "n_clusters": leiden.n_clusters,
            "modularity": leiden.modularity,
            "resolution": leiden.resolution,
            "seed": leiden.seed,
        },
        "hdbscan_contender": hdbscan_meta,
    }
    return nodes, themes, clustering_meta


# Label du groupe « bruit » HDBSCAN (cluster_id = -1). UI, pas un mot de corpus.
NOISE_LABEL = "non classé"


def _build_hdbscan(ideas, vecs, weights, *,
                   min_cluster_size, min_samples, umap_n_neighbors, seed,
                   dup_threshold=None):
    """Mode UMAP+HDBSCAN : clusters PLATS (level=0) + bruit (cluster_id=-1).

    UMAP(5D)→HDBSCAN sur les vecteurs cachés (pas de ré-embed). Les nœuds non
    assignés tombent dans le groupe « non classé » (-1). Une UMAP-2D fournit des
    coords `(x,y)` par nœud (affichage 2D futur ; le circle packing reste défaut).
    Même shape GraphPayload que les autres modes.
    """
    from pipeline.cluster import hdbscan_contender as hc

    if not hc.available():
        raise RuntimeError(
            "Méthode 'hdbscan' indisponible : installe les extras contender "
            "(uv run --extra contender …) pour umap-learn + hdbscan."
        )

    res = hc.run_hdbscan(
        vecs,
        n_neighbors=umap_n_neighbors,
        n_components=hc.N_COMPONENTS,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        seed=seed,
        compute_2d=True,
    )
    membership = res.membership
    coords = res.coords_2d or [[0.0, 0.0]] * len(ideas)

    members: dict[int, list[int]] = defaultdict(list)
    for idx, cid in enumerate(membership):
        members[cid].append(idx)

    scores = {cid: score_cluster(idxs, vecs, weights, dup_threshold=dup_threshold)
              for cid, idxs in members.items()}

    # Naming TF-IDF des clusters réels (le bruit a un label fixe « non classé »).
    real_ids = [cid for cid in members if cid >= 0]
    cluster_docs = {cid: [ideas[i].text for i in members[cid]] for cid in real_ids}
    corpus_stop, _ = derive_corpus_stopwords([idea.text for idea in ideas])
    names = name_clusters(cluster_docs, corpus_stopwords=corpus_stop) if cluster_docs else {}

    n_colors = max(1, res.n_clusters)
    nodes = []
    for idx, idea in enumerate(ideas):
        cid = int(membership[idx])
        x, y = coords[idx]
        nodes.append({
            "id": idea.id,
            "type": "idea",
            "label": _node_label(idea),
            "props": _node_props(idea, weights[idx]),
            "cluster_id": cid,
            "macro_id": cid,        # plat : le nœud EST son macro (pas de hiérarchie)
            "color": color_for(cid, n_colors),
            "x": x,
            "y": y,
        })

    # Thèmes plats : clusters réels rangés par intérêt, puis le bruit en dernier.
    themes = []
    for cid in rank_clusters({c: scores[c] for c in real_ids}):
        nm = names.get(cid, {"label": f"thème {cid}", "keywords": []})
        themes.append(_theme_entry(
            cid, members[cid], ideas, scores[cid], nm, color_for(cid, n_colors),
            level=0, parent_id=None, children=[],
        ))
    if -1 in members:
        themes.append(_theme_entry(
            -1, members[-1], ideas, scores[-1],
            {"label": NOISE_LABEL, "keywords": []}, color_for(-1),
            level=0, parent_id=None, children=[],
        ))

    clustering_meta = {
        "mode": "flat",
        "primary": "hdbscan",
        "hdbscan": {
            "n_clusters": res.n_clusters,
            "n_noise": res.n_noise,
            "params": res.params,
        },
    }
    return nodes, themes, clustering_meta


def _build_hierarchical(ideas, vecs, weights, knn, *,
                        resolution_macro, resolution_sub, min_sub_size, seed,
                        dup_threshold=None):
    """Mode HIÉRARCHIQUE : macro (level=0) → sous-thèmes (level=1).

    Nœud coloré par le MACRO parent ; `cluster_id` = feuille (sous-thème),
    `macro_id` = macro parent. Naming TF-IDF inchangé : macro inter-macros,
    sous-thème contrasté DANS son macro.
    """
    h = run_hierarchical(
        knn,
        vecs,
        resolution_macro=resolution_macro,
        resolution_sub=resolution_sub,
        min_sub_size=min_sub_size,
        seed=seed,
    )

    # Regroupe les index de nœuds par macro et par feuille.
    macro_members: dict[int, list[int]] = defaultdict(list)
    leaf_members: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(ideas)):
        macro_members[h.macro_membership[idx]].append(idx)
        leaf_members[h.leaf_membership[idx]].append(idx)

    # Scores aux deux niveaux.
    macro_scores = {m: score_cluster(idxs, vecs, weights, dup_threshold=dup_threshold)
                    for m, idxs in macro_members.items()}
    leaf_scores = {l: score_cluster(idxs, vecs, weights, dup_threshold=dup_threshold)
                   for l, idxs in leaf_members.items()}

    # Mots-vides saturants dérivés du corpus GLOBAL (partagés macro + sous-thèmes).
    corpus_stop, _ = derive_corpus_stopwords([idea.text for idea in ideas])

    # Naming macro : c-TF-IDF inter-macros (chaque macro = un document).
    macro_docs = {m: [ideas[i].text for i in idxs] for m, idxs in macro_members.items()}
    macro_names = name_clusters(macro_docs, corpus_stopwords=corpus_stop)

    # Naming sous-thèmes : c-TF-IDF CONTRASTÉ dans chaque macro (sous-thèmes entre eux).
    leaf_names: dict[int, dict] = {}
    for m, children in h.macro_children.items():
        sub_docs = {l: [ideas[i].text for i in leaf_members[l]] for l in children}
        leaf_names.update(name_clusters(sub_docs, corpus_stopwords=corpus_stop))

    macro_color = {m: color_for(m, len(macro_members)) for m in macro_members}

    # Nœuds : feuille + macro_id + couleur du macro.
    nodes = []
    for idx, idea in enumerate(ideas):
        leaf = int(h.leaf_membership[idx])
        macro = int(h.macro_membership[idx])
        nodes.append({
            "id": idea.id,
            "type": "idea",
            "label": _node_label(idea),
            "props": _node_props(idea, weights[idx]),
            "cluster_id": leaf,
            "macro_id": macro,
            "color": macro_color[macro],
        })

    # Thèmes : pour chaque macro (rangé par intérêt), le macro puis ses sous-thèmes.
    themes = []
    for m in rank_clusters(macro_scores):
        mnm = macro_names.get(m, {"label": f"macro {m}", "keywords": []})
        children = h.macro_children[m]
        themes.append(_theme_entry(
            m, macro_members[m], ideas, macro_scores[m], mnm, macro_color[m],
            level=0, parent_id=None, children=children,
        ))
        # sous-thèmes du macro, rangés par intérêt (entre eux)
        ranked_children = rank_clusters({c: leaf_scores[c] for c in children})
        for l in ranked_children:
            lnm = leaf_names.get(l, {"label": f"sous-thème {l}", "keywords": []})
            themes.append(_theme_entry(
                l, leaf_members[l], ideas, leaf_scores[l], lnm, macro_color[m],
                level=1, parent_id=m, children=[],
            ))

    clustering_meta = {
        "mode": "hierarchical",
        "primary": "leiden",
        "leiden_hierarchy": {
            "n_macros": h.n_macros,
            "n_leaves": h.n_leaves,
            "resolution_macro": h.resolution_macro,
            "resolution_sub": h.resolution_sub,
            "min_sub_size": h.min_sub_size,
            "macro_modularity": h.macro_modularity,
            "seed": h.seed,
        },
        "hdbscan_contender": None,
    }
    return nodes, themes, clustering_meta


def build_payload(
    input_path: str | None = None,
    k: int | None = None,
    threshold: float | None = None,
    resolution: float = 1.5,
    seed: int = 42,
    model_id: str | None = None,
    with_hdbscan: bool = False,
    source: str | None = None,
    lang: str | None = None,
    min_chars: int = 0,
    dedup_threshold: float | None = None,
    max_links: int | None = None,
    hierarchical: bool = False,
    resolution_macro: float = DEFAULT_RESOLUTION_MACRO,
    resolution_sub: float = DEFAULT_RESOLUTION_SUB,
    min_sub_size: int | None = None,
    dup_threshold: float | None = None,
) -> dict:
    """Construit le GraphPayload. `k`, `threshold`, `min_sub_size`, `dup_threshold`
    valant ``None`` sont **dérivés des données** (cf. `pipeline.cluster.adaptive`,
    audit #6/#7/#9) ; passer une valeur explicite la force (knob)."""
    ideas = io.load_ideas(input_path)
    src = io.resolve_input(input_path)

    # 0) Subset réel (ex. consultation TikTok FR) + nettoyage des avis trop
    #    courts/vides. On filtre AVANT d'embedder (moins de calcul) et on trace
    #    le détail dans meta pour audit.
    n_loaded = len(ideas)
    if source:
        ideas = [i for i in ideas if i.source == source]
    n_after_source = len(ideas)
    if lang:
        ideas = [i for i in ideas if i.lang == lang]
    n_after_lang = len(ideas)
    if min_chars:
        ideas = [i for i in ideas if len((i.text_clean or i.text).strip()) >= min_chars]
    n_after_minlen = len(ideas)

    n = len(ideas)
    if n == 0:
        raise SystemExit("Aucun avis à traiter (filtres trop stricts ?).")

    # 1) Embeddings ---------------------------------------------------------
    embedder = Embedder(model_id=model_id) if model_id else Embedder()
    texts = [idea.text_clean or idea.text for idea in ideas]
    vecs = embedder.embed(texts)
    weights = np.array([idea.weight for idea in ideas], dtype=np.float32)

    # 1.5) Déduplication near-dup (les gens répètent) -----------------------
    #      cosine > seuil → on garde 1 représentant, on cumule son `weight`.
    dedup_meta = None
    if dedup_threshold is not None:
        dd = dedup_near(vecs, weights, threshold=dedup_threshold)
        ideas = [ideas[i] for i in dd.keep]
        vecs = np.ascontiguousarray(vecs[dd.keep])
        weights = dd.weights
        dedup_meta = {
            "threshold": dedup_threshold,
            "n_in": dd.n_in,
            "n_out": dd.n_out,
            "n_collapsed": dd.n_collapsed,
        }

    # 1.75) Défauts DÉRIVÉS des données (audit #6/#7/#9) ---------------------
    #       Tout knob laissé à None est dérivé de la distribution des cosinus
    #       k-NN du corpus DÉDUPLIQUÉ (pas de ré-embed : on réutilise `vecs`).
    #       Une valeur explicite est toujours respectée.
    derived = derive_defaults(vecs, k=k)
    k = k if k is not None else derived.k
    if threshold is None:
        threshold = derived.threshold
    if min_sub_size is None:
        min_sub_size = derived.min_sub_size
    # near-dup (diversity) DÉRIVÉ de la distribution (p98), sous le seuil `dedup`
    # (dedup collapse les paires > dedup ; diversity mesure le résidu juste en
    # dessous). Forçable en knob (`dup_threshold`).
    if dup_threshold is None:
        dup_threshold = derived.dup_threshold

    # 2) Graphe k-NN sémantique --------------------------------------------
    knn = build_knn_graph(vecs, k=k, threshold=threshold)

    # 3-5) Clustering + scoring + naming + nœuds ----------------------------
    #      Deux modes : plat (Leiden 1 niveau, défaut) ou hiérarchique
    #      (macro→sous-thèmes, Leiden 2 niveaux). Les nœuds sont colorés
    #      différemment (plat = communauté ; hiérarchique = macro-thème), d'où
    #      deux constructeurs distincts mais des `links` communs.
    if hierarchical:
        nodes, themes, clustering_meta = _build_hierarchical(
            ideas, vecs, weights, knn,
            resolution_macro=resolution_macro,
            resolution_sub=resolution_sub,
            min_sub_size=min_sub_size,
            seed=seed,
            dup_threshold=dup_threshold,
        )
    else:
        nodes, themes, clustering_meta = _build_flat(
            ideas, vecs, weights, knn,
            resolution=resolution, seed=seed, with_hdbscan=with_hdbscan,
            dup_threshold=dup_threshold,
        )

    id_by_idx = [idea.id for idea in ideas]
    # Leiden clustering ci-dessus utilise le graphe k-NN COMPLET. Pour le rendu,
    # on peut plafonner les arêtes AFFICHÉES (les plus fortes en cosine) si elles
    # sont trop denses — sans jamais retirer un nœud (aucune voix sacrifiée).
    edges = knn.edges
    n_links_total = len(edges)
    links_capped = False
    if max_links is not None and n_links_total > max_links:
        edges = sorted(edges, key=lambda e: e[2], reverse=True)[:max_links]
        links_capped = True
    links = [
        {
            "source": id_by_idx[i],
            "target": id_by_idx[j],
            "type": "knn",
            "props": {"weight": round(float(w), 4)},
        }
        for (i, j, w) in edges
    ]

    payload = {
        "meta": {
            "model_id": embedder.model_id,
            "embedding_dim": int(vecs.shape[1]),
            "n_nodes": len(nodes),
            "n_links": len(links),
            "n_themes": len(themes),
            "input": str(src.relative_to(REPO_ROOT)) if src.is_relative_to(REPO_ROOT) else str(src),
            "subset": {
                "source": source,
                "lang": lang,
                "min_chars": min_chars,
                "n_loaded": n_loaded,
                "n_after_source": n_after_source,
                "n_after_lang": n_after_lang,
                "n_after_minlen": n_after_minlen,
            },
            "dedup": dedup_meta,
            "params": {
                "k": k,
                "threshold": round(float(threshold), 4),
                "resolution": resolution,
                "min_sub_size": min_sub_size,
                "dup_threshold": round(float(dup_threshold), 4),
                "seed": seed,
                "knn_backend": knn.backend,
                "avg_degree": round(knn.avg_degree, 3),
                "max_links": max_links,
                "n_links_total": n_links_total,
                "links_capped": links_capped,
            },
            # Traçabilité des défauts DÉRIVÉS (audit #6/#7/#9) : ce qui a été
            # auto-calculé vs forcé, + la distribution dont le seuil sort.
            "derived": {
                "k": derived.k,
                "threshold": round(derived.threshold, 4),
                "min_sub_size": derived.min_sub_size,
                "dup_threshold": round(derived.dup_threshold, 4),
                "knn_sim_mean": derived.pool_mean,
                "knn_sim_std": derived.pool_std,
                "edge_sigma": EDGE_SIGMA,
            },
            "clustering": clustering_meta,
        },
        "nodes": nodes,
        "links": links,
        "themes": themes,
    }
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Build GraphPayload (NLP batch).")
    ap.add_argument("--input", default=None, help="ideas.jsonl (sinon auto-résolu)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="chemin graph.json")
    ap.add_argument("--k", type=int, default=None,
                    help="voisins k-NN (défaut: DÉRIVÉ ∝ log10(N))")
    ap.add_argument("--threshold", type=float, default=None,
                    help="seuil d'arête cosine (défaut: DÉRIVÉ μ−σ·k des cosinus k-NN)")
    ap.add_argument("--resolution", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default=None, help="override model_id")
    ap.add_argument("--source", default=None,
                    help="ne garder que cette source (ex. tiktok)")
    ap.add_argument("--lang", default=None, help="ne garder que cette langue (ex. fr)")
    ap.add_argument("--min-chars", type=int, default=0,
                    help="retire les avis dont text_clean < N caractères")
    ap.add_argument("--dedup", type=float, default=None, metavar="COSINE",
                    help="déduplique les near-dups (cosine > COSINE, ex. 0.95)")
    ap.add_argument("--max-links", type=int, default=None,
                    help="plafonne les arêtes affichées (garde les + fortes ; tous les nœuds restent)")
    ap.add_argument("--with-hdbscan", action="store_true", help="trace le contender")
    ap.add_argument("--hierarchical", action="store_true",
                    help="thèmes hiérarchiques macro→sous-thèmes (Leiden 2 niveaux)")
    ap.add_argument("--resolution-macro", type=float, default=DEFAULT_RESOLUTION_MACRO,
                    help="résolution Leiden niveau macro (basse → peu de grandes communautés)")
    ap.add_argument("--resolution-sub", type=float, default=DEFAULT_RESOLUTION_SUB,
                    help="résolution Leiden des sous-thèmes (haute → finesse intra-macro)")
    ap.add_argument("--min-sub-size", type=int, default=None,
                    help="taille mini d'un sous-thème (défaut: DÉRIVÉ, relatif à N)")
    ap.add_argument("--dup-threshold", type=float, default=None, metavar="COSINE",
                    help="seuil near-dup pour diversity (défaut: DÉRIVÉ p98, ou lié à --dedup)")
    ap.add_argument("--fixture", action="store_true",
                    help="écrit aussi le fixture viz graph.sample.json")
    args = ap.parse_args()

    payload = build_payload(
        input_path=args.input,
        k=args.k,
        threshold=args.threshold,
        resolution=args.resolution,
        seed=args.seed,
        model_id=args.model,
        with_hdbscan=args.with_hdbscan,
        source=args.source,
        lang=args.lang,
        min_chars=args.min_chars,
        dedup_threshold=args.dedup,
        max_links=args.max_links,
        hierarchical=args.hierarchical,
        resolution_macro=args.resolution_macro,
        resolution_sub=args.resolution_sub,
        min_sub_size=args.min_sub_size,
        dup_threshold=args.dup_threshold,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.fixture:
        FIXTURE_OUT.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE_OUT.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    m = payload["meta"]
    print(f"✓ {out}")
    sub = m.get("subset") or {}
    if sub.get("source") or sub.get("lang"):
        print(f"  subset     : source={sub.get('source')} lang={sub.get('lang')} "
              f"min_chars={sub.get('min_chars')}  "
              f"({sub.get('n_loaded')}→{sub.get('n_after_minlen')})")
    if m.get("dedup"):
        d = m["dedup"]
        print(f"  dedup      : cosine>{d['threshold']}  "
              f"{d['n_in']}→{d['n_out']} (−{d['n_collapsed']} near-dups)")
    print(f"  model_id   : {m['model_id']} (dim {m['embedding_dim']})")
    d = m.get("derived", {})
    p = m["params"]
    print(f"  DÉRIVÉ     : k={d.get('k')}  seuil={d.get('threshold')} "
          f"(μ={d.get('knn_sim_mean')} −{d.get('edge_sigma')}·σ={d.get('knn_sim_std')})  "
          f"min_sub={d.get('min_sub_size')}  dup={d.get('dup_threshold')}")
    print(f"  effectif   : k={p['k']}  seuil={p['threshold']}  "
          f"min_sub={p.get('min_sub_size')}  dup={p.get('dup_threshold')}")
    print(f"  nodes/links: {m['n_nodes']} / {m['n_links']}  (avg_deg {p['avg_degree']})")
    cl = m["clustering"]
    if cl.get("mode") == "hierarchical":
        lh = cl["leiden_hierarchy"]
        print(f"  hiérarchie : {lh['n_macros']} macros → {lh['n_leaves']} sous-thèmes "
              f"(modul. macro {lh['macro_modularity']}, "
              f"res {lh['resolution_macro']}/{lh['resolution_sub']}, backend {m['params']['knn_backend']})")
        print("  arbre      :")
        for t in payload["themes"]:
            if t["level"] == 0:
                print(f"    ▸ [{t['cluster_id']}] {t['label']}  "
                      f"(n={t['size']}, w={t['weight_sum']}, {len(t['children'])} sous-thèmes)")
            else:
                print(f"        └ [{t['cluster_id']}] {t['label']}  "
                      f"(n={t['size']}, w={t['weight_sum']}, cons={t['consensus']})")
    else:
        print(f"  leiden     : {cl['leiden']['n_clusters']} communautés "
              f"(modularité {cl['leiden']['modularity']}, backend {m['params']['knn_backend']})")
        print("  thèmes     :")
        for t in payload["themes"]:
            print(f"    [{t['cluster_id']}] {t['label']}  "
                  f"(n={t['size']}, w={t['weight_sum']}, div={t['diversity']}, cons={t['consensus']})")


if __name__ == "__main__":
    main()
