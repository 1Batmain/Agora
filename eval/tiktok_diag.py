"""DIAGNOSTIC — « pourquoi tiktok domine » (lane eval, LECTURE/ANALYSE).

Quantifie les DEUX mécanismes par lesquels le mot « tiktok » (et tik/tok/appli/
application/réseau·x) écrase la consultation TikTok FR :

  1. **NOMMAGE** : le terme sature le label TF-IDF des clusters (mot-vide de
     domaine, comme « le »).
  2. **GÉOMÉTRIE** : le terme tire-t-il l'espace d'embedding (une « direction
     tiktok » commune) au point de DIRIGER le clustering, pas seulement de le
     nommer ?

READ-ONLY sur le pipeline de prod : on réutilise `pipeline.cluster.*`,
`pipeline.embed.*` et le cache backend (`backend/cache/*`). On n'écrit que dans
`eval/`. Aucune modification du pipeline.

Expériences (cf. brief) :
  1. Saturation lexicale
  2. Composante commune (cos centroïde global + PCA + corrélations PC1)
  3. Ablation géométrie : (a) masquage pré-embedding (ré-embed nomic-v2 UNE fois)
     (b) common-component removal (all-but-the-top, sans ré-embed)
  4. Isolation du nommage : TF-IDF (prod) vs c-TF-IDF
  5. Les clusters « tiktok » : cohérents (→ nommage) ou mélangés (→ géométrie) ?

Usage :
    uv run --extra embed-contender python -m eval.tiktok_diag
    uv run python -m eval.tiktok_diag --no-reembed   # saute le ré-embed masqué

Sorties :
    eval/tiktok_diag_results.json   (numéros bruts, reproductibles)
    eval/tiktok_diag_two_clusters.md (annexe : 2 clusters tiktok + avis)
Le rapport rédigé (`eval/tiktok_diagnostic.md`) cite ces numéros.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from backend.recluster import load_cache
from pipeline.cluster.dedup import dedup_near
from pipeline.cluster.hierarchy import run_hierarchical
from pipeline.cluster.knn import build_knn_graph
from pipeline.cluster.naming import FRENCH_STOPWORDS, _tokenizer, name_clusters
from pipeline.cluster.scoring import score_cluster

SEED = 42
EVAL_DIR = Path(__file__).resolve().parent
CACHE_DIR = EVAL_DIR / ".cache"          # cache local (gitignoré) du ré-embed masqué
RESULTS_PATH = EVAL_DIR / "tiktok_diag_results.json"
TWO_CLUSTERS_PATH = EVAL_DIR / "tiktok_diag_two_clusters.md"

# Paramètres baseline = défauts de prod (console live :8010 / contrat cross-lane).
BASELINE = dict(
    dedup=0.95, min_chars=12, k=12, threshold=0.60,
    resolution_macro=1.0, resolution_sub=1.5, min_sub_size=18,
)

# Famille lexicale « tiktok » : le terme + ses satellites de domaine.
# Couvre « tik tok », « tik-tok », pluriels et appli/application/réseau·x.
TIKTOK_RE = re.compile(
    r"\b(?:tik[\s\-]?tok|tiktok|tik|tok|toktok|appli(?:cation|s|cations)?|"
    r"r[ée]seaux?)\b",
    re.IGNORECASE,
)
# Tokens (après _tokenizer) considérés « famille tiktok » pour le comptage label.
TIKTOK_TOKENS = {
    "tiktok", "tik", "tok", "appli", "application", "applications", "applis",
    "reseau", "reseaux", "réseau", "réseaux", "tiktoks",
}


# --------------------------------------------------------------------------- #
# Préparation : baseline (set + clustering hiérarchique aux défauts de prod).
# --------------------------------------------------------------------------- #
def prepare_baseline():
    """Charge le cache, applique min_chars + dedup (défauts prod), clusterise.

    Pour que les ARI d'ablation soient bien définis, on FIGE le set de nœuds ici
    (post min_chars + dedup) ; les variantes (masquée, all-but-the-top) re-
    clusterisent EXACTEMENT ces nœuds, vecteurs modifiés. Retourne tout le
    contexte partagé par les expériences.
    """
    ideas_all, vecs_all, weights_all = load_cache()
    n_cached = len(ideas_all)

    # 1) min_chars (filtre texte, indépendant des embeddings → déterministe).
    keep = [
        i for i, idea in enumerate(ideas_all)
        if len((idea.text_clean or idea.text).strip()) >= BASELINE["min_chars"]
    ]
    ideas = [ideas_all[i] for i in keep]
    vecs = np.ascontiguousarray(vecs_all[keep])
    weights = weights_all[keep]
    n_after_minlen = len(ideas)

    # 2) dedup near-dup (cosine > 0.95) — comme la prod.
    dd = dedup_near(vecs, weights, threshold=BASELINE["dedup"])
    ideas = [ideas[i] for i in dd.keep]
    vecs = np.ascontiguousarray(vecs[dd.keep])
    weights = dd.weights
    n_baseline = len(ideas)

    texts = [idea.text_clean or idea.text for idea in ideas]

    # 3) clustering hiérarchique baseline (sur les vecteurs cachés).
    macro_b, leaf_b, hres = cluster(vecs)

    return {
        "ideas": ideas,
        "texts": texts,
        "vecs": vecs,            # L2-normalisés (cache nomic-v2)
        "weights": weights,
        "macro": macro_b,
        "leaf": leaf_b,
        "hres": hres,
        "n_cached": n_cached,
        "n_after_minlen": n_after_minlen,
        "n_baseline": n_baseline,
    }


def cluster(vecs: np.ndarray):
    """k-NN + Leiden hiérarchique aux défauts de prod. Retourne (macro, leaf, h)."""
    knn = build_knn_graph(vecs, k=BASELINE["k"], threshold=BASELINE["threshold"])
    h = run_hierarchical(
        knn, vecs,
        resolution_macro=BASELINE["resolution_macro"],
        resolution_sub=BASELINE["resolution_sub"],
        min_sub_size=BASELINE["min_sub_size"],
        seed=SEED,
    )
    return list(h.macro_membership), list(h.leaf_membership), h


def cluster_graph(knn, vecs: np.ndarray):
    """Leiden hiérarchique sur un graphe k-NN DÉJÀ construit (seuil custom)."""
    h = run_hierarchical(
        knn, vecs,
        resolution_macro=BASELINE["resolution_macro"],
        resolution_sub=BASELINE["resolution_sub"],
        min_sub_size=BASELINE["min_sub_size"],
        seed=SEED,
    )
    return list(h.macro_membership), list(h.leaf_membership), h


def l2norm(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def mentions_tiktok(texts: list[str]) -> np.ndarray:
    """Booléen par doc : contient un token de la famille tiktok."""
    return np.array([bool(TIKTOK_RE.search(t)) for t in texts])


# --------------------------------------------------------------------------- #
# Exp 1 — Saturation lexicale.
# --------------------------------------------------------------------------- #
def exp1_saturation(texts: list[str]) -> dict:
    n = len(texts)
    # Comptage de docs contenant chaque mot-clé (présence booléenne).
    keywords = {
        "tiktok": re.compile(r"\btik[\s\-]?tok\b|\btiktok\b", re.I),
        "tik": re.compile(r"\btik\b", re.I),
        "tok": re.compile(r"\btok\b", re.I),
        "appli/application": re.compile(r"\bappli(?:cation|s|cations)?\b", re.I),
        "réseau(x)": re.compile(r"\br[ée]seaux?\b", re.I),
    }
    doc_share = {
        kw: round(sum(bool(rx.search(t)) for t in texts) / n, 4)
        for kw, rx in keywords.items()
    }
    family_share = round(float(mentions_tiktok(texts).mean()), 4)

    # Rang de fréquence : tokens (mots de contenu) toutes occurrences confondues,
    # AVANT retrait des stopwords (pour montrer que tiktok rivalise avec « le »).
    tok_counts: Counter = Counter()
    for t in texts:
        # tokenizer « brut » : tous les mots (≥1 lettre), pas de filtre longueur.
        for w in re.findall(r"[a-zàâäéèêëîïôöùûüçœ]+", t.lower()):
            tok_counts[w] += 1
    ranking = tok_counts.most_common()
    rank_of = {w: i + 1 for i, (w, _) in enumerate(ranking)}

    def rank_info(word: str) -> dict:
        c = tok_counts.get(word, 0)
        return {"count": c, "rank": rank_of.get(word)}

    # Rang parmi les mots de CONTENU (stopwords retirés) — « tiktok » est-il
    # le 1er mot « plein » ?
    content_counts = Counter(
        {w: c for w, c in tok_counts.items()
         if w not in FRENCH_STOPWORDS and len(w) > 2}
    )
    content_ranking = content_counts.most_common(15)

    return {
        "n_docs": n,
        "doc_share_per_keyword": doc_share,
        "doc_share_family": family_share,
        "top10_tokens_global": [(w, c) for w, c in ranking[:10]],
        "rank_tiktok_among_all": rank_info("tiktok"),
        "rank_le_among_all": rank_info("le"),
        "rank_la_among_all": rank_info("la"),
        "top15_content_tokens": content_ranking,
    }


# --------------------------------------------------------------------------- #
# Exp 2 — Composante commune (géométrie).
# --------------------------------------------------------------------------- #
def exp2_common_component(vecs: np.ndarray, texts: list[str]) -> dict:
    n, d = vecs.shape
    # Cosinus moyen au centroïde global (vecteurs déjà L2-normalisés).
    centroid = vecs.mean(axis=0)
    cnorm = np.linalg.norm(centroid)
    centroid_u = centroid / cnorm if cnorm > 0 else centroid
    cos_to_centroid = vecs @ centroid_u
    # Cosinus moyen inter-docs (échantillon de paires si n grand) pour le contexte.
    rng = np.random.default_rng(SEED)
    idx = rng.choice(n, size=min(n, 2000), replace=False)
    sub = vecs[idx]
    sims = sub @ sub.T
    iu = np.triu_indices(sub.shape[0], k=1)
    mean_pair_cos = float(sims[iu].mean())

    # PCA (sur vecteurs CENTRÉS) : variance expliquée PC1..PC5.
    Xc = vecs - vecs.mean(axis=0, keepdims=True)
    # SVD économe.
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2)
    evr = var / var.sum()
    pc_scores = U * S  # projections (n, d) ; colonne k = score sur PCk

    ment = mentions_tiktok(texts).astype(float)
    lengths = np.array([len(t) for t in texts], dtype=float)

    def corr(a, b):
        if a.std() == 0 or b.std() == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    pc_corr = []
    for k in range(min(5, pc_scores.shape[1])):
        s = pc_scores[:, k]
        pc_corr.append({
            "pc": k + 1,
            "evr": round(float(evr[k]), 4),
            "corr_mentions_tiktok": round(corr(s, ment), 4),
            "corr_doc_length": round(corr(s, lengths), 4),
        })

    return {
        "cos_to_global_centroid_mean": round(float(cos_to_centroid.mean()), 4),
        "cos_to_global_centroid_std": round(float(cos_to_centroid.std()), 4),
        "mean_pairwise_cos": round(mean_pair_cos, 4),
        "centroid_norm": round(float(cnorm), 4),
        "evr_pc1_5": [round(float(x), 4) for x in evr[:5]],
        "evr_cumulative_pc1_5": round(float(evr[:5].sum()), 4),
        "pc_correlations": pc_corr,
        "share_mentions_tiktok": round(float(ment.mean()), 4),
    }


# --------------------------------------------------------------------------- #
# Exp 3 — Ablation géométrie.
# --------------------------------------------------------------------------- #
def mask_texts(texts: list[str]) -> list[str]:
    """Neutralise les tokens de la famille tiktok (remplacés par un espace)."""
    out = []
    for t in texts:
        masked = TIKTOK_RE.sub(" ", t)
        masked = re.sub(r"\s+", " ", masked).strip()
        out.append(masked or ".")  # évite le texte vide pour l'embedder
    return out


def reembed_masked(texts: list[str], use_cache: bool = True) -> np.ndarray:
    """Ré-embedde la variante masquée avec nomic-v2 (UN seul appel torch).

    Met en cache le résultat dans eval/.cache pour des reruns instantanés.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    masked = mask_texts(texts)
    sig = str(len(masked)) + ":" + str(sum(len(t) for t in masked))
    cache_npy = CACHE_DIR / "masked_embeddings.npy"
    cache_sig = CACHE_DIR / "masked_embeddings.sig"
    if use_cache and cache_npy.exists() and cache_sig.exists():
        if cache_sig.read_text().strip() == sig:
            return np.load(cache_npy).astype(np.float32)

    from pipeline.embed.embedder import Embedder

    emb = Embedder(model_id="nomic-v2")
    vecs = emb.embed(masked).astype(np.float32)  # L2-normalisés (spec nomic)
    np.save(cache_npy, vecs)
    cache_sig.write_text(sig)
    return vecs


def build_knn_target_degree(vecs, k, target_deg, lo=0.0, hi=1.0, iters=18):
    """k-NN dont le SEUIL est choisi pour atteindre `target_deg` (avg_degree).

    Retirer la composante commune dé-comprime l'espace : tous les cosinus
    chutent et le seuil de prod (0.60) viderait le graphe (→ singletons). Pour
    un test de TOPOLOGIE honnête, on apparie la densité : on cherche par
    bissection le seuil qui donne ~`target_deg`. Retourne (KnnGraph, threshold).
    """
    best = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        g = build_knn_graph(vecs, k=k, threshold=mid)
        deg = g.avg_degree
        best = (g, mid)
        if deg > target_deg:
            lo = mid          # trop dense → monter le seuil
        else:
            hi = mid          # trop clairsemé → baisser le seuil
    return best


def all_but_the_top(vecs: np.ndarray, n_pc: int) -> np.ndarray:
    """Retire les top-`n_pc` composantes principales puis re-normalise (L2).

    « All-but-the-top » (Mu & Viswanath 2018) : on soustrait la moyenne et la
    projection sur les top PCs — la « direction commune » de l'espace.
    """
    mean = vecs.mean(axis=0, keepdims=True)
    Xc = vecs - mean
    if n_pc <= 0:
        return l2norm(Xc.copy())
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    top = Vt[:n_pc]                       # (n_pc, d)
    proj = (Xc @ top.T) @ top             # composante à retirer
    cleaned = Xc - proj
    return l2norm(cleaned)


def topk_neighbors(vecs: np.ndarray, k: int) -> list[set]:
    """Ensemble des k plus proches voisins (cosine) par nœud, hors soi-même."""
    n = vecs.shape[0]
    out = []
    for start in range(0, n, 512):
        stop = min(start + 512, n)
        sims = vecs[start:stop] @ vecs.T
        for r in range(stop - start):
            i = start + r
            sims[r, i] = -np.inf
            nbr = np.argpartition(sims[r], -k)[-k:]
            out.append(set(int(x) for x in nbr))
    return out


def mean_neighbor_jaccard(vecs_a: np.ndarray, vecs_b: np.ndarray, k: int) -> float:
    """Jaccard moyen des k-PPV de chaque nœud entre deux espaces (THRESHOLD-FREE).

    Mesure directe de stabilité GÉOMÉTRIQUE locale : 1.0 = voisinages
    identiques (le token n'a pas bougé la géométrie), ~0 = voisinages
    totalement remaniés (le token dirigeait la géométrie locale).
    """
    na, nb = topk_neighbors(vecs_a, k), topk_neighbors(vecs_b, k)
    js = []
    for a, b in zip(na, nb):
        u = len(a | b)
        js.append(len(a & b) / u if u else 1.0)
    return float(np.mean(js))


def silhouette_safe(vecs, labels):
    from eval.metrics import silhouette
    return silhouette(vecs, labels, exclude_noise=False)


def exp3_ablation(base: dict, do_reembed: bool) -> dict:
    vecs = base["vecs"]
    leaf_b = base["leaf"]
    macro_b = base["macro"]
    texts = base["texts"]

    sil_base_leaf = silhouette_safe(vecs, leaf_b)
    sil_base_macro = silhouette_safe(vecs, macro_b)

    out = {
        "baseline": {
            "n_macros": base["hres"].n_macros,
            "n_leaves": base["hres"].n_leaves,
            "silhouette_leaf": _r(sil_base_leaf),
            "silhouette_macro": _r(sil_base_macro),
        },
        "variants": {},
    }

    kk = BASELINE["k"]
    # (a) Masquage pré-embedding (ré-embed nomic-v2).
    if do_reembed:
        v_mask = reembed_masked(texts)
        macro_m, leaf_m, h_m = cluster(v_mask)
        out["variants"]["mask_preembed"] = {
            "ari_leaf": round(float(adjusted_rand_score(leaf_b, leaf_m)), 4),
            "ari_macro": round(float(adjusted_rand_score(macro_b, macro_m)), 4),
            "n_macros": h_m.n_macros,
            "n_leaves": h_m.n_leaves,
            # silhouette des NOUVEAUX clusters dans le NOUVEL espace masqué.
            "silhouette_leaf": _r(silhouette_safe(v_mask, leaf_m)),
            # cos moyen au centroïde global après masquage (compression résiduelle).
            "cos_to_centroid_mean": _r(_cos_centroid(v_mask)),
            # stabilité géométrique locale (threshold-free).
            "neighbor_jaccard_vs_baseline": round(
                mean_neighbor_jaccard(vecs, v_mask, kk), 4),
        }

    # (b) Common-component removal (all-but-the-top), sans ré-embed.
    #
    # Deux lectures :
    #   - FIXED : mêmes params que la prod (seuil 0.60). Dé-comprimer l'espace
    #     fait chuter tous les cosinus < 0.60 → le graphe se vide → singletons.
    #     ARI≈0 ici reflète l'effondrement d'ÉCHELLE, pas la topologie ; on le
    #     rapporte mais on ne le sur-interprète pas.
    #   - MATCHED : on apparie la densité (avg_degree cible) entre baseline et
    #     variante → test de TOPOLOGIE honnête. La baseline appariée sert de
    #     référence ARI (et on vérifie qu'elle reste ~ la prod).
    target_deg = 16.0    # plafond atteignable par l'espace abt (~17) → apparié
    knn_bm, thr_bm = build_knn_target_degree(vecs, BASELINE["k"], target_deg)
    macro_bm, leaf_bm, h_bm = cluster_graph(knn_bm, vecs)
    out["baseline_matched"] = {
        "target_avg_degree": target_deg,
        "threshold": round(thr_bm, 4),
        "avg_degree": round(knn_bm.avg_degree, 2),
        "n_macros": h_bm.n_macros,
        "n_leaves": h_bm.n_leaves,
        "ari_leaf_vs_prod": round(float(adjusted_rand_score(leaf_b, leaf_bm)), 4),
    }
    for n_pc in (1, 2, 3):
        v_abt = all_but_the_top(vecs, n_pc)
        # FIXED (seuil prod 0.60).
        macro_a, leaf_a, h_a = cluster(v_abt)
        # MATCHED (densité appariée à la baseline).
        knn_am, thr_am = build_knn_target_degree(v_abt, BASELINE["k"], target_deg)
        macro_am, leaf_am, h_am = cluster_graph(knn_am, v_abt)
        out["variants"][f"all_but_top_{n_pc}"] = {
            "fixed": {
                "ari_leaf": round(float(adjusted_rand_score(leaf_b, leaf_a)), 4),
                "ari_macro": round(float(adjusted_rand_score(macro_b, macro_a)), 4),
                "n_macros": h_a.n_macros,
                "n_leaves": h_a.n_leaves,
                "note": "seuil 0.60 → graphe effondré (artefact d'échelle)",
            },
            "matched": {
                "threshold": round(thr_am, 4),
                "avg_degree": round(knn_am.avg_degree, 2),
                "ari_leaf": round(float(adjusted_rand_score(leaf_bm, leaf_am)), 4),
                "ari_macro": round(float(adjusted_rand_score(macro_bm, macro_am)), 4),
                "n_macros": h_am.n_macros,
                "n_leaves": h_am.n_leaves,
                "silhouette_leaf": _r(silhouette_safe(v_abt, leaf_am)),
            },
            "cos_to_centroid_mean": _r(_cos_centroid(v_abt)),
            "neighbor_jaccard_vs_baseline": round(
                mean_neighbor_jaccard(vecs, v_abt, kk), 4),
        }
    return out


def _cos_centroid(vecs):
    c = vecs.mean(axis=0)
    nrm = np.linalg.norm(c)
    cu = c / nrm if nrm > 0 else c
    return float((vecs @ cu).mean())


def _r(x):
    return None if x is None else round(float(x), 4)


# --------------------------------------------------------------------------- #
# Exp 4 — Isolation du nommage : TF-IDF (prod) vs c-TF-IDF.
# --------------------------------------------------------------------------- #
def ctfidf_name(cluster_docs: dict[int, list[str]], top_k: int = 6,
                label_k: int = 3, extra_stop: set | None = None) -> dict[int, dict]:
    """c-TF-IDF (BERTopic) : tf du terme DANS la classe × idf inter-classes.

    idf_t = log(1 + A / f_t), A = moyenne de mots par classe, f_t = fréquence
    totale du terme. Un terme omniprésent (tiktok) a f_t énorme → idf bas → il
    cesse de saturer les labels, révélant le terme DISTINCTIF de chaque classe.
    """
    cids = sorted(cluster_docs.keys())
    stop = set(FRENCH_STOPWORDS) | (extra_stop or set())
    # tf par classe (compte d'occurrences, tokenizer + stopwords de prod).
    tf: dict[int, Counter] = {}
    for c in cids:
        cnt: Counter = Counter()
        for doc in cluster_docs[c]:
            for w in _tokenizer(doc):
                if w in stop:
                    continue
                cnt[w] += 1
        tf[c] = cnt
    # f_t = fréquence totale du terme sur tout le corpus de classes.
    total_freq: Counter = Counter()
    class_sizes = {}
    for c in cids:
        class_sizes[c] = sum(tf[c].values())
        total_freq.update(tf[c])
    A = (sum(class_sizes.values()) / len(cids)) if cids else 0.0

    import math

    out: dict[int, dict] = {}
    for c in cids:
        size = class_sizes[c] or 1
        scores = {}
        for w, f in tf[c].items():
            idf = math.log(1.0 + A / total_freq[w]) if total_freq[w] else 0.0
            scores[w] = (f / size) * idf
        order = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        keywords = [w for w, s in order[:top_k] if s > 0]
        label = " · ".join(keywords[:label_k]) if keywords else f"thème {c}"
        out[c] = {"label": label, "keywords": keywords}
    return out


def _label_has_tiktok(name: dict) -> bool:
    kws = name.get("keywords", [])[:3]
    toks = set()
    for kw in kws:
        toks.update(kw.split())
    return bool(toks & TIKTOK_TOKENS)


def exp4_naming(base: dict) -> dict:
    ideas = base["ideas"]
    leaf_b = base["leaf"]
    macro_b = base["macro"]

    def docs_by(membership):
        d: dict[int, list[str]] = defaultdict(list)
        for idx, c in enumerate(membership):
            d[c].append(ideas[idx].text)
        return d

    results = {}
    for level, membership in (("leaf", leaf_b), ("macro", macro_b)):
        cdocs = docs_by(membership)
        tfidf = name_clusters(cdocs)                       # prod (TF-IDF inter-clusters)
        ctfidf = ctfidf_name(cdocs)                        # c-TF-IDF
        # FIX recommandé : c-TF-IDF + famille tiktok en STOPWORDS de domaine.
        fix = ctfidf_name(cdocs, extra_stop=TIKTOK_TOKENS)
        n_clusters = len(cdocs)
        tfidf_hits = sum(_label_has_tiktok(tfidf[c]) for c in cdocs)
        ctfidf_hits = sum(_label_has_tiktok(ctfidf[c]) for c in cdocs)
        fix_hits = sum(_label_has_tiktok(fix[c]) for c in cdocs)
        # détail par cluster (trié par taille) pour le rapport.
        sizes = {c: len(v) for c, v in cdocs.items()}
        per_cluster = []
        for c in sorted(cdocs, key=lambda x: sizes[x], reverse=True):
            per_cluster.append({
                "cluster_id": c,
                "size": sizes[c],
                "tfidf_label": tfidf[c]["label"],
                "ctfidf_label": ctfidf[c]["label"],
                "fix_label": fix[c]["label"],
            })
        results[level] = {
            "n_clusters": n_clusters,
            "tfidf_tiktok_labels": tfidf_hits,
            "ctfidf_tiktok_labels": ctfidf_hits,
            "fix_tiktok_labels": fix_hits,
            "tfidf_tiktok_rate": round(tfidf_hits / n_clusters, 4) if n_clusters else 0,
            "ctfidf_tiktok_rate": round(ctfidf_hits / n_clusters, 4) if n_clusters else 0,
            "fix_tiktok_rate": round(fix_hits / n_clusters, 4) if n_clusters else 0,
            "per_cluster": per_cluster,
        }
    return results


# --------------------------------------------------------------------------- #
# Exp 5 — Les clusters « tiktok » : nommage vs géométrie.
# --------------------------------------------------------------------------- #
def exp5_two_clusters(base: dict) -> dict:
    ideas = base["ideas"]
    vecs = base["vecs"]
    weights = base["weights"]
    leaf_b = base["leaf"]

    members: dict[int, list[int]] = defaultdict(list)
    for idx, c in enumerate(leaf_b):
        members[c].append(idx)

    cdocs = {c: [ideas[i].text for i in idxs] for c, idxs in members.items()}
    tfidf = name_clusters(cdocs)
    ctfidf = ctfidf_name(cdocs)

    # Clusters dont le top-1 terme TF-IDF appartient à la famille tiktok.
    tiktok_clusters = []
    for c in members:
        kws = tfidf[c].get("keywords", [])
        if kws and set(kws[0].split()) & TIKTOK_TOKENS:
            tiktok_clusters.append(c)

    details = []
    for c in tiktok_clusters:
        idxs = members[c]
        score = score_cluster(idxs, vecs, weights)
        # 3 avis représentatifs = les plus proches du centroïde du cluster.
        cen = vecs[idxs].mean(axis=0)
        cn = np.linalg.norm(cen)
        cen_u = cen / cn if cn > 0 else cen
        sims = vecs[idxs] @ cen_u
        top3 = [idxs[i] for i in np.argsort(sims)[::-1][:3]]
        # terme c-TF-IDF distinctif (top non-tiktok).
        distinctive = [w for w in ctfidf[c]["keywords"]
                       if not (set(w.split()) & TIKTOK_TOKENS)]
        details.append({
            "cluster_id": c,
            "size": score.size,
            "consensus": score.consensus,
            "diversity": score.diversity,
            "tfidf_label": tfidf[c]["label"],
            "ctfidf_label": ctfidf[c]["label"],
            "ctfidf_distinctive": distinctive[:4],
            "sample_avis": [ideas[i].text[:280] for i in top3],
        })

    # Cohérence comparative : consensus médian de TOUS les clusters, pour situer.
    all_cons = []
    for c, idxs in members.items():
        if len(idxs) >= 2:
            all_cons.append(score_cluster(idxs, vecs, weights).consensus)
    median_cons = float(np.median(all_cons)) if all_cons else None

    return {
        "n_tiktok_top_clusters": len(tiktok_clusters),
        "median_consensus_all_clusters": _r(median_cons),
        "clusters": details,
    }


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #
def write_two_clusters_md(exp5: dict) -> None:
    lines = ["# Annexe — les clusters « tiktok » (avis représentatifs)\n"]
    lines.append(f"Clusters dont le top-terme TF-IDF est de la famille tiktok : "
                 f"**{exp5['n_tiktok_top_clusters']}**. "
                 f"Consensus médian (tous clusters) = "
                 f"{exp5['median_consensus_all_clusters']}.\n")
    for d in exp5["clusters"]:
        lines.append(f"## Cluster {d['cluster_id']} (n={d['size']}, "
                     f"consensus={d['consensus']}, diversity={d['diversity']})\n")
        lines.append(f"- **Label TF-IDF (prod)** : {d['tfidf_label']}")
        lines.append(f"- **Label c-TF-IDF**     : {d['ctfidf_label']}")
        lines.append(f"- **Terme distinctif (c-TF-IDF, hors tiktok)** : "
                     f"{', '.join(d['ctfidf_distinctive']) or '—'}\n")
        lines.append("- **3 avis représentatifs** (proches du centroïde) :")
        for s in d["sample_avis"]:
            lines.append(f"  - « {s.strip()} »")
        lines.append("")
    TWO_CLUSTERS_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnostic tiktok (lane eval).")
    ap.add_argument("--no-reembed", action="store_true",
                    help="saute le masquage pré-embed (pas d'appel torch).")
    args = ap.parse_args()

    print("→ Préparation baseline (cache nomic-v2, défauts prod)…")
    base = prepare_baseline()
    print(f"  n_cached={base['n_cached']}  après min_chars={BASELINE['min_chars']}: "
          f"{base['n_after_minlen']}  après dedup>{BASELINE['dedup']}: {base['n_baseline']}")
    print(f"  baseline: {base['hres'].n_macros} macros, {base['hres'].n_leaves} sous-thèmes")

    print("→ Exp1 saturation lexicale…")
    e1 = exp1_saturation(base["texts"])
    print("→ Exp2 composante commune (PCA)…")
    e2 = exp2_common_component(base["vecs"], base["texts"])
    print(f"→ Exp3 ablation géométrie (reembed={not args.no_reembed})…")
    e3 = exp3_ablation(base, do_reembed=not args.no_reembed)
    print("→ Exp4 nommage TF-IDF vs c-TF-IDF…")
    e4 = exp4_naming(base)
    print("→ Exp5 clusters tiktok…")
    e5 = exp5_two_clusters(base)
    write_two_clusters_md(e5)

    results = {
        "seed": SEED,
        "baseline_params": BASELINE,
        "set": {
            "n_cached": base["n_cached"],
            "n_after_minlen": base["n_after_minlen"],
            "n_baseline": base["n_baseline"],
            "n_macros": base["hres"].n_macros,
            "n_leaves": base["hres"].n_leaves,
        },
        "exp1_saturation": e1,
        "exp2_common_component": e2,
        "exp3_ablation": e3,
        "exp4_naming": e4,
        "exp5_two_clusters": e5,
    }
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"\n✓ Résultats : {RESULTS_PATH}")
    print(f"✓ Annexe    : {TWO_CLUSTERS_PATH}")
    _print_summary(results)


def _print_summary(r: dict) -> None:
    print("\n================ RÉSUMÉ ================")
    e1 = r["exp1_saturation"]
    print(f"[1] famille tiktok dans {100*e1['doc_share_family']:.1f}% des docs ; "
          f"'tiktok' rang {e1['rank_tiktok_among_all']['rank']} "
          f"(count {e1['rank_tiktok_among_all']['count']}) vs 'le' rang "
          f"{e1['rank_le_among_all']['rank']}")
    e2 = r["exp2_common_component"]
    print(f"[2] cos moyen au centroïde global = {e2['cos_to_global_centroid_mean']} "
          f"(compression) ; EVR PC1..5 = {e2['evr_pc1_5']}")
    for pc in e2["pc_correlations"][:2]:
        print(f"    PC{pc['pc']}: evr={pc['evr']} corr(tiktok)={pc['corr_mentions_tiktok']} "
              f"corr(len)={pc['corr_doc_length']}")
    e3 = r["exp3_ablation"]
    mp = e3["variants"].get("mask_preembed")
    if mp:
        print(f"[3] mask_preembed: ARI_leaf={mp['ari_leaf']} ARI_macro={mp['ari_macro']} "
              f"(→ {mp['n_macros']}macros/{mp['n_leaves']}sous) "
              f"nbrJaccard={mp['neighbor_jaccard_vs_baseline']} cos_cent={mp.get('cos_to_centroid_mean')}")
    bm = e3["baseline_matched"]
    print(f"[3] baseline_matched (deg≈{bm['avg_degree']}, thr={bm['threshold']}): "
          f"{bm['n_macros']}macros/{bm['n_leaves']}sous (ARI vs prod={bm['ari_leaf_vs_prod']})")
    for name, v in e3["variants"].items():
        if name == "mask_preembed":
            continue
        m = v["matched"]
        print(f"[3] {name} MATCHED (thr={m['threshold']}): ARI_leaf={m['ari_leaf']} "
              f"ARI_macro={m['ari_macro']} (→ {m['n_macros']}macros/{m['n_leaves']}sous) "
              f"nbrJaccard={v['neighbor_jaccard_vs_baseline']} ; "
              f"FIXED ARI_leaf={v['fixed']['ari_leaf']} (effondré)")
    e4 = r["exp4_naming"]
    for lvl, v in e4.items():
        print(f"[4] {lvl}: labels tiktok TF-IDF={v['tfidf_tiktok_labels']}/{v['n_clusters']} "
              f"→ c-TF-IDF={v['ctfidf_tiktok_labels']}/{v['n_clusters']} "
              f"→ FIX(c-TF-IDF+domain-stop)={v['fix_tiktok_labels']}/{v['n_clusters']}")
    e5 = r["exp5_two_clusters"]
    print(f"[5] {e5['n_tiktok_top_clusters']} clusters top-TF-IDF=tiktok ; "
          f"consensus médian={e5['median_consensus_all_clusters']}")
    for d in e5["clusters"]:
        print(f"    cluster {d['cluster_id']} n={d['size']} cons={d['consensus']} "
              f"distinctif={d['ctfidf_distinctive']}")


if __name__ == "__main__":
    main()
