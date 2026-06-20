"""T-N6 · Naming des thèmes — c-TF-IDF + mots-vides DÉRIVÉS du corpus.

But : labels **distinctifs** par cluster, **sans aucun mot de domaine codé en dur**.
L'outil tourne sur des centaines de consultations (sujets et langues variés) ; un terme
qui sature la consultation courante (« tiktok », « vélo », « retraite »…) doit être
neutralisé *parce que les statistiques du corpus le révèlent*, pas parce qu'on l'a
écrit dans le code. Le corpus TikTok n'est qu'un cas de test, jamais une cible.

Trois mécanismes, du plus important au complément :

1. **Mots-vides de DOMAINE — cutoff `max_df` au niveau DOCUMENT (le levier principal).**
   On mesure la *document-frequency* (DF) de chaque terme sur TOUT le corpus
   (un avis = un document — PAS une présence par cluster, qui raterait un terme massif
   concentré dans peu de clusters). Un terme dont la DF dépasse un **seuil dérivé des
   données** est un mot-vide *de ce corpus*. Le seuil n'est pas un magic number : on
   isole le « plateau saturant » du haut de la distribution via le plus grand **écart
   relatif** (gap) entre termes de tête contenus (cf. `_domain_stopwords`). Corpus-
   relatif, langue-agnostique, zéro liste en dur.

2. **Mots-vides FONCTIONNELS (linguistique, pas domaine).** Les mots-outils (le, la, et
   / the, and / der, die…) sont chargés DYNAMIQUEMENT depuis un set de stopwords
   multilingue standard (`stopwordsiso`, 50+ langues), pris en **union** — aucune
   détection de langue, rien de spécifique à un domaine. Repli statistique si la lib
   est absente : token court ET très fréquent (faible longueur + haute DF), un signal
   universel de mot-outil.

3. **c-TF-IDF (class-based, COMPLÉMENT).** Chaque cluster = un « document » (concat de
   ses avis). Pondération BERTopic : ctfidf(t,c) = tf(t,c) · log(1 + A / f(t)), où
   f(t) = fréquence totale du terme sur toutes les classes, A = mots moyens par classe.
   Fait remonter le terme **distinctif** d'un cluster et écrase ce qui est commun à
   tous — le complément « soft » du retrait « hard » des mots-vides ci-dessus.

API publique inchangée : `name_clusters(cluster_docs, top_k, label_k)` →
`{cluster_id: {"label": str, "keywords": [str, ...]}}`. Params optionnels (rétro-
compatibles) : `corpus_stopwords` (set pré-dérivé pour partager le même vocabulaire
saturant entre niveaux macro/sous) et `return_diagnostics`.
"""

from __future__ import annotations

import math
import re

# Convention statistique générique (PAS un mot de domaine) pour repérer une valeur
# aberrante haute sur une distribution : k écarts-types au-dessus de la moyenne.
SIGMA_K = 3.0
# Écart relatif minimal (ratio de DF) qui sépare le « plateau saturant » du corps.
GAP_RATIO = 1.3
# Repli fonctionnel statistique (utilisé seulement si la lib de stopwords manque) :
# un mot-outil est court ET fréquent.
SHORT_LEN = 4
MIN_TOKEN_LEN = 3

# Tokenizer Unicode générique : suites de lettres (toutes langues), aucun mot listé.
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

_LING_CACHE: set[str] | None = None


def _tokenizer(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text) if len(w) >= MIN_TOKEN_LEN]


def _linguistic_stopwords() -> set[str]:
    """Union multilingue de mots-outils, chargée DYNAMIQUEMENT (linguistique, pas
    domaine). Filtrée aux tokens cohérents avec notre tokenizer. Vide si lib absente.
    """
    global _LING_CACHE
    if _LING_CACHE is not None:
        return _LING_CACHE
    out: set[str] = set()
    try:
        import stopwordsiso

        raw = set().union(*(stopwordsiso.stopwords(lang) for lang in stopwordsiso.langs()))
        out = {w for w in raw if len(w) >= MIN_TOKEN_LEN and _WORD_RE.fullmatch(w)}
    except Exception:
        out = set()
    _LING_CACHE = out
    return out


def _df_fractions(avis: list[str]) -> tuple[dict[str, float], int]:
    """Document-frequency (fraction d'avis contenant le terme) — un avis = un document."""
    n_docs = len(avis)
    df: dict[str, int] = {}
    for text in avis:
        for term in set(_tokenizer(text)):
            df[term] = df.get(term, 0) + 1
    return ({t: c / n_docs for t, c in df.items()}, n_docs) if n_docs else ({}, 0)


def _domain_stopwords(
    fracs: dict[str, float],
    linguistic: set[str],
    sigma_k: float = SIGMA_K,
    gap_ratio: float = GAP_RATIO,
) -> tuple[set[str], float | None]:
    """Mots-vides de DOMAINE via cutoff max_df-document dérivé des données.

    On part des termes *de contenu* (hors mots-outils) dont la DF est aberrante haute
    (> μ + k·σ : ils saturent le corpus). On isole ensuite le plateau de tête : si un
    grand écart relatif (gap) sépare un sous-ensemble du reste, seul ce sous-ensemble
    (les plus saturants) est retenu — ainsi un mot de contenu modérément fréquent reste
    disponible pour le naming. Renvoie (stopwords, cutoff_df).
    """
    if not fracs:
        return set(), None
    vals = list(fracs.values())
    mean = sum(vals) / len(vals)
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
    elevated_cut = mean + sigma_k * std

    # Candidats : termes de contenu (non fonctionnels) au-dessus du seuil d'aberration.
    cand = sorted(
        ((t, f) for t, f in fracs.items() if f > elevated_cut and t not in linguistic),
        key=lambda kv: -kv[1],
    )
    if not cand:
        return set(), None
    if len(cand) == 1:
        return {cand[0][0]}, cand[0][1]

    # Plus grand écart RELATIF entre termes de tête consécutifs → frontière du plateau.
    vals_c = [f for _, f in cand]
    ratios = [(vals_c[i] / vals_c[i + 1], i) for i in range(len(vals_c) - 1)]
    best_ratio, cut_i = max(ratios)
    if best_ratio >= gap_ratio:
        domain_terms = cand[: cut_i + 1]          # au-dessus du gap = les saturants
    else:
        domain_terms = cand                       # pas de séparation nette : tout sature
    cutoff_df = domain_terms[-1][1]
    return {t for t, _ in domain_terms}, cutoff_df


def _statistical_functional(fracs: dict[str, float]) -> set[str]:
    """Repli langue-agnostique pour mots-outils : court ET fréquent (faible longueur +
    DF dans la queue haute). Utilisé seulement si la lib de stopwords est absente."""
    if not fracs:
        return set()
    vals = list(fracs.values())
    mean = sum(vals) / len(vals)
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
    thr = mean + std
    return {t for t, f in fracs.items() if len(t) <= SHORT_LEN and f > thr}


def derive_corpus_stopwords(
    avis: list[str],
    sigma_k: float = SIGMA_K,
) -> tuple[set[str], dict]:
    """Dérive l'ensemble des mots-vides (domaine + fonctionnels) à partir du corpus.

    `avis` = liste d'avis bruts (un avis = un document). AUCUN mot codé en dur :
    le domaine vient du cutoff max_df-document, le fonctionnel d'une lib multilingue
    (repli statistique). Renvoie `(stopwords, diagnostics)`.
    """
    fracs, n_docs = _df_fractions(avis)
    if not fracs:
        return set(), {"n_docs": n_docs, "domain_cutoff": None, "domain_examples": []}

    linguistic = _linguistic_stopwords()
    functional = linguistic if linguistic else _statistical_functional(fracs)
    domain, cutoff = _domain_stopwords(fracs, functional, sigma_k)

    stop = functional | domain
    domain_examples = sorted(((t, round(fracs[t], 3)) for t in domain), key=lambda kv: -kv[1])
    diag = {
        "n_docs": n_docs,
        "vocab": len(fracs),
        "functional_source": "stopwordsiso(multilingual-union)" if linguistic else "statistical(short+frequent)",
        "n_functional": len(functional),
        "domain_cutoff_df": round(cutoff, 5) if cutoff is not None else None,
        "n_domain": len(domain),
        "domain_examples": domain_examples,
        "n_stopwords": len(stop),
    }
    return stop, diag


def name_clusters(
    cluster_docs: dict[int, list[str]],
    top_k: int = 6,
    label_k: int = 3,
    *,
    corpus_stopwords: set[str] | None = None,
    return_diagnostics: bool = False,
):
    """Nomme chaque cluster via c-TF-IDF + mots-vides corpus-dérivés.

    `cluster_docs` : {cluster_id: [avis, ...]}. Renvoie
    `{cluster_id: {"label": str, "keywords": [str, ...]}}`.

    Si `corpus_stopwords` est fourni (set pré-dérivé sur le corpus GLOBAL), il est
    réutilisé — utile pour partager le même vocabulaire saturant entre le naming macro
    et celui des sous-thèmes. Sinon il est dérivé des avis de `cluster_docs`.
    """
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer

    cids = sorted(cluster_docs.keys())
    docs = [" ".join(cluster_docs[c]) for c in cids]

    if not docs or all(not d.strip() for d in docs):
        empty = {c: {"label": f"thème {c}", "keywords": []} for c in cids}
        return (empty, {"domain_cutoff_df": None}) if return_diagnostics else empty

    # 1) Mots-vides DÉRIVÉS du corpus (domaine max_df-document + fonctionnels).
    if corpus_stopwords is None:
        all_avis = [a for c in cids for a in cluster_docs[c]]
        stop, diag = derive_corpus_stopwords(all_avis)
    else:
        stop, diag = set(corpus_stopwords), {"reused": True, "n_stopwords": len(corpus_stopwords)}

    # 2) Comptage des n-grammes (1,2). On ne passe PAS `stop_words` au vectorizer
    #    (évite l'avertissement de re-tokenisation) : on filtre `stop` au moment du tri.
    vectorizer = CountVectorizer(
        tokenizer=_tokenizer, token_pattern=None, ngram_range=(1, 2), min_df=1,
    )
    counts = vectorizer.fit_transform(docs)            # (n_clusters, n_terms)
    vocab = vectorizer.get_feature_names_out()

    # 3) c-TF-IDF (BERTopic) : tf(t,c) · log(1 + A / f(t)).
    X = counts.toarray().astype(float)
    words_per_class = X.sum(axis=1)
    words_per_class[words_per_class == 0] = 1.0
    tf = X / words_per_class[:, None]
    f_t = X.sum(axis=0)
    f_t[f_t == 0] = 1.0
    A = words_per_class.mean()
    idf = np.log(1.0 + A / f_t)
    ctfidf = tf * idf[None, :]

    out: dict[int, dict] = {}
    for row, cid in enumerate(cids):
        scores = ctfidf[row]
        order = scores.argsort()[::-1]
        keywords: list[str] = []
        for j in order:
            if scores[j] <= 0:
                break
            term = vocab[j]
            # retire tout n-gramme contenant un mot-vide (domaine ou fonctionnel)
            if any(tok in stop for tok in term.split()):
                continue
            # évite la redondance unigramme/bigramme déjà couverte
            if any(term != kw and (term in kw.split() or kw in term.split()) for kw in keywords):
                continue
            keywords.append(term)
            if len(keywords) >= top_k:
                break
        label = " · ".join(keywords[:label_k]) if keywords else f"thème {cid}"
        out[cid] = {"label": label, "keywords": keywords}

    return (out, diag) if return_diagnostics else out
