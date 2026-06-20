"""Cohérence de thèmes (NPMI) — la mesure INTRINSÈQUE du banc qualité.

On note la qualité d'un clustering par la **cohérence** de ses thèmes : les
top-mots TF-IDF d'un même cluster doivent **co-occurrer** dans le corpus de
référence. NPMI ∈ [-1, 1] : 1 = co-occurrence parfaite, 0 = indépendance,
< 0 = anti-corrélation. Plus haut = thèmes plus cohérents.

⚠️ Confusion multilingue évitée. Le corpus est trilingue (DE/FR/IT) et chaque
commentaire est monolingue : un mot allemand et un mot français ne co-occurrent
jamais. Si on mélangeait les langues, un BON cluster trans-langues serait puni
(top-mots de langues différentes → co-occurrence ≈ 0). On calcule donc la
cohérence **par langue** (top-mots et co-occurrences restreints à une langue),
puis on **moyenne pondérée** par le nombre de documents de chaque langue. C'est
la cohérence thématique « intra-langue », non biaisée par le multilinguisme.

Implémentation maison (pas de gensim) : NPMI sur la **co-occurrence
document** (présence booléenne par commentaire). Les commentaires x-stance
sont courts → fenêtre = document. Déterministe, sans dépendance lourde.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

# Stopwords compacts DE/FR/IT (sklearn ne couvre que l'anglais). On retire le
# bruit grammatical pour que les top-mots portent du sens thématique.
STOPWORDS: set[str] = {
    # --- français ---
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "à", "au",
    "aux", "en", "dans", "pour", "par", "sur", "avec", "sans", "sous", "vers",
    "ce", "cet", "cette", "ces", "se", "sa", "son", "ses", "leur", "leurs",
    "notre", "nos", "votre", "vos", "mon", "ma", "mes", "il", "elle", "ils",
    "elles", "on", "nous", "vous", "que", "qui", "quoi", "dont", "où", "ne",
    "pas", "plus", "moins", "très", "trop", "peu", "tout", "tous", "toute",
    "toutes", "est", "sont", "être", "était", "ont", "avoir", "fait", "faire",
    "faut", "doit", "comme", "aussi", "encore", "déjà", "donc", "car", "mais",
    "afin", "ainsi", "entre", "si", "non", "oui", "cela", "ça", "les", "ces",
    "été", "avait", "aussi", "leur", "même", "alors", "bien", "deux",
    # --- allemand ---
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem",
    "einer", "und", "oder", "aber", "doch", "denn", "weil", "dass", "wenn",
    "als", "wie", "auch", "noch", "schon", "nur", "nicht", "kein", "keine",
    "ist", "sind", "war", "waren", "sein", "haben", "hat", "hatte", "werden",
    "wird", "wurde", "muss", "soll", "kann", "können", "für", "mit", "von",
    "zu", "zum", "zur", "auf", "aus", "bei", "nach", "über", "unter", "vor",
    "durch", "gegen", "ohne", "um", "im", "in", "an", "am", "es", "sie", "er",
    "ich", "wir", "ihr", "man", "sich", "dieser", "diese", "dieses", "mehr",
    "sehr", "schon", "immer", "wieder", "hier", "dann", "also", "etwa",
    # --- italien ---
    "il", "lo", "la", "le", "gli", "un", "uno", "una", "di", "del", "della",
    "dei", "delle", "e", "ed", "o", "ma", "se", "che", "chi", "cui", "non",
    "più", "meno", "molto", "poco", "tutto", "tutti", "è", "sono", "era",
    "essere", "avere", "ha", "hanno", "per", "con", "su", "tra", "fra", "da",
    "in", "nel", "nella", "al", "alla", "come", "anche", "ancora", "già",
    "quindi", "perché", "questo", "questa", "quello", "si", "ci", "ne", "lui",
    "lei", "loro", "noi", "voi", "io", "tu", "essi", "deve", "può", "sempre",
}

_WORD_RE = re.compile(r"[a-zàâäéèêëîïôöùûüçœßüäö]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Tokens alphabétiques minuscules, > 2 lettres, hors stopwords."""
    return [
        w for w in (m.lower() for m in _WORD_RE.findall(text))
        if len(w) > 2 and w not in STOPWORDS
    ]


def top_words_per_cluster(
    cluster_docs: dict[int, list[str]], top_n: int = 10
) -> dict[int, list[str]]:
    """Top-N mots (unigrammes) par cluster via TF-IDF inter-clusters.

    Chaque cluster = un document concaténé ; le TF-IDF fait ressortir les mots
    distinctifs. Unigrammes seulement (la cohérence compte des co-occurrences
    de mots, pas de bigrammes).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    cids = sorted(cluster_docs.keys())
    docs = [" ".join(cluster_docs[c]) for c in cids]
    if not any(docs):
        return {c: [] for c in cids}

    vec = TfidfVectorizer(
        tokenizer=tokenize, token_pattern=None, ngram_range=(1, 1),
        min_df=1, sublinear_tf=True,
    )
    tfidf = vec.fit_transform(docs)
    vocab = vec.get_feature_names_out()
    out: dict[int, list[str]] = {}
    for row, cid in enumerate(cids):
        scores = tfidf[row].toarray().ravel()
        order = scores.argsort()[::-1]
        words = [vocab[j] for j in order if scores[j] > 0][:top_n]
        out[cid] = words
    return out


def npmi_coherence(
    topics_words: list[list[str]],
    docs_tokens: list[set[str]],
    eps: float = 1e-12,
) -> float | None:
    """NPMI moyen sur les paires de top-mots, par co-occurrence document.

    `topics_words` : liste de listes de mots (un par cluster).
    `docs_tokens`  : corpus de référence, un set de tokens par document.
    Retourne la moyenne (sur les clusters) du NPMI moyen des paires, ou None.
    """
    D = len(docs_tokens)
    if D == 0:
        return None

    vocab = {w for t in topics_words for w in t}
    if not vocab:
        return None

    # Présence document restreinte aux mots des thèmes (rapide).
    doc_sets = [s & vocab for s in docs_tokens]
    df: Counter[str] = Counter()
    codf: Counter[tuple[str, str]] = Counter()
    for s in doc_sets:
        if not s:
            continue
        ws = sorted(s)
        for w in ws:
            df[w] += 1
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                codf[(ws[i], ws[j])] += 1

    def npmi(a: str, b: str) -> float | None:
        p_a = df[a] / D
        p_b = df[b] / D
        if p_a == 0 or p_b == 0:
            return None  # mot absent de la référence → paire indécidable
        key = (a, b) if a < b else (b, a)
        p_ab = codf.get(key, 0) / D
        p_ab_s = p_ab + eps
        denom = -math.log(p_ab_s)
        if denom == 0:
            return None
        return math.log(p_ab_s / (p_a * p_b)) / denom

    per_topic: list[float] = []
    for t in topics_words:
        words = [w for w in t if df[w] > 0]
        pairs = [
            v for i in range(len(words)) for j in range(i + 1, len(words))
            if (v := npmi(words[i], words[j])) is not None
        ]
        if pairs:
            per_topic.append(float(np.mean(pairs)))
    return float(np.mean(per_topic)) if per_topic else None


def per_language_coherence(
    membership: list[int],
    texts: list[str],
    langs: list[str],
    top_n: int = 10,
    min_cluster_docs: int = 3,
) -> dict:
    """Cohérence NPMI calculée PAR LANGUE puis moyennée (pondérée par #docs).

    Pour chaque langue : on restreint corpus + appartenance à cette langue, on
    extrait les top-mots par cluster sur ce sous-corpus, et on calcule le NPMI
    avec ce sous-corpus comme référence. La moyenne pondérée évite le biais
    « top-mots de langues différentes qui ne co-occurrent jamais ».
    """
    by_lang: dict[str, list[int]] = {}
    for i, lg in enumerate(langs):
        by_lang.setdefault(lg, []).append(i)

    per_lang: dict[str, float | None] = {}
    weights: dict[str, int] = {}
    for lg, idxs in by_lang.items():
        # clusters → docs de CETTE langue
        cdocs: dict[int, list[str]] = {}
        for i in idxs:
            cdocs.setdefault(membership[i], []).append(texts[i])
        cdocs = {c: d for c, d in cdocs.items() if len(d) >= min_cluster_docs}
        if not cdocs:
            per_lang[lg] = None
            weights[lg] = len(idxs)
            continue
        tw = top_words_per_cluster(cdocs, top_n=top_n)
        ref = [set(tokenize(texts[i])) for i in idxs]
        per_lang[lg] = npmi_coherence(list(tw.values()), ref)
        weights[lg] = len(idxs)

    num = sum(
        per_lang[lg] * weights[lg]
        for lg in per_lang
        if per_lang[lg] is not None
    )
    den = sum(weights[lg] for lg in per_lang if per_lang[lg] is not None)
    overall = float(num / den) if den > 0 else None
    return {"overall": overall, "per_lang": per_lang}
