"""Score de DÉVELOPPEMENT d'un claim — D1 : valoriser les arguments ÉTOFFÉS plutôt
que la reformulation générique courte, tout en gardant la **centralité en garde-fou**
(pas de hors-sujet).

Mesure préalable (cf. `backend/measure_develop.py` et `.agent/notes/DEVELOP_NOTE.md`) : la longueur du
claim corrèle POSITIVEMENT avec la distance au centroïde (tiktok +0.46, granddebat
+0.38 intra-feuille). Autrement dit le pur médoïde (plus proche du centroïde) surface
les claims COURTS et génériques ; les arguments développés sont plus loin. D'où ce
re-ranking `centralité(garde-fou) × développement`.

Développement = trois signaux, tous DÉRIVÉS des données (zéro mot de domaine codé) :
  - **longueur** (rang relatif dans le pool — robuste, langue-agnostique) ;
  - **spécificité** (idf moyen des tokens : mots rares = contenu, pas du remplissage) ;
  - **raisonnement** (bonus LÉGER : connecteurs argumentatifs multilingues + chiffres).

Le garde-fou centralité est un **gate multiplicatif** : dans le gros du nuage (claims
on-topic) le gate vaut 1 et seul le développement départage ; les vrais outliers de
centralité sont rabaissés proportionnellement → jamais de hors-sujet en tête.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from pipeline.cluster.naming import _tokenizer

# Connecteurs de raisonnement (FR/EN/DE/IT/ES) — bonus LÉGER, pas un critère dur.
# Signal d'argumentation, pas du domaine ; volontairement court et multilingue.
_REASON_MARKERS = (
    "parce que", "parce qu", "car ", "afin de", "afin d", "donc", "puisque",
    "en effet", "c'est pourquoi", "pour que", "grâce à", "à cause de",
    "because", "therefore", "in order to", "since ", "thus", "hence", "so that",
    "weil", "damit", "deshalb", "denn ",
    "perché", "perciò", "quindi", "poiché",
    "porque", "por lo tanto", "para que",
)
_DIGIT_RE = re.compile(r"\d")

# Poids des trois signaux de développement (longueur dominante car proxy le + fort).
_W_LEN, _W_SPEC, _W_REASON = 0.50, 0.35, 0.15

# Garde-fou : quantile de centralité sous lequel un claim est considéré périphérique.
GUARD_QUANTILE = 0.20


def corpus_idf(texts: list[str]) -> dict[str, float]:
    """idf des tokens sur TOUT le corpus de claims (df-document, un claim = un doc)."""
    n = len(texts)
    if not n:
        return {}
    df: dict[str, int] = {}
    for t in texts:
        for tok in set(_tokenizer(t)):
            df[tok] = df.get(tok, 0) + 1
    return {tok: math.log(1.0 + n / c) for tok, c in df.items()}


def _specificity(text: str, idf: dict[str, float], idf_max: float) -> float:
    """Spécificité ∈ [0,1] : idf moyen des tokens de contenu, normalisé par l'idf max."""
    toks = _tokenizer(text)
    if not toks or idf_max <= 0:
        return 0.0
    mean_idf = sum(idf.get(t, idf_max) for t in toks) / len(toks)
    return min(1.0, mean_idf / idf_max)


def _reasoning(text: str) -> float:
    """Bonus raisonnement ∈ [0,1] : présence de connecteurs argumentatifs et de chiffres."""
    low = text.lower()
    hits = sum(1 for m in _REASON_MARKERS if m in low)
    if _DIGIT_RE.search(text):
        hits += 1
    return min(1.0, hits / 2.0)


def development_scores(texts: list[str], idf: dict[str, float]) -> np.ndarray:
    """Score de développement ∈ [0,1] par claim : longueur(rang) + spécificité + raisonnement."""
    n = len(texts)
    if n == 0:
        return np.zeros(0)
    lengths = np.array([len(t) for t in texts], float)
    # Longueur en RANG relatif au pool (0 = le plus court, 1 = le plus long) : robuste,
    # sans échelle magique ; constante si pool homogène.
    if n > 1 and np.ptp(lengths) > 0:
        order = lengths.argsort()
        rank = np.empty(n)
        rank[order] = np.arange(n) / (n - 1)
        len_score = rank
    else:
        len_score = np.zeros(n)
    idf_max = max(idf.values()) if idf else 0.0
    spec = np.array([_specificity(t, idf, idf_max) for t in texts])
    reason = np.array([_reasoning(t) for t in texts])
    return _W_LEN * len_score + _W_SPEC * spec + _W_REASON * reason


def guard_gate(sims: np.ndarray, quantile: float = GUARD_QUANTILE) -> np.ndarray:
    """Gate de centralité ∈ [0,1] : 1 dans le gros du nuage, décroît pour les outliers.

    `sims` = cos au centroïde (centralité). On fixe un plancher au quantile bas des
    similarités ; au-dessus le gate vaut 1 (le développement décide seul), en-dessous
    il décroît proportionnellement (garde-fou anti hors-sujet). Pas de cliff dur.
    """
    if len(sims) == 0:
        return sims
    floor = max(0.05, float(np.quantile(sims, quantile)))
    return np.clip(sims / floor, 0.0, 1.0)


def rerank_order(members: list[int], sims: np.ndarray, texts: list[str],
                 idf: dict[str, float]) -> np.ndarray:
    """Ordre des `members` par `centralité(garde-fou) × développement` décroissant.

    Renvoie des indices LOCAUX (dans `members`). Départage par centralité décroissante.
    `texts` est aligné sur `members` (textes des claims du pool), `sims` sur `members`.
    """
    n = len(members)
    if n == 0:
        return np.zeros(0, dtype=int)
    dev = development_scores(texts, idf)
    gate = guard_gate(sims)
    score = gate * dev
    # Tri stable : score desc, puis centralité desc en départage.
    return np.lexsort((-sims, -score))


# --------------------------------------------------------------------------- #
# Sélection DISTINCTIVE — claims denses dans le vocabulaire c-TF-IDF du cluster.
#
# Dual du re-ranking DÉVELOPPEMENT ci-dessus (`rerank_order`) mais objectif opposé :
# non pas l'argument le plus ÉTOFFÉ, mais la contribution qui PORTE le mieux les termes
# CARACTÉRISTIQUES du cluster. Motivation (cf. `research/cluster_merge_note.md`, §5) :
# sous l'anisotropie de l'embedding (centroïdes quasi colinéaires, +0.04 de contraste
# inter-cluster à peine), les claims proches du centroïde sont GÉNÉRIQUES — elles
# partagent la composante commune du corpus — et font tomber le titrage LLM sur des
# quasi-synonymes (« addiction » / « temps perdu » répétés entre thèmes DISTINCTS). On
# ANCRE plutôt le titrage dans les claims riches en vocabulaire distinctif. Déterministe,
# zéro LLM, zéro mot de domaine en dur.
#
# Helper PARTAGÉ : sert le titrage ancré (`backend.titles`) ET la lane stance (sélection
# des claims saillants d'un pôle par le même critère de distinctivité).
# --------------------------------------------------------------------------- #
def cluster_term_weights(texts: list[str], idf: dict[str, float]) -> dict[str, float]:
    """Poids c-TF-IDF (class-based) des termes d'un cluster : tf_cluster(t) · idf(t).

    `tf_cluster` = nombre TOTAL d'occurrences du terme dans les claims du cluster ; `idf`
    = idf corpus des claims (cf. `corpus_idf`), qui écrase les termes présents PARTOUT
    (« tiktok », « vélo »…). Un terme fréquent DANS le cluster et rare AILLEURS pèse
    lourd = distinctif ; un one-off rare (tf 1) reste modeste. Déterministe, langue-
    agnostique (même tokenizer que le naming c-TF-IDF). Renvoie {} si aucun token.
    """
    tf: Counter[str] = Counter()
    for t in texts:
        tf.update(_tokenizer(t))
    return {tok: c * idf.get(tok, 0.0) for tok, c in tf.items()}


def select_distinctive_claims(texts: list[str], idf: dict[str, float], k: int = 5,
                              *, anchor_terms: list[str] | None = None) -> list[int]:
    """Indices LOCAUX des ≤`k` claims les plus DENSES dans le vocabulaire distinctif.

    Densité d'un claim = poids c-TF-IDF MOYEN de ses tokens DISTINCTS (cf.
    `cluster_term_weights`) : on moyenne sur le vocabulaire du claim (set), pas sur les
    répétitions, pour qu'un claim ne remonte pas juste en martelant un même terme
    porteur — c'est la RICHESSE en vocabulaire caractéristique qui compte. On surface
    ainsi les contributions ancrées dans les termes du cluster plutôt que le médoïde
    générique (centroïde-proche sous anisotropie). Déterministe : tri stable par densité
    décroissante puis index croissant → même sortie à chaque run, aucun appel LLM.

    `texts` = claims du cluster (pas tout le corpus). `idf` = idf corpus des claims
    (`corpus_idf`) ; recalculé LOCALEMENT s'il est vide (repli autonome). `anchor_terms`
    (optionnel, p.ex. les mots-clés c-TF-IDF déjà nommés du nœud) restreint le vocabulaire
    porteur à ces termes → sélection ALIGNÉE sur les ancres montrées ensuite au LLM.

    API partagée titrage ↔ lane stance : renvoie des indices (le caller mappe vers les
    textes ou les ids de claims selon son besoin).
    """
    n = len(texts)
    if n == 0 or k <= 0:
        return []
    if not idf:
        idf = corpus_idf(texts)
    weights = cluster_term_weights(texts, idf)
    if anchor_terms is not None:
        allow = {a.lower() for a in anchor_terms}
        weights = {t: w for t, w in weights.items() if t in allow}

    scored: list[tuple[float, int]] = []
    for i, text in enumerate(texts):
        toks = set(_tokenizer(text))
        density = sum(weights.get(t, 0.0) for t in toks) / len(toks) if toks else 0.0
        scored.append((density, i))
    # Tri stable : densité desc, index asc en départage (déterminisme total).
    scored.sort(key=lambda si: (-si[0], si[1]))
    return [i for _, i in scored[:k]]
