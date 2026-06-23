"""Score de DÉVELOPPEMENT d'un claim — D1 : valoriser les arguments ÉTOFFÉS plutôt
que la reformulation générique courte, tout en gardant la **centralité en garde-fou**
(pas de hors-sujet).

Mesure préalable (cf. `backend/measure_develop.py` et `DEVELOP_NOTE.md`) : la longueur du
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
