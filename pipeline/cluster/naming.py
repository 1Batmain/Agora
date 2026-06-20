"""T-N6 · Naming des thèmes (TF-IDF seul — décision Bob, pas de LLM).

Chaque communauté est traitée comme un document (concaténation de ses avis).
Un TF-IDF inter-clusters fait ressortir les termes distinctifs de chaque thème.
`keywords[]` = top termes ; `label` = 2-3 mots-clés joints (lisible).

Si `keybert` est installé on pourrait l'utiliser, mais le défaut reste TF-IDF
pour rester léger et sans modèle additionnel.
"""

from __future__ import annotations

import re

# Stopwords FR (sklearn ne fournit pas le français). Liste compacte mais utile.
FRENCH_STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "à", "au",
    "aux", "en", "dans", "pour", "par", "sur", "avec", "sans", "sous", "vers",
    "chez", "ce", "cet", "cette", "ces", "se", "sa", "son", "ses", "leur",
    "leurs", "notre", "nos", "votre", "vos", "mon", "ma", "mes", "ton", "ta",
    "tes", "il", "elle", "ils", "elles", "on", "nous", "vous", "je", "tu",
    "que", "qui", "quoi", "dont", "où", "ne", "pas", "plus", "moins", "très",
    "trop", "peu", "tout", "tous", "toute", "toutes", "est", "sont", "être",
    "étaient", "était", "ont", "avoir", "fait", "faire", "faut", "il faut",
    "doit", "devrait", "devraient", "comme", "aussi", "encore", "déjà", "donc",
    "car", "mais", "afin", "ainsi", "entre", "leurs", "y", "d", "l", "qu", "s",
    "n", "c", "j", "m", "t", "si", "non", "oui", "cela", "ça", "leur",
}

_WORD_RE = re.compile(r"[a-zàâäéèêëîïôöùûüçœ]+", re.IGNORECASE)


def _tokenizer(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text) if len(w) > 2]


def name_clusters(
    cluster_docs: dict[int, list[str]],
    top_k: int = 6,
    label_k: int = 3,
) -> dict[int, dict]:
    """Retourne {cluster_id: {"label": str, "keywords": [str, ...]}}."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    cids = sorted(cluster_docs.keys())
    docs = [" ".join(cluster_docs[c]) for c in cids]

    if not docs:
        return {}

    stop = sorted(FRENCH_STOPWORDS)
    vectorizer = TfidfVectorizer(
        tokenizer=_tokenizer,
        token_pattern=None,
        stop_words=stop,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(docs)
    vocab = vectorizer.get_feature_names_out()

    out: dict[int, dict] = {}
    for row, cid in enumerate(cids):
        scores = tfidf[row].toarray().ravel()
        order = scores.argsort()[::-1]
        keywords: list[str] = []
        for j in order:
            if scores[j] <= 0:
                break
            term = vocab[j]
            # évite qu'un bigramme noie ses unigrammes déjà retenus
            if any(term != kw and term in kw.split() for kw in keywords):
                pass
            keywords.append(term)
            if len(keywords) >= top_k:
                break
        label = " · ".join(keywords[:label_k]) if keywords else f"thème {cid}"
        out[cid] = {"label": label, "keywords": keywords}
    return out
