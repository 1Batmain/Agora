"""T-D4 (partie langue) — détection de langue.

Utilise `langdetect` si disponible (recommandé, cf. README), sinon retombe sur
une heuristique FR/EN basée sur des mots fonctionnels pour ne jamais bloquer le
pipeline. La langue est calculée sur le texte nettoyé.
"""
from __future__ import annotations

try:  # dépendance légère, fournie via `uv run --with langdetect`
    from langdetect import DetectorFactory, LangDetectException
    from langdetect import detect as _ld_detect

    DetectorFactory.seed = 0  # déterminisme
    _HAS_LANGDETECT = True
except Exception:  # pragma: no cover - chemin de repli
    _HAS_LANGDETECT = False

# Mots fonctionnels très fréquents par langue (repli sans dépendance).
_STOP = {
    "fr": {"le", "la", "les", "de", "des", "et", "est", "que", "pour", "pas",
           "une", "un", "je", "ne", "sur", "qui", "dans", "plus", "ce", "il"},
    "en": {"the", "and", "is", "of", "to", "that", "for", "not", "with", "this",
           "you", "are", "it", "on", "they", "we", "have", "but", "as", "be"},
    "de": {"der", "die", "das", "und", "ist", "nicht", "ein", "eine", "zu",
           "mit", "auf", "den", "von", "wir", "es", "auch", "für", "dem"},
}


def _heuristic(text: str) -> str:
    toks = [t for t in "".join(c if c.isalpha() else " " for c in text.lower()).split()]
    if not toks:
        return "und"  # undetermined
    scores = {lang: sum(t in words for t in toks) for lang, words in _STOP.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "und"


def detect_lang(text: str, default: str = "fr") -> str:
    """Code langue ISO-639-1 (fr/en/de/...) ; 'und' si indéterminé."""
    text = (text or "").strip()
    if len(text) < 3:
        return "und"
    if _HAS_LANGDETECT:
        try:
            return _ld_detect(text)
        except LangDetectException:
            return _heuristic(text)
    return _heuristic(text) or default


def has_langdetect() -> bool:
    return _HAS_LANGDETECT
