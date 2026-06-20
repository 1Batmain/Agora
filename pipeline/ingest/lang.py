"""T-D4 (partie langue) — détection de langue.

Multilingue par défaut (audit #10) : `langdetect` est le chemin recommandé
(via `uv run --with langdetect`). Sans lui, un **repli élargi** couvre plusieurs
langues européennes (fr/en/de/es/pt/it/nl) au lieu de FR/EN/DE seulement, et le
défaut est **`"und"`** (indéterminé) — jamais `"fr"`, qui étiquetterait à tort un
corpus non-FR. La langue est calculée sur le texte nettoyé.
"""
from __future__ import annotations

try:  # dépendance légère, fournie via `uv run --with langdetect`
    from langdetect import DetectorFactory, LangDetectException
    from langdetect import detect as _ld_detect

    DetectorFactory.seed = 0  # déterminisme
    _HAS_LANGDETECT = True
except Exception:  # pragma: no cover - chemin de repli
    _HAS_LANGDETECT = False

# Mots fonctionnels très fréquents par langue (repli élargi sans dépendance).
_STOP = {
    "fr": {"le", "la", "les", "de", "des", "et", "est", "que", "pour", "pas",
           "une", "un", "je", "ne", "sur", "qui", "dans", "plus", "ce", "il"},
    "en": {"the", "and", "is", "of", "to", "that", "for", "not", "with", "this",
           "you", "are", "it", "on", "they", "we", "have", "but", "as", "be"},
    "de": {"der", "die", "das", "und", "ist", "nicht", "ein", "eine", "zu",
           "mit", "auf", "den", "von", "wir", "es", "auch", "für", "dem"},
    "es": {"el", "la", "los", "las", "de", "que", "no", "por", "para", "con",
           "una", "uno", "es", "en", "se", "su", "lo", "como", "más", "pero"},
    "pt": {"o", "a", "os", "as", "de", "que", "não", "por", "para", "com",
           "uma", "um", "é", "em", "se", "do", "da", "como", "mais", "mas"},
    "it": {"il", "lo", "la", "le", "di", "che", "non", "per", "con", "una",
           "uno", "è", "in", "si", "del", "della", "come", "più", "ma", "sono"},
    "nl": {"de", "het", "een", "en", "van", "is", "niet", "dat", "op", "te",
           "met", "voor", "die", "in", "zijn", "ook", "maar", "om", "aan", "we"},
}


def _heuristic(text: str) -> str:
    toks = [t for t in "".join(c if c.isalpha() else " " for c in text.lower()).split()]
    if not toks:
        return "und"  # undetermined
    scores = {lang: sum(t in words for t in toks) for lang, words in _STOP.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "und"


def detect_lang(text: str, default: str = "und") -> str:
    """Code langue ISO-639-1 (fr/en/de/...) ; 'und' (défaut) si indéterminé."""
    text = (text or "").strip()
    if len(text) < 3:
        return default
    if _HAS_LANGDETECT:
        try:
            return _ld_detect(text)
        except LangDetectException:
            return _heuristic(text) or default
    return _heuristic(text) or default


def has_langdetect() -> bool:
    return _HAS_LANGDETECT
