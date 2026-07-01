"""T-D2 — nettoyage et normalisation du texte libre.

`clean_text` produit `text_clean` : espaces normalisés, ponctuation/casse
raisonnables, retrait des PII évidentes. Conserve le sens (pas de lowercasing
agressif, important pour les embeddings multilingues type BGE-m3/e5).
"""
from __future__ import annotations

import re
import unicodedata

# PII évidentes à masquer dans le texte (l'anonymisation porte aussi sur l'auteur).
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\s.\-]?){9,}\d(?!\d)")
_URL = re.compile(r"https?://\S+|www\.\S+")
_HANDLE = re.compile(r"(?<!\w)@[A-Za-z0-9_.]{2,}")

_WS = re.compile(r"\s+")
# Caractères de contrôle (hors tab/newline déjà gérés par le collapse d'espaces).
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_REPEAT_PUNCT = re.compile(r"([!?.,])\1{2,}")  # "!!!!" -> "!"


def strip_pii(text: str) -> str:
    """Masque emails, téléphones, URLs et @mentions dans le texte libre."""
    text = _EMAIL.sub("[email]", text)
    text = _URL.sub("[url]", text)
    text = _HANDLE.sub("[mention]", text)
    text = _PHONE.sub("[tel]", text)
    return text


def clean_text(text: str) -> str:
    """Normalise un avis citoyen. Retourne '' si le contenu est vide/quasi-vide."""
    if not text:
        return ""
    # Normalisation Unicode + espaces insécables (fréquents dans les exports FR).
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\xa0", " ").replace(" ", " ")
    text = _CTRL.sub(" ", text)
    text = strip_pii(text)
    text = _REPEAT_PUNCT.sub(r"\1", text)
    text = _WS.sub(" ", text).strip()
    # Guillemets/tirets typographiques -> formes simples (stabilise le dédoublonnage aval).
    text = (
        text.replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
    )
    return text


def is_empty_like(text_clean: str, min_chars: int = 3, min_alpha: int = 2) -> bool:
    """Vrai si le texte nettoyé est vide ou quasi-vide (à retirer du corpus)."""
    if len(text_clean) < min_chars:
        return True
    alpha = sum(c.isalpha() for c in text_clean)
    return alpha < min_alpha


def make_label(text_clean: str, maxlen: int) -> str:
    """Libellé d'affichage court (tronqué proprement sur un mot)."""
    if len(text_clean) <= maxlen:
        return text_clean
    cut = text_clean[:maxlen].rsplit(" ", 1)[0].rstrip(",;:.- ")
    return (cut or text_clean[:maxlen]).rstrip() + "…"
