"""Traduction des avis vers le français (cœur pur, batché). Cf. `backend.translate`
pour l'orchestration build (détection de langue + cache idempotent aligné aux avis)."""

from .translate import (
    DEFAULT_TRANSLATE_MODEL,
    FR,
    is_french,
    translate_batch,
)

__all__ = ["DEFAULT_TRANSLATE_MODEL", "FR", "is_french", "translate_batch"]
