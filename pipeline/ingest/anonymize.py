"""T-D4 (partie anonymisation) — author_hash sans PII en clair."""
from __future__ import annotations

import hashlib

from . import config


def author_hash(raw_author: str, source: str) -> str:
    """sha256 salé et tronqué d'un identifiant auteur.

    Le sel (`AGORA_HASH_SALT`) rend le hash stable entre exécutions mais non
    ré-identifiable sans connaître le sel. Le préfixe `source` évite les
    collisions entre jeux de données. Retourne toujours une valeur opaque,
    jamais l'identifiant d'origine.
    """
    raw_author = (raw_author or "").strip()
    payload = f"{config.HASH_SALT}|{source}|{raw_author}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[: config.AUTHOR_HASH_LEN]
