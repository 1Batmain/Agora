"""Chemins et constantes GÉNÉRIQUES partagés par la lane data.

Plus aucune constante corpus-spécifique ici (audit #1) : les particularités de
chaque consultation (encoding, délimiteur, indices de colonnes, URL…) vivent
désormais dans son **descripteur** (`descriptors/*.json`), pas dans le code.
"""
from __future__ import annotations

import os
from pathlib import Path

# Racine du dépôt = deux niveaux au-dessus de ce fichier (pipeline/ingest/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]

# `data/` est gitignored : rien de ce qui s'y trouve n'est versionné.
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
IDEAS_JSONL = PROCESSED_DIR / "ideas.jsonl"

# Fixture committé (synthétique/anonyme) pour démarrer nlp + eval sans données réelles.
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ideas.sample.jsonl"

# Descripteurs de sources déclaratifs auto-découverts par `build`/`download`.
# Un corpus = un fichier JSON ici (cf. descriptors/README ou pipeline/ingest/README.md).
DESCRIPTORS_DIR = Path(__file__).resolve().parent / "descriptors"

# ---------------------------------------------------------------------------
# Réglages génériques
# ---------------------------------------------------------------------------
# Sel d'anonymisation (RGPD). AUCUN défaut : un sel committé rendrait
# `author_hash` réversible. Doit être fourni via l'env AGORA_HASH_SALT et
# validé au démarrage de l'INGESTION (cf. `require_hash_salt`). Le backend
# `serve` ne ré-ingère pas et n'est donc pas impacté.
HASH_SALT = os.environ.get("AGORA_HASH_SALT")
HASH_SALT_MIN_LEN = 32
AUTHOR_HASH_LEN = 16  # caractères hex conservés du sha256


def require_hash_salt() -> str:
    """Exige un sel d'anonymisation valide ; à appeler au point d'entrée de
    l'ingestion. Lève `SystemExit` avec un message clair si `AGORA_HASH_SALT`
    est absent ou trop court (< HASH_SALT_MIN_LEN caractères).
    """
    salt = HASH_SALT
    if not salt or len(salt) < HASH_SALT_MIN_LEN:
        problem = "absent" if not salt else f"trop court ({len(salt)} < {HASH_SALT_MIN_LEN} caractères)"
        raise SystemExit(
            f"ERREUR : sel d'anonymisation AGORA_HASH_SALT {problem}.\n"
            "  L'ingestion hashe les auteurs (RGPD) ; un sel absent ou faible rend "
            "`author_hash` ré-identifiable.\n"
            f"  Fournis un sel d'au moins {HASH_SALT_MIN_LEN} caractères, p. ex. :\n"
            "    export AGORA_HASH_SALT=\"$(python -c 'import secrets;print(secrets.token_hex(32))')\"\n"
            "  (les caches déjà construits ne sont PAS re-hashés ; seule une "
            "ré-ingestion exige le sel.)"
        )
    return salt

# Longueur max du libellé d'affichage (label).
LABEL_MAXLEN = 80
