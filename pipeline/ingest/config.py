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
# Sel d'anonymisation. Surchargeable via l'env pour que les hash soient stables
# entre exécutions mais non ré-identifiables sans le sel.
HASH_SALT = os.environ.get("AGORA_HASH_SALT", "agora-an-2026")
AUTHOR_HASH_LEN = 16  # caractères hex conservés du sha256

# Longueur max du libellé d'affichage (label).
LABEL_MAXLEN = 80
