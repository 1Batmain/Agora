"""Chemins, URLs des sources et constantes partagées par la lane data."""
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

# ---------------------------------------------------------------------------
# Sources autoritatives
# ---------------------------------------------------------------------------
# x-stance (ZurichNLP) : questions politiques + commentaires labellisés FAVOR/AGAINST.
XSTANCE_URL = (
    "https://github.com/ZurichNLP/xstance/raw/master/data/xstance-data-v1.0.zip"
)
XSTANCE_ZIP = RAW_DIR / "xstance-data-v1.0.zip"
# Fichiers du zip contenant les commentaires (questions.*.jsonl = libellés des questions).
XSTANCE_COMMENT_FILES = ("train.jsonl", "valid.jsonl", "test.jsonl")

# Consultation citoyenne TikTok (open data Assemblée nationale, ~33 609 réponses).
# CSV LimeSurvey encodé cp1252, séparateur ';'.
TIKTOK_URL = (
    "https://data.assemblee-nationale.fr/static/openData/repository/"
    "CONSULTATIONS_CITOYENNES/TIKTOK/tiktok_appel_a_temoignages.csv"
)
TIKTOK_CSV = RAW_DIR / "tiktok_appel_a_temoignages.csv"
TIKTOK_ENCODING = "cp1252"

# Colonnes du CSV TikTok exploitées (index 0-based, schéma figé au 2025).
TIKTOK_ID_COL = 0  # "ID de la réponse" -> base de l'author_hash
TIKTOK_TS_COL = 1  # "Date de soumission"
# Question ouverte la plus riche en texte libre (témoignages mal-être / harcèlement).
TIKTOK_TEXT_COL = 141  # "Souhaitez-vous décrire ce sentiment de mal-être ... ?"

# ---------------------------------------------------------------------------
# Réglages
# ---------------------------------------------------------------------------
# Sel d'anonymisation. Surchargeable via l'env pour que les hash soient stables
# entre exécutions mais non ré-identifiables sans le sel.
HASH_SALT = os.environ.get("AGORA_HASH_SALT", "agora-an-2026")
AUTHOR_HASH_LEN = 16  # caractères hex conservés du sha256

# Longueur max du libellé d'affichage (label).
LABEL_MAXLEN = 80

# x-stance est multilingue (de/fr/it) ; la démo est FR -> on ne garde que le FR
# par défaut. Mettre AGORA_XSTANCE_ALL_LANGS=1 pour tout conserver.
XSTANCE_FR_ONLY = os.environ.get("AGORA_XSTANCE_ALL_LANGS", "0") != "1"
