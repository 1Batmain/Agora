"""Réglages du collecteur : chemins, politesse réseau, seuils de l'heuristique.

Tous les tunables vivent ici (aucun nom de consultation — la liste vient du
scraping de l'index).
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Point d'entrée unique du portail (une page d'index, pas une liste de corpus).
INDEX_URL = "https://data.assemblee-nationale.fr/autres/consultations-citoyennes"

COLLECT_DIR = REPO_ROOT / "data" / "collect"
RAW_DIR = COLLECT_DIR / "raw"
DB_PATH = COLLECT_DIR / "consultations.duckdb"

# Réseau : on est un bon citoyen du portail open data.
USER_AGENT = "agora-an-2026/collect (open-data)"
REQUEST_DELAY_S = 1.0
TIMEOUT_S = 60
# Cap de téléchargement — garde-fou contre les fichiers pathologiques, sans
# nommer aucune consultation. 400 MiB laisse passer les plus gros CSV légitimes
# observés (~253 Mo) tout en bornant le pire cas.
MAX_DOWNLOAD_BYTES = 400 * 2**20

# Heuristique de classification des colonnes (voir classify.py).
DATE_SHARE_MIN = 0.90       # part de valeurs "date" pour kind=date
NUMERIC_SHARE_MIN = 0.95    # part de valeurs numériques pour kind=numeric
OPEN_MIN_ANSWERS = 10       # en deçà, pas assez de signal pour "texte libre"
OPEN_AVG_LEN_STRONG = 60    # longueur moyenne suffisante (avec plancher de diversité)
OPEN_AVG_LEN_WEAK = 25      # longueur moyenne + forte diversité requise
OPEN_DISTINCT_RATIO_MIN = 0.5
# Plancher de diversité de la règle forte : écarte les libellés longs RÉPÉTÉS
# (colonne "Question" des exports agrégés) qui ne sont pas du texte libre.
OPEN_DISTINCT_RATIO_FLOOR = 0.05
# Règle "payload dupliqué" : réponses courtes très dupliquées mais vraies
# contributions longues présentes (colonne "Contribution" des exports agrégés).
OPEN_MAX_LEN_LONG = 500     # au moins une valeur de cette longueur
OPEN_DISTINCT_ABS_MIN = 100  # et une diversité absolue suffisante
DISTINCT_CAP = 50_000       # cap mémoire du comptage de valeurs distinctes

# Chargement DuckDB : taille des lots executemany.
INSERT_BATCH_SIZE = 10_000
