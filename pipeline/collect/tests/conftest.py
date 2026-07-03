"""Conftest léger du collecteur — AUCUN import lourd (pas de fastapi/torch).

Les tests se skippent proprement si l'extra `collect` (duckdb) n'est pas installé,
convention du dépôt (jamais d'échec parasite).
"""
import sys
from pathlib import Path

import pytest

# `pipeline` est un namespace package (pas de __init__.py à sa racine) : pytest
# n'insère donc pas la racine du dépôt dans sys.path comme pour `backend/tests`.
# On l'ajoute explicitement, sans effet de bord hors tests.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

duckdb = pytest.importorskip("duckdb")
