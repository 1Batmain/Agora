#!/usr/bin/env bash
# Installe Agora en local, d'un coup : deps back+front, caches d'analyse, secrets. Idempotent.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
REPO="1Batmain/Analyse-des-consultations-citoyennes"
say() { printf '\033[1;34m▸ %s\033[0m\n' "$*"; }

say "Prérequis"
command -v uv  >/dev/null || { echo "  uv manquant : curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
command -v npm >/dev/null || { echo "  node/npm manquant (nvm : https://github.com/nvm-sh/nvm)"; exit 1; }

say "Dépendances backend (uv · extras contender/embed/faiss)"
uv sync --extra contender --extra embed-contender --extra faiss

say "Dépendances frontend (npm)"
( cd frontend && npm install )

say "Secrets locaux (var/, gitignoré)"
mkdir -p var
[ -f var/deploy.env ] || printf 'AGORA_API_TOKEN=%s\nAGORA_HASH_SALT=%s\n' \
  "$(openssl rand -hex 24)" "$(openssl rand -hex 32)" > var/deploy.env
[ -f var/mistral.key ] || echo "  (optionnel) clé Mistral pour CONSTRUIRE des analyses → echo 'ta-cle' > var/mistral.key"

say "Caches d'analyse (~250 Mo, release GitHub 'caches')"
if ls backend/cache/*/analysis/analysis.json >/dev/null 2>&1; then
  echo "  déjà présents — ok"
elif command -v gh >/dev/null; then
  gh release download caches -R "$REPO" -p 'agora-caches.tar.gz' -O /tmp/agora-caches.tar.gz --clobber \
    && tar xzf /tmp/agora-caches.tar.gz -C "$ROOT" && echo "  caches installés" \
    || echo "  (release 'caches' indispo — l'app marche, mais sans données tant que non construites)"
else
  echo "  gh absent → récupère l'asset 'agora-caches.tar.gz' de la release 'caches' puis : tar xzf agora-caches.tar.gz"
fi

say "✓ Prêt.  Lance :  make dev   (backend :8010 + front :5180 → http://localhost:5180)"
