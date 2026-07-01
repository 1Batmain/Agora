#!/usr/bin/env bash
# Promotion des caches d'analyse DEV → PROD — le SEUL chemin par lequel prod reçoit des données.
# Prod ne construit jamais (mode public, aucune clé) → reste stable. À lancer depuis DEV après un
# rebuild validé. Usage : deploy/promote-cache.sh [dataset]  (sans argument = tous les caches).
set -euo pipefail
DEV=/home/bat/agora-dev
PROD=/home/bat/projects/Analyse-des-consultations-citoyennes
DS="${1:-}"
echo "[promote] $DEV → $PROD (dataset: ${DS:-TOUS})"
rsync -a --info=stats1 "$DEV/backend/cache/${DS:+$DS/}" "$PROD/backend/cache/${DS:+$DS/}"
XDG_RUNTIME_DIR=/run/user/1000 systemctl --user restart agora-backend
echo "[promote] ✓ caches promus + backend prod redémarré"
