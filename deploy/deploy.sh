#!/usr/bin/env bash
# Déploiement du VPS : met à jour main, rebuild le front, redémarre les services (mode PUBLIC).
# Appelé par le runner self-hosted (workflow Deploy) OU à la main. Idempotent.
set -euo pipefail
REPO=~/projects/Analyse-des-consultations-citoyennes
cd "$REPO"

echo "[deploy] fetch + reset main"
git fetch origin main
git reset --hard origin/main

echo "[deploy] build frontend"
( cd frontend && npm ci && npm run build )

echo "[deploy] restart services (systemd user)"
# Les services systemd (agora-backend, agora-frontend) portent le MODE PUBLIC + les secrets.
systemctl --user restart agora-backend agora-frontend || {
  echo "[deploy] systemd user indisponible — fallback nohup"
  bash deploy/serve.sh
}
echo "[deploy] OK — $(date)"
