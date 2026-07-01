#!/usr/bin/env bash
# Déploiement du VPS : met à jour main, rebuild le front, redémarre les services (mode PUBLIC).
# Appelé par le runner self-hosted (workflow Deploy) OU à la main. Idempotent.
set -euo pipefail
REPO=~/projects/Analyse-des-consultations-citoyennes
cd "$REPO"
export PATH="~/.local/bin:~/.nvm/versions/node/v24.16.0/bin:$PATH"
export XDG_RUNTIME_DIR=/run/user/1000
export DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/1000/bus"

echo "[deploy] fetch + reset main"
git fetch origin main
git reset --hard origin/main

echo "[deploy] build frontend"
( cd frontend && npm ci && npm run build )

# Filet anti-doublon : tue tout uvicorn/vite MANUEL (nohup) qui tiendrait encore un port —
# sinon le service systemd ne peut pas reprendre :8010/:5180 et sert du code périmé (bug vu
# en prod). La SOURCE DE VÉRITÉ, ce sont les services systemd ci-dessous, rien d'autre.
pkill -f 'nohup.*uvicorn backend.server' 2>/dev/null || true
pkill -f 'vite preview' 2>/dev/null || true

echo "[deploy] restart services (systemd user)"
# Les services systemd (agora-backend, agora-frontend) portent le MODE PUBLIC + les secrets.
# PAS de fallback nohup : si systemd échoue, le deploy ÉCHOUE (visible) au lieu de lancer un
# process manuel concurrent qui masquerait le nouveau code.
systemctl --user restart agora-backend agora-frontend
sleep 8
systemctl --user is-active agora-backend agora-frontend
echo "[deploy] OK — $(date)"
