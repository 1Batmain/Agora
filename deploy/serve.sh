#!/usr/bin/env bash
# Fallback de service (sans systemd) : backend en MODE PUBLIC + front (build servi). nohup.
set -uo pipefail
REPO=~/projects/Analyse-des-consultations-citoyennes
cd "$REPO"
: "${AGORA_API_TOKEN:?AGORA_API_TOKEN requis}"; : "${AGORA_HASH_SALT:?AGORA_HASH_SALT requis}"
export MISTRAL_API_KEY="$(cat var/mistral.key 2>/dev/null || true)"
# backend en mode PUBLIC fail-closed (auto-build off, endpoints coûteux/mutants protégés)
pkill -f 'uvicorn backend.server:app' 2>/dev/null || true; sleep 2
AGORA_PUBLIC=1 AGORA_AUTOBUILD=0 AGORA_CLAIMS_BACKEND=api \
  nohup uv run --extra contender --extra embed-contender --extra faiss --extra serve \
  uvicorn backend.server:app --host 0.0.0.0 --port 8010 > /tmp/agora_backend.log 2>&1 &
# front : build statique servi (vite preview, stable)
pkill -f 'vite preview' 2>/dev/null || true; sleep 1
( cd frontend && nohup npm run preview -- --host 0.0.0.0 --port 5180 --strictPort > /tmp/agora_front.log 2>&1 & )
echo "[serve] backend (public) + front (preview) lancés"
