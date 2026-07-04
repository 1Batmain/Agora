#!/usr/bin/env bash
# Promotion des caches d'analyse DEV → PROD — le SEUL chemin par lequel prod reçoit des données.
# Prod ne construit jamais (mode public, aucune clé) → reste stable. À lancer depuis DEV après un
# rebuild validé. Usage : deploy/promote-cache.sh [dataset]  (sans argument = tous les caches).
set -euo pipefail
DEV=~/agora-dev
PROD=~/projects/Analyse-des-consultations-citoyennes
DS="${1:-}"
echo "[promote] $DEV → $PROD (dataset: ${DS:-TOUS})"
# Index DuckDB de LECTURE du hot path /avis_list : (re)baké ICI, sur DEV, juste avant le
# rsync — il part alors avec le reste de `analysis/` vers prod (analysis.duckdb est gitignoré,
# ce chemin de promotion est sa seule route). Best-effort : si le bake échoue, on n'empêche
# pas la promotion (le serve retombe gracieusement sur le scan Python). Signature de sources
# recalculée au bake ⇒ l'index promu est toujours cohérent avec l'avis.json/claim_stance promus.
( cd "$DEV" && AGORA_CLAIMS_BACKEND=api uv run --extra collect --extra serve \
    python -m backend.bake_duckdb "${DS:---all}" ) || echo "[promote] ⚠ bake DuckDB ignoré (fallback Python)"
rsync -a --info=stats1 "$DEV/backend/cache/${DS:+$DS/}" "$PROD/backend/cache/${DS:+$DS/}"
XDG_RUNTIME_DIR=/run/user/1000 systemctl --user restart agora-backend
echo "[promote] ✓ caches promus + backend prod redémarré"
