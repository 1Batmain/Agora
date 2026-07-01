# Agora — instructions projet (Claude Code)

Le guide agent complet est dans **`.agent/README.md`** (architecture, dev/prod, conventions,
carte du dépôt). Lis-le en début de session, ainsi que `.agent/notes/` pour le domaine touché.

Rappels critiques :
- **Dev/prod séparés** (`.agent/notes/DEV_PROD.md`) : coder en dev (a la clé Mistral) ; la
  **prod est possédée par le runner, sans clé**, sert le cache.
- **Flux** : branche → **PR vers `main`** → CI (pytest + build) verte → auto-deploy du VPS.
  Jamais d'état local non poussé (le deploy fait `reset --hard origin/main`).
- **Secrets** dans `var/` (jamais commités). Public = `AGORA_PUBLIC=1` fail-closed.
- **R&D pilotée par verdict** : mesurer avant d'adopter, écrire le verdict (OUI/NON + pourquoi).
- **Généricité** : zéro nom de corpus en dur.
