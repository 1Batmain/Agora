# AGENTS.md

Un agent (ou un humain) démarre sur Agora ? **Le guide complet est dans
[`.agent/README.md`](.agent/README.md)** — architecture, flux dev/prod, conventions de
R&D, et la carte du dépôt.

Démarrage express :
```bash
./scripts/setup.sh   # installe tout (deps, caches, .env)
make dev             # backend :8010 + front :5180
```
Contribuer : branche → **PR vers `main`** (la CI doit passer) → auto-déploiement.
Détails dans [`CONTRIBUTING.md`](CONTRIBUTING.md).
