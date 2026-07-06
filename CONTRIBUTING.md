# Contribuer à Agora

## Installation — une commande
```bash
git clone git@github.com:1Batmain/Analyse-des-consultations-citoyennes.git
cd Analyse-des-consultations-citoyennes
./scripts/setup.sh     # deps back+front · caches d'analyse (release) · secrets locaux
make dev               # backend :8010 + front :5180 → http://localhost:5180
```
Prérequis : **uv** et **node/npm**. La clé Mistral est **optionnelle** (le front et la
lecture des caches marchent sans ; elle ne sert qu'à *construire* de nouvelles analyses).

## Workflow (PR obligatoire)
**Branche d'intégration : `dev`** — tous les collaborateurs y poussent librement ;
`main` (protégée, review @1Batmain requise) ne reçoit que des PR depuis `dev`.

1. `git checkout dev && git pull` puis `git checkout -b feat/mon-sujet` (branche courte depuis dev)
2. Code — guidelines dans **`.agent/README.md`**, décisions passées dans `.agent/notes/`.
3. Vérifie : `make test && make build`
4. Pousse + merge dans **`dev`** (push direct ou petite PR d'équipe) ; régulièrement, une **PR `dev` → `main`** est soumise pour validation.
5. La **CI** (pytest + build) doit passer → un mainteneur merge → le serveur **se déploie seul**.

⚠️ Ne laisse pas d'état local non poussé : le déploiement fait `reset --hard origin/main`.

## Repères
- Architecture · dev/prod · conventions → **`.agent/README.md`**
- Contrat d'API front↔back → `frontend/src/redesign/contract.ts`
- Coût LLM d'une analyse → `GET /cost` (transparence des coûts)
