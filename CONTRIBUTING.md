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
1. `git checkout -b feat/mon-sujet`
2. Code — guidelines dans **`.agent/README.md`**, décisions passées dans `.agent/notes/`.
3. Vérifie : `make test && make build`
4. Pousse + ouvre une **PR vers `main`**.
5. La **CI** (pytest + build) doit passer → un mainteneur merge → le VPS **se déploie seul**.

⚠️ Ne laisse pas d'état local non poussé : le déploiement fait `reset --hard origin/main`.

## Repères
- Architecture · dev/prod · conventions → **`.agent/README.md`**
- Contrat d'API front↔back → `frontend/src/redesign/contract.ts`
- Coût LLM d'une analyse → `GET /cost` (transparence des coûts)
