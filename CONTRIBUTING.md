# Contribuer à Agora

## Installation — une commande
```bash
git clone git@github.com:1Batmain/Agora.git
cd Agora
./scripts/setup.sh     # deps back+front · caches d'analyse (release) · secrets locaux
make dev               # backend :8010 + front :5180 → http://localhost:5180
```
Prérequis : **uv** et **node/npm**. La clé Mistral est **optionnelle** (le front et la
lecture des caches marchent sans ; elle ne sert qu'à *construire* de nouvelles analyses).

## Workflow (PR obligatoire — GitHub Flow)

`main` est la seule branche au long cours ; elle est **protégée** (review du mainteneur + CI verte + **signature du [CLA](CLA.md)** requises).

1. `git checkout main && git pull` puis `git checkout -b feat/mon-sujet` (branche courte)
2. Code — guidelines dans **`.agent/README.md`**, décisions passées dans `.agent/notes/`.
3. Vérifie : `make test && make build`
4. Pousse ta branche et ouvre une **Pull Request vers `main`**. Le bot te fera signer le CLA ; le mainteneur relit et merge.
5. La **CI** (pytest + build) doit passer → un mainteneur merge → le serveur **se déploie seul**.

⚠️ Ne laisse pas d'état local non poussé : le déploiement fait `reset --hard origin/main`.

## Repères
- Architecture · dev/prod · conventions → **`.agent/README.md`**
- Contrat d'API front↔back → `frontend/src/redesign/contract.ts`
- Coût LLM d'une analyse → `GET /cost` (transparence des coûts)


## Licence & CLA

Agora est sous **AGPL-3.0** (voir `LICENSE`). En contribuant, vous acceptez le
[CLA](CLA.md) : signez vos commits avec `git commit -s` (`Signed-off-by:`) — cela vaut
acceptation. Le CLA permet de financer l'édition civique **gratuite** d'Agora par une
édition commerciale ; vous restez propriétaire de votre contribution, votre nom reste
au générique.
