# Architecture Dev / Prod (concepts projet)

Deux rôles, séparation stricte (ordre + sécurité + budget). Ce document décrit les
**concepts projet** ; les specs machine (chemins, services, exposition réseau) vivent
hors du dépôt.

## PROD — le serveur servi
- Repo **servi**, mis à jour par l'automatisation de déploiement. Personne n'y code directement.
- **Aucune clé Mistral** : mode public = sert le cache, **zéro appel LLM au runtime** → aucune
  clé sur la machine publique.
- Backend en **mode public fail-closed** (`AGORA_PUBLIC=1`) + frontend (build statique servi).
- Mis à jour UNIQUEMENT par le workflow **Deploy** (push `main` → pull + build front + restart).
  La procédure de déploiement est tenue hors-repo. Les caches (untracked) survivent au reset.

## DEV — le clone de travail
- **A la clé Mistral** (budget cappé) → c'est ici qu'on **construit**.
- On y code et on y (re)construit les caches d'analyse (extraction / clustering / enrichment / opinion).

## Les caches d'analyse
`backend/cache/<dataset>/` : claims, embeddings, arbre de thèmes, enrichissement LLM, opinion/stance.
Volumineux et **gitignorés** (dérivés) — seuls ideas/embeddings/meta sont dans git. Prod ne les
construit jamais.

## Flux de travail
1. **Code (front/back)** : dev → commit → push → **PR** → merge `main` → la prod **se met à jour
   automatiquement**.
2. **Rebuild de données** (nouveau pipeline / dataset) : construire en **DEV** (clé dev) → valider →
   **promotion de cache** (sync dev→prod + restart). La promotion est le SEUL chemin par lequel la
   prod reçoit des données (elle ne construit jamais).

> ⚠️ Ne laisse pas d'état local non poussé : le déploiement fait `reset --hard origin/main`.

## ⚠️ Gotcha : `avis.json` et `claim_stance.json` doivent venir du MÊME build
Les claims sont identifiés `f"{avis_id}#{index_global}"` où l'index global vient de
`tree.prepared`. `build_analysis` écrit `avis.json` (+ l'arbre), `build_opinion` écrit
`claim_stance.json` (+ `opinion.json`). Si on re-bake **l'un sans l'autre**, les index de
claims divergent → les ids ne matchent plus (stance par avis + filtre `/avis_list?stance=`
cassés, ~1% de recouvrement). **Toujours re-lancer `build_analysis` PUIS `build_opinion`
ensemble** pour un dataset, et **promouvoir le dossier `analysis/` entier** (jamais un seul
fichier). Vérif rapide : `intersection(ids avis.json, clés claim_stance)` doit être ≈ la
part de claims dans les feuilles non-`impur` (≈80-100%).
