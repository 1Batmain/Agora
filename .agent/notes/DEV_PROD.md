# Architecture Dev / Prod

Deux checkouts, deux rôles — séparation stricte (ordre + sécurité + budget).

## PROD — `/home/bat/projects/Analyse-des-consultations-citoyennes`
- Repo **servi**, possédé par le **runner GitHub Actions**. Personne n'y code directement.
- **Aucune clé Mistral** (mode public = sert le cache, zéro appel LLM au runtime → aucune clé sur la machine publique).
- Services systemd : `agora-backend` (mode public fail-closed, :8010) + `agora-frontend` (build servi, :5180).
- Exposé via Tailscale Funnel : **https://forge.tail0b8aa8.ts.net**
- Mis à jour UNIQUEMENT par le workflow **Deploy** (push `main` → `deploy/deploy.sh` : `reset --hard` + build front + restart). Les caches (untracked) survivent au reset.

## DEV — `/home/bat/agora-dev`
- Clone de travail. **A la clé Mistral** (`agora-dev`, budget cappé) → c'est ici qu'on **construit**.
- On y code et on y (re)construit les caches d'analyse (extraction / clustering / enrichment / opinion).

## Les caches d'analyse
`backend/cache/<dataset>/` : claims, embeddings, arbre de thèmes, enrichissement LLM, opinion/stance.
~258 Mo, **gitignorés** (dérivés + volumineux) — seuls ideas/embeddings/meta sont dans git. Prod ne les construit jamais.

## Flux de travail
1. **Code (front/back)** : dev → commit → push → **PR** → merge `main` → le runner **déploie sur prod automatiquement**.
2. **Rebuild de données** (nouveau pipeline / dataset) : construire en **DEV** (clé dev) → valider → `deploy/promote-cache.sh [dataset]` (rsync dev→prod + restart prod).

## ⚠️ Gotcha : `avis.json` et `claim_stance.json` doivent venir du MÊME build
Les claims sont identifiés `f"{avis_id}#{index_global}"` où l'index global vient de
`tree.prepared`. `build_analysis` écrit `avis.json` (+ l'arbre), `build_opinion` écrit
`claim_stance.json` (+ `opinion.json`). Si on re-bake **l'un sans l'autre**, les index de
claims divergent → les ids ne matchent plus (stance par avis + filtre `/avis_list?stance=`
cassés, ~1% de recouvrement). **Toujours re-lancer `build_analysis` PUIS `build_opinion`
ensemble** pour un dataset, et **promouvoir le dossier `analysis/` entier** (jamais un seul
fichier). Vérif rapide : `intersection(ids avis.json, clés claim_stance)` doit être ≈ la
part de claims dans les feuilles non-`impur` (≈80-100%).
