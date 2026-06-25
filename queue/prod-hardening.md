# 🛡️ LANE PROD-HARDENING — durcissement avant mise en production
> Issu de l'audit prod (2026-06). Verdict : NO-GO public en l'état ; GO interne/restreint après P1+P3.
> **Ordre : sécurité d'abord (P1, P3, P5), puis scaling (P2, P4).** À lancer après la consolidation.
> Garde-fous transverses : ne pas casser le verbatim-gate, /avis, le multilingue, le rendu par défaut ; zéro-hardcoding.

## P1 — SEC1 ✅ FAIT (34f8e07) : verrouiller l'exposition du backend
**Constat** : FastAPI sur `0.0.0.0:8010` sans auth, CORS `allow_origins=["*"]` (`server.py:88-93`) ; `POST /build`, `/sandbox`, `/flag`
déclenchent calcul lourd / écriture → DoS + abus de facture Mistral trivial.
- **Auth** : token/clé (env `AGORA_API_TOKEN`) requis au minimum sur les endpoints MUTATIFS/coûteux (`/build`, `/sandbox`, `/flag`,
  `/explain`). Lecture (`/analysis`,`/avis`,`/citations`,`/insights`) : derrière le proxy aussi, ou token léger.
- **CORS** : restreindre aux origines connues (env `AGORA_ALLOWED_ORIGINS`), pas de `*`.
- **Rate-limiting** : par IP/token sur les endpoints coûteux (slowapi ou middleware maison).
- **Bind** : `127.0.0.1` derrière un reverse-proxy (doc déploiement) plutôt que `0.0.0.0` exposé.
- **Acceptance** : appels mutatifs refusés (401) sans token ; CORS limité ; rate-limit testé ; doc proxy. Lecture inchangée pour le front (token injecté).

## P2 — S1+S2 (CRITIQUE, scaling) : découpler build/serve + borner la RAM
**Constat** : process unique ; tous datasets en RAM au boot (`server.py:62-69`) ; `_PREP_CACHE` (`sandbox.py:46`) + `_GRAPH_CACHE`
jamais évincés (float64 ×2) ; build en thread daemon DANS le serve (`build_manager.py:67`) ; mono-worker (GIL).
- **Lazy-load** des datasets (charger `_Dataset` à la 1re requête, pas au boot).
- **LRU borné** sur `_PREP_CACHE` et `_GRAPH_CACHE` (taille configurable) → éviction.
- **Vecteurs en float32 en mémoire** (S4 : `claims_endpoint.py:288`, `state.py:81` castent en float64 inutilement) → ½ RAM.
- **Découpler build et serve** : worker de jobs en **process séparé** (file type RQ/arq/Celery) ; serve en **multi-worker**
  (`gunicorn -k uvicorn.workers.UvicornWorker --workers N`) derrière proxy.
- **Acceptance** : RAM bornée (LRU vérifié) ; un build ne bloque plus le serve ; serve multi-worker ; vecteurs float32.

## P3 — SEC2 (ÉLEVÉ, RGPD) : sel d'anonymisation
**Constat** : `pipeline/ingest/config.py:33` `HASH_SALT` défaut committé `"agora-an-2026"` → `author_hash` réversible (force brute).
- **Pas de défaut** : exiger `AGORA_HASH_SALT` (≥32 octets aléatoires) ; **échec au démarrage de l'ingestion** si absent.
- **Acceptance** : ingestion refuse de tourner sans sel secret ; doc de génération du sel.

## P4 — S3 (ÉLEVÉ, scaling/coût) : industrialiser le coût LLM
**Constat** : extraction mistral-large (`build_analysis.py:54`) batchée 8/avis mais **boucle séquentielle** ; enrichissement
mistral-small **sérial par thème** (`build_analysis.py:129-180`). Backoff (`backend.py:42`) sérialise sous RPM bas.
- **Paralléliser** les lots d'extraction (asyncio + **sémaphore respectant le RPM**) et les boucles d'enrichissement (ThreadPool borné).
- **Tiering** : extraction sur modèle medium par défaut, large en option.
- **`pip-audit`** en CI + **épingler la révision** du modèle nomic (`trust_remote_code` = exécution de code distant au chargement).
- **Acceptance** : build sensiblement plus rapide à concurrence contrôlée ; pip-audit vert ; révision nomic épinglée.

## P5 — SEC3 (ÉLEVÉ, RGPD) : PII servie
**Constat** : `text` brut (PII non masquée) servi/persisté par `/avis` (`avis.py:90`, `build.py:66`) ; `strip_pii` regex-only
(rate noms/adresses/n° de dossier). L'extraction LLM utilise déjà `text_clean` (bon).
- **Servir/persister `text_clean`** (ou variante masquée) dans `/avis` et `avis.json` ; spans ré-ancrés sur le texte servi.
- **Renforcer `strip_pii`** (NER) si l'original doit être gardé. Documenter rétention + base légale RGPD.
- **Acceptance** : /avis ne renvoie plus de PII non masquée ; spans cohérents ; note RGPD.

## SECONDAIRE (prod-readiness)
- **Tests** : pytest configuré + tests de régression API (`server.py`, build, `analysis_store`) + sécurité (auth, CORS, traversal).
- **Observabilité** : remplacer les `print` par `logging` (niveaux), métriques basiques, `/health` détaillé.
- **Conteneurisation** : Dockerfile + compose (serve + worker + proxy) ; manifeste de déploiement.
- **Backups** : stratégie pour les caches LLM (un rebuild perdu = re-payer mistral-large).
- **Défense en profondeur** : durcir `recluster.dataset_dir` (rejeter `..`/`/`, ré-ancrer sous CACHE_DIR) même si `_resolve` mitige déjà.
