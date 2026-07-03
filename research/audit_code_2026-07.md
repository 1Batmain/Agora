# Audit de la BASE DE CODE — Agora (juillet 2026)

> Portée : qualité du **code** (pas du produit — voir `research/audit_capacites_2026-07.md`).
> Axes : (1) single source of truth · (2) séparation des responsabilités · (3) sécurité ·
> (4) efficacité · (5) robustesse/élégance. Chaque constat cite `fichier:ligne`, classé
> **CRITIQUE / MAJEUR / MINEUR**, avec un correctif concret. Aucun fichier de code modifié.
> Méthode : lecture directe du code (`backend/`, `pipeline/`, `frontend/src/`, `deploy/`)
> + trois sous-audits ciblés (SSOT, hot-paths serve, frontend), tous re-vérifiés.

---

## Résumé exécutif — Top 10 par gravité

| # | Gravité | Axe | Constat | Localisation |
|---|---|---|---|---|
| 1 | **CRITIQUE** | Efficacité | `/avis_list` **scanne les N avis et matérialise TOUS les matchs (texte entier + claims)** avant de paginer → ~15 Mo d'allocations + fold Unicode intégral **par requête** (granddebat = 22 174 avis) pour ne renvoyer que 15 items | `backend/avis.py:227-267` (via `server.py:588`) |
| 2 | **CRITIQUE** | Efficacité (front) | **three.js chargé statiquement** dans le bundle principal (vue Densité 3D importée en dur) → cause dominante du bundle ~725 kB, servi à **tout** visiteur même sans ouvrir la 3D | `frontend/src/redesign/RedesignApp.tsx:17` + `Density3D.tsx:2-3` |
| 3 | **CRITIQUE** | SSOT | **Deux implémentations de subdivision hiérarchique** (batch `pipeline.cluster.hierarchy` vs serve `backend.analysis._subdivide`) censées produire « la MÊME structure » **sans code ni constante partagés** ; la résolution `1.5` réconciliée (bug audit #6 : 3.0 vs 1.5) ne contraint QUE le batch → vecteur de re-divergence identique au bug corrigé | `pipeline/cluster/hierarchy.py:46` vs `backend/analysis.py:154,483,586` |
| 4 | **MAJEUR** | Sécurité + Efficacité | Les endpoints de **lecture publique** (`/analysis`, `/avis_list`, `/avis`, `/citations`, `/insights`, `/opinion`) n'ont **NI auth NI rate-limit**, alors qu'ils font un travail non-trivial par requête (parse JSON ≤1 Mo, fold-scan O(N) sur 16 Mo) → **DoS CPU/mémoire anonyme** avant exposition Internet. La dépendance `rate_limit` existe mais n'est appliquée qu'à `/submit`, `/recluster`, `/build`, `/flag` | `backend/server.py:415-594` ; `auth.py:85-97` |
| 5 | **MAJEUR** | SSOT / Robustesse | **`fastapi`/`uvicorn`/`pydantic`/`starlette` NON déclarés** dans `pyproject.toml` : injectés **non-pinnés** via `uv run --with fastapi --with uvicorn` → une version majeure FastAPI/pydantic peut casser la prod silencieusement, aucun lockfile pour la stack web runtime | `pyproject.toml` (absents) ; `deploy/serve.sh:11` |
| 6 | **MAJEUR** | Efficacité | `read_analysis` **non caché** : re-parse `analysis.json` (≤1 Mo) à **chaque** `/analysis` ET `/avis_list` ; pire, `/avis_list` le parse **même sans filtre thème** (cas par défaut) alors que `themes` n'y sert à rien | `backend/analysis_store.py:170-171` ; `server.py:583-584` |
| 7 | **MAJEUR** | SSOT | `resolution = 1.0` **codé en dur dans ~13 sites** ; `DEFAULT_RESOLUTION` existe mais n'est **importé nulle part**. `DEFAULT_SEED = 42` **redéclaré 3×** indépendamment | `pipeline/cluster/leiden_cluster.py:14` (jamais importé) ; sites listés en §1-M1/M2 |
| 8 | **MAJEUR** | Séparation | **Implémentation de clustering dupliquée** : `live_cluster` « **Reproduit le corps de** `analysis.build_theme_tree` » à la main → deux constructeurs d'arbre (console vs serve) à synchroniser manuellement | `backend/live_cluster.py:72` vs `backend/analysis.py:476` |
| 9 | **MAJEUR** | Efficacité (front) | **Fetchs redondants, aucun cache partagé** : `/analysis` fetché par 3 composants, `/flags` par 3, `/datasets` 2× ; + `segments()` **recalculé à chaque render** de chaque carte, `AvisCard` non-`memo` | `RedesignApp.tsx:164`, `ConsultationOverview.tsx:83`, `AvisExplorer.tsx:129` ; `AvisDetail.tsx:151` |
| 10 | **MAJEUR** | Robustesse / SSOT | `INSIGHTS_DIRNAME = "insights"` **déclaré 2×** vers des dossiers parents ET schémas de nom **différents** (baké `analysis/insights/<name>` vs live `insights/<hash>`) → deux caches distincts sous un nom commun trompeur | `backend/analysis_store.py:37` vs `backend/insights.py:29` |

**Quick-wins (< 1 h) recommandés avant la démo — voir §6.**

**Ce qui est solide (vérifié) :** fail-closed public re-lu et correct (`auth.py`) ; path-traversal gardé (whitelist `_resolve` + `_safe`) ; secrets propres (`var/` + `*.key` gitignorés, zéro clé en dur, exceptions sanitizées avant envoi client) ; extraction LLM **batchée + concurrente** (aucun N+1) ; build/serve **réellement séparés** (subprocess `Popen`, aucun module de build importé au top-level du serveur) ; caches mtime corrects pour les 2 plus gros JSON (`avis.json` 16 Mo, `claim_stance.json` 8 Mo) ; gate verbatim + masquage PII **couverts hors-cache** par `test_avis_pii.py` (tourne en CI) ; constantes k-NN (`derive_k`, `K_MAX`, `EDGE_SIGMA`) et pricing LLM à source unique.

---

## Axe 1 — Single source of truth

### CRITIQUE
**C1. Divergence de résolution de subdivision — le bug audit #6 peut resurgir.**
`pipeline/cluster/hierarchy.py:43-46` documente littéralement le précédent (« était 3.0 dans build, 1.5 dans le backend »). La constante réconciliée `DEFAULT_RESOLUTION_SUB = 1.5` (`hierarchy.py:46`) n'est importée QUE par `pipeline/cluster/build.py` (chemin batch/recherche). Le chemin **servi** — `backend/analysis.py._subdivide` (`:154`, `base_resolution: float = 1.0` en `:483,586`) — est une **hiérarchie ré-implémentée** qui ne référence jamais `DEFAULT_RESOLUTION_SUB`. Deux algos, promesse « MÊME structure », zéro partage.
*Fix :* soit le serve consomme `DEFAULT_RESOLUTION_SUB`, soit unifier les deux hiérarchies, soit supprimer la promesse « MÊME structure » et documenter la divergence volontaire.

### MAJEUR
**M1. `resolution = 1.0` en dur dans ~13 sites ; `DEFAULT_RESOLUTION` (`leiden_cluster.py:14`) importé nulle part.**
Sites : `analysis.py:483,586`, `live_cluster.py:67,208`, `build_analysis.py:127,306`, `build_opinion.py:370,537`, `citations.py:68`, `insights.py:271`, `claims_endpoint.py:332`, `server.py:412,630`, `pipeline/claims/pipeline.py:236,310`. *Fix :* importer `DEFAULT_RESOLUTION` partout (ou une `backend/config.py`).

**M2. `DEFAULT_SEED = 42` redéclaré 3×.** `leiden_cluster.py:13`, `pipeline/claims/pipeline.py:43`, `hdbscan_contender.py:31` (+ littéraux `42` dans `build_cache.py:69,135,245`, `build.py:358,537`, `hierarchy.py:144`, `synthetic.py:95`). *Fix :* une seule source importée.

**M3. `INSIGHTS_DIRNAME` déclaré 2× vers des chemins/schémas différents** (cf. Top-10 #10). *Fix :* `insights.py` importe `INSIGHTS_DIRNAME` d'`analysis_store` ; documenter baké vs live.

**M4. `CACHE_DIR` recalculé** dans `submissions.py:27` (copie de `recluster.py:33`) — délibéré (éviter torch) mais fragile. *Fix :* extraire un `backend/paths.py` léger (sans torch) importé par les deux.

**M5. Port backend `8010` en dur** dans `deploy/serve.sh:12` ET `frontend/vite.config.ts:13` (+ unités systemd). *Fix :* `AGORA_BACKEND_PORT` unique.

**M6. Port frontend `5180` couplé en dur** entre `deploy/serve.sh:15` et le défaut CORS `server.py:173-174` → changer l'un sans l'autre casse le CORS silencieusement. *Fix :* dériver l'origine du port front partagé.

**M7. `MODEL_ID` nomic en littéral report-only** (`recluster.py:40`, exposé par `/params`/`/health` `server.py:194`) ≠ modèle réellement utilisé (`pipeline/embed/registry.py:56` via l'alias `nomic-v2`). Changer l'embedder ferait **mentir** `/health`. *Fix :* résoudre l'ID depuis `pipeline.embed.registry`.

**M8. `contract.ts` mirroir MANUEL des dicts backend** (`SpatialTheme` ↔ `analysis.py:734-755` ; `ThemeOpinion` ↔ `build_opinion.py:256-265` ; enum `profil` ↔ `build_opinion.py:250-254`). Aligné aujourd'hui, divergence garantie à terme. *Fix :* a minima un **test de contrat** comparant les clés servies aux champs `contract.ts` (codegen pydantic→TS idéal). Le docstring de `consultation_schema.py:7` référence d'ailleurs un `frontend/src/types.ts` **qui n'existe pas** (le type vit dans `contract.ts`) — doc périmée.

### MINEUR
`"mistral-small-latest"` copié 4× comme défaut (`build_analysis.py:61`, `build_opinion.py:47`, `mistral_client.py:24`, `translate/translate.py:26` — knobs distincts, mais littéral dupliqué ; `translate.py:31` importe **bien** la constante = bon exemple) · `DEFAULT_EMBEDDER="nomic-v2"` (`pipeline/claims/pipeline.py:42`) non importé par `build_cache.py:129,238` qui hardcode `"nomic-v2"` · `"meta.json"` en dur `claims_endpoint.py:264` au lieu de `recluster.META_NAME` · docstring « 96×96 » `server.py:604` vs `density.py:29 GRID=96` · `AGORA_LLM_MAX_WORKERS` défaut `"4"` répété 3× (`build_analysis.py:67`, `build_opinion.py:58`, `extract.py:43`).

---

## Axe 2 — Séparation des responsabilités / modularité

### MAJEUR
**S1. Clustering dupliqué (Top-10 #8).** `live_cluster.recluster_payload` déclare « Reproduit le corps de `analysis.build_theme_tree` » (`live_cluster.py:72`). Deux constructeurs d'arbre parallèles (console live vs build servi) qui doivent rester alignés à la main — plus C1 (deux subdivisions). *Fix :* factoriser un cœur de construction d'arbre partagé (paramétré source-de-vecteurs), consommé par les deux.

### MINEUR
**S2. `backend/analysis.py` = quasi-god-module (814 lignes, 24 fonctions).** Mélange : subdivision variance-adaptative (`_subdivide`, `_coarsen_roots`), stats (`_dataset_stats`, `_gini`), couleurs/convergence (`_assign_colors`, `_assign_convergence`), nommage (`_name_nodes`), co-occurrence (`_cooccurrence`), sérialisation payload (`analysis_payload`, `theme_dict`). Cohérent (« construire l'arbre ») mais dense. *Fix :* extraire stats + colorisation dans des sous-modules `analysis_stats.py` / `analysis_style.py`.

**S3. Imports paresseux baladeurs** (`recut.py:160`, `build_opinion.py:281,516`, `server.py:264-265`, `keywords_fr.py:107`, `submissions.py:101`) — la plupart **justifiés et commentés** (éviter torch au boot). RAS, à garder documentés. **Bon signal :** `research/` n'est importé ni par `backend/` ni par `pipeline/` (direction de dépendance propre).

---

## Axe 3 — Sécurité

### MAJEUR
**SEC1. Lectures publiques sans rate-limit ni auth (Top-10 #4).** En mode public exposé, un anonyme peut marteler `/avis_list?q=...` (fold-scan O(N) sur 16 Mo, cf. C1-efficacité) et `/analysis` (parse ≤1 Mo) sans aucune borne → DoS CPU/mémoire. `rate_limit` (`auth.py:85`) n'est posé que sur les mutations/compute. *Fix :* ajouter `Depends(rate_limit)` sur tous les endpoints de lecture (fenêtre plus large que les mutations si besoin) ; garder nginx en défense de profondeur.

### MINEUR
**SEC2. `deploy/serve.sh:7` charge `MISTRAL_API_KEY` depuis `var/mistral.key` même en nœud PUBLIC** (`|| true`, tolérant) — contredit l'invariant « prod keyless ». Pas de fuite (la clé n'est jamais servie), mais un nœud public ne devrait pas monter la clé. *Fix :* ne pas exporter la clé quand `AGORA_PUBLIC=1`.
**SEC3. `submissions.jsonl` append-only sans cap total** (`submissions.py:74-87`) : borné par rate-limit + 5000 car/contrib, mais aucun plafond global → croissance disque non bornée sur une consultation ouverte spammée. *Fix :* cap de lignes / rotation, ou dédup.
**Vérifié OK :** fail-closed public (`auth.py:47-52`), whitelist `_resolve` (path-traversal, `server.py:100-105`), `_safe()` sur `theme_id`→chemin (`analysis_store.py:85-96`), body-size 64 Ko + headers sécurité + CORS restreint (jamais `*`), `hmac.compare_digest`, exceptions sanitizées (`_sanitize_progress`, messages génériques), `/docs` désactivés en public. Secrets : zéro clé en dur, `var/`+`*.key` gitignorés.

---

## Axe 4 — Efficacité

### CRITIQUE
**E1. `/avis_list` : scan O(N) + matérialisation de tous les matchs avant pagination (Top-10 #1).** `avis.py:228` boucle sur les 22 174 avis ; `:254-262` construit un dict complet (texte entier + claims) par match ; `matched[start:start+limit]` (`:264-266`) ne pagine qu'après. Avec `q`, `_fold(text)` (normalisation NFD + casefold, `:155-164,250`) recalculé sur **tous** les textes à chaque frappe. *Fix :* (a) filtrer d'abord vers des clés légères, compter `total`, puis ne construire les items lourds que pour la tranche `[offset:offset+limit]` ; (b) mémoïser par `(dataset, mtime)` le `_fold(text)` par avis et la map `avis_id→leaf_ids`.

### MAJEUR
**E2. `read_analysis` non caché (Top-10 #6)** — re-parse ≤1 Mo par `/analysis` ET `/avis_list`, alors que `read_avis_all` et `read_claim_stance` sont, eux, cachés par mtime (`analysis_store.py:181-201,226-246`). *Fix :* `_ANALYSIS_CACHE` mtime calqué ; pour `/analysis`, muter une **copie** (l'enrich serve-time écrit en place) ; pour `/avis_list`, ne lire que `themes` (lecture seule, cache direct).
**E3. `/avis_list` lit `analysis.json` même sans `theme_id`** (`server.py:583`) → parse ≤1 Mo gaspillé sur le cas par défaut. *Fix :* `themes = read_analysis(...)["themes"] if theme_id else []`.
**E4. `/recluster` re-`load_cache()` disque** (`live_cluster.py:218`) au lieu de réutiliser `_Dataset._LOADED` déjà en RAM, + double copie numpy float64 (`:84-87`). Atténué (gated `COMPUTE`, dev/console). *Fix :* passer l'objet `_Dataset` résolu.

### MINEUR
Petits artefacts re-parsés sans cache mtime : `read_insights`/`read_citations`/`read_opinion` (jusqu'à 119 Ko)/`read_status` (systématique, ~220 o) — `analysis_store.py:174-176,213-221` + `cost.py:66`. *Fix :* helper de cache mtime générique. — **Front :** `segments()` recalculé par render (`AvisDetail.tsx:151`), `AvisCard` non-`memo` + `toProvenance` literal par carte (`AvisExplorer.tsx:357,415`), `SpatialMap.computeLayout` 360 itérations O(n²) synchrones (correctement `useMemo`, peut janker). **Build-time : aucun N+1** (extraction batch 8 + `ThreadPoolExecutor` `extract.py:348` ; opinion/insights parallélisés).

---

## Axe 5 — Robustesse / élégance

### MAJEUR
**R1. Stack web runtime non-pinnée (Top-10 #5).** `fastapi/uvicorn/pydantic` hors `pyproject.toml`, tirés non-versionnés à chaque `uv run --with`. *Fix :* les déclarer + pinner (`fastapi>=0.11x,<0.12`, `pydantic>=2,<3`), lockfile.
**R2. Front : erreurs de build-state avalées.** `.catch(() => null)` sur `fetchAnalysis`/`fetchInsights` (`ConsultationOverview.tsx:83-85`, `AvisExplorer.tsx:129`) jette le `source: building|error` calculé par `analysisApi` → l'Overview affiche « indisponible » au lieu de « Analyse en cours… ». *Fix :* propager le `source`.
**R3. Code mort front expédié dans le bundle.** `SpatialMap` props `query`/`minConsensus`/`live` jamais passées (`RedesignApp.tsx:441`) → ~40 lignes de transitions d3 + dimming/filtrage morts (`SpatialMap.tsx:59-61,189-273`) ; `edges` calculées et passées mais jamais dessinées (`RedesignApp.tsx:344,446`). *Fix :* supprimer ou câbler.

### MINEUR
`isPct` classe un compte entier `0/1` comme pourcentage → « 100 % » erroné (`IndicesDashboard.tsx:50`). — `except Exception: pass` large en `build_cache.py:222` (enregistrement de coût — masque toute erreur d'I/O cost). — Types hors-contrat (`Flag`, `DensityPayload`, `ScatterPoint`, `Sourced`) + interface `Theme` morte (`contract.ts:298`). — **Tests :** couverture backend solide (auth, public_mode, ratelimit, whitelist, pii, invariants, recut, opinion) ; **mais `pipeline/` n'a AUCUN test unitaire** et la CI ne lance que `backend/tests` — le cœur algo (span/extract parsing, adaptive, hierarchy) n'est couvert qu'indirectement ; les tests d'invariants sur **vrais caches** (`test_avis_invariants`) **skippent en CI** (cache `analysis/` gitignoré, pas de clé Mistral) — mitigé par `test_avis_pii` (gate verbatim+PII hors-cache, tourne en CI). *Fix :* tests unitaires directs sur `pipeline/claims/span.as_claim` (cas d'ancrage non-verbatim/quasi-verbatim) et `parse_batch_claims` (JSON LLM malformé).

### Note d'hygiène
`git status` (agora-dev) montre des **caches binaires committés modifiés non commités** (`granddebat/embeddings.npy`, `ideas.jsonl`, `meta.json`, `tiktok/meta.json`) : le deploy fait `reset --hard origin/main` → ces changements locaux seraient perdus, et versionner des `.npy` qui dérivent est fragile. *Fix :* décider si les caches d'entrée sont sources de vérité versionnées ou régénérables ; ne pas laisser dériver silencieusement.

---

## 6. Quick-wins (< 1 h chacun) recommandés avant la démo de vendredi

1. **Lazy-load `Density3D` (three.js)** — `const Density3D = React.lazy(() => import('./Density3D'))` + `<Suspense>` autour du site de rendu (`RedesignApp.tsx:17,459`). Gain : la majorité des ~725 kB retirés du bundle initial → **page qui charge vite en démo**, three.js fetché seulement si on ouvre la 3D. Risque quasi nul.
2. **Cacher `read_analysis` par mtime + le sauter dans `/avis_list` sans `theme_id`** (`analysis_store.py` + `server.py:583`). Gain : supprime le re-parse ≤1 Mo sur les deux endpoints les plus sollicités (carte + explorateur) → **explorateur/carte plus réactifs**. Servir une **copie** côté `/analysis` (enrich mute en place).
3. **Poser `Depends(rate_limit)` sur les endpoints de lecture publics** (`/analysis`, `/avis_list`, `/avis`, `/citations`, `/insights`, `/opinion`). Gain : ferme le DoS CPU/mémoire anonyme **avant** l'exposition Internet de la démo ; la dépendance existe déjà, changement trivial et sûr.

*(Bonus si le temps le permet : `useMemo(segments(...))` + `React.memo(AvisCard)` pour un scroll d'explorateur sans à-coups pendant la démo — `AvisDetail.tsx:151`, `AvisExplorer.tsx:415`.)*
