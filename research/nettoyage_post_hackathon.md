# Grand nettoyage post-hackathon — bilan (2026-07-04)

> Mandat (Bob, carte blanche) : nettoyage en profondeur post-hackathon — corriger
> duplications / erreurs / incohérences, par **commits atomiques** sur
> `chore/grand-nettoyage` (branché sur `main` à jour), **gate à chaque commit**
> (pytest CI + `npm run build`/`tsc -b`), puis merge dans `dev` **et** `main`.
> Sources : `research/audit_code_2026-07.md` (top-10) + `research/audit_capacites_2026-07.md`.
>
> **Interdits respectés** : aucun `backend/cache/` (dev/prod) ni `cached_data/` touché ;
> aucun build LLM lancé ; rien supprimé dans `research/` ni `hackathon-an-2026/` ;
> invariants intacts (verbatim, généricité zéro-hardcoding, fail-closed public).

---

## 1. Ce qui est FAIT (8 commits atomiques, tous gatés verts)

| Commit | Type | Résumé |
|---|---|---|
| `f634fad` | fix | **Clé de cache des titres déterministe** (bug connu du 4/07) |
| `15f9fc7` | ssot | **Seed de clustering** à source unique |
| `92a6fdb` | ssot | **Résolution Leiden 1er niveau** à source unique |
| `dcaaa99` | ssot | **INSIGHTS_DIRNAME** unifié + baké vs live documenté |
| `4e4cf86` | deps | **Stack web SERVE épinglée** (pyproject + uv.lock) |
| `3513dd7` | dedup | **Cœur de forêt partagé** analysis ↔ live_cluster + doc contrat |
| `54ca760` | front | **Deux exports morts retirés** (AvisFlag, Theme) |
| `d8d958c` | agent | **Post-mortem** sur le PLAN DE NUIT (queue périmée) |

### 1.1 Bug connu — clé de cache des titres instable (`fix titles`)
`backend/titles.py::_content_key` hashait `"|".join(anchors)` dans l'**ordre incident**
de sélection des claims d'ancrage. Or c'est l'**ensemble** des ancres qui définit le titre,
pas leur ordre — et cet ordre varie entre builds (re-ranking « développement » des
`representative_claims`, régime plein-ancrage vs repli). Une simple permutation flippait la
clé → cache MISS → **régénération en avalanche** (incident du 4/07 : 257/281 thèmes tiktok
retombés en mots-clés au rebuild parallèle, aggravé par le 429 transitoire déjà corrigé en
amont via `cache_fallback=False`).
**Correctif** : `sorted(anchors)` **dans la clé** (le prompt garde l'ordre de sélection).
**Test de régression** : `test_content_key_invariant_to_anchor_order` — la clé est
invariante par permutation. ⚠️ *Effet unique attendu* : la clé change une fois pour tous
les titres → une re-génération ciblée au **prochain** build (idempotente ensuite). Non
déclenchée ici (aucun build LLM).

### 1.2 SSOT — trois sources uniques rétablies
- **Seed de clustering** : `DEFAULT_SEED=42` était redéclaré 3× (`leiden_cluster`,
  `claims/pipeline`, `hdbscan_contender`) + littéraux dans `hierarchy`/`build`. Tout pointe
  désormais sur `pipeline.cluster.leiden_cluster.DEFAULT_SEED`. **Non conflaté** avec les
  seeds d'**échantillonnage** (`build_cache`, `ingest.synthetic`) qui sont un concept
  distinct et gardent leur propre défaut — regrouper par coïncidence de valeur (42) aurait
  été une fausse SSOT.
- **Résolution Leiden 1er niveau** : `resolution=1.0` codé en dur dans ~17 sites
  (signatures, `Field` pydantic, argparse) ; `DEFAULT_RESOLUTION` n'était importé nulle
  part. Tout l'importe maintenant (ré-exporté par `claims.pipeline` puis `backend.analysis`,
  même chemin que le seed ; `server.py` l'importe du module léger sans torch au boot).
  Commentaire à la source rappelant de **ne pas confondre** avec
  `hierarchy.DEFAULT_RESOLUTION_SUB=1.5` (subdivision intra-thème). **Ferme le vecteur de
  re-divergence** de l'audit (C1/M1) — comportement inchangé (1.0 partout).
- **INSIGHTS_DIRNAME** : littéral `"insights"` déclaré 2×. `insights.py` l'importe
  d'`analysis_store`. Surtout : **documenté aux deux sites** que ce ne sont PAS des doublons
  mais deux étages distincts — **baké** (`<dataset>/analysis/insights/<nom>.json`, nom
  sémantique) vs **live** (`<dataset>/insights/<hash>.json`, repli caché par hash).

### 1.3 Dépendances — stack web épinglée (`deps`)
`fastapi`/`uvicorn`/`starlette`/`pydantic` étaient tirés **non-versionnés** via
`uv run --with fastapi --with uvicorn` (aucune borne, hors lockfile) : une montée majeure
pouvait casser la prod en silence (audit R1/#5). Désormais **extra `serve`** borné
(`fastapi>=0.115,<1.0 · uvicorn>=0.30,<1.0 · starlette>=0.37,<2.0 · pydantic>=2,<3`) et figé
dans `uv.lock`. Invocations basculées `--with fastapi --with uvicorn` → `--extra serve`
(Makefile, `deploy/serve.sh`, CI, READMEs). L'extra respecte la séparation build/serve
(le cœur pipeline n'importe jamais fastapi).
**Effet de bord assumé** : `uv lock` a **purgé des paquets CUDA/nvidia/triton fantômes** que
l'ancien lock traînait alors que torch est épinglé sur l'index **CPU-only** (`pytorch-cpu`).
Le lock est maintenant **cohérent** avec `pyproject` — vérifié `--locked` : `torch 2.x+cpu`,
`cuda.is_available()=False`. (Le gros diff `uv.lock` est surtout du reformat uv + ces
ajouts/retraits.)

### 1.4 Dé-duplication — cœur de construction d'arbre partagé (`dedup`)
`live_cluster.build_live_tree` **recopiait à la main** le corps de
`analysis.build_theme_tree` (coarsening racine + `tau` dérivé + sous-arbres
variance-adaptatifs + nommage/couleurs/convergence) — deux orchestrations à synchroniser à
la main (audit S1). Le bloc **identique** est extrait dans `analysis._build_macro_forest`,
appelé par les deux. La **seule divergence** (source du graphe RACINE : seuil dérivé pour
`/analysis` vs seuil **donné** pour la Console) reste **en amont**, côté appelant.
Comportement inchangé (bloc déplacé tel quel, couvert par `test_recluster` + tests d'arbre).
Corrigé au passage : la doc de `consultation_schema.py` pointait vers
`frontend/src/types.ts` **inexistant** → c'est `frontend/src/redesign/contract.ts`
(interface `Consultation`).

### 1.5 Code mort & hygiène
- Front : `AvisFlag` (alias `@deprecated` de `Flag`, référencé nulle part) et l'interface
  `Theme` (forme legacy supplantée par `SpatialTheme`, jamais importée) retirées ; mention
  nettoyée dans la doc du contrat. `tsc -b` vert.
- `.agent/queue/night-plan.md` (ledger d'intention périmé) : bandeau **post-mortem** 1 ligne.

---

## 2. Vérification (gate à chaque commit)

- **pytest** (commande CI exacte : `backend/tests` + `pipeline/{collect,ingest_full,cluster}/tests`,
  extras `contender embed-contender faiss collect serve`) : **207 passed, 2 skipped**
  (les 2 skips = analyse `granddebat` non précalculée en local, cache `analysis/` absent —
  comportement normal, jamais un échec). Départ : 206 → +1 (nouveau test de régression titres).
- **`npm run build`** = `tsc -b && vite build` : **vert** à chaque commit touchant le front.
- **`uv run --locked …`** : le lock résout sans re-résolution → cohérence confirmée.

---

## 3. Périmètre — ce qui a été DÉLIBÉRÉMENT laissé (avec raison)

Findings de l'audit **hors scope** de ce nettoyage (ou déjà résolus), non touchés :

- **Audit R3 (SpatialMap : props/branches mortes)** : **déjà résolu** en amont — le
  composant `SpatialMap.tsx` n'existe plus dans `frontend/src/`. Rien à faire.
- **Three.js / bundle 725 kB (audit #2)** : **déjà corrigé** — le bundle servi est ~239 kB
  (lazy-load effectif). Rien à faire.
- **Types du contrat non importés directement** (`AnalysisPayload`, `ThemeOpinion`…) :
  **conservés volontairement**. `contract.ts` est le **miroir documentaire** des payloads
  backend (rôle du fichier) ; plusieurs sont consommés via `any`/inline. Les retirer
  abîmerait le contrat, pour un gain nul (types = zéro runtime).
- **`InsightsPanel` `@deprecated`** : **conservé** — déprécié mais **toujours utilisé**
  (lié à la vue Graphe encore vivante). Ce n'est pas du code mort.
- **`research/run_stance_validation.log`** (seul fichier « temp » tracké) : dans `research/`,
  **interdit de suppression**. Laissé.
- **`cached_data/lutte-contre-les-fausses-informations/`** : donnée **intentionnelle**
  (commit `feat: add lutte désinformation data`), pas un temp ; et cache → interdit. Laissé.
- **Audit #1/#4/#6 (efficacité/sécurité serve : `/avis_list` O(N), rate-limit lectures,
  cache `read_analysis`)** : ce sont des **optimisations de perf/durcissement**, pas des
  duplications/incohérences — hors mandat « nettoyage ». À traiter comme lot séparé
  (voir §4). Note : `MODEL_ID`/ports en dur (M5–M7) restent, refactor à part.

---

## 4. Chantiers à FROID restants — décisions PRODUIT (à ne pas trancher seul)

Ces trois chantiers dépassent le nettoyage : ce sont des **arbitrages produit/données**
qui engagent la validation et le récit. À décider **avec Bob**, verdict écrit avant action.

1. **Re-bake `granddebat` nouvelle génération.** Le cache servi est un échantillon
   (3 000/28 384 ≈ 11 %) sur pipeline claims+cible. Passer au corpus complet via l'API
   **batch Mistral** (extraction v3 batchée déjà validée non-dégradante, −50 % de coût) est
   la **preuve d'échelle** la moins chère (cf. `audit_capacites` B.1). Décisions ouvertes :
   quel corpus cible (3 k vs 28 k), coût/temps réel à publier, et **sensibilité à
   l'échantillon** (2 tirages, recouvrement des thèmes) jamais mesurée. **Coûte des appels
   LLM → interdit ici.**

2. **Calibration `stance-large`.** La stance servie tourne sur le modèle courant ; la cible
   de clivage v2 est **auto-invalidée** (panel aveugle 6-6, `cleavage_fit` = proxy faible
   0.58). Décider s'il faut (a) re-bencher une stance « large » contre le gold x-stance,
   (b) afficher une bande de fiabilité, (c) masquer les % sous n minimal. **Arbitrage
   validation + UI**, pas du nettoyage.

3. **Consolidation `granddebat` / `granddebat-complet`.** Deux workspaces divergent (cf.
   mémoire projet : le cache **dev** est un espace démo 22k, le **commité** est un 3k prod).
   Il faut **décider la source de vérité** (lequel sert, lequel est régénérable) et si les
   caches d'entrée `.npy`/`ideas.jsonl` doivent rester **versionnés** ou passer en
   DVC/LFS (audit — note d'hygiène : versionner des `.npy` qui dérivent est fragile).
   **Touche des caches → interdit ici ; décision produit.**

**Note d'hygiène liée** : `git status` (dev) montre des caches binaires committés modifiés
non commités (`granddebat/embeddings.npy`, `ideas.jsonl`, `meta.json`, `tiktok/meta.json`).
Ils ont été **parqués via `git stash`** au début du nettoyage (jamais commités, conformément
à la mémoire projet). À arbitrer dans le chantier #3.

---

## 5. Invariants — préservés (revérifiés)

- **Verbatim & traçabilité** : aucun toucher au gate `is_verbatim` ni au masquage PII
  (`test_avis_pii` passe). Refactors = déplacement de code identique + SSOT (valeurs
  inchangées).
- **Généricité (zéro hardcoding)** : les SSOT remplacent des littéraux par des constantes
  dérivées, ne réintroduisent aucun nom de corpus. Contrat `cross-lane` respecté.
- **Fail-closed public** : `server.py` inchangé côté auth/public (seuls deux défauts
  `Field(1.0)` → `Field(DEFAULT_RESOLUTION)`, valeur identique). `test_public_mode`/
  `test_hardening`/`test_ratelimit` passent.

---

*Branche : `chore/grand-nettoyage` (8 commits, depuis `origin/main`). Merge prévu dans
`dev` ET `main` (push direct admin, autorisation explicite de Bob).*
