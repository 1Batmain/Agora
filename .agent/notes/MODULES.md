# MODULES — carte des frontières (ségmentation 5 personnes)

> Question de Bob : les couches du pipeline sont-elles assez découplées pour qu'une
> personne possède un module sans casser les autres ? **Réponse courte : OUI pour le
> CODE, avec 3 contrats transverses à geler et 2 fichiers « bibliothèque partagée » à
> gardienner.** Détail ci-dessous, module par module, puis verdict + règles.

Les **frontières = fichiers de cache** (`backend/cache/<dataset>/…`). Chaque flèche est un
schéma. Ce qui traverse plusieurs couches (la « colonne vertébrale ») est listé en §0.

```
data/raw + descriptors ─▶ ideas.jsonl ─▶ claims.json ─▶ claims_emb.npz
   [L1 ingest+claims]                          │
                                               ▼
                            analysis/analysis.json (arbre)   [L2 embed+cluster]
                                               │
                        ┌──────────────────────┴─────────────────────┐
                        ▼                                             ▼
        analysis/avis.json (+ recut)              analysis/opinion.json + claim_stance.json
              [L2/build_analysis]                          [L3 build_opinion]
                        └──────────────┬──────────────────────────────┘
                                       ▼
                          API :8010  (server, avis, analysis_store)  [L4]
                                       ▼
                          contract.ts ─▶ redesign/*  [L5 front]
```

---

## §0 — La COLONNE VERTÉBRALE (contrats transverses = à geler en cross-lane)

Trois choses traversent les couches et **ne se possèdent pas individuellement**. Toute
modif passe par accord cross-lane (le CODEOWNERS `@1Batmain-sur-tout` est le backstop).

1. **Constantes de déterminisme** — `pipeline/claims/pipeline.py:41-43`
   `DEFAULT_MODEL="ministral-3:latest"`, `DEFAULT_EMBEDDER="nomic-v2"`, `DEFAULT_SEED=42`
   + `EXTRACT_MODEL` (`backend/build_analysis.py:60`, défaut `mistral-large-latest`).
   Importées à l'identique par L2 (`analysis.py:35`) et L3 (`build_opinion.py:40-41`). Elles
   lient L1↔L2↔L3 : changer l'une ré-extrait/re-clusterise TOUT et décale les ids.

2. **L'identité claim `f"{avis_id}#{i}"`** où `i` = **index GLOBAL** du claim (position dans
   `prepared.claim_texts`, issue de l'aplatissement `_flatten`, `pipeline/claims/pipeline.py:248`
   — ordre = avis × claims dans `claims.json`). Émise 2× indépendamment :
   `avis.py:92` (`ci = enumerate(claim_owner)`) et `build_opinion.py:352` (`gi ∈ node.members`).
   Les deux ne matchent QUE si les deux builds ont tourné sur le **même `claims.json` + mêmes
   params**. C'est la clé de jointure de la stance côté API.

3. **`contract.ts`** (`frontend/src/redesign/contract.ts`) = miroir des shapes de réponse L4.
   Toute modif de shape API et son type front partent **dans la même PR**.

Deux fichiers sont des **bibliothèques partagées** (un seul propriétaire, le reste read-only) :
- `backend/analysis.py` — `build_theme_tree`/`ThemeTree` (possédé L2 ; consommé L3, L4).
- `backend/claims_endpoint.py` — `prepare_claims`/cache claims (possédé L1 ; consommé L2, L3).

---

## §1 — Couche 1 · Ingestion + identification des claims

**Périmètre** : `pipeline/ingest/*` (build, config, sources, normalize, anonymize),
`pipeline/claims/*` (pipeline, extract, backend, span), `backend/build_cache.py`,
`backend/claims_endpoint.py` (cache claims + `prepare_claims`).

**Entrée** : `data/raw/` (gitignoré) piloté par un **descripteur déclaratif**
`pipeline/ingest/descriptors/<name>.json` (schéma `sources.py:10-40` : `name, format, path,
columns{id,text,…}, lang_keep, keep_where, props, status`). Zéro nom de corpus en dur.

**Sorties (CONTRATS)** :
- `ideas.jsonl` — `{id, type:"idea", label, props:{text, text_clean, ts, lang, author_hash,
  source, weight}}`, écrit `build.py:61-84`. **`id = f"{source}:{raw_id}"`** (`build.py:51`) —
  c'est l'`avis_id`.
- `claims.json` — `{"model": str, "claims": {avis_id: [{text, spans:[[s,e],…],
  target:[s,e]|null}]}}`. Schéma dict = `Claim.to_dict` (`span.py:77-83`) ; écrit
  `claims_endpoint.py:84-88`. **Clé de cache = MODÈLE** : `if rec.get("model") != model:
  return {}` (`claims_endpoint.py:76`) → modèle différent ⇒ ré-extraction.
- `claims_emb.npz` — `{vecs: float32 (n,768), fingerprint}` ; clé = `sha256(embedder + \x00 +
  join(claim_texts))` (`claims_endpoint.py:91-110`). Texte d'un claim change ⇒ fingerprint
  différent ⇒ ré-embed automatique.
- `meta.json` — `{id, status, model_id, dim, question, built_with{min_chars,seed,…}, …}`
  (`build_cache.py:183-204`). `question` **cadre la granularité d'extraction** (`claims_endpoint.py:260-267`).

**Librement modifiable** : descripteurs, patterns PII (si `strip_pii` appliqué partout),
knobs `min_chars`/`dedup`/`balance`/`cap`, choix backend LLM (`api`/`mac`/`auto`) — le format
de sortie est stable.

**CONTRAT GELÉ** : le dict claim `{text,spans,target}` ; le cache clé-par-modèle ;
`avis_id = source:raw_id` ; offsets `[start,end)` ancrés sur `text_clean` (PII masquée) ;
le sel d'anonymisation (`anonymize.py`) — le changer casse tous les `author_hash`.

**Risques croisés** : changer le prompt/modèle d'extraction ⇒ `claims.json` change ⇒ **l'ordre
d'aplatissement et donc l'index global changent** ⇒ tous les ids `#i` en aval bougent.
Se fait délibérément via `build_analysis --reextract` (`build_analysis.py:309`), **jamais à
moitié**. Tests : `backend/tests/test_avis_pii.py`, `test_avis_invariants.py`, `test_serve_metrics.py`.

---

## §2 — Couche 2 · Embedding + clustering + hiérarchie

**Périmètre** : `pipeline/embed/*` (embedder, registry), `pipeline/cluster/*` (knn, leiden,
hierarchy, adaptive, naming, io), `backend/analysis.py` (`build_theme_tree`, `ThemeTree`,
`ThemeNode`), `backend/recut.py`, `backend/build_analysis.py`.

**Entrée** : `claims.json` via `prepare_claims` (`claims_endpoint.py:233`) → `PreparedClaims`
(claim_texts / claim_owner / claim_vecs alignés, `claims_endpoint.py:181-206`).

**Sorties (CONTRATS)** :
- `analysis/analysis.json` — `{themes:[ThemeNode], edges:[{a,b,weight}], params:{seed,
  resolution, derived, recut, adaptive}, dataset_stats, backend_used}`. `ThemeNode` sérialisé
  `analysis.py:731-756` (`id, parent_id, level, label, title, keywords, n_avis, n_claims,
  color, has_children, representative_claims, cohesion/consensus/dispersion…`). Écrit via
  `store.write_analysis` (`build_analysis.py:228`).
- **L'arbre `ThemeTree`** (`analysis.py:79-100`) : `.nodes`, `.order` (préfixe), `.macros`,
  `.prepared`. Construit par `build_theme_tree` (`analysis.py:476-567`), **déterministe** pour
  `(dataset, extract_model, embedder, seed=42, resolution=1.0)`. Ids de nœuds = compteur
  séquentiel en parcours préfixe (`analysis.py:295-327`) ; **préservés par le recut**
  (`build_analysis.py:170` « Ids de nœuds inchangés »).

**Params** : `seed=DEFAULT_SEED=42` (`analysis.py:484`), `resolution=1.0` (`analysis.py:483`),
défauts adaptatifs **dérivés des données** (`adaptive.py:40-47` : `K_LOG_COEF`, `EDGE_SIGMA`,
`MIN_SUB_FRAC`, `DUP_PERCENTILE`) via `derive_defaults` (déterministe, pas d'aléa).

**Librement modifiable** : labels, keywords, couleurs (palette), `representative_claims`,
titres/hooks/descriptions (LLM, cachés à part sous `analysis/titles|hooks|descriptions/`).

**CONTRAT GELÉ** : `seed=42` ; **l'ordre d'attribution des ids de nœuds** et **l'index global
de claim** ; le schéma `ThemeNode` (ajouter un champ optionnel OK, retirer/renommer casse
citations/insights/avis/opinion qui référencent par id) ; dtype `float32` / dim 768 des embeddings.

**Risques croisés** : changer `seed`/`resolution`/une constante `adaptive.py` ⇒ arbre différent
⇒ ids de nœuds décalés ⇒ **tout l'aval keyé par id casse**. Il faut alors tout rebuild.
Tests : `backend/tests/test_recluster.py` (forme + monotonie du seuil), `test_integration.py`
(chaîne `/analysis`→insights/citations/avis/opinion), `hierarchy.check_integrity` (invariants).

---

## §3 — Couche 3 · Stance / opinion par cluster

**Périmètre** : `backend/build_opinion.py` (+ `build_theme_tree` importé de L2, `title_for_node`
de `titles.py`, écritures via `analysis_store`).

**Entrée** : **RE-DÉRIVE l'arbre** — `build_theme_tree(ds, model=extract_model,
embedder=embedder, resolution=resolution, seed=seed)` (`build_opinion.py:394`). Réutilise le
`claims.json`/`claims_emb.npz` déjà cachés **si et seulement si** `extract_model` et `seed`
matchent ceux de `build_analysis` (`build_opinion.py:380-383`, défaut commun `EXTRACT_MODEL`
importé de `build_analysis.py:41`). N'écrit JAMAIS l'arbre / `avis.json` (`build_opinion.py:18`).

**Sorties (CONTRATS)** :
- `analysis/opinion.json` — `{dataset, model, seed, thresholds, counts,
  themes:[{theme_id, proposition, fav, def, nuance, n, engagement, opposition,
  pct_favorable, profil:"clivant|consensuel|impur", title, cleavage_justif,
  cleavage_fit…}]}`. Écrit `build_opinion.py:512` → `analysis_store.write_opinion:270`.
- `analysis/claim_stance.json` — `{ "avis_id#gi": {stance:"favorable|defavorable|nuance",
  stance_confidence, justif, proposition, theme_id} }`. Écrit `build_opinion.py:513` →
  `analysis_store.write_claim_stance:275`. **Émis seulement pour les feuilles PURES**
  (`build_opinion.py:347` : `if opinion["profil"] != "impur"`).

**LE COUPLAGE CRITIQUE** (gotcha `DEV_PROD.md:24-32`, vérifié en code) : les clés de
`claim_stance` (`build_opinion.py:352`, `gi ∈ node.members`) doivent égaler les ids de claim
d'`avis.json` (`avis.py:92`, `ci = enumerate(claim_owner)`). Les deux dérivent du **même
`prepared.claim_owner`** → identiques **uniquement si `build_analysis` et `build_opinion` ont
tourné sur le même `claims.json` avec `extract_model`+`seed`+`min_chars` identiques**. Rebake
de l'un sans l'autre ⇒ index divergent ⇒ jointure stance ~1 % (cassée).

**Librement modifiable** (idempotent, ne touche ni l'arbre ni les index) : `model`
cleavage+stance (`build_opinion.py:46`), `CAP` claims/feuille, `MIN_ENGAGEMENT`, `MIN_CLAIMS`,
seuil `cleavage_fit_low`.

**CONTRAT GELÉ** : format de clé `avis_id#index_global` ; vocabulaire stance
`favorable|defavorable|nuance` ; **co-build obligatoire avec `build_analysis`**.

**Détection / règle** : `intersection(ids avis.json, clés claim_stance) ≈ part de claims dans
feuilles non-impur (≈80-100 %)`. **Toujours** `build_analysis` PUIS `build_opinion` ensemble,
puis promouvoir le dossier `analysis/` **entier** (`deploy/promote-cache.sh`, jamais un fichier
seul). Test : `backend/tests/test_opinion.py` (round-trip stance + jointure). *Gap : pas de
test d'intersection dédié — candidat à ajouter en CI.*

---

## §4 — Couche 4 · Serve / API

**Périmètre** : `backend/server.py` (routes), `backend/avis.py` (avis + jointure stance),
`backend/analysis_store.py` (I/O cache), `backend/auth.py` (fail-closed public).

**Entrée** : lit `analysis.json`, `avis.json`, `opinion.json`, `claim_stance.json`, `claims.json`
via `analysis_store` (`read_analysis`, `read_avis_all`, `read_opinion`, `read_claim_stance`),
**mémoïsé par mtime**. Écritures atomiques temp+`os.replace` (`analysis_store.py:111-121`) ⇒
pas de lecture torn.

**Sortie (CONTRAT = shapes de réponse, gelées vs le front)** : `/analysis` (POST, arbre+edges+
dataset_stats), `/avis/{id}`, `/avis_list`, `/insights`, `/citations`, `/opinion`, `/datasets`,
`/health`, `/build_status`, `/flags`. Claims servis toujours `{id, cluster_id, leaf_id, color,
spans, target, theme_title}`, **enrichis** `{stance, stance_confidence, proposition,
stance_justif}` si `claim_stance.json` présent (gracieux si absent).

**Jointure stance** `/avis_list?stance=` : `read_claim_stance` → `avis.join_claim_stance`
matche `claim.id` ↔ clé stance (`avis.py:240-248`, `server.py:587-593`). **Casse si `avis.json`
et `claim_stance.json` viennent de builds différents** (cf. §3).

**Fail-closed public** (`auth.py:32` `PUBLIC_MODE`) : `forbid_in_public` → 403 sur
build/recluster/density ; `require_token` fail-closed si pas d'`API_TOKEN` ; pas d'autobuild
en public (`server.py:376-397`) ; `/docs` désactivé. Lectures (`/analysis`, `/avis…`, `/opinion`)
ouvertes.

**Librement modifiable** : chemins de cache, impl interne, mémoïsation. **GELÉ** : shapes de
réponse, routes, format d'id claim, vocab stance.

**Tests** : `test_read_shape.py`, `test_avis_list.py`, `test_opinion.py`, `test_public_mode.py`,
`test_auth.py`, `test_hardening.py`.

---

## §5 — Couche 5 · Front

**Périmètre** : `frontend/src/redesign/*` + `frontend/src/api.ts`. Contrat de types
`redesign/contract.ts` ; base HTTP `redesign/http.ts` (`BASE='/api'` → proxy Vite `:8010`,
`TIMEOUT=180 s`) ; clients `redesign/analysisApi.ts` + `src/api.ts`.

**Entrée (CONTRAT)** : `contract.ts` = miroir des réponses L4 — `SpatialTheme` (15-49),
`AnalysisPayload` (79-93), `ThemeOpinion` (109-127), `AvisClaim` (192-219), `AvisProvenance`
(228-234), `AvisListItem` (242-256), `Citation` (169-176), `Consultation` (319-353),
`SubmitResult`. Endpoints appelés : `/analysis`, `/insights`, `/citations`, `/avis`,
`/avis_list`, `/opinion`, `/cost`, `/flags` (`analysisApi.ts`) ; `/datasets`, `/submit`
(`api.ts`) ; `/density` (`densityApi.ts`).

**Librement modifiable** : composants React, CSS, viz D3, strings i18n, `mock.ts` (doit refléter
`contract.ts`). **GELÉ** : `contract.ts` doit matcher les shapes L4.

**Détection casse croisée** : `npm run build` = `tsc -b && vite build` (strict mode) → un
`contract.ts` désaligné avec les consommateurs échoue au typecheck ; gardes runtime partielles
(`Array.isArray` dans `analysisApi.ts`). **Angle mort** : un champ optionnel retiré côté back
n'est PAS attrapé par tsc (dégradation silencieuse) — d'où la règle §0.3 (PR appariée back+front).

---

## §6 — VERDICT : la ségmentation 5 personnes est-elle sûre ?

**OUI pour le code, à condition de tenir 5 règles.** Les couches sont réellement séparées par
des fichiers de cache à schéma stable, et chaque frontière est couverte par des tests. Deux
réserves : (a) `analysis.py` et `claims_endpoint.py` sont des **bibliothèques partagées**
(un owner, le reste read-only) ; (b) au **build de données**, L2 et L3 sont **couplés** (index
de claim) et **non splittables** — ils se rebuild ensemble.

**Répartition proposée**
| # | Owner | Couche | Possède (schéma) |
|---|---|---|---|
| P1 | ingest+claims | L1 | `claims.json`, `avis_id`, modèle d'extraction, `claims_endpoint.py` |
| P2 | embed+cluster | L2 | `analysis.json`/arbre, `seed`, index global, `analysis.py` |
| P3 | opinion+stance | L3 | `opinion.json`, `claim_stance.json` |
| P4 | serve/API | L4 | shapes de réponse (`server`, `avis`, `analysis_store`) |
| P5 | front | L5 | UI, consomme `contract.ts` |

**Les 5 règles**
1. **Geler la colonne vertébrale** (§0) : `DEFAULT_SEED/EMBEDDER/EXTRACT_MODEL`, le template
   `avis_id#index_global`, l'ordre d'aplatissement `_flatten` — modif = cross-lane + CODEOWNERS.
2. **`analysis.py` et `claims_endpoint.py` = read-only** pour tout le monde sauf leur owner
   (P2, P1) ; les autres consomment l'API `ThemeTree`/`prepare_claims`, ne l'éditent pas.
3. **Rebuild de données couplé** : toujours `build_analysis` PUIS `build_opinion` sur le même
   `claims.json` ; promouvoir `analysis/` **entier** (jamais un fichier). Vérif intersection ≥80 %.
4. **L4↔L5** : toute modif de shape API part avec son édit `contract.ts` dans **la même PR**.
5. **CI = le garde-frontières** : `pytest backend/tests` (couvre chaque frontière) vert avant
   merge ; front `npm run build` (tsc strict). Le flux PR→CI→auto-deploy est la barrière.

**Hotspots (là où le découpage est le plus fin)** : la jointure stance `/avis_list?stance=`
(3 couches : L2 index → L3 clés → L4 join) ; et `contract.ts` (silencieux si un champ optionnel
disparaît). Les surveiller en priorité pendant le hackathon.
