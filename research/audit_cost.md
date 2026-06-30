# AUDIT SÉCURITÉ — Exposition COÛT (LLM / compute)

**Périmètre :** backend FastAPI `backend/server.py` + routers, branche `work/sec-cost`.
**Question directrice (adversaire) :** *que peut faire dépenser un visiteur ANONYME, par
requête et en boucle, avant l'exposition PUBLIC ?*
**Date :** 2026-06-30 · **Auditeur :** sec-cost · **Méthode :** lecture exhaustive du
call-graph de chaque endpoint (cache sûr vs. appel Mistral/Ollama vs. build/recluster runtime).

> ⚠️ **Verdict global : NE PAS exposer en l'état.** Cinq endpoints de LECTURE non
> authentifiés et non rate-limités peuvent **déclencher un build LLM complet**
> (extraction `mistral-large-latest` sur tout le corpus). Combiné à l'absence de
> sauvegarde incrémentale de l'extraction et au re-déclenchement sur état `error`, cela
> crée une **boucle de ré-extraction LLM non bornée** pilotable par un simple polling
> anonyme (ou même par le front légitime sur un dataset cassé).

---

## 1. Cartographie des endpoints (call-graph + classe de coût)

Légende coût : 🟢 cache/fichier (sûr) · 🟡 compute CPU au runtime · 🔴 appel LLM (Mistral/Ollama).
Légende garde : `TOKEN` = `require_token`, `RL` = `rate_limit` (cf. `PROTECTED`, server.py:40).

| Endpoint | Méthode | Garde | Coût | Call-graph / déclencheur |
|---|---|---|---|---|
| `/health` | GET | — | 🟢 | meta léger |
| `/datasets` | GET | — | 🟢 | descripteurs (meta.json/ideas) |
| `/todo` | GET | — | 🟢 | lecture `todo.json` |
| `/todo` | POST | `RL` | 🟢 | écriture `todo.json` (pas de TOKEN) |
| `/todo/{id}` | PATCH | `RL` | 🟢 | écriture `todo.json` (pas de TOKEN) |
| `/submit` | POST | `TOKEN`+`RL` | 🟡 | embed **nomic LOCAL** (pas de LLM, pas de clé) — server.py:240,268 |
| **`/analysis`** | POST | **— (aucune)** | 🟢→**🔴** | si analyse non `ready` → `_not_ready_response` → **`ensure_build`** — server.py:351-369 |
| **`/insights`** | GET | **— (aucune)** | 🟢→**🔴** | idem si non `ready` — server.py:391-392 |
| **`/citations`** | GET | **— (aucune)** | 🟢→**🔴** | idem si non `ready` — server.py:413-414 |
| **`/avis/{id}`** | GET | **— (aucune)** | 🟢→**🔴** | idem si non `ready` — server.py:454-455 |
| **`/avis_list`** | GET | **— (aucune)** | 🟢→**🔴** | idem si non `ready` — server.py:488-492 |
| `/opinion` | GET | — | 🟢 | cache only ; absent → `{themes:[]}` (NE build PAS) — server.py:422-436 ✅ |
| **`/recluster`** | POST | **— (aucune)** | **🟡 lourd** | Leiden + subdivision + UMAP **À CHAQUE appel**, `load_cache` disque à chaque fois — server.py:539-550 |
| `/density` | GET | — | 🟡 (1×/dataset) | UMAP paresseux au 1ᵉʳ appel puis caché — server.py:505-522 |
| `/build` | POST | `TOKEN`+`RL` | 🔴 | build de fond ; `force=true` efface puis rebuild — server.py:563-574 ✅ |
| `/build_status` | GET | — | 🟢 | lecture `status.json` |
| `/flags` | GET | — | 🟢 | lecture `flags.json` |
| `/flag` | POST/DELETE | `TOKEN`+`RL` | 🟢 | upsert `flags.json` ✅ |

**Constat structurant :** les endpoints *réputés* « SERVE-only / lecture cache » ne le
sont PAS inconditionnellement. Cinq d'entre eux **basculent en déclencheur de build LLM**
dès que l'analyse du dataset n'est pas `ready`, via le chemin commun
`_not_ready_response()` (server.py:322-333) → `build_manager.ensure_build(ds)`.

---

## 2. Réponse aux questions du brief

**« Un dataset INCONNU déclenche-t-il un build ? »** → **NON.** `_resolve` (server.py:91-107)
oppose une whitelist `_IDSET` découverte au boot ; un id absent = 404 *avant* toute
construction. Le path-traversal est correctement gardé. ✅

**« Un PARAMÈTRE déclenche-t-il une ré-extraction ? »** → **NON via les reads.**
`_not_ready_response` appelle `ensure_build(ds)` **sans kwargs** : `body.model` /
`body.backend` de `/analysis` sont ignorés (server.py:360,328). Pas d'escalade de modèle
par paramètre. La ré-extraction propre n'est offerte que par `/build` (protégé) +
`--reextract` (CLI). ✅ *(mitigation réelle, à conserver.)*

**« Un dataset CONNU mais non prêt déclenche-t-il un build ? »** → **OUI, et sans aucune
garde.** C'est le cœur du risque (§3, F1).

---

## 3. Findings priorisés

### 🟥 CRITIQUE — F1. Les reads non authentifiés déclenchent un build LLM ; boucle de ré-extraction non bornée sur `error`

**Preuve :**
- `server.py:351` `do_analysis` (aucune `dependencies=`), `:369` → `_not_ready_response`.
- `server.py:372` `/insights`, `:400` `/citations`, `:439` `/avis/{id}`, `:467` `/avis_list` — toutes sans garde, toutes appelant `_not_ready_response`.
- `server.py:322-333` `_not_ready_response` → `build_manager.ensure_build(ds)`.
- `build_manager.py:81-118` `ensure_build` : si `state != READY` **et** dataset pas dans `_procs` → `subprocess.Popen([... "backend.build_analysis" ...])` qui **hérite de l'environnement (clé Mistral)**.
- `build_analysis.py:57` `EXTRACT_MODEL = "mistral-large-latest"`, `:155-164` extraction LLM sur **tout le corpus**.
- `analysis_store.py:139-151` `state()` : un dataset en `error` n'est PAS `ready` → re-déclenche.
- `build_manager.py:96-99` : le garde anti-double-build (`_procs`) ne tient **que tant que le sous-process est vivant**. Sitôt qu'un build échoue et est *reapé* (`_reap_locked`, build_manager.py:39-43), l'`ensure_build` **suivant relance un build neuf**.
- `claims_endpoint.py:270-278` : `_save_claims_cache` n'est appelé **qu'après** le retour COMPLET de `extract_claims`. Si l'extraction lève (un lot qui échoue après backoff propage l'exception — extract.py:347), **rien n'est mis en cache** → le build suivant **ré-extrait tout le corpus**.

**Scénario d'échec concret :**
1. Un dataset whitelisté est non-`ready` (jamais buildé, ou `AGORA_AUTOBUILD=0`, ou build cassé) ;
2. un visiteur anonyme (ou le **front légitime qui poll `/analysis` toutes les ~2 s** pendant « Analyse en cours… ») touche `/analysis` ;
3. → `ensure_build` lance `mistral-large` sur ~3000 avis ;
4. l'extraction échoue partway (panne réseau Mistral, 429 soutenu, donnée pathologique, OOM du sous-process) → `status=error`, **claims.json jamais écrit** ;
5. le poll suivant retrouve `state=error`, `_procs` vidé → **relance une extraction complète à neuf** ;
6. boucle : ré-extraction intégrale du corpus à chaque cycle de build, **indéfiniment, sans intervention ni plafond**, pilotée par du trafic anonyme banal.

Aucun de ces 5 endpoints n'a `TOKEN` ni `RL`. Il n'existe **aucun plafond de coût LLM**,
aucun disjoncteur (« N échecs → ne plus retenter »), aucune borne de concurrence
inter-datasets (un attaquant peut amorcer les 5 datasets en parallèle = 5 sous-process
d'extraction `mistral-large` simultanés).

**Chiffrage du pire cas (ordre de grandeur, hypothèses explicites) :**
extraction par lots de 8 avis (`BATCH_SIZE=8`, extract.py:33), sortie plafonnée
`min(8192, 400×8)=3200` tok/lot. Pour 3000 avis ≈ **375 appels `mistral-large`/build**.
À ~2 900 tok entrée + 3 200 tok sortie par lot et un tarif d'ordre `large` (~2 $/M in,
~6 $/M out) ⇒ ≈ **9–15 $ par build complet d'un dataset**, ~**50–75 $** pour balayer les
5 datasets une fois. **En boucle `error`, ce montant se répète à chaque cycle de build
(quelques minutes), sans borne haute** — la facture est en pratique illimitée tant que le
dataset reste cassé et qu'un client (humain ou bot) sonde l'endpoint.

**Remédiation (avant expo, par ordre) :**
1. **Découpler reads et build.** Les 5 reads NE doivent JAMAIS appeler `ensure_build`. Si
   non-`ready` : renvoyer `202 {status}` **sans** déclencher quoi que ce soit. Le build
   ne se déclenche que par `/build` (déjà protégé) ou l'autobuild de démarrage.
2. **Disjoncteur sur `error`.** `ensure_build` ne doit pas relancer un build qui vient
   d'échouer : compteur d'échecs + backoff persistés dans `status.json` (N échecs →
   `error` collant, relance manuelle/opérateur uniquement).
3. **Sauvegarde incrémentale de l'extraction.** Persister `claims.json` au fil des lots
   (claims_endpoint.py:276) pour qu'un échec ne jette pas le travail déjà payé et ne force
   pas une ré-extraction intégrale.
4. **Plafond de coût LLM** (compteur de tokens/jour + kill-switch) côté build.
5. À défaut de (1) immédiat : poser au minimum `dependencies=PROTECTED` sur les 5 reads,
   et un plafond de concurrence globale des builds (1 à la fois, file d'attente).

---

### 🟥 HAUT — F2. `/recluster` : compute lourd, anonyme, sans rate-limit → DoS CPU/RAM

**Preuve :** `server.py:539-550` `do_recluster` (aucune `dependencies=`). Le payload
(live_cluster.py:203-254) fait à **chaque appel** : `load_cache` disque (jamais mémoïsé
ici), construction k-NN, `run_leiden`, subdivision variance-adaptative, coarsening,
nommage c-TF-IDF, indices, points UMAP. `k` accepté jusqu'à **200** (server.py:535),
`resolution` non bornée en haut (`gt=0.0`, server.py:536). Plus `k` monte, plus le graphe
et le calcul gonflent.

**Scénario :** boucle anonyme `POST /recluster {k:200, resolution: <grand>}` → saturation
CPU + relectures disque + allocations NumPy répétées sur des datasets de milliers de
vecteurs. Zéro coût LLM, mais **DoS de disponibilité** (et coût compute cloud) trivial à
soutenir, sans aucune garde.

**Remédiation :** `dependencies=PROTECTED` (au moins `rate_limit`) sur `/recluster` ;
borner `resolution` (ex. `le=5.0`) ; envisager un cache court par `(dataset,k,thr,res)`.

---

### 🟧 HAUT — F3. Auth « fail-OPEN » : sans `AGORA_API_TOKEN`, tout le PROTECTED est ouvert

**Preuve :** `auth.py:23` `API_TOKEN = os.environ.get("AGORA_API_TOKEN") or None` ;
`auth.py:33-40` : si `API_TOKEN is None` → `require_token` **retourne sans rien exiger**
(simple `print` d'avertissement). Conséquence : si la variable d'env est oubliée au
déploiement, **`/build` (le plus cher : `force=true` efface le cache puis ré-extrait tout
au `mistral-large`), `/submit`, `/flag`** deviennent **anonymes**. La protection la plus
sensible dépend entièrement du fait de *se souvenir* d'une variable d'env, et **échoue en
mode ouvert**, pas fermé.

**Remédiation :** échouer FERMÉ en production. Introduire `AGORA_ENV=prod` (ou détecter
`AGORA_ALLOWED_ORIGINS` non-localhost) qui rend `AGORA_API_TOKEN` **obligatoire** : refuser
le démarrage (ou répondre 503 sur les routes protégées) si le token est absent en prod.

---

### 🟨 MOYEN — F4. Le rate-limit est inefficace derrière reverse-proxy / multi-worker

**Preuve :** `auth.py:50` `_hits` est un dict **en mémoire de process** ; `auth.py:55`
clé = `request.client.host`. Deux failles cumulées :
- **Multi-worker** : la note de déploiement recommande `gunicorn --workers N`
  (build_manager.py:130-135). Chaque worker a son propre `_hits` ⇒ limite effective
  **×N**.
- **Derrière un proxy** : `request.client.host` est l'IP du **proxy**, pas du client
  (aucune lecture de `X-Forwarded-For`). Résultat : soit tous les clients partagent un
  unique seau (faux positifs), soit, si le proxy masque l'IP, la limite par-IP ne sépare
  plus les attaquants. Le seul garde-fou anti-abus de facture est donc affaibli.

Note connexe : `_hits` croît sans bornes (un deque par IP, jamais purgé) → fuite mémoire
lente sous charge distribuée / IP spoofées.

**Remédiation :** rate-limit partagé (Redis/limiteur au niveau proxy) ; honorer
`X-Forwarded-For` **uniquement** derrière un proxy de confiance ; purge périodique des
seaux vides.

---

### 🟨 MOYEN — F5. `/density` (et `/recluster`) déclenchent un calcul UMAP au 1ᵉʳ appel

**Preuve :** `server.py:505-522` → `density.density_payload` → `compute_umap2d`
(density.py:60-90) : UMAP cosine sur tous les embeddings, **non protégé**. Borné (le
résultat est caché `umap2d.npy`/`density.json` après le 1ᵉʳ calcul), mais le **premier**
appel anonyme sur chaque dataset froid paie un calcul lourd ; couplé à F2, un attaquant
choisit le moment et multiplie par datasets.

**Remédiation :** soit pré-calculer UMAP au build (jamais à la requête), soit poser `RL`
sur `/density` et `/recluster`.

---

### 🟦 BAS — F6. `/todo` POST/PATCH : écriture anonyme (intégrité, hors coût)

**Preuve :** server.py:187,207 — `dependencies=[Depends(rate_limit)]` mais **pas** de
`require_token`. Un anonyme peut polluer `todo.json` (vandalisme/intégrité). Pas de coût
LLM ; signalé pour complétude.
**Remédiation :** `TOKEN` si la feuille de route est exposée publiquement, ou la passer en
lecture seule côté public.

---

## 4. Synthèse — quoi fermer/protéger/plafonner AVANT l'exposition

| Priorité | Action | Endpoints |
|---|---|---|
| **P0** | **Découpler reads ↔ build** : ne JAMAIS appeler `ensure_build` depuis un read (renvoyer l'état sans déclencher) | `/analysis`, `/insights`, `/citations`, `/avis/{id}`, `/avis_list` |
| **P0** | **Disjoncteur** `error` + **sauvegarde incrémentale** des claims + **plafond de coût LLM** | build_manager / build_analysis / claims_endpoint |
| **P1** | `dependencies=PROTECTED` (au moins `RL`) | `/recluster`, `/density`, et les 5 reads ci-dessus en filet de sécurité |
| **P1** | **Auth fail-CLOSED** : `AGORA_API_TOKEN` obligatoire en prod (refus de démarrage sinon) | `auth.py` |
| **P2** | Rate-limit **partagé** + `X-Forwarded-For` derrière proxy de confiance ; borner `resolution` | `auth.py`, `/recluster` |
| **P3** | `TOKEN` sur écritures `/todo` si exposées | `/todo` POST/PATCH |

**En une phrase :** le risque « des mille et des cents » est réel et concret — il vient
**moins du modèle que du câblage** : des endpoints de lecture qui retombent silencieusement
sur un build `mistral-large`, sans authentification, sans rate-limit, sans sauvegarde
incrémentale, sans disjoncteur et sans plafond. À corriger **avant** toute exposition
publique.
