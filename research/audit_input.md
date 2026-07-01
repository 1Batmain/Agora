# Audit sécurité — Validation d'entrée, injection, path-traversal, DoS

**Périmètre** : surface HTTP du backend FastAPI (`backend/server.py` :8010) et ses
dépendances de stockage/calcul. Audit **adversaire** (PRO → PUBLIC), avant partage.
**Branche** : `work/sec-input`. **Date** : 2026-06-30.
**Posture** : aucune logique applicative modifiée — rapport seul.

Méthode : revue de chaque endpoint, traçage de chaque paramètre attaquant-contrôlé
jusqu'à un effet de bord (lecture/écriture disque, calcul lourd, sous-process, mémoire),
en cherchant systématiquement le contournement.

---

## Résumé exécutif

Le **path-traversal par `dataset` est correctement neutralisé** (whitelist `_IDSET`
appliquée *partout* via `_resolve`, + `_safe()` sur les ids de thème) — c'est le point
fort, déjà couvert par `test_security_whitelist.py`. **Aucune injection de commande**
(subprocess en liste d'args, pas de `shell=True`, pas d'`eval`/`pickle` sur entrée).

Les angles morts sont ailleurs et concernent surtout la **disponibilité** :

| # | Sévérité | Titre | Lieu |
|---|----------|-------|------|
| 1 | **CRITIQUE** | Auth « fail-open » : sans `AGORA_API_TOKEN`, tous les endpoints « protégés » sont ouverts | `backend/auth.py:33-40` |
| 2 | **HAUT** | Aucune limite de taille de corps de requête → DoS mémoire sur tous les POST | `backend/server.py` (global) |
| 3 | **HAUT** | `/recluster` : calcul lourd, NON authentifié, NON rate-limité, recharge le cache disque à chaque appel | `backend/server.py:539-550`, `live_cluster.py:218` |
| 4 | **HAUT** | `resolution` sans borne supérieure → amplificateur de coût Leiden | `backend/server.py:536`, `348` |
| 5 | **MOYEN** | Rate-limit par `client.host` : s'effondre derrière un proxy / spoofable si proxy-headers activés | `backend/auth.py:53-65` |
| 6 | **MOYEN** | Dictionnaire `_hits` du rate-limiter jamais purgé → fuite mémoire / DoS par rotation d'IP | `backend/auth.py:50,57` |
| 7 | **MOYEN** | Comparaison de token non constante en temps (`!=`) → oracle temporel | `backend/auth.py:44` |
| 8 | **MOYEN** | `/avis_list?q=` : scan + normalisation Unicode du corpus entier à chaque requête, non protégé | `backend/avis.py:201-224` |
| 9 | **MOYEN** | `/submit` : store append-only non borné → coût O(N) croissant + remplissage disque | `backend/submissions.py:74-80` |
| 10 | **BAS** | Champs texte non bornés (`/flag.text`, `/todo.title|note`) | `backend/server.py:607`, `todo_store.py` |
| 11 | **BAS** | `/todo` POST/PATCH mutateurs sans token (rate-limit seul) | `backend/server.py:187,207` |
| 12 | **BAS** | `/density` 1ᵉʳ appel (UMAP) lourd et non protégé (atténué par cache) | `backend/density.py:128` |

---

## CRITIQUE

### 1. Auth « fail-open » — sans `AGORA_API_TOKEN`, la protection est inerte

**Preuve** — `backend/auth.py:33-40` :

```python
def require_token(...):
    if API_TOKEN is None:                 # AGORA_API_TOKEN non défini
        if not _warned:
            print("[auth] AGORA_API_TOKEN non defini -> endpoints couteux NON proteges ...")
        return                            # ← laisse PASSER toute requête
```

`API_TOKEN = os.environ.get("AGORA_API_TOKEN") or None` (`auth.py:23`). Si la variable
n'est pas exportée — ce qui est l'état **par défaut** et le mode hackathon —, *toutes*
les dépendances `require_token` deviennent un no-op. Les endpoints `PROTECTED`
(`/build`, `/flag`, `DELETE /flag`, `/submit`) sont alors **entièrement ouverts** :
n'importe qui peut déclencher un rebuild (`POST /build` lance un sous-process lourd,
cf. `build_manager.py:111`), écrire/supprimer des flags, ou injecter des contributions.

C'est un défaut **« fail-open »** : l'absence de configuration *désactive* la sécurité au
lieu de *fermer* l'accès. La seule trace est un `print` au démarrage, facile à manquer.

**Scénario d'attaque** : déploiement public sans `AGORA_API_TOKEN` exporté (oubli, env
non propagé au worker gunicorn) → `for i in ...: curl -XPOST :8010/build -d '{"dataset":"tiktok","force":true}'`
relance des builds en boucle (chacun = un `subprocess.Popen` python lourd, voir aussi #3).

**Remédiation** :
- **Fail-closed en production** : si un drapeau d'environnement `AGORA_ENV=prod` (ou
  `AGORA_REQUIRE_AUTH=1`) est posé et que `AGORA_API_TOKEN` est absent, **lever au
  démarrage** (`raise RuntimeError`) au lieu de servir ouvert.
- À défaut d'environnement explicite, transformer le `print` en `logging.warning` de
  niveau visible et documenter dans le README de déploiement que l'absence de token =
  ouverture totale.
- Vérifier que gunicorn/uvicorn héritent bien de l'env (`--env` ou unit systemd).

---

## HAUT

### 2. Aucune limite de taille de corps → DoS mémoire sur tout POST

**Preuve** — `backend/server.py:119-125` : la seule middleware est `CORSMiddleware`.
Ni Starlette ni uvicorn n'imposent de limite de taille de corps par défaut. Tout handler
POST (`/analysis`, `/recluster`, `/build`, `/flag`, `/submit`, `/todo`) lit **l'intégralité
du corps en mémoire** avant validation Pydantic.

Le garde-fou de `/submit` est **post-parse** et donc inefficace contre l'attaque
mémoire — `server.py:259-266` :

```python
text = (body.text or "").strip()        # body.text déjà MATÉRIALISÉ en RAM
if len(text) > SUBMIT_MAX_CHARS:        # vérifié APRÈS coup
    raise HTTPException(422, ...)
```

Un `POST /submit` (ou n'importe quel POST, même `/analysis`) avec un corps JSON de
plusieurs centaines de Mo est lu, désérialisé et la chaîne `text` construite **avant** que
la limite de 5000 caractères ne s'applique. Répété en parallèle, c'est un épuisement
mémoire trivial — aucun token requis (cf. #1) ni même nécessaire pour `/analysis`.

**Scénario** : `curl -XPOST :8010/analysis --data-binary @500mb.json` ×N → OOM du worker.

**Remédiation** :
- Ajouter une middleware ASGI qui **rejette (413)** si `Content-Length` (ou le cumul des
  chunks lus) dépasse une borne (ex. 64 Ko pour les endpoints JSON ; 32 Ko suffisent pour
  `/submit` à 5000 caractères UTF-8). Refuser aussi l'absence de `Content-Length` au-delà
  d'un quota de chunks lus.
- Configurer le reverse-proxy en amont (`client_max_body_size` nginx) — défense en
  profondeur, mais ne pas s'y fier seul (le backend est lançable sans proxy, cf. docstring
  `server.py:26`).

---

### 3. `/recluster` — calcul lourd, ouvert, non rate-limité, recharge disque par appel

**Preuve** — `backend/server.py:539-550` : `do_recluster` n'a **aucune** dépendance
(`PROTECTED` absent). Il appelle `live_cluster.recluster_payload`, qui à
`live_cluster.py:218` fait :

```python
ideas, vecs, weights = load_cache(dataset)   # relit embeddings.npy + ideas.jsonl du DISQUE
```

à **chaque requête** (aucune mémoïsation, contrairement aux `_Dataset` de `server.py`),
puis k-NN + Leiden hiérarchique + subdivision variance-adaptative + nommage c-TF-IDF +
projection UMAP (`live_cluster.py:102-148`). La docstring annonce « < ~2 s » *par appel* —
mais rien n'empêche de boucler.

**Scénario** : `while true; do curl -XPOST :8010/recluster -d '{"dataset":"<grand_dataset>","k":200}'; done`
sature CPU **et** I/O disque (rechargement du `.npy` complet à chaque fois). `k=200`
(plafond autorisé, `server.py:535`) densifie le graphe → coût maximal.

**Remédiation** :
- Poser `dependencies=[Depends(rate_limit)]` (au minimum) sur `/recluster` — c'est un
  endpoint **COMPUTE**, pas une simple lecture de cache ; il n'a aucune raison d'être dans
  la même catégorie « lecture ouverte » qu'`/analysis`.
- Mémoïser `load_cache(dataset)` (les vecteurs sont déjà chargés en RAM par `_Dataset`
  côté serveur — réutiliser `_resolve(dataset).vecs` au lieu de relire le disque).
- Envisager un cache LRU des payloads `recluster` par `(dataset, k, threshold, resolution)`.

---

### 4. `resolution` sans borne supérieure → amplificateur de coût

**Preuve** — `ReclusterBody.resolution = Field(1.0, gt=0.0)` (`server.py:536`) et
`AnalysisBody.resolution = Field(1.0, gt=0.0)` (`server.py:348`) : `gt=0.0` mais **pas de
`le=`**. Pour `/recluster`, `resolution` est passé tel quel à Leiden
(`live_cluster.py:112 run_leiden(graph, resolution=resolution, ...)`).

Une `resolution` très élevée (ex. `1e6`) force Leiden vers une partition ultra-fine
(quasi un cluster par nœud) → `fine_groups` explose → `_build_subtree`/`_name_nodes`
(c-TF-IDF) itèrent sur un nombre de groupes démesuré. Couplé à #3 (endpoint ouvert),
c'est un multiplicateur de coût gratuit pour l'attaquant.

**Remédiation** : borner — `Field(1.0, gt=0.0, le=10.0)` (ou une valeur métier raisonnable)
sur `ReclusterBody.resolution`. Pour `AnalysisBody`, `resolution` est *ignoré*
(SERVE-only, cf. docstring `server.py:341`) → soit le borner aussi, soit le retirer du
modèle pour ne pas laisser croire qu'il agit.

---

## MOYEN

### 5. Rate-limit par `client.host` — fragile derrière un proxy, spoofable selon config

**Preuve** — `backend/auth.py:55` : `ip = request.client.host`. C'est l'IP du **pair TCP**.

Deux modes de défaillance, selon le déploiement (cf. note multi-worker
`build_manager.py:127-144` qui recommande gunicorn derrière un proxy) :

1. **Derrière un reverse-proxy sans `--forwarded-allow-ips`** : `client.host` vaut
   *toujours* l'IP du proxy. **Tous** les clients partagent alors un seul compteur → soit
   le quota global bloque des utilisateurs légitimes (déni de service induit), soit il est
   relevé et devient inopérant.
2. **Avec uvicorn `--proxy-headers`/`--forwarded-allow-ips=*`** : `client.host` est dérivé
   du `X-Forwarded-For` fourni par le client → **spoofable** ; l'attaquant fait tourner un
   en-tête `X-Forwarded-For` bidon et obtient un compteur neuf à chaque requête, contournant
   totalement le rate-limit.

Le code ne lit pas lui-même `X-Forwarded-For`, donc le risque dépend de la config — mais
aucune des deux configurations courantes ne donne un rate-limit correct.

**Remédiation** : décider explicitement de la chaîne de confiance. Idéalement, faire le
rate-limit **dans le reverse-proxy** (nginx `limit_req`) où l'IP source est fiable. Si
gardé applicatif, documenter qu'uvicorn doit tourner avec `--forwarded-allow-ips=<IP du
proxy uniquement>` et lire l'IP réelle via le `X-Forwarded-For` *de confiance* (1ᵉʳ saut),
jamais l'en-tête brut du client.

### 6. `_hits` jamais purgé → fuite mémoire / DoS par rotation d'IP

**Preuve** — `backend/auth.py:50` : `_hits: dict[str, deque] = defaultdict(deque)`. Dans
`rate_limit` (`auth.py:57-65`), on purge les *timestamps* anciens d'une deque, mais
l'**entrée `_hits[ip]` (clé IP) n'est jamais supprimée**, même vidée. Chaque IP distincte
vue crée une entrée permanente.

**Scénario** : un attaquant (ou simplement le passage du temps en IPv6, ou #5 mode 2 avec
`X-Forwarded-For` rotatif) fait croître `_hits` sans borne → fuite mémoire monotone du
process serve.

**Remédiation** : purger périodiquement les IP dont la deque est vide (ou expirée), ou
remplacer par un cache borné (`cachetools.TTLCache` à capacité fixe). Au minimum, supprimer
`_hits[ip]` quand `dq` devient vide après éviction.

### 7. Comparaison de token non constante en temps

**Preuve** — `backend/auth.py:44` : `if not token or token != API_TOKEN:`. L'opérateur `!=`
sur des `str` court-circuite au 1ᵉʳ octet différent → **oracle temporel** permettant de
reconstruire le token octet par octet (exploitabilité réelle faible sur réseau bruité, mais
gratuit à corriger et standard).

**Remédiation** : `import secrets; if not token or not secrets.compare_digest(token, API_TOKEN):`.

### 8. `/avis_list?q=` — scan + normalisation Unicode du corpus à chaque requête, non protégé

**Preuve** — `backend/avis.py:201,212` : pour chaque requête, `avis_list` parcourt **tous**
les avis et appelle `_fold(text)` (`avis.py:132-141` : `unicodedata.normalize("NFD", …)` +
filtrage des diacritiques + `casefold`) sur le **texte intégral de chaque avis**, sans
aucun cache du résultat folding. L'endpoint `/avis_list` (`server.py:467`) est **ouvert**
(aucune dépendance).

Ce n'est pas du ReDoS (recherche par sous-chaîne `in`, pas de regex) mais un coût CPU
O(N · longueur) **par requête**, répétable à volonté.

**Scénario** : `while true; do curl ':8010/avis_list?q=z&dataset=<grand>'; done` → CPU lié à
la re-normalisation NFD de tout le corpus à chaque appel.

**Remédiation** : précalculer et mettre en cache la forme `_fold(text)` par avis (à côté du
cache `_AVIS_CACHE` mémoïsé par mtime dans `analysis_store.py:181`), de sorte que la
recherche `q` ne refolde que la *needle*. Borner aussi la longueur de `q` (ex. 200 car.).
Optionnellement poser `rate_limit` sur les lectures lourdes.

### 9. `/submit` — store append-only non borné

**Preuve** — `backend/submissions.py:74-80` : `append_submission` ajoute une ligne au
`submissions.jsonl` sans plafond. À chaque `/submit`, `correlate` (`submissions.py:100-122`)
charge **toutes** les contributions et construit une matrice `mat = np.array([...])` →
coût mémoire/CPU **O(N)** croissant à mesure que le store gonfle. Chaque ligne stocke aussi
le vecteur 768-d complet (`submissions.py:78`) → croissance disque ~6 Ko/contribution.

Protégé par token+rate-limit, mais inerte si #1 (token absent). Même rate-limité, la
croissance est cumulative et persistante (le fichier ne se vide jamais).

**Remédiation** : plafonner le nombre de contributions live par consultation (ou fenêtre
glissante), borner la taille de `submissions.jsonl`, et/ou indexer les vecteurs (FAISS) au
lieu d'une matrice rechargée intégralement à chaque appel.

---

## BAS

### 10. Champs texte non bornés

- `FlagBody.text: str = ""` (`server.py:607`) : aucune limite de longueur ; persisté dans
  `flags.json` (`flags_store.upsert_flag`). Upserts répétés (clés `target_id` variées) →
  croissance disque. Protégé (token+rate), donc gravité basse.
- `TodoCreateBody.title/note` et `TodoPatchBody.assignee` (`server.py:181-205`) : non
  bornés, persistés dans `todo.json` à la racine du repo.

**Remédiation** : `Field(max_length=…)` sur ces champs (ex. 2000 pour `text`, 200 pour
`title`/`assignee`). Borner le nombre d'entrées des stores.

### 11. `/todo` POST/PATCH mutateurs sans token

**Preuve** — `server.py:187,207` : `dependencies=[Depends(rate_limit)]` seulement (pas de
`require_token`). Choix assumé (« outil collaboratif ouvert », docstring `server.py:192`).
Reste une écriture **non authentifiée** dans un fichier versionné (`todo.json`) ;
acceptable pour un hackathon, à revoir avant exposition publique durable. `lane` est
toutefois bien validé contre `known_lanes()` (`todo_store.py:94`) → pas d'injection de
valeur arbitraire ni de path.

**Remédiation** : si exposition publique, passer `/todo` mutateur derrière `require_token`
ou un captcha ; sinon documenter le risque d'abus (spam de tâches).

### 12. `/density` — 1ᵉʳ appel lourd, non protégé

**Preuve** — `server.py:505-522` ouvert ; `density.density_payload` calcule UMAP+KDE au
1ᵉʳ appel par dataset (`density.py:128-146`). Fortement atténué : le résultat est **caché
sur disque** (`umap2d.npy` + `density.json`) et les builds le précalculent généralement →
seuls les datasets « froids » sans cache déclenchent le calcul, et une seule fois.

**Remédiation** : s'assurer que `density` est précalculé au BUILD pour tous les datasets
publics (de sorte qu'aucun chemin requête ne déclenche UMAP) ; sinon, `rate_limit` dessus.

---

## Points vérifiés — SÛRS (non-findings, documentés pour la traçabilité)

- **Path-traversal par `dataset`** : neutralisé. `_resolve` (`server.py:91-107`) vérifie
  l'appartenance à `_IDSET` (whitelist O(1) découverte au boot) **avant** toute
  construction de `_Dataset`/accès disque. Appliqué sur *tous* les endpoints par-dataset
  (vérifié un à un). Couvert par `test_security_whitelist.py` (incl. `../etc`,
  `../../etc/passwd`, `tiktok/../secret`).
- **Path-traversal par `theme_id` / `id` d'insight** : `analysis_store._safe()`
  (`analysis_store.py:85-96`) remplace tout caractère hors `[alnum-_]` par `_` avant de
  composer `citations/<theme_id>.json` et `insights/<name>.json`. `../` → `___`. Sûr.
- **`consultation_id` (`/submit`)** : validé contre `set(list_open_consultations())`
  (`server.py:254`) avant tout accès aux chemins `submissions.*`. Sûr.
- **Injection de commande** : `build_manager.py:111` lance `subprocess.Popen(argv)` avec
  `argv` en **liste** (pas de `shell=True`) ; le seul élément variable, `dataset`, est déjà
  whitelisté ; aucun champ de requête (`model`/`backend`) n'est passé en argv depuis les
  endpoints (`do_build`/`_not_ready_response` appellent `ensure_build(ds)` sans kwargs).
  Pas d'`eval`/`exec`/`pickle.load`/`yaml.load` sur entrée utilisateur.
- **ReDoS** : aucune regex appliquée à une entrée utilisateur au moment de la requête. La
  recherche `q` utilise `in` (sous-chaîne). Les regex de `_slugify` (`todo_store.py:78`,
  `[^a-z0-9]+`) et de normalisation PII sont linéaires et/ou exécutées au BUILD, pas à la
  requête.
- **CORS** : origines explicites, jamais `"*"` (`server.py:112-125`) ; `allow_credentials`
  cohabite avec une liste fermée. Correct (surveiller que `AGORA_ALLOWED_ORIGINS` ne soit
  pas mis à `*` en prod, ce qui serait d'ailleurs rejeté par Starlette avec credentials).
- **Pagination** : `limit` borné `ge=1, le=200` (`server.py:473`) ; `offset ge=0` sans
  borne haute mais inoffensif (slice sur liste vide). Pas d'abus mémoire.

---

## Priorités de remédiation (ordre conseillé)

1. **#1 (CRITIQUE)** — fail-closed sur token absent en prod. *Une ligne au démarrage.*
2. **#2 (HAUT)** — middleware de limite de taille de corps (413). *Couvre tous les POST.*
3. **#3 + #4 (HAUT)** — `rate_limit` + borne `resolution` sur `/recluster` ;
   réutiliser les vecteurs déjà en RAM. *Ferme le DoS applicatif principal.*
4. **#5–#9 (MOYEN)** — fiabiliser le rate-limit (IP, purge, constant-time) et borner les
   scans/stores ouverts.
5. **#10–#12 (BAS)** — `max_length` sur les champs texte, revoir `/todo` ouvert.

Aucune logique applicative n'a été modifiée dans le cadre de cet audit.
