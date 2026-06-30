# Audit sécurité — Auth, endpoints mutants & CORS

**Périmètre** : `backend/server.py`, `backend/auth.py`, `backend/todo_store.py`,
`backend/flags_store.py`, `backend/submissions.py`, `backend/analysis_store.py`,
`backend/recluster.py`, `backend/live_cluster.py`.
**Contexte** : audit PRO avant exposition PUBLIQUE du serveur FastAPI `:8010`.
**Posture** : adversaire (attaquant anonyme sur Internet), exhaustif.
**Méthode** : revue statique de tous les endpoints + dépendances de sécurité +
stores de persistance + comportement CORS vérifié dans la source Starlette installée.
**Aucune logique applicative modifiée** (rapport seul).

---

## Matrice de protection (état actuel)

| Endpoint | Méthode | Mutant ? | Coûteux ? | `require_token` | `rate_limit` |
|---|---|---|---|---|---|
| `/todo` | POST | **OUI** (écrit `todo.json` repo-root) | non | ❌ | ✅ |
| `/todo/{id}` | PATCH | **OUI** | non | ❌ | ✅ |
| `/recluster` | POST | non | **OUI** (Leiden+UMAP ~2 s) | ❌ | ❌ |
| `/analysis` | POST | déclenche build | moyen | ❌ | ❌ |
| `/density` | GET | non | **OUI** (UMAP au 1ᵉʳ appel) | ❌ | ❌ |
| `/build` | POST | **OUI** (rebuild LLM) | **OUI** | ✅ | ✅ |
| `/submit` | POST | **OUI** (append store) | moyen (embed) | ✅ | ✅ |
| `/flag` | POST/DELETE | **OUI** | non | ✅ | ✅ |
| lectures (`/insights`,`/citations`,`/avis*`,`/flags`,`/opinion`,`/build_status`,`/datasets`,`/health`) | GET | non | non | ❌ | ❌ |

Deux constats structurants sortent de cette matrice : (1) la protection `require_token`
est **fail-open** (cf. CRIT-1), donc toute la colonne ✅ s'effondre si une variable
d'env manque ; (2) trois endpoints mutants/coûteux échappent à toute protection
(`/todo` ×2, `/recluster`), cf. HAUT-1 et HAUT-2.

---

## Findings priorisés

### CRITIQUE

#### CRIT-1 — Auth *fail-open* : un `AGORA_API_TOKEN` non défini ouvre TOUT en silence
**Preuve** : `backend/auth.py:23` `API_TOKEN = os.environ.get("AGORA_API_TOKEN") or None`,
puis `backend/auth.py:33-40` :
```python
if API_TOKEN is None:
    if not _warned:
        print("[auth] AGORA_API_TOKEN non defini -> endpoints couteux NON proteges ...")
        _warned = True
    return            # <-- laisse passer SANS token
```
Si la variable n'est pas exportée (oubli, mauvais unit systemd, conteneur sans
secret monté), **tous** les endpoints « protégés » (`/build`, `/submit`, `/flag`,
`DELETE /flag`) deviennent anonymes. Le seul signal est un `print` unique sur stdout,
invisible derrière un superviseur, et `_warned` le supprime ensuite. Pour une expo
publique c'est le pire mode de défaillance : silencieux et total.

**Impact** : déclenchement anonyme de rebuilds LLM (coût facture + CPU), spam du store
de contributions, écriture/suppression de flags — toute la surface mutante coûteuse.

**Remédiation** :
- Ajouter un mode strict : `AGORA_REQUIRE_AUTH=1` (recommandé par défaut en prod) qui
  fait **échouer le démarrage** (`raise RuntimeError`) si `AGORA_API_TOKEN` est absent,
  au lieu de dégrader en mode ouvert.
- À défaut, exposer dans `GET /health` un champ `auth_enforced: bool` pour qu'une sonde
  externe détecte l'état ouvert, et logger en `WARNING`/`stderr` à chaque requête mutante
  non protégée (pas seulement une fois).
- Documenter dans le runbook de déploiement que l'absence de token = ouverture totale.

---

### HAUT

#### HAUT-1 — `/recluster` : calcul lourd anonyme, sans token NI rate-limit → DoS trivial
**Preuve** : `backend/server.py:539-550` :
```python
@app.post("/recluster")                       # <-- AUCUNE dependencies=
def do_recluster(body: ReclusterBody) -> dict:
    ds = _resolve(body.dataset)
    return live_cluster.recluster_payload(ds.id, body.knn_threshold, k=body.k, resolution=body.resolution)
```
`recluster_payload` (`backend/live_cluster.py:203-254`, doc l.10 « < ~2 s ») exécute à
chaque appel : `load_cache` (I/O), k-NN, **Leiden hiérarchique**, coarsening, subdivision
variance-adaptative, c-TF-IDF, **projection UMAP** (`_points`). C'est l'endpoint le plus
coûteux du serveur — et le seul mutant/coûteux **sans aucune dépendance de sécurité**
(ni `require_token`, ni `rate_limit`). Quelques requêtes/seconde saturent le CPU et
rendent le service indisponible (pas de worker pool isolé : FastAPI sync = thread pool
borné, vite épuisé).

Amplificateur : `resolution: float = Field(1.0, gt=0.0)` (`server.py:536`) n'a **pas de
borne haute** → une `resolution` énorme alourdit Leiden gratuitement.

**Impact** : déni de service applicatif anonyme et bon marché (asymétrie attaquant/serveur
très favorable à l'attaquant).

**Remédiation** :
- Poser au minimum `dependencies=[Depends(rate_limit)]` sur `/recluster` (comme `/todo`),
  idéalement `PROTECTED` si la Console est un outil interne.
- Borner `resolution` (`Field(1.0, gt=0.0, le=5.0)` p.ex.) et envisager un cache LRU
  court par `(dataset, k, threshold, resolution)` pour absorber les rejeux.
- Même traitement pour `/density` (`server.py:505`) et `/analysis` (`server.py:351`) :
  pas de rate-limit aujourd'hui (cf. MOY-3).

#### HAUT-2 — `/todo` POST & PATCH : mutation anonyme d'un fichier versionné, sans borne de taille
**Preuve** : `backend/server.py:187` et `:207` — `dependencies=[Depends(rate_limit)]`
**uniquement** (jamais `require_token`, *par design « outil collaboratif »*). La cible
d'écriture est `todo.json` **à la racine du repo** : `backend/todo_store.py:26-27`
`REPO_ROOT = Path(__file__).resolve().parent.parent ; TODO_PATH = REPO_ROOT / "todo.json"`.
`add_todo` (`todo_store.py:87-110`) ne borne **ni la longueur du titre ni celle de la
note**, et il n'y a **aucun plafond sur le nombre d'items**. Le corps `TodoCreateBody`
(`server.py:180-185`) ne contraint pas non plus les longueurs.

Conséquences pour une expo publique :
1. **Intégrité** : n'importe quel anonyme écrase/pollue un fichier *suivi par git* —
   l'outil de coordination de l'équipe devient une page blanche pour graffitis.
2. **DoS disque / mémoire** : titres/notes de plusieurs Mo × N items → `todo.json`
   grossit sans limite ; chaque écriture est un read-modify-write **de tout le fichier**
   (`_write` → `write_json`, `todo_store.py:82-84`) → coût quadratique au fil du spam.
3. La validation de `lane` est contournable trivialement : `known_lanes()`
   (`todo_store.py:67-73`) **accepte toute lane déjà présente**, donc il suffit d'une lane
   de base (`backend`, etc.) — aucun frein réel.

**Impact** : corruption d'état partagé + DoS disque, le tout anonyme **même quand
`AGORA_API_TOKEN` est défini** (le token n'est jamais exigé ici).

**Remédiation** :
- Pour une expo publique, passer `/todo` POST/PATCH derrière `require_token` (le `/todo`
  GET peut rester ouvert). Si la collaboration ouverte est volontaire, l'isoler derrière
  le reverse-proxy / un réseau de confiance, pas sur l'Internet public.
- Borner les entrées : `title` (p.ex. `max_length=200`), `note` (`max_length=2000`) via
  Pydantic, et plafonner le nombre d'items (rejet 422 au-delà de N).
- Ne pas écrire dans un fichier *versionné* depuis un endpoint exposé : déplacer
  `todo.json` hors de l'arbre git (dossier d'état runtime) pour éviter de salir le repo.

---

### MOYEN

#### MOY-1 — Comparaison de token NON à temps constant (canal temporel)
**Preuve** : `backend/auth.py:44` `if not token or token != API_TOKEN:`. L'opérateur `!=`
sur `str` court-circuite au premier octet différent → le temps de réponse fuit la longueur
du préfixe correct. Sur un secret deviné octet par octet, c'est exploitable en théorie
(en pratique atténué par le bruit réseau et le `rate_limit`, d'où MOYEN et non HAUT).

**Remédiation** : `import hmac` puis
`if not token or not hmac.compare_digest(token, API_TOKEN): raise ...`.
Comparer aussi des `bytes`/`str` de longueurs cohérentes.

#### MOY-2 — CORS : `allow_credentials=True` + piège du wildcard via env
**Preuve** : `backend/server.py:119-125` :
```python
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
```
avec `ALLOWED_ORIGINS` dérivé de `AGORA_ALLOWED_ORIGINS` (`server.py:114-118`). Le défaut
(`localhost:5180`) est sain, mais le réglage « naturel » pour une mise en ligne —
`AGORA_ALLOWED_ORIGINS="*"` — est un **piège** : Starlette, quand `"*" ∈ allow_origins`
**et** `allow_credentials=True`, **renvoie l'Origin du demandeur** + `Access-Control-Allow-
Credentials: true` (vérifié dans la source installée, `starlette/middleware/cors.py:166-171`
`if self.allow_all_origins and self.allow_credentials: self.allow_explicit_origin(...)`).
Résultat : **n'importe quel site** peut faire des requêtes *créditées* cross-origin.
`allow_methods=["*"]` / `allow_headers=["*"]` élargissent encore la surface.

Aujourd'hui l'auth est *header-based* (pas de cookie), donc l'exfiltration via CORS crédité
est limitée — mais `allow_credentials=True` est **inutile** dans ce modèle et n'existe que
pour devenir dangereux le jour où un cookie/session apparaît.

**Remédiation** :
- Mettre `allow_credentials=False` (l'app n'utilise pas de cookies) — supprime le piège.
- Refuser explicitement `"*"` dans le parseur d'origines (rejeter ou logger un WARNING si
  `"*"` est fourni avec credentials), ou exiger une liste blanche stricte.
- Restreindre `allow_methods` aux verbes réellement utilisés (`GET, POST, PATCH, DELETE`).

#### MOY-3 — `/analysis` & `/density` : coûteux/effets de bord, sans rate-limit
**Preuve** : `server.py:351` (`/analysis`) et `server.py:505` (`/density`) — aucune
dépendance. `/analysis` peut déclencher un build de fond via `_not_ready_response`
(`server.py:322-333` → `build_manager.ensure_build`) et exécute `serve_metrics.enrich_indices`
à chaque appel ; `/density` calcule une **projection UMAP complète** au 1ᵉʳ appel
(`density.density_payload`). Spammables anonymement.

**Remédiation** : poser `dependencies=[Depends(rate_limit)]` sur ces deux endpoints (et
sur `/recluster`, cf. HAUT-1). Lecture pure du cache OK sans token ; le rate-limit suffit.

#### MOY-4 — Rate-limit inopérant derrière un reverse-proxy + fuite mémoire `_hits`
**Preuve** : `backend/auth.py:55` `ip = request.client.host if request.client else "?"`.
Aucune prise en compte de `X-Forwarded-For`. Derrière un reverse-proxy (déploiement
attendu, cf. docstring `auth.py:13`), **toutes** les requêtes portent l'IP du proxy →
un **seul** seau partagé : soit on bloque collectivement les usagers légitimes
(faux positifs), soit la limite par-IP n'a plus de sens. Par ailleurs `_hits`
(`auth.py:50`, `defaultdict(deque)`) n'est **jamais purgé** des IP inactives : les deques
vides s'accumulent (fuite mémoire lente ; surface d'amplification si l'IP source varie,
ex. pool IPv6).

**Remédiation** :
- Dériver l'IP client de `X-Forwarded-For` **uniquement** si la requête vient d'un proxy
  de confiance (liste blanche d'IP amont), sinon `request.client.host`.
- Purger périodiquement les seaux vides (ou utiliser un store TTL / `cachetools.TTLCache`).
- Idéalement, déléguer le rate-limiting au reverse-proxy (nginx `limit_req`) pour une
  limite robuste avant d'atteindre l'app.

---

### BAS

#### BAS-1 — Fuite d'information : `/_resolve` renvoie la liste des datasets
**Preuve** : `backend/server.py:100-103` — un dataset inconnu renvoie
`detail=f"dataset inconnu: {ds!r} (disponibles: {_ids})"`, exposant l'inventaire interne
des datasets à un anonyme. Mineur (les ids sont aussi listés par `/datasets`), mais inutile
sur des erreurs.

**Remédiation** : message générique `"dataset inconnu"` sans énumérer `_ids`.

#### BAS-2 — Longueurs non bornées sur `/flag` (et croissance `submissions.jsonl`)
**Preuve** : `FlagBody.text` (`server.py:607`) sans `max_length` → `flags.json` peut enfler
(`flags_store.upsert_flag`, `flags_store.py:85-113`). Atténué : `/flag` est sous `PROTECTED`,
donc fermé tant que CRIT-1 n'est pas réalisé. De même `append_submission`
(`submissions.py:74-80`) est append-only sans plafond du nombre de contributions par
consultation (DoS disque lent ; `/submit` est `PROTECTED` + borné en taille de texte 3–5000,
`server.py:230-231`).

**Remédiation** : borner `FlagBody.text` (`max_length=2000`), plafonner le nombre de
contributions/flags par cible, et purger/roter les stores `*.jsonl`.

---

## Points vérifiés SAINS (non-findings, pour traçabilité)

- **Path-traversal datasets** : `_resolve` (`server.py:91-107`) garde une whitelist O(1)
  `_IDSET` découverte au boot ; tout id hors-liste → 404 **avant** tout accès disque.
  Couvert par `tests/test_security_whitelist.py` (`../etc`, `../../etc/passwd`, etc.).
  `/submit` valide `consultation_id` contre `list_open_consultations()` (`server.py:254`).
  `analysis_store._safe` (`analysis_store.py:85-87`) durcit en plus les noms de fichiers
  dérivés de `theme_id`.
- **Écritures atomiques** : `write_json` (`analysis_store.py:111-121`, temp+`os.replace`)
  empêche un GET concurrent de lire un JSON à moitié écrit ; même schéma dans `flags_store`.
- **Lectures tolérantes** : `read_todo`/`_read_json` n'exceptionnent pas sur fichier
  corrompu (pas de 500 exploitable).
- **`/submit`** : bornes de texte présentes (3–5000 car., `server.py:230-231,259-266`) ;
  embedding **local** (aucune clé/LLM réseau) ; whitelist consultation.
- **Pas de cookie/session** : l'auth est par header (`X-API-Token`/`Bearer`), donc la
  surface CSRF est faible *aujourd'hui* (à préserver — cf. MOY-2).

---

## Plan d'action recommandé (ordre)

1. **CRIT-1** — Mode auth strict (fail-closed) : démarrage qui échoue sans token en prod.
2. **HAUT-1 / HAUT-2** — Poser `rate_limit` sur `/recluster` (+`/density`,`/analysis`) ;
   fermer ou borner `/todo` POST/PATCH (token + `max_length` + plafond d'items).
3. **MOY-2** — `allow_credentials=False`, rejet du wildcard, `allow_methods` restreint.
4. **MOY-1** — `hmac.compare_digest` pour le token.
5. **MOY-4 / MOY-3 / BAS-*** — IP réelle derrière proxy + purge `_hits`, rate-limit lecture,
   bornes de longueur, message d'erreur générique.

*Aucune logique applicative n'a été modifiée dans le cadre de cet audit — rapport seul.*
