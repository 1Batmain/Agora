# Audit sécurité — Vie privée / PII & infra/erreurs

- **Périmètre** : backend FastAPI `:8010` (SERVE) + pipeline d'ingestion/anonymisation + caches committés, **avant passage du dépôt en PUBLIC**.
- **Posture** : adversaire, exhaustif. Aucune logique applicative modifiée (rapport seul).
- **Branche** : `work/sec-privacy` · **Date** : 2026-06-30.
- **Méthode** : revue statique des endpoints servant du texte citoyen (`/avis`, `/avis_list`, `/citations`, `/insights`, `/submit`), du masquage PII (`pipeline.ingest.normalize`), du hash auteur (`pipeline.ingest.anonymize`), de l'auth/rate-limit (`backend.auth`), de la gestion d'erreur et des artefacts committés (`backend/cache/**`).

## Synthèse exécutive

| # | Sévérité | Titre | Preuve |
|---|----------|-------|--------|
| 1 | **CRITIQUE** | `/submit` stocke le texte citoyen NON masqué et **renvoie verbatim l'avis d'un autre citoyen** (`nearest_excerpt`) | `backend/server.py:259`, `backend/submissions.py:74`, `:120` |
| 2 | **HAUT** | Texte BRUT non masqué + `author_hash` **committés** dans le dépôt (bientôt public), de façon permanente dans l'historique git | `backend/cache/*/ideas.jsonl` (champ `text`), tracké git |
| 3 | **HAUT** | Masquage PII **regex-only** : noms, adresses postales, âges, signatures NON masqués — alors que `text_clean` est servi publiquement | `pipeline/ingest/normalize.py:12-30` |
| 4 | **HAUT** | Bypass d'auth de fait : `AGORA_API_TOKEN` non posé par défaut **et** le front n'envoie aucun token → `/build` `/flag` `/submit` `/todo` ouverts (DoS coût LLM) | `backend/auth.py:33-40`, `frontend/src/api.ts:42` |
| 5 | MOYEN | Message d'exception brut (`str(exc)`) divulgué aux clients **non authentifiés** (`/build_status`, `/analysis` 503) | `backend/build_analysis.py:265`, `backend/analysis_store.py:163` |
| 6 | MOYEN | Aucun en-tête de sécurité HTTP (X-Content-Type-Options, X-Frame-Options, CSP, HSTS, Referrer-Policy) | `backend/server.py:119` |
| 7 | MOYEN | `/todo` POST/PATCH non authentifié → écriture/altération arbitraire de `todo.json` à la racine du repo | `backend/server.py:187`, `:207`, `backend/todo_store.py:27` |
| 8 | MOYEN | `author_hash` committé permet la **corrélation / ré-identification par recoupement** (lien inter-contributions d'un même auteur) | `backend/cache/*/ideas.jsonl`, `pipeline/ingest/anonymize.py:19` |
| 9 | BAS | `/docs` & `/openapi.json` exposés (divulgation du schéma d'API) | `backend/server.py:110` |
| 10 | BAS | Comparaison de token non constante en temps (timing side-channel) | `backend/auth.py:44` |
| 11 | BAS | CORS `allow_credentials=True` + méthodes/headers `*` (fragile si `AGORA_ALLOWED_ORIGINS` mal réglé) | `backend/server.py:119-125` |
| 12 | BAS | Rate-limit sur `request.client.host` (pas de `X-Forwarded-For`), en mémoire, non partagé entre workers | `backend/auth.py:53-65` |

**Points positifs vérifiés** : path-traversal correctement gardé (whitelist `_IDSET` sur `dataset` + `_safe()` qui n'autorise que `[A-Za-z0-9_-]` sur `theme_id`/`level`, `backend/analysis_store.py:84-97`) ; `author_hash` **n'est servi par aucun endpoint** (grep backend+front = 0) ; le sel d'anonymisation n'a **aucun défaut** et est validé `≥32` à l'ingestion (`pipeline/ingest/config.py:40`) ; aucune clé/secret committé (`var/`, `*.key` gitignorés) ; FastAPI sans `debug=True` (pas de stack-trace dans le corps des 500) ; submissions live (`submissions.jsonl`) gitignorées.

---

## CRITIQUE

### 1. `/submit` : stockage non masqué + fuite croisée de PII verbatim entre citoyens

**Fichiers** : `backend/server.py:259-272`, `backend/submissions.py:74-80`, `:100-122`.

Le flux `/submit` (consultations OUVERTES) ne fait que `strip()` le texte — **aucun appel** à `normalize.strip_pii` / `clean_text` (vérifié : `grep strip_pii|clean_text|normalize backend/server.py backend/submissions.py` = vide) :

```python
# server.py:259
text = (body.text or "").strip()
...
vec = submissions.embed_text(text)
existing = submissions.load_submissions(body.consultation_id)
corr = submissions.correlate(vec, existing)
submissions.append_submission(body.consultation_id, text, vec, ts)  # texte BRUT persisté
```

Et `correlate` **renvoie verbatim le texte d'une autre contribution** :

```python
# submissions.py:120
"nearest_excerpt": existing[best].get("text"),   # texte BRUT d'un AUTRE citoyen
```

**Impact** : un citoyen qui saisit un email, un numéro de téléphone, son nom complet ou une adresse voit ces données (a) **persistées en clair** dans `submissions.jsonl`, et (b) **réémises mot pour mot à un autre citoyen** dès qu'une contribution sémantiquement proche arrive (`nearest_excerpt`). C'est une divulgation de PII inter-utilisateurs sur un endpoint vivant, aggravée par le bypass d'auth (finding #4) qui le rend public.

**Remédiation** :
1. Masquer **avant** embed/stockage : `text = normalize.clean_text(body.text)` (réutilise `strip_pii`).
2. Masquer aussi la sortie : `nearest_excerpt = normalize.strip_pii(existing[best]["text"])` (défense en profondeur pour le seed et l'historique déjà stocké).
3. Idéalement, ne jamais renvoyer le texte intégral d'autrui : tronquer en aperçu court (≤120 c.) après masquage.

---

## HAUT

### 2. Texte brut non masqué + `author_hash` committés dans un dépôt bientôt public

**Fichiers** : `backend/cache/{tiktok,granddebat,republique-numerique,xstance}/ideas.jsonl` (trackés git).

Chaque ligne committée contient **le texte source BRUT** (`text`) à côté de `text_clean` masqué, **plus** `author_hash` :

```json
{"id":"tiktok:532","text":"...vidéo Youtube ... (https://www.youtube.com/watch?v=...) ...",
 "text_clean":"...vidéo Youtube ... ([url] ...","author_hash":"7b1ca42e0ad8ddbd", ...}
```

Preuve concrète : sur `tiktok`, **699/1621** lignes ont `text != text_clean`, et au moins 1 (`tiktok:532`) conserve dans `text` une URL que `text_clean` a masquée en `[url]`. Le masquage est donc **annulé** par la présence du champ brut, et ce de manière **permanente dans l'historique git** une fois le dépôt public.

**Impact** : publier le dépôt publie l'intégralité des avis citoyens d'origine + un pseudonyme stable par auteur. Tout ce que le pipeline d'anonymisation retire est restauré par le champ `text`.

**Remédiation** :
- Régénérer les caches committés **sans** le champ `text` brut (ne committer que `text_clean`, `lang`, `id`, `weight`) ; le serveur n'utilise le brut qu'en repli si `text_clean` est vide (`backend/claims_endpoint.py:57`), repli évitable en garantissant `text_clean` non vide.
- Si l'historique contient déjà du brut sensible : réécrire l'historique (git filter-repo) **avant** de rendre public, sinon le passé reste exposé.
- Décider explicitement si `author_hash` doit être committé (cf. #8).

### 3. Masquage PII regex-only — noms/adresses/signatures non couverts

**Fichier** : `pipeline/ingest/normalize.py:12-30`.

`strip_pii` ne masque que 4 motifs : email, téléphone (`≥10` chiffres), URL, `@mention`. Les PII les plus fréquentes en consultation citoyenne FR — **nom/prénom, adresse postale, âge, profession, signatures (« Cordialement, Jean D. »), mentions de tiers** — ne sont **pas** masquées. Or `text_clean` est servi tel quel par `/avis`, `/avis_list`, `/citations` (et committé, cf. #2). Le regex téléphone (`:14`) a aussi des faux positifs (dates/IDs longs).

**Impact** : la promesse « on ne sert JAMAIS la PII brute (SEC3) » (`backend/avis.py:20`) est partielle — elle ne tient que pour 4 motifs structurés. Du texte libre identifiant transite vers le public.

**Remédiation** :
- Documenter honnêtement la portée du masquage (ne couvre PAS les noms en clair).
- Pour un usage public, ajouter une passe NER (ex. spaCy `fr_core_news` PER/LOC, ou un masquage Presidio) sur `text_clean` à l'ingestion ; au minimum, un avertissement + revue manuelle des datasets publiés.
- Élargir le regex téléphone avec contexte (préfixes FR) pour réduire les faux positifs sans rater les vrais numéros.

### 4. Bypass d'authentification de fait sur les endpoints mutatifs/coûteux

**Fichiers** : `backend/auth.py:33-45`, `frontend/src/api.ts:42`, `backend/server.py:240,563,634`.

`require_token` **laisse passer** si `AGORA_API_TOKEN` est absent (mode dev, simple `print` d'avertissement, `auth.py:33-40`). Or le front **n'envoie jamais** de token (`api.ts` ne pose que `Content-Type`, aucun `X-API-Token`/`Authorization`). Donc soit le déploiement tourne **sans** token (endpoints ouverts), soit poser le token **casse** le front. En pratique : `/build`, `/flag`, `/submit`, `/todo` exposés à Internet, gardés seulement par le rate-limit (#12).

**Impact** : `/build` déclenche le pipeline lourd (embeddings + appels LLM) → **DoS sur la facture LLM** et la CPU par un anonyme. `/submit`/`/flag`/`/todo` permettent l'injection de contenu non sollicité.

**Remédiation** :
- Décider du modèle de confiance : si ces routes doivent rester publiques, retirer la fausse protection et durcir le rate-limit + plafonds de coût ; sinon, placer un reverse-proxy authentifiant **devant** et exiger `AGORA_API_TOKEN` (échec dur si absent en prod, pas un simple `print`).
- Ne jamais embarquer le token dans une SPA (il serait public) : auth côté proxy, pas côté navigateur.

---

## MOYEN

### 5. Divulgation de messages d'exception internes aux clients non authentifiés

**Fichiers** : `backend/build_analysis.py:264-265`, `backend/build_manager.py:113-114`, `backend/analysis_store.py:154-164`, `backend/server.py:322-333,577-585`.

En cas d'échec de build, l'exception est stockée brute : `error=str(exc)`. `analysis_store.progress()` la replace dans le payload (`:163`), renvoyé par `/build_status` (GET, **non protégé**) et par le chemin 503 de `/analysis`, `/insights`, `/citations`. `str(exc)` peut contenir des chemins disque absolus, des noms de fichiers internes, des détails d'API (clé/modèle Mistral), etc.

**Remédiation** : renvoyer un message générique au client (`detail="échec du build"` existe déjà) et **ne pas** propager `error=str(exc)` ; logguer le détail côté serveur uniquement. Retirer `error` du dict `progress()` ou le réserver à un endpoint protégé.

### 6. Aucun en-tête de sécurité HTTP

**Fichier** : `backend/server.py:119` (seul `CORSMiddleware`).

Aucun `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy`, `Content-Security-Policy`, ni `Strict-Transport-Security`. Le front sert du Markdown/HTML d'insights → surface de clickjacking / sniffing MIME.

**Remédiation** : middleware ajoutant ces en-têtes (au minimum `nosniff`, `X-Frame-Options`, `Referrer-Policy: no-referrer`, HSTS derrière TLS, CSP adaptée au front).

### 7. `/todo` POST/PATCH non authentifié → écriture arbitraire sur disque

**Fichiers** : `backend/server.py:187,207`, `backend/todo_store.py:26-27`.

`POST /todo` et `PATCH /todo/{id}` ne portent que `rate_limit` (pas `require_token`) et écrivent `todo.json` à la **racine du repo**. Un anonyme peut polluer / défigurer la feuille de route servie par `/todo`.

**Remédiation** : protéger par `require_token` (comme `/flag`) ou cantonner l'écriture à un store hors-repo ; valider/plafonner la taille des champs.

### 8. `author_hash` committé → ré-identification par recoupement

**Fichiers** : `backend/cache/*/ideas.jsonl`, `pipeline/ingest/anonymize.py:9-19`, `pipeline/ingest/config.py:37`.

Le hash est salé (sel secret `≥32`) et tronqué à 16 hex (64 bits) — **non réversible** tant que le sel reste secret. Mais committer le hash le rend **stable et public** : toutes les contributions d'un même auteur partagent le même pseudonyme → regroupement/profilage par recoupement (date, sujets, style), et ré-identification possible si le sel fuite un jour (le sel signe tout l'historique committé). C'est un pseudonyme, pas une anonymisation, au sens RGPD.

**Remédiation** : ne pas committer `author_hash` dans les caches publics si la traçabilité par auteur n'est pas un besoin servi ; sinon, re-saler par dataset et documenter le risque de linkage. Conserver le sel hors de tout dépôt, à rotation.

---

## BAS

### 9. `/docs` & `/openapi.json` exposés
`FastAPI(...)` sans `docs_url=None, redoc_url=None, openapi_url=None` (`server.py:110`) → schéma d'API public. **Remédiation** : désactiver en prod ou placer derrière auth.

### 10. Comparaison de token non constante en temps
`token != API_TOKEN` (`auth.py:44`) est sensible au timing. **Remédiation** : `hmac.compare_digest(token, API_TOKEN)`.

### 11. CORS `allow_credentials=True` + `allow_methods/headers=["*"]`
Correct avec origines fixes (`server.py:114-125`), mais fragile : si un opérateur met `AGORA_ALLOWED_ORIGINS="*"`, le combo credentials+wildcard est invalide/risqué. **Remédiation** : refuser `*` explicitement quand `allow_credentials=True`, restreindre les méthodes réellement utilisées.

### 12. Rate-limit naïf
`request.client.host` sans prise en compte de `X-Forwarded-For` (`auth.py:55`) : derrière un reverse-proxy, tous les clients partagent l'IP du proxy (sur-throttle) ; en mémoire et par-process (non partagé entre workers, remis à zéro au redémarrage) → contournable par rotation d'IP / redémarrage. **Remédiation** : dériver l'IP du `X-Forwarded-For` de confiance, store partagé (Redis) si multi-worker, plafonds par endpoint coûteux.

---

## Réponses directes au brief

- **Seuls les `text_clean` (masqués) sont servis ?** Côté endpoints SERVE (`/avis`, `/avis_list`, `/citations`) : **oui** (ancrage `text_clean`, repli brut seulement si `text_clean` vide). **MAIS** : (a) `/submit` sert/stocke du brut non masqué (#1) ; (b) le dépôt committé contient le champ `text` brut (#2) ; (c) le masquage lui-même est superficiel (#3).
- **`author_hash` réversible / métadonnées identifiantes ?** Non réversible sans le sel, mais committé → linkage/profilage (#8). Non servi par les endpoints (bon).
- **`AGORA_HASH_SALT` vrai secret ≥32, jamais en dur ?** **Oui** : aucun défaut, validé `≥32` à l'ingestion, jamais committé. Robuste.
- **Masquage PII appliqué partout ?** **Non** : absent de `/submit` (#1), regex-only (#3), contourné par le champ brut committé (#2).
- **Mode debug FastAPI off ?** Oui (pas de `debug=True` ; pas de stack-trace dans le corps des 500).
- **Les erreurs fuient-elles des détails internes ?** **Oui** via `error=str(exc)` exposé sans auth (#5).
- **En-têtes de sécurité ?** **Absents** (#6). **Verbosité logs** : `str(exc)` en clair côté status (#5) ; warning auth en clair (acceptable).

## Priorisation de remédiation
1. **#1** (CRITIQUE) — masquer `/submit` (entrée + `nearest_excerpt`) : correctif localisé, fort impact.
2. **#2 / #3** (HAUT) — re-générer les caches sans champ brut + décider de la portée du masquage **avant** de rendre public (et nettoyer l'historique git).
3. **#4** (HAUT) — clarifier le modèle d'auth (proxy authentifiant + plafonds coût LLM).
4. **#5–#8** (MOYEN) — message d'erreur générique, en-têtes de sécurité, protéger `/todo`, statuer sur `author_hash` committé.
5. **#9–#12** (BAS) — durcissements défensifs.
