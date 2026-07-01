# Audit sécurité — Secrets & chaîne d'approvisionnement

**Périmètre** : Agora (audit PRO avant passage PUBLIC). Lane `work/sec-secrets`.
**Méthode** : adversaire & exhaustive. Historique git complet (518 commits, **toutes branches** : `agora`, `main`, 70+ `work/*`), arbre de travail, caches committés, bundle frontend, dépendances Python (`uv.lock`, 55 paquets) et npm (`frontend/package-lock.json`).
**Date** : 2026-06-30. **Contrainte respectée** : aucune modification de la logique applicative.

---

## Verdict synthétique

**Aucune fuite de secret détectée** — ni dans l'arbre courant, ni dans l'historique complet, ni dans aucune branche. L'hygiène des secrets est **forte et délibérée** (clé jamais loggée/codée en dur, URL réseau privé masquée, `var/`+`*.key`+`.env` ignorés et **jamais committés en 518 révisions**).

Les findings restants concernent la **chaîne d'approvisionnement frontend** (outils de build dev vulnérables) et un **durcissement** (comparaison de token non constante en temps). Rien de CRITIQUE ou HAUT au sens « secret exposé publiquement ».

| Priorité | # | Sujet |
|----------|---|-------|
| CRITIQUE | 0 | — |
| HAUT     | 0 | — |
| MOYEN    | 2 | Vulns build frontend (vite/esbuild) ; comparaison de token non constante |
| BAS      | 3 | Durcissement gitignore `.cache` ; `print` d'URL de téléchargement ; détails d'erreur `str(e)` |
| INFO     | 4 | Constats positifs (preuves de non-fuite) |

---

## MOYEN

### M1 — Dépendances de build frontend vulnérables (`vite` ≤6.4.2, `esbuild` ≤0.24.2)

**Preuve** : `frontend/package.json:devDependencies` → `"vite": "^5.4.10"`. `npm audit` :
- **vite** (HIGH advisory) — *Path Traversal in Optimized Deps `.map` Handling* + `launch-editor` NTLMv2 hash disclosure (UNC, Windows). GHSA côté serveur de dev.
- **esbuild** (MODERATE, transitif via vite) — *enables any website to send any requests to the development server and read the response* (SSRF du serveur de dev), GHSA-67mh-4wv8-2f99.

**Analyse de surface (adversaire mais honnête)** : ces deux paquets sont des **`devDependencies`** — ils n'apparaissent **PAS** dans le bundle de production (`react`, `react-dom`, `three`, `d3-*` sont les seules `dependencies`). Le bundle `frontend/dist` **n'est pas committé** (vérifié : `git ls-files frontend/dist` = vide), donc aucun artefact vulnérable n'est publié. L'exposition réelle se limite au **poste du développeur lançant `vite dev`** (un site web malveillant ouvert en parallèle peut lire des réponses du serveur de dev local). Sévérité rétrogradée HIGH→**MOYEN** car hors artefact de production.

**Remédiation** :
```bash
cd frontend && npm install -D vite@^6.5.0   # corrige vite + esbuild transitif
npm audit            # doit retomber à 0
```
Vérifier que le passage de major (5→6) ne casse pas le build (`npm run build`). Si rétro-compat impossible avant la deadline : documenter que `vite dev` ne doit pas tourner avec des onglets non fiables ouverts ; le **build de prod n'est pas affecté**.

### M2 — Comparaison du token API non constante en temps (`backend/auth.py:73`)

**Preuve** : `backend/auth.py:73` → `if not token or token != API_TOKEN:`. La comparaison `!=` de chaînes Python court-circuite au premier octet différent → **canal auxiliaire temporel** permettant, en théorie, de reconstituer `AGORA_API_TOKEN` octet par octet par mesure de latence.

**Analyse** : recoupe la lane `work/sec-auth` (porte d'authentification), mais relève aussi de la **protection d'un secret** (le token lui-même), d'où sa présence ici. Exploitation réaliste rendue difficile par le bruit réseau + le rate-limit (`backend/auth.py:48`, 30 req/60 s par IP), mais le durcissement est trivial et sans risque.

**Remédiation** (changement local, hors logique métier — à coordonner avec sec-auth) :
```python
import hmac
if not token or not hmac.compare_digest(token, API_TOKEN):
    raise HTTPException(status_code=401, detail="Token API invalide ou manquant.")
```

---

## BAS

### B1 — `.cache` couverte mais non explicite dans `.gitignore`

**Preuve** : `pipeline/claims/.cache/ollama/` (cache LLM local) est bien ignoré (`git check-ignore` = IGNORED) et **n'a jamais été committé**. Mais `.gitignore` ne contient **pas** de règle `.cache/` explicite — la couverture actuelle est incidentelle. Le contenu écrit (`pipeline/claims/ollama.py:140`) ne stocke que `{content, seconds, eval_count}` — **pas** la `base_url` du tailnet (confirmé). Risque faible, mais ces caches contiennent de la **sortie LLM sur données citoyennes**.

**Remédiation** : ajouter une ligne défensive à `.gitignore` :
```gitignore
# Caches LLM locaux (sortie sur données citoyennes) — jamais versionnés
**/.cache/
```

### B2 — `print` de l'URL de téléchargement (`pipeline/ingest/download.py:33`)

**Preuve** : `pipeline/ingest/download.py:33` → `print(f"  [get ] {url}")`. Ces URL proviennent des **descripteurs de datasets publics** (jeux de données officiels téléchargés), **pas** d'un endpoint réseau privé. L'URL Ollama/tailnet (`AGORA_OLLAMA_URL`) n'est **jamais** loggée — elle est systématiquement masquée par `_redact()` (`pipeline/claims/ollama.py:43`, `_redact(self.base_url)` aux lignes 108/134). Aucun secret exposé ; mention pour exhaustivité.

**Remédiation** : aucune nécessaire. Conserver la discipline `_redact` pour toute future URL réseau privé.

### B3 — Détails d'exception renvoyés en clair dans quelques réponses API (`backend/server.py:197,217,522`)

**Preuve** : `detail=str(e)` / `detail=str(exc)` sur `ValueError` (todo_store, lignes 197/217) et `DensityUnavailable` (ligne 522). Vérifié : ces exceptions portent des **messages contrôlés** (validation de saisie, projection UMAP indisponible) — **pas** de chemin disque, pas de trace, pas de secret. Aucun `str(FileNotFoundError)`/traceback n'atteint le client. Risque résiduel : un futur refactor pourrait y faire transiter une exception plus bavarde.

**Remédiation** : à terme, mapper les exceptions internes vers des messages fixes plutôt que `str(exc)` brut. Non bloquant pour le partage public.

---

## INFO — Constats positifs (preuves de NON-fuite)

Documentés pour la revue : ce qui a été activement vérifié et trouvé **propre**.

### I1 — Secrets jamais committés (historique complet, toutes branches)
- `git log --all --diff-filter=A --name-only` sur **518 commits / toutes branches** : **aucun** `.env`, `var/`, `*.key`, `mistral.key`, `MAC_LOCAL_OLLAMA`, `.pem`, `id_rsa`, `credentials`, `*.secret` jamais ajouté.
- `.gitignore:12-13` ignore `var/` et `*.key` ; `.gitignore:23` ignore `.env`. **Cohérent et effectif.**
- Recherche de valeurs codées en dur (`(api_key|secret|salt|token|password)= "..."` haute-entropie) sur tout l'historique : **zéro** correspondance.

### I2 — Clé Mistral : lue, jamais loggée, jamais en erreur
- `pipeline/cluster/mistral_client.py:58-82` : résolution env → `backend/.env` → `var/mistral.key`. Jamais codée en dur.
- `MistralError` (`:90-99`) et `_safe_reason` (`:102-109`) : message ≤200 car, **jamais** la clé ni l'`Authorization`. Erreur réseau réduite au **type** d'exception (`:153`), pas d'URL/headers.

### I3 — URL réseau privé (tailnet) masquée et jamais committée
- Recherche `ts.net|tailscale|:11434|/home/|/Users/` sur l'arbre committé : seules occurrences = **commentaires** (`pipeline/claims/ollama.py:5,28`, `backend.py:10`). **Aucune** URL/host tailnet réel committé.
- `_redact()` masque le host dans tous les logs d'erreur Ollama.

### I4 — Frontend & caches propres
- **Aucune** variable `VITE_` interne ni URL backend en dur dans `frontend/src` (seul `VITE_FORCE_MOCK`, un flag de dev). Aucun `.env` frontend tracké. `frontend/dist` non committé.
- Caches committés d'**entrée** (`backend/cache/*/{meta.json,ideas.jsonl,embeddings.npy}`) : `meta.json` contient des **métadonnées de build** (modèle, dims, langues) — **aucun** secret/URL/chemin. Scan d'entropie des fichiers texte trackés : seules « chaînes longues » = fragments d'URL **publiques** dans le texte citoyen (pas des secrets). *(Note : `ideas.jsonl` contient du texte citoyen → relève de la lane `work/sec-privacy`, hors périmètre secrets.)*

### I5 — Chaîne d'appro Python saine
- `uv.lock` (55 paquets) : **aucune** source non officielle hormis `https://download.pytorch.org/whl/cpu` (index PyTorch **officiel**, déclaré explicitement dans `pyproject.toml:54-60`). Pas de `git+`, pas de typosquat repéré.
- Versions récentes, sans CVE notoire ouverte : `transformers 5.12.1`, `torch 2.12.1`, `httpx 0.28.1`, `numpy 2.4.6`, `scikit-learn 1.9.0`, `jinja2 3.1.6`, `certifi 2026.6.17`.
- **`trust_remote_code=True`** (modèle `nomic-embed-text-v2-moe`, exécute du code amont au chargement) est **épinglé** à une révision figée (`pipeline/embed/registry.py:64`, `revision="1066b6599..."`) → aucun push amont ne peut altérer le code exécuté. Bonne hygiène supply-chain.

---

## Récapitulatif des actions recommandées (par effort croissant)

1. **B1** (1 ligne) : ajouter `**/.cache/` à `.gitignore`. *Trivial, défensif.*
2. **M1** (build) : `npm install -D vite@^6` + `npm run build` + `npm audit`. *À valider avant publication ; n'affecte pas le bundle de prod.*
3. **M2** (coord. sec-auth) : `hmac.compare_digest` pour la comparaison de token.
4. **B3** (refactor à terme) : messages d'erreur fixes au lieu de `str(exc)`.

**Bloquant pour le partage public** : **aucun** — sous réserve de M1 traité (ou documenté comme dev-only) avant publication du dépôt.
