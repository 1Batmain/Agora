# Nommage LLM (Mistral) + Synthèse — note technique

Le nommage `llm` des clusters et l'endpoint `/synthesize` passent désormais par
l'**API Mistral** (EU, souverain au sens RGPD/UE). L'ancien backend **Ollama
local est abandonné** (le VPS de déploiement ne peut pas l'exécuter). Tout est
générique, langue-agnostique, zéro hardcoding de domaine ; **repli gracieux** si
la clé manque ou si l'API échoue.

## Clé API — résolution (jamais commit, jamais loggée)

`pipeline/cluster/mistral_client.load_api_key()` cherche, premier trouvé gagnant :

1. variable d'environnement **`MISTRAL_API_KEY`** ;
2. **`backend/.env`** — ligne `MISTRAL_API_KEY=...` (fichier gitignoré) ;
3. **`var/mistral.key`** à la racine du repo — fichier brut (gitignoré).

La clé n'est **jamais** écrite dans le code, ni loggée, ni renvoyée dans une
erreur. `.gitignore` couvre `.env`, `var/`, `*.key`.

Lancer le backend avec la clé :

```bash
export MISTRAL_API_KEY="$(cat var/mistral.key)"   # ou un backend/.env
uv run --extra contender --with fastapi --with uvicorn \
    uvicorn backend.server:app --host 0.0.0.0 --port 8010
```

## Variables d'environnement (toutes surchargeables)

| Env | Défaut | Rôle |
|-----|--------|------|
| `MISTRAL_API_KEY` | — | clé API (obligatoire pour `llm`/synthèse) |
| `AGORA_MISTRAL_URL` | `https://api.mistral.ai/v1/chat/completions` | endpoint |
| `AGORA_MISTRAL_MODEL` | `mistral-small-latest` | modèle de **nommage** |
| `AGORA_MISTRAL_SYNTH_MODEL` | = modèle nommage | modèle de **synthèse** (p.ex. `mistral-large-latest`) |
| `AGORA_MISTRAL_TIMEOUT` | `60` (s) | timeout par appel |

## 1) Nommage `llm` — batché

`pipeline/cluster/naming_methods.py::_name_llm`. **UN seul appel** Mistral pour
tous les clusters (`response_format={"type":"json_object"}`). Entrée par cluster :
keywords c-TF-IDF + jusqu'à 3 témoignages représentatifs (médoïdes, proches du
centroïde). Prompt langue-agnostique : titre court (≤ 6 mots) dans la langue
dominante de **chaque** cluster.

Format de réponse attendu : `{"<cluster_id>": "titre court", ...}` (parsing
tolérant : retire un éventuel bloc ```json, isole le premier objet `{...}`).

`meta.naming_meta` trace : `naming` (méthode réelle), `requested`, `fallback`,
`reason`, `model`, `n_llm`, `n_fallback`, `took_ms`.

**Repli** : pas de clé → `ctfidf` (`reason=no_api_key`) ; erreur/timeout API →
`ctfidf` (`reason=api_error:<status>:<msg>`) ; cluster sans titre exploitable →
repli ctfidf de CE cluster (`partial_fallback`).

## 2) Synthèse — `POST /synthesize`

`backend/synthesize.py`. Body `{dataset?, method?, naming?}`. Réutilise
`recluster` puis bâtit un **résumé compact de tous les macro-thèmes** (label,
keywords, taille/poids, cohérence/diversité, 2 témoignages médoïdes par thème ;
exclut le bruit HDBSCAN `cluster_id=-1`, cap à 40 thèmes). **UN appel** Mistral
→ rapport Markdown court en deux parties imposées :

- `## Synthèse` — grands thèmes de la parole citoyenne ;
- `## Pertinence du découpage` — cohérence, redondances, couverture, clusters
  douteux.

Rédigé dans la **langue dominante** (descripteur dataset, sinon dérivée des
nœuds). Réponse : `{ report_markdown, meta:{ model, took_ms, n_clusters,
truncated, fallback?, reason?, lang? } }`.

**Repli** : pas de clé → `report_markdown` = « Synthèse indisponible : clé
Mistral manquante » + `meta.fallback=true` (HTTP 200, pas un crash). Erreur API
→ message « l'appel à Mistral a échoué (statut N) » + `meta.fallback=true`.

## Front

- Picker de nommage : option **« LLM (Mistral) »** → `/recluster {naming:"llm"}`
  (contrat inchangé). Repli signalé discrètement sous le toggle.
- **Panneau « Synthèse »** (`frontend/src/SynthesisPanel.tsx`) : repliable, bouton
  « Générer la synthèse » → `POST /api/synthesize` → rendu Markdown (mini-renderer
  maison, sans dépendance). État de chargement, métadonnées (modèle, durée).

## Sortie de données (à noter)

Le nommage `llm` et la synthèse envoient des **résumés de clusters + échantillons
de témoignages** (déjà anonymisés) à l'API Mistral (EU). Choix produit assumé
(VPS sans LLM local). Aucune autre donnée ne quitte le serveur.

## Validé sans vraie clé / reste à valider

Validé (clé absente ou bidon) :

- **Structure de requête OK** : un appel avec une clé bidon renvoie **401** de
  Mistral (et non une erreur de code) → la requête est bien formée.
- **Repli gracieux** : sans clé, `naming:"llm"` → `ctfidf` (`fallback=true`,
  `reason=no_api_key`) ; `/synthesize` → message « clé manquante », pas de crash.
- **Build front** : `npm run build` OK.

À valider avec la **vraie clé** (architecte) :

- qualité réelle des titres batchés (langue, longueur ≤ 6 mots, pertinence) ;
- qualité/longueur du rapport de synthèse (FR + corpus multilingue p.ex.
  `xstance`) ;
- latence bout-à-bout (un appel batché nommage ; un appel synthèse) ;
- choix éventuel d'un modèle de synthèse plus fort
  (`AGORA_MISTRAL_SYNTH_MODEL=mistral-large-latest`).
