# Nommage switchable des thèmes — c-TF-IDF / Centroïde / LLM local

Le **nommage** des clusters est désormais un axe orthogonal au *clustering*
(`method` = leiden/hdbscan) et au *dataset*. Param **`naming`** sur
`POST /recluster` (défaut `"ctfidf"`, rétro-compat : absent ⇒ ctfidf). Interface
unique côté pipeline : `pipeline/cluster/naming_methods.py::name_clusters_method`.

## Les trois méthodes

| `naming`   | Ce que devient le label d'un cluster | Coût | Déterministe |
|------------|--------------------------------------|------|--------------|
| `ctfidf`   | mots-clés distinctifs dérivés du corpus (c-TF-IDF + mots-vides corpus-dérivés, `naming.py`) — **inchangé** | ~0 (déjà calculé) | oui (seed) |
| `centroid` | le **témoignage le plus représentatif** : membre (médoïde) dont l'embedding est le plus proche du centroïde du cluster (cosinus max), verbatim tronqué proprement | ~0.1 s / dataset | oui |
| `llm`      | **titre court généré par un LLM LOCAL** (Ollama), prompt langue-agnostique, **repli c-TF-IDF** si Ollama injoignable | 3–5 s / appel (Ollama) | non (best-effort) |

Toujours générique / souverain : aucun mot de corpus en dur, LLM **local only**
(pas d'API externe), prompt langue-agnostique (« titre court dans LEUR langue »).
c-TF-IDF est **toujours** calculé en base : il fournit les `keywords` (affichés),
l'entrée du LLM et le **repli**.

## Modèle Ollama & le cas qwen3:4b

- Appel : `POST http://localhost:11434/api/generate` (non-stream), modèle par
  défaut **`llama3.2:3b`**, surchargeable par env `AGORA_OLLAMA_MODEL`
  (+ `AGORA_OLLAMA_URL`, `AGORA_OLLAMA_TIMEOUT`).
- **Pourquoi pas `qwen3:4b`** (suggéré au contrat) : c'est un modèle *à
  raisonnement*. L'Ollama installé (**0.30.0**) n'arrive pas à désactiver sa
  « pensée » : `think:false` ET `/no_think` sont ignorés → le modèle consomme tout
  le budget de tokens en chain-of-thought et ne renvoie **aucun titre exploitable**
  (~50 s, sortie « Okay, let's tackle this… »). On garde donc un défaut **qui
  marche** (`llama3.2:3b`, non-raisonneur, multilingue, ~3 s/appel). qwen3:4b
  reste sélectionnable via `AGORA_OLLAMA_MODEL=qwen3:4b` quand un Ollama sachant
  couper le raisonnement sera dispo.
- Le client reste défensif : il déballe un éventuel `<think>…</think>`, prend la
  première ligne, retire guillemets/préfixes, borne à ~6 mots.

## Parcimonie (Ollama partagé)

- **Leiden (hiérarchique)** : seuls les **macros** (level 0) passent par
  centroïde/LLM ; les **sous-thèmes restent c-TF-IDF** (bien plus nombreux → on
  épargne Ollama). `meta.naming_meta.scope = "macros (sous-thèmes=ctfidf)"`.
- **HDBSCAN (plat)** : tous les clusters réels sont nommés par la méthode choisie ;
  le groupe **bruit** garde son label fixe « non classé ».
- Appels LLM **parallélisés** (≤ 4 workers, `AGORA_LLM_MAX_WORKERS`).
- **Repli gracieux** : une sonde `GET /api/tags` courte (2.5 s) ; si Ollama est
  down → repli GLOBAL immédiat sur c-TF-IDF (`naming="ctfidf"`,
  `fallback=true`, `reason="ollama_unreachable"`). Un échec/timeout isolé retombe
  sur le label c-TF-IDF de SON cluster (`reason="partial_fallback"`,
  `n_fallback>0`). `meta.naming` reflète **la méthode réellement appliquée**.

## Latence mesurée (:8011, embeddings cachés nomic-v2)

`took_ms` = recluster COMPLET (clustering + nommage). Le delta « nommage » :

| dataset · méthode | ctfidf | centroid | llm (n macros/clusters) |
|---|---|---|---|
| tiktok · leiden  | ~4.1 s | ~0.1 s de plus | ~40 s (8 macros) |
| tiktok · hdbscan | — | — | ~35 s (4 clusters) |
| xstance · leiden | — | ~2.6 s total | ~84 s (17 macros) |

(LLM : ~3–5 s/appel à chaud, dominé par Ollama partagé ; le 1er appel paie le
chargement du modèle. centroïde/ctfidf : négligeable.)

## Exemples d'en-têtes par méthode

### tiktok (FR) — macros leiden

| ctfidf | centroïde (verbatim) | LLM (llama3.2:3b) |
|---|---|---|
| sentiment · perte · culpabilité | « Sentiments de mal être : passer trop de temps dessus et gacher son temps » | Perdre le Temps |
| application · faire · passer | « Simplement mal a l'aise à l'idée d'utiliser Tiktok… » | Addiction aux réseaux sociaux |
| fille · harcèlement · haine | « Messages haineux et répétitifs sur un avis personnel très banal » | Harcèlement en ligne |
| vidéos · contenus · vidéo | « vidéos choquantes pouvant mettre mal à l'aise un certain temps » | Contenu choquant visuel |
| algorithme · vidéos · triste | « avec l'algorithme, lorsqu'on est mal, tiktok nous fait voir… » | Algorithme de tristesse |
| corps · parfait · comparaison | « Se comparer physiquement au autres » | Comparaison corporelle |

### xstance (DE/FR/IT) — le titre suit la langue des contributions

Les clusters x-stance sont **trans-langues** (regroupés par THÈME, ~⅓ DE/FR/IT) :
le LLM choisit une des langues présentes du cluster.

| LLM (langue du titre) | langues membres |
|---|---|
| Souveraineté Suisse (fr) | fr 140 / de 137 / it 98 |
| Salario Minimo Garantito (it) | it 64 / de 52 / fr 29 |
| Énergie écosensible (fr) | fr 43 / de 61 / it 42 |
| Congé parental équilibré (fr/it) | it 76 / de 68 / fr 64 |

- centroïde xstance (médoïde) : « Mi sembra che in questa discussione vi sia molta
  confusione e poca chiarezza. » (verbatim IT réel).

## Front

`NamingPicker` (c-TF-IDF / Centroïde / LLM local) à côté des sélecteurs
méthode/dataset → `POST /recluster {naming}` → re-render des labels (pas de
re-pull des knobs : le nommage ne change pas les sliders). Si l'API signale un
repli LLM→c-TF-IDF, un avertissement discret s'affiche sous le toggle.

## Test

```bash
# Backend sur :8011 (NE PAS toucher la démo live :8010/:5180)
uv run --extra contender --with fastapi --with uvicorn --with httpx \
    uvicorn backend.server:app --host 127.0.0.1 --port 8011

curl -s :8011/recluster -d '{"dataset":"tiktok","naming":"centroid"}' -H 'content-type: application/json'
curl -s :8011/recluster -d '{"dataset":"xstance","naming":"llm"}'     -H 'content-type: application/json'
```
