# Pipeline d'analyse — état des lieux

Carte du pipeline BUILD (analyse) d'Agora, pour contributeurs. Architecture **BUILD / SERVE
séparés** : le pipeline lourd s'exécute une fois (`backend/build_analysis.py` +
`backend/build_opinion.py`) et écrit des caches sous `backend/cache/<dataset>/` ; l'API
(`backend/server.py`) ne fait que LIRE ces caches (serve-only, zéro calcul à la requête).

## Les 10 étapes

```
ideas.jsonl (avis nettoyés + PII masquée à l'ingestion)
   │
   ① CLAIMS ──────── extraction verbatim (mistral-large, lots de 8 avis)
   │                 → claims.json  {avis_id: [{text, spans, target}]}
   ② EMBEDDINGS ──── nomic-embed-v2 LOCAL, 768d L2-normalisé → claims_emb.npz
   ③ GRAPHE k-NN ─── faiss ; k ≈ 3.8·log₁₀(N) borné [8,30], seuil μ−3.2σ (dérivés)
   ④ LEIDEN ──────── détection de communautés → clusters fins (poids = cosinus brut)
   ⑤ HIÉRARCHIE ──── subdivision variance-adaptative (τ = plus grand gap de dispersion)
   ⑥ RECUT ───────── sauce_magique : re-coupe l'arbre + regroupe la poussière
   │                 → analysis.json  (arbre de thèmes, ids nX)
   ⑦ NAMING ──────── titres ancrés c-TF-IDF (mistral-large, cachés par contenu)
   ⑧ ENRICH ──────── hooks + descriptions + traductions FR + mots-clés
   ⑨ INSIGHTS ────── synthèses bottom-up sectionnées (feuilles→racines)
   ⑩ OPINION ─────── objet de clivage + stance par claim (build SÉPARÉ)
                     → opinion.json + claim_stance.json
```

## Détail par étage

### ① Extraction de claims — le socle
- `mistral-large-latest`, lots de 8 avis/appel (`AGORA_CLAIMS_BATCH`).
- Un claim = 1..N portions **verbatim** (sous-chaînes exactes) + **cible** optionnelle,
  ancrées par offsets sur `text_clean`. **Gate dur** : toute portion non retrouvée est
  rejetée (zéro hallucination). Cf. `pipeline/claims/`.
- Cache `claims.json` — **clé = le modèle** : changer `AGORA_EXTRACT_MODEL` ré-extrait tout.
- Validé : verbatim ~98,8 %, prompt v2 validé en panel aveugle.

### ②–③ Embeddings + graphe
- **nomic-embed-v2 local** (souverain, hors-ligne). Choix validé contre gold (clusterise
  par THÈME NMI 0.407, pas par langue 0.008). `k`/seuil **dérivés des données** (aucune
  constante de corpus, `pipeline/cluster/adaptive.py`).

### ④–⑤ Leiden + hiérarchie variance-adaptative
- Leiden pondéré (cosinus brut — transformations testées, aucune ne gagne).
- Un nœud se subdivise seulement si dispersion > τ dérivé ET si Leiden dégage ≥2
  sous-thèmes viables. Profondeur max 4.

### ⑥ Recut — sauce_magique (`backend/recut.py`)
- Résout l'effondrement macro à l'échelle (à 22k claims, sans recut : 1 communauté à
  99,9 %). Fonction objectif minimisée `α(1−cohésion)+β|log(N_eff/N_cible)|+γ·poussière+δ·top1`
  par descente gloutonne sur les **coupes de l'arbre existant** (zéro re-clustering, zéro LLM).
- Regroupement poussière : nœuds < 0,5 % des voix → nœud « Contributions isolées »
  (navigable, ids intacts, rien supprimé).
- Validé au témoin Grand Débat 22k : 18 macros, 14/14 sous-thèmes officiels, 0 mismatch.
- ⚠️ Poids v1 NON calibrés (la calibration par gold est invalide : les golds ne mesurent
  que la granularité, cf. `research/sauce_magique_calibration.md`).

### ⑦–⑨ Nommage + enrichissement + synthèses
- **Titres ancrés** : claims d'entrée sélectionnés par distinctivité c-TF-IDF (pas par
  proximité centroïde, qui donnait des titres génériques par anisotropie).
- **Synthèses bottom-up** : un parent agrège les synthèses de ses enfants ; sectionnées ;
  cache à clé de contenu (hash des synthèses enfants inclus).
- Enrichissement en mistral-large, cachés (rebuild ne les repaye que s'ils changent).

### ⑩ Opinion — `backend/build_opinion.py` (build séparé)
- Par feuille : **objet de clivage** (proposition polaire conditionnée sur titre +
  mots-clés distinctifs, variante « b » validée en panel aveugle) puis **stance de chaque
  claim** envers cette proposition.
- **Stance servie** : `mistral-small`, T=0, lots de 10, + confiance auto-déclarée. Prompt
  clé : juge le SOUTIEN À L'ACTION, pas le sentiment envers le sujet (décrire un méfait de X
  = favorable à « réguler X »).
- Garde-fous : thème « impur » si engagement < 0.35 ou < 8 claims → pas de répartition.
- Validé : accuracy 0.79 sur gold x-stance, confiance calibrée. Bench large fait : large
  sur-abstient sur corpus réel → on garde small (`research/stance_large_bench.md`).

## Frontières (fichiers de cache = contrats entre modules)
| Fichier | Produit par | Contrat |
|---|---|---|
| `ideas.jsonl`, `embeddings.npy`, `meta.json` | `build_cache` | entrées (committées) |
| `claims.json`, `claims_emb.npz` | extraction | claims verbatim (clé = modèle) |
| `analysis/analysis.json` | `build_analysis` | arbre de thèmes, ids `nX` |
| `analysis/avis.json` | `build_analysis` | avis + spans surlignables (id claim = `avis_id#index`) |
| `analysis/opinion.json`, `claim_stance.json` | `build_opinion` | opinion par thème / stance par claim |
| `analysis/{insights,citations,titles,hooks,descriptions}/` | enrich | 1 JSON par nœud, cachés |
| `analysis/cost.json` | build | tokens · $ estimé · durées |

⚠️ **Invariant critique** : `avis.json` et `claim_stance.json` doivent venir du MÊME build
(ids `avis_id#index` alignés) — les régénérer séparément casse le filtre par sentiment.
Toujours `build_analysis` PUIS `build_opinion`, promouvoir le dossier `analysis/` entier.

## Fragilités connues (par impact)
1. **Argument mining** (`/arguments`) : codé au hackathon, ne respecte pas le verbatim — chantier en cours.
2. **Stance sur corpus réel** : cibles dérivées dures → engagement fragile.
3. **Poids sauce_magique v1** non calibrés.
4. **Anisotropie** des embeddings sur corpus mono-sujet (cos aléatoire ~0.59) — mitigé, pas résolu.
5. **Sensibilité à l'échantillon** jamais mesurée (re-tirage).
6. **SSOT stockage** : fichiers + DuckDB (+ futur PG) — vigilance divergence.

## Forces prouvées (par la mesure)
Extraction verbatim gate-dur · thèmes validés au gold officiel (14/14, 0 mismatch) ·
stance calibrée 0.79 · souveraineté (embeddings locaux) · coût mesuré (~24 $/22k avis) ·
culture de verdict (chaque idée benchée avant adoption, résultats négatifs publiés).
