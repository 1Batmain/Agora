# Pare-feu de provenance — jina-v3 (CC-BY-NC) comme embedder de build

**Décision (2026-07-08)** : jina-v3 devient l'embedder de **build par défaut** (`DEFAULT_EMBEDDER
= "jina-v3"`) pour générer des golds/datasets d'analyse de **meilleure qualité** (NMI thème 0.482
vs nomic 0.407). C'est le **meilleur modèle mesuré** de toute la campagne (cf. `bench_jina.md`,
`bench_veille.md`). **Assumé en connaissance de la licence.**

## Le fait légal (à ne jamais perdre de vue)

Les poids `jina-embeddings-v3` sont **CC-BY-NC-4.0 = NON-COMMERCIAL**. Agora a une **édition
commerciale** (double-licence AGPL + commerciale). Donc :

- ✅ **Autorisé (phase actuelle)** : usage **RECHERCHE / non-commercial** — générer des synthèses,
  des golds, des datasets d'analyse, faire des régressions/ablations de modèles. Tant que RIEN de
  ce qui en découle n'est vendu ni expédié dans l'édition commerciale, on est dans le périmètre NC.
- ❌ **Interdit** : que les **sorties de jina** (embeddings, clusters, golds, synthèses) servent à
  **entraîner / distiller un modèle EXPÉDIÉ commercialement**. Le NC **contamine l'élève** : un
  modèle maison entraîné sur des golds jina hériterait de la restriction et ne serait pas
  commercialisable — ce qui **défait** l'objectif « créer notre propre modèle vendable ».

## La règle (pare-feu)

> **Tout artefact dérivé de jina-v3 est estampillé NON-COMMERCIAL et cantonné à la R&D.**
> **Avant toute commercialisation, re-dériver l'artefact avec un embedder Apache** (nomic-v2 ou,
> mieux, **arctic-l** — 0.455, la meilleure qualité *permissive* mesurée ; ou granite-311m).

Concrètement :
1. **Traçabilité** : chaque build enregistre son embedder dans `meta.json` (`model_id`,
   `embedder`). Un dataset dont `embedder` = `tomaarsen/jina-embeddings-v3-hf` est **NC** →
   ne jamais l'utiliser comme source de training d'un modèle destiné à la vente.
2. **Pour un gold « propre » destiné à entraîner un modèle vendable** : passer explicitement
   `embedder="arctic-l"` (ou `nomic-v2`) — pas le défaut jina.
3. **Étape de commercialisation** : re-embed all + re-générer les golds avec un embedder Apache,
   re-valider (le témoin), et vérifier qu'aucun artefact NC ne subsiste dans la chaîne de training.
   arctic-l est à ~94 % de la qualité de jina → la perte à la re-dérivation est faible.

## Impacts techniques du changement de défaut

- `DEFAULT_EMBEDDER = "jina-v3"` (`pipeline/claims/pipeline.py`) + `DEFAULT_MODEL_ID`
  (`pipeline/embed/embedder.py`). jina-v3 charge via un **loader natif** `hf_mean_pool`
  (`AutoModel` + mean-pooling) car sentence-transformers/trust_remote_code casse sur ce
  transformers. Port `tomaarsen/jina-embeddings-v3-hf`, commit épinglé.
- **Coût** : ~4-5× plus lent que nomic à l'embed (CPU), dim **1024** (vs 768 → caches +33 %).
  Les builds de golds sont donc plus longs — acceptable en phase R&D (qualité > débit).
- **Prod inchangée au SERVE** : la prod sert le cache (sans clé, ne rebuild pas). Le changement
  n'affecte que les **builds** (en dev). Les caches servis existants restent en nomic jusqu'à un
  rebuild explicite + commit.
- **Réversible** : remettre `DEFAULT_EMBEDDER = "nomic-v2"` suffit à revenir à l'embedder Apache.

## En une phrase
jina-v3 = **meilleure qualité, licence NC** → on l'exploite pour la **qualité des golds en R&D**,
avec un **pare-feu strict** : re-dérivation Apache (arctic-l) obligatoire avant toute vente.
