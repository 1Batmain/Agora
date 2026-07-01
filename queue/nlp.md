# Lane nlp — embeddings · graphe k-NN · Leiden · scoring · naming

Owns: `pipeline/embed/`, `pipeline/cluster/`.

## T-N1 · Service d'embeddings (in-process)
- Goal : embeddings sémantiques robustes au paraphrasing via **sentence-transformers
  in-process** (BGE-m3 / multilingual-e5 — décision Bob). API : `embed(texts) ->
  vectors`, batch + single. Pas d'API externe, pas de dépendance Ollama.
- Accept : latence/throughput mesurés ; même `model_id` traçable.
- Deps : —. Contract : remplit `Embedding`.

## T-N2 · Graphe k-NN sémantique
- Goal : index vectoriel (FAISS in-proc) → arêtes cosine > seuil → `Edge`.
- Accept : graphe construit sur batch TikTok ; degré moyen raisonnable.
- Deps : T-N1.

## T-N3 · Clustering Leiden (+ HDBSCAN contender)
- Goal : Leiden (igraph/leidenalg) sur le graphe k-NN = approche primaire ;
  UMAP+HDBSCAN comme contender pour le banc d'éval.
- Accept : communautés stables sur batch ; les 2 enregistrées dans l'éval.
- Deps : T-N2.

## T-N4 · Assigneur incrémental (le cœur "live")
- Goal : fast path (nouvel avis → assign à la communauté la plus proche) +
  slow path (re-Leiden périodique → restructure, emit merged/split). Stratégie = fork #1.
- Accept : un flux d'avis produit une séquence d'events cohérente (cf. WS contract).
- Deps : T-N3.

## T-N5 · Scoring des thèmes
- Goal : `weight_sum`, `diversity` (1−densité dup), `consensus` (formulations variées
  → même intention). Une idée minoritaire mais forte > bruit majoritaire.
- Accept : scores reproductibles ; classement thèmes lisible.
- Deps : T-N3.

## T-N6 · Naming des thèmes (TF-IDF/KeyBERT seul)
- Goal : `label` + `keywords[]` par thème via **TF-IDF / KeyBERT uniquement**
  (décision Bob — pas de LLM pour l'instant). Titrage LLM = amélioration ultérieure.
- Accept : chaque thème a un `label` lisible + `keywords[]`.
- Deps : T-N3.
