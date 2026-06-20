# Lane NLP — `pipeline/cluster`

Pipeline **batch** qui transforme un JSONL d'avis citoyens en un **GraphPayload
coloré par thème** que la viz charge directement (contrat `cross-lane.md`).

```
ideas.jsonl
   │  embed (sentence-transformers e5-small, in-process, CPU)   ── pipeline/embed/
   ▼
vecteurs L2-normalisés (dim 384)
   │  graphe k-NN cosine > seuil   (sklearn NearestNeighbors, faiss-cpu si dispo)
   ▼
graphe sémantique pondéré
   │  Leiden (igraph + leidenalg, seed fixé)        ← clustering PRIMAIRE
   │  UMAP+HDBSCAN                                  ← contender (option, --with-hdbscan)
   ▼
communautés = thèmes
   │  scoring (weight_sum, diversity, consensus) + naming TF-IDF (FR)
   ▼
graph.json  { meta, nodes, links, themes }
```

## Commande

```bash
# génère data/graph.json (gitignored)
uv run python -m pipeline.cluster.build

# + écrit le fixture viz committé pipeline/cluster/fixtures/graph.sample.json
uv run python -m pipeline.cluster.build --fixture

# trace aussi le contender HDBSCAN dans meta (nécessite l'extra `contender`)
uv run python -m pipeline.cluster.build --with-hdbscan
```

Options : `--input PATH` `--out PATH` `--k 8` `--threshold 0.84`
`--resolution 1.5` `--seed 42` `--model <model_id>`.

### Dépendances optionnelles

```bash
uv sync --extra contender   # umap-learn + hdbscan (banc d'éval)
uv sync --extra faiss       # faiss-cpu (k-NN accéléré ; sinon fallback sklearn)
```

## Entrée

Résolution automatique, dans l'ordre (ne bloque jamais sur la lane data) :

1. `data/processed/ideas.jsonl` (produit par la lane data)
2. `pipeline/ingest/fixtures/ideas.sample.jsonl` (fixture lane data)
3. `pipeline/cluster/fixtures/ideas.sample.jsonl` (fixture de dev, committé)

Chaque ligne = un avis : au minimum `{ "id", "text" }`, optionnellement
`text_clean, ts, lang, author_hash, source, weight`.

## Sortie — GraphPayload (contrat)

```jsonc
{
  "meta": {
    "model_id": "intfloat/multilingual-e5-small", "embedding_dim": 384,
    "n_nodes": ..., "n_links": ..., "n_themes": ...,
    "params": { "k", "threshold", "resolution", "seed", "knn_backend", "avg_degree" },
    "clustering": {
      "primary": "leiden",
      "leiden": { "n_clusters", "modularity", "resolution", "seed" },
      "hdbscan_contender": null            // ou { n_clusters, n_noise, params }
    }
  },
  "nodes": [ { "id", "type":"idea", "label", "props": { text, text_clean, ts,
               lang, author_hash, source, weight }, "cluster_id", "color" } ],
  "links": [ { "source", "target", "type":"knn", "props": { "weight": cosine } } ],
  "themes": [ { "cluster_id", "member_ids", "size", "weight_sum", "diversity",
                "consensus", "centroid", "label", "keywords", "color" } ]
}
```

- **couleur** du nœud = `cluster_id` Leiden (palette qualitative type `dummy`).
- `themes` est **trié** par intérêt : `weight_sum × (0.5 + consensus·diversity)`
  → une idée minoritaire mais cohérente et non redondante remonte face au bruit
  majoritaire.

## Scores des thèmes

| score        | définition |
|--------------|------------|
| `weight_sum` | somme des poids sociaux (`weight`) des avis du thème |
| `diversity`  | `1 − densité de duplicats` (fraction de paires cosine > 0.93). 1.0 = aucune redondance littérale |
| `consensus`  | cosinus moyen intra-thème (cohérence sémantique). Haut = même intention |
| `centroid`   | barycentre L2-normalisé (sert à l'assignation live — Phase 2) |

`consensus` haut + `diversity` haut ⇒ « même intention, formulations variées ».

## Naming (T-N6)

TF-IDF inter-clusters (uni + bigrammes, stopwords FR) — **pas de LLM**
(décision Bob). `keywords[]` = top termes distinctifs, `label` = 3 premiers.

## Reproductibilité

Seed fixé (`42`) pour Leiden et HDBSCAN ; embeddings déterministes (CPU). Deux
exécutions produisent un `graph.json` identique. `model_id` tracé dans `meta`.

## Params par défaut (calés sur le fixture FR)

`k=8`, `threshold=0.84`, `resolution=1.5` → 6 communautés nettes
(transport·environnement / école / santé / sécurité / services publics /
numérique·territoire). La lane **eval** ajustera ces params sur le batch TikTok
(33 609 réponses) et arbitrera Leiden vs HDBSCAN.
