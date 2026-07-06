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

```bash
# thèmes HIÉRARCHIQUES (macro → sous-thèmes, Leiden 2 niveaux)
uv run python -m pipeline.cluster.build --hierarchical
```

Options hiérarchie : `--resolution-macro 1.0` (basse → grandes communautés)
`--resolution-sub 3.0` (plus fine, par sous-graphe induit) `--min-sub-size 15`
(fusion des miettes). Voir **`HIERARCHY_NOTE.md`** pour l'arbre obtenu sur le réel.

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
               lang, author_hash, source, weight }, "cluster_id", "color",
               "macro_id"? } ],   // macro_id présent en mode --hierarchical
  "links": [ { "source", "target", "type":"knn", "props": { "weight": cosine } } ],
  "themes": [ { "cluster_id", "level", "parent_id", "children", "member_ids",
                "size", "weight_sum", "diversity", "consensus", "centroid",
                "label", "keywords", "color" } ]
}
```

- **couleur** du nœud = `cluster_id` Leiden (palette qualitative).
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

## Thèmes hiérarchiques (`--hierarchical`)

Deux niveaux interprétables pour un·e député·e : quelques **macro-thèmes**
(`level=0`) qu'on ouvre en **sous-thèmes** (`level=1`). Cf. `hierarchy.py`.

- **Niveau 0 (macro)** : Leiden **basse** résolution (`--resolution-macro`) sur le
  graphe k-NN complet → grandes communautés.
- **Niveau 1 (sous-thèmes)** : pour chaque macro, Leiden **haute** résolution
  (`--resolution-sub`) sur son **sous-graphe induit**. Les sous-clusters
  < `--min-sub-size` sont fusionnés dans le sous-thème viable **le plus proche**
  (cosine) → pas de poussière de singletons.
- **Naming inchangé** (TF-IDF) appliqué aux deux niveaux : macro = TF-IDF
  inter-macros ; sous-thème = TF-IDF **contrasté dans son macro**.

Forme de sortie (contrat `Theme` étendu) :

| niveau | `level` | `parent_id` | `children` |
|--------|---------|-------------|------------|
| macro  | `0`     | `null`      | `[ids des sous-thèmes]` |
| sous   | `1`     | `<macro_id>`| `[]` |

- **`cluster_id`** d'un thème = id du macro (level 0) **ou** de la feuille (level 1).
  Espaces d'ids **disjoints** (macros `[0,M)`, feuilles `[M,M+L)`).
- **Nœuds** : `cluster_id` = la **feuille** (sous-thème) ; `macro_id` = le macro
  parent ; `color` = couleur du **macro** (l'essaim se lit par macro-thème, la
  finesse apparaît au drill-down). Les trois sont au **top-level** du nœud.
- `meta.clustering` trace `mode`, `resolution_macro/sub`, `min_sub_size`,
  `macro_modularity`, `n_macros`, `n_leaves`, `seed`.

**Intégrité** : `hierarchy.check_integrity(payload)` renvoie la liste des erreurs
(vide = arbre cohérent : `children` d'un macro = exactement les feuilles dont
`parent_id` = ce macro ; chaque nœud → feuille valide + `macro_id` concordant).

Le **mode plat** (Leiden 1 niveau, défaut sans `--hierarchical`) reste disponible
pour non-régression ; ses thèmes portent `level=0, parent_id=null, children=[]`.

## Reproductibilité

Seed fixé (`42`) pour Leiden et HDBSCAN ; embeddings déterministes (CPU). Deux
exécutions produisent un `graph.json` identique. `model_id` tracé dans `meta`.

## Params par défaut (calés sur le fixture FR)

`k=8`, `threshold=0.84`, `resolution=1.5` → 6 communautés nettes
(transport·environnement / école / santé / sécurité / services publics /
numérique·territoire). La lane **eval** ajustera ces params sur le batch TikTok
(33 609 réponses) et arbitrera Leiden vs HDBSCAN.

## Subset réel + dédup (consultation TikTok FR)

Pour brancher la **vraie** consultation à la place du fixture, on filtre le
corpus, on déduplique les répétitions, et on re-tune Leiden pour ~1,5 k avis
bruités (cf. `REALDATA_NOTE.md`) :

```bash
uv run python -m pipeline.cluster.build \
  --source tiktok --lang fr --min-chars 12 --dedup 0.95 \
  --k 12 --threshold 0.84 --resolution 2.0 \
  --out frontend/public/graph.json
```

Nouvelles options :

| Option | Effet |
|--------|-------|
| `--source tiktok` | ne garde que cette source |
| `--lang fr` | ne garde que cette langue |
| `--min-chars 12` | retire les avis trop courts (« Néant », « Déprime »…) |
| `--dedup 0.95` | fusionne les near-dups (cosine > seuil) ; le `weight` du représentant cumule celui des copies (`dedup.py`) |
| `--max-links N` | plafonne les arêtes **affichées** (garde les plus fortes) ; tous les nœuds restent — Leiden, lui, voit le graphe complet |

Le subset accepte aussi bien le JSONL **niché** `props{...}` de la lane data que
le format **plat** du fixture de dev (`io.from_row` gère les deux).
Réglages retenus : `k=12, threshold=0.84, resolution=2.0` → **15 thèmes**
(modularité 0.53) sur 1 514 nœuds après dédup.
