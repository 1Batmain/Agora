# Cross-lane — CONTRAT (Phase-0 freeze) · v0 (forks résolus)

> Architecte = seul auteur. Les lanes lisent, ne réécrivent pas. Freeze imminent
> (commit du contrat) une fois ce v0 validé par Bob.

## Vision produit
Consultation **batch d'abord, live ensuite** : on bâtit le pipeline + l'éval sur
batch, puis on rejoue/streame les avis pour l'animation d'un essaim de nœuds qui
s'auto-organise en **communautés (Leiden) = thèmes**. Démo headline = Consultation
TikTok (33 609 réponses, FR).

## Décisions (forks tranchés par Bob — 2026-06-20)
1. **Stratégie live** = **batch d'abord, live ensuite**. Phase 1 = pipeline+éval
   statiques. Phase 2 = animation via streaming (la base viz supporte déjà l'ajout
   incrémental, cf. `addNodes`).
2. **Embeddings** = **sentence-transformers in-process** (BGE-m3 / multilingual-e5).
   Souverain, offline, n'encombre pas l'Ollama partagé. Pas d'API externe.
3. **Animation front** = **forker la base viz de `dummy`** (`~/forge/dummy/frontend`,
   read-only) : React + `@react-three/fiber` + `d3-force-3d` dans un **web worker**.
   Le protocole worker (`forceLayout.protocol.ts`) est STABLE et expose déjà
   `init` / `addNodes` (extension live, positions préservées) / `focus` / `setParams`.
4. **Naming des thèmes** = **TF-IDF / KeyBERT seul** pour l'instant (pas de LLM).
   Titrage LLM = amélioration ultérieure.

## Décisions — itération « amélioration clustering » (2026-06-20 bis)
- **Multilingue = contrainte de 1er ordre** (usage européen/international visé).
  Les avis peuvent être en plusieurs langues ; un bon clustering regroupe par
  THÈME, pas par langue.
- **Embeddings pluggables** : registre de modèles derrière une interface unique,
  chacun avec sa convention de préfixe. Contenders au banc : `multilingual-e5-small`
  (baseline), `nomic-embed-text-v2-moe`, `BAAI/bge-m3`. Gagnant choisi par la mesure.
- **Naming** : reste TF-IDF (inchangé — décision Bob).
- **Thèmes hiérarchiques** : 2 niveaux (macro `level=0` → sous-thèmes `level=1`),
  via `parent_id`/`children[]`. Le drill-down viz devient un arbre.
- **Banc qualité de clustering** (étend la lane eval) : cohérence de thèmes
  (NPMI / c_v) + intrinsèques (silhouette, modularité) + **mixité linguistique**
  (NMI cluster↔langue, qu'on veut BAS = clusters trans-langues) + stabilité.
  x-stance (DE/FR/IT) sert le test cross-lingue ; TikTok (FR) la cohérence.

## Modèle de données (canonique — aligné sur les shapes viz de dummy)
```
Idea  → GraphNode { id, type, label, props{ text, text_clean, ts, lang,
                                            author_hash, source, weight=1.0 } }
Edge  → GraphLink { source, target, type, props{ weight=cosine } }   # k-NN, > seuil
Theme            { cluster_id, member_ids[], size, weight_sum,
                   diversity, consensus, centroid, label, keywords[], color,
                   level, parent_id, children[] }   # hiérarchie macro→sous-thèmes
Embedding        { idea_id, vector[d], model_id }
```
- `type` du nœud = `idea` (extensible). `author_hash` = anonymisation. `weight` = social.
- **PRÉCISION (post-merge nlp)** : `cluster_id` (int communauté Leiden) et `color`
  (hex palette) vivent **au TOP-LEVEL du nœud** (à côté de `id/type/label/props`),
  PAS dans `props`. La lane viz colore l'essaim par `node.cluster_id` / `node.color`.
  Artefact batch de référence = `pipeline/cluster/fixtures/graph.sample.json`.
- `diversity` = 1 − densité de duplicats. `consensus` = même intention, formulations
  variées.

## Protocole de transport
- **Phase 1 (batch)** : le front charge un `GraphPayload { meta, nodes, links }`
  statique (comme `mock/graph.json` de dummy) → rendu de l'essaim complet + thèmes.
- **Phase 2 (live)** : WS pousse les avis ; le client appelle `addNodes` au worker.
  Événements WS cibles :
```
idea_added     { node:GraphNode, x?, y?, provisional_cluster }
edges_added    { idea_id, edges:[GraphLink] }
cluster_updated{ cluster_id, size, weight_sum, label?, keywords?, color }
cluster_merged { from:[..], into }
cluster_split  { from, into:[..] }
snapshot       { GraphPayload + themes }    # late-joiners / reconnect
```

## Carte d'ownership (fichiers disjoints — anti-conflit)
| Lane   | Possède                                   | Port    |
|--------|-------------------------------------------|---------|
| data   | `data/`, `pipeline/ingest/`               | —       |
| nlp    | `pipeline/embed/`, `pipeline/cluster/`    | —       |
| stream | `backend/` (FastAPI + WS + replay)        | `:8010` |
| viz    | `frontend/` (fork base dummy)             | `:5180` |
| eval   | `eval/`                                   | —       |

Ports interdits (dummy/Ollama) : `:8000 :5173 :8765 :11434`.
Isolation : dummy/`~/forge` = **lecture seule** (inspiration viz), jamais lancé/modifié.

## Ordre de dépendances (batch-first)
`contract figé` → **data** → **nlp** → **eval** (arbitre Leiden vs HDBSCAN) →
**viz batch** (essaim statique + thèmes) → **stream + viz live** (Phase 2).
