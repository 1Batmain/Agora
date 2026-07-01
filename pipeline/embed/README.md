# `pipeline/embed` — embeddings multilingues pluggables

Service d'embeddings in-process (sentence-transformers, **CPU**, offline). Le
multilingue est une **contrainte de 1er ordre** : on veut regrouper les avis par
**thème**, pas par langue. Plusieurs modèles vivent derrière **une seule interface**,
chacun avec **sa convention de préfixe** — un mauvais préfixe dégrade la qualité
silencieusement, c'est subtil et critique.

## Modèles du registre

| `model_id` | alias | préfixe **doc** / **query** | dim native | flags |
|---|---|---|---|---|
| `intfloat/multilingual-e5-small` *(défaut, baseline)* | `e5`, `e5-small` | `passage: ` / `query: ` | 384 | — |
| `nomic-ai/nomic-embed-text-v2-moe` | `nomic`, `nomic-v2` | `search_document: ` / `search_query: ` | 768 | `trust_remote_code`, dépend de `einops`, MoE, matryoshka (dim native gardée) |
| `BAAI/bge-m3` | `bge-m3`, `bge` | **aucun préfixe** | 1024 | dense vectors |

La convention exacte par modèle vit dans `registry.py` (`REGISTRY: model_id ->
ModelSpec{doc_prefix, query_prefix, trust_remote_code, normalize}`). **Ajouter un
contender = ajouter une `ModelSpec`** ; aucun autre changement requis.

## Utilisation

```python
from pipeline.embed import Embedder

emb = Embedder()                       # défaut e5-small
emb = Embedder(model_id="bge-m3")      # alias OK
emb = Embedder(model_id="nomic-ai/nomic-embed-text-v2-moe")

docs    = emb.embed(["il faut plus de pistes cyclables", "wir brauchen mehr Radwege"])
queries = emb.embed("pistes cyclables", is_query=True)   # bon préfixe auto
```

- Vecteurs **L2-normalisés** par défaut → cosine = produit scalaire.
- `is_query` choisit le préfixe (requête vs document) selon le modèle.
- `emb.model_id` est **tracé** pour remplir le contrat `Embedding{idea_id, vector[d], model_id}`.
- **Chargement paresseux** (aucun coût torch tant qu'on n'encode rien), **CPU**,
  **un seul modèle chargé par instance** — n'instanciez pas les 3 simultanément
  (RAM ~7 Gi partagée).
- Helper one-shot : `from pipeline.embed import embed; embed(texts, model_id=...)`.

### Sélectionner un modèle
- Par code : `Embedder(model_id="<id ou alias>")`.
- En CLI : `--model <id ou alias>` (cf. smoke ci-dessous).
- Le défaut reste `e5-small` (`DEFAULT_MODEL_ID`) — **l'API et le défaut ne changent pas**
  pour cluster/eval. Le kwarg legacy `e5_prefix=` est toujours accepté (alias de `use_prefix`).

## Smoke-test (critère d'acceptation)

Encode un échantillon **multilingue parallèle** (même idée en FR/DE/EN) et imprime
`dim` + le **cosinus cross-lingue** des paraphrases (doit être **élevé** pour un bon
modèle multilingue) vs le cosinus inter-thèmes (doit être bas) :

```bash
python -m pipeline.embed.embedder --model e5      --smoke
python -m pipeline.embed.embedder --model nomic   --smoke
python -m pipeline.embed.embedder --model bge-m3  --smoke
```

Mesures observées (CPU, 1er run = téléchargement du modèle) :

| modèle | dim | cos cross-lingue (paraphrases) ↑ | cos inter-thèmes ↓ | marge ↑ |
|---|---|---|---|---|
| e5-small | 384 | 0.918 | 0.832 | +0.087 |
| nomic-v2-moe | 768 | 0.873 | 0.575 | +0.298 |
| bge-m3 | 1024 | 0.890 | 0.476 | **+0.414** |

> La **marge** (cross-lingue − inter-thèmes) est le signal qui nous intéresse : plus
> elle est haute, mieux le modèle sépare les thèmes tout en rapprochant les langues.
> Sur cet échantillon, **bge-m3** sépare nettement le mieux. Le choix final du modèle
> du pipeline revient au **banc qualité** (lane eval), pas à ce micro-smoke.

## Dépendances

`sentence-transformers` + `torch` (déjà requis). **nomic-v2-moe** charge du code
custom (`trust_remote_code`) qui dépend de **`einops`** → extra `embed-contender` :

```bash
uv sync --extra embed-contender
```

e5-small et bge-m3 ne nécessitent rien de plus. Les imports lourds (torch/einops)
restent **paresseux**.
