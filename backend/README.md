# Backend — re-clustering LIVE (`:8010`)

Serveur FastAPI **léger** qui re-clusterise les avis citoyens **en live** quand le
front (console, `:5180`) bouge les knobs. Les embeddings **nomic-v2** sont
**précalculés une fois** et chargés depuis un `.npy` : le serveur **ne charge
jamais torch** et ne ré-embedde **jamais** → re-clustering en ~1–3 s pour ~1600 avis.

Réutilise tel quel `pipeline.cluster.{dedup,knn,hierarchy,scoring,naming}` (mode
hiérarchique macro→sous-thèmes). Aucune écriture hors `backend/`. Port **8010** uniquement.

## Architecture

```
data/processed/ideas.jsonl ──(build_cache, 1×)──▶ backend/cache/embeddings.npy   (1621×768 f32)
                                                  backend/cache/ideas.jsonl      (texte+meta alignés)
                                                          │
                              GET /params  ◀── server.py (charge le .npy au démarrage, PAS de torch)
                              POST /recluster ──▶ min_chars → dedup → k-NN → Leiden hiérarchique
                                                  → scoring → naming TF-IDF → GraphPayload
```

- `build_cache.py` — **one-shot** : embedde le superset TikTok/FR (`source=tiktok`,
  `lang=fr`, `min_chars≥1`) avec nomic-v2. SEUL appel au modèle.
- `recluster.py` — cœur live : à partir des vecteurs cachés, applique la chaîne du
  contrat et renvoie le `GraphPayload` hiérarchique + `meta.stats`.
- `server.py` — FastAPI :8010 (CORS permissif en dev).

## Cache (à refaire seulement si les avis changent)

```bash
# 0) données (évite le download flaky)
mkdir -p data/raw
cp /home/bat/agora-worktrees/realdata/data/raw/tiktok_appel_a_temoignages.csv data/raw/
uv run --with langdetect python -m pipeline.ingest.build      # → data/processed/ideas.jsonl

# 1) cache d'embeddings nomic-v2 (~1 min, une fois)
uv run --extra embed-contender python -m backend.build_cache --model nomic-v2
# → backend/cache/embeddings.npy + backend/cache/ideas.jsonl
```

Le cache est versionné (≈5 Mo) : le serveur démarre sans rejouer l'embedding.

## Lancer le serveur

```bash
uv run --extra contender --extra serve \
    uvicorn backend.server:app --host 0.0.0.0 --port 8010
```

(`--extra contender` apporte hdbscan/umap importés par le pipeline ; k-NN tombe sur
sklearn si faiss absent.)

## Endpoints

### `GET /health`
```json
{ "ok": true, "n_cached": 1621, "model_id": "nomic-ai/nomic-embed-text-v2-moe", "dim": 768 }
```

### `GET /params`
Table des knobs pour construire les sliders (`name, label, default, min, max, step, help`)
+ `defaults` + `seed`.

| knob | défaut | borne | effet |
|---|---|---|---|
| `dedup` (cosine) | 0.95 | 0.90–0.99 | fusion near-dups (cumule le poids) |
| `min_chars` | 12 | 0–40 | filtre avis courts |
| `k` (voisins) | 12 | 5–30 | densité k-NN |
| `threshold` (cosine) | 0.60 | 0.40–0.85 | coupe les arêtes |
| `resolution_macro` | 1.0 | 0.3–3.0 | granularité macros |
| `resolution_sub` | 1.5 | 0.5–4.0 | granularité sous-thèmes |
| `min_sub_size` | 18 | 5–40 | fusion des miettes |

### `POST /recluster`
Body (tous optionnels → repli sur défaut, bornes validées) :
```json
{ "dedup":0.95, "min_chars":12, "k":12, "threshold":0.60,
  "resolution_macro":1.0, "resolution_sub":1.5, "min_sub_size":18 }
```
Réponse = **GraphPayload hiérarchique** (même shape que `data/graph.json`) :
- `meta` — dont `meta.stats { n_macros, n_subs, n_nodes, modularity, took_ms }`
- `nodes[]` — `{ id, type, label, props{...}, cluster_id, macro_id, color }`
  (`cluster_id`/`macro_id`/`color` au top-level, cf. contrat)
- `links[]` — `{ source, target, type:"knn", props{weight} }`
- `themes[]` — macro (`level:0`) + sous-thèmes (`level:1`, `parent_id`),
  avec `member_ids, size, weight_sum, diversity, consensus, centroid, label, keywords, color, children`

Exemples :
```bash
curl -s localhost:8010/recluster -H 'content-type: application/json' -d '{}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["meta"]["stats"])'
# {'n_macros': 8, 'n_subs': 47, 'n_nodes': 1597, 'modularity': 0.6015, 'took_ms': ~2500}

curl -s localhost:8010/recluster -H 'content-type: application/json' -d '{"threshold":0.70}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["meta"]["stats"])'
# {'n_macros': 40, ...}   ← le live marche : threshold change le nb de communautés
```

## Front / proxy
Le front (`:5180`) appelle via un proxy vite `/api` → `:8010`. CORS est permissif en
dev (`allow_origins=["*"]`) pour couvrir aussi l'accès direct localhost/forge.

## Reproductibilité
`seed=42` partout. Même body → même payload.
