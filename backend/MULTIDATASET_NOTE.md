# Multi-dataset — note (lane stream)

La console expose **plusieurs jeux de données** sélectionnables. Générique :
**un dataset = un descripteur + un cache**, zéro code spécifique à un corpus.

## Caches construits

Cache **par dataset** : `backend/cache/<dataset>/{embeddings.npy, ideas.jsonl, meta.json}`.
Embeddings **nomic-v2** (`nomic-ai/nomic-embed-text-v2-moe`, dim 768), précalculés
une fois — le serveur ne ré-embedde JAMAIS.

| dataset  | n avis | langues            | source  | échantillonnage |
|----------|--------|--------------------|---------|-----------------|
| `tiktok` | 1 621  | fr                 | tiktok  | superset (min_chars≥1, FR) |
| `xstance`| 3 000  | de 1000 · fr 1000 · it 1000 | xstance | équilibré par langue, cap 3000, min_chars≥12, dédup exacte |

`meta.json` (langues / n / source / label) est lu par `GET /datasets` ; s'il
manque, le serveur le **dérive** des `ideas.jsonl` cachés (aucune valeur en dur).

## Vitrine multilingue (x-stance)

nomic-v2 regroupe par **THÈME**, pas par langue. Mesuré sur le recluster x-stance
(défauts dérivés) :

- **NMI(macro, langue) = 0.012**, **NMI(sous-thème, langue) = 0.038** — TRÈS BAS
  ⇒ les clusters sont **trans-langues** (un faible NMI = bon, cf. cross-lane.md).
- Répartition typique d'un macro-thème : ~30 % de / 30 % fr / 30 % it, et les
  mots-clés mélangent les langues (ex. `energia · nucleare · nucléaire`,
  `schweiz · svizzera · pays`). C'est le point de la démo.

Non-régression : `tiktok` (sans `dataset`) → **8 macros / 47 sous-thèmes** comme avant.

## Endpoints (rétro-compat)

- `GET  /datasets` → `[{id, label, n_nodes, languages, lang_counts, source}]`
  (datasets = sous-dossiers de `cache/` avec un cache complet, découverts).
- `GET  /params?dataset=<id>` → knobs + **défauts dérivés du cache de CE dataset**.
- `POST /recluster {…, dataset:"<id>"}` → GraphPayload. `dataset` **défaut `"tiktok"`**.
- `GET  /health` → datasets chargés + dims.

Tous les caches sont chargés en RAM au démarrage (vecteurs `.npy` uniquement, pas
de torch) ; `_resolve(dataset)` route chaque requête vers le bon cache.

## Ajouter un nouveau dataset (généricité)

1. **Descripteur** : déposer `pipeline/ingest/descriptors/<id>.json` (format,
   path, url, colonnes canoniques `id`/`text`/`lang`/…). Cf.
   `pipeline/ingest/sources.py`. Aucun code à écrire.
2. **Cache** : construire ses embeddings nomic-v2 (un seul appel torch) :
   ```bash
   uv run --extra contender --extra embed-contender \
       python -m backend.build_cache --dataset <id> \
       [--balance lang] [--cap 3000] [--min-chars 12] [--label "Mon corpus"]
   ```
   Options de sous-échantillonnage **déclaratives et génériques** :
   `--balance <champ>` (échantillon équilibré par valeur, ex. `lang`),
   `--cap N` (plafond pour un rendu fluide), `--min-chars`, `--no-dedup-exact`.
   Si la source est absente, elle est téléchargée via l'`url` du descripteur.
3. **Rien d'autre** : `GET /datasets` le découvre automatiquement ; le sélecteur
   du front se peuple tout seul.

## Lancer

```bash
# backend :8010 (fastapi/uvicorn ne sont pas des deps du projet → --with)
uv run --extra contender --with fastapi --with uvicorn \
    uvicorn backend.server:app --host 0.0.0.0 --port 8010
# front :5180
cd frontend && npm run dev
```
