# Méthode HDBSCAN switchable — recette & repères

> La console compare **deux méthodes de clustering** sur le même corpus et les
> mêmes embeddings cachés (nomic-v2) : **Leiden** (hiérarchique, défaut) et
> **UMAP-5D → HDBSCAN** (clusters plats + bruit). Switch côté front.

## Recette (méthode `hdbscan`)

```
vecteurs cachés (nomic-v2)            # PAS de ré-embed
  → min_chars + dedup near-dup        # mêmes filtres que Leiden (set partagé)
  → UMAP(n_components=5, metric=cosine)   # densification, n_components FIXE
  → HDBSCAN(min_cluster_size, min_samples, metric=euclidean)
  → clusters PLATS (level=0) + bruit (cluster_id=-1, label « non classé »)
  → scoring + naming TF-IDF (réutilisés du pipeline Leiden)
  → UMAP(n_components=2)               # coords (x,y) par nœud, affichage 2D futur
  → GraphPayload (même shape) + meta.method + meta.stats
```

Code : `pipeline/cluster/hdbscan_contender.py` (UMAP+HDBSCAN, défauts dérivés,
coords 2D) → `pipeline/cluster/build.py::_build_hdbscan` (GraphPayload plat) →
`backend/recluster.py` (routage `method`) → `backend/server.py` (`/params` par
méthode, champs `method`/knobs hdbscan sur `/recluster`).

## Knobs & défauts DÉRIVÉS (zéro magic-number corpus)

Mêmes **formes** que le reste du pipeline (`pipeline.cluster.adaptive`) ; la
valeur s'adapte à N, rien n'est calé sur un corpus.

| knob | défaut dérivé | forme | effet |
|---|---|---|---|
| `min_cluster_size` | ∝ N (`derive_min_sub_size`, cf. `min_sub_size`) | `max(5, round(0.011·N))` | ↑ → moins de clusters, plus gros |
| `min_samples` | **1** (`MIN_SAMPLES_FLOOR`) | plancher absolu | ↑ → lisse la densité : moins de clusters, plus de bruit |
| `umap_n_neighbors` | ∝ log N (`derive_k`, cf. `k`) | `round(3.8·log10 N)` borné | voisinage UMAP |
| `n_components` | **5 — FIXE** (contrat) | — | dim. de l'espace UMAP de clustering |

**Pourquoi `min_samples=1` ?** Empiriquement, le défaut HDBSCAN
(`min_samples=min_cluster_size`) écrase tout en **2 macro-blobs** après l'UMAP
(densité trop lissée). Le plancher `min_samples=1` = sensibilité maximale à la
densité → révèle la vraie structure fine. C'est un plancher structurel, pas un
réglage corpus. Monter le knob raréfie les clusters et augmente le bruit.

## Repères obtenus (backend testé sur :8011, seed=42, reproductible)

Filtres par défaut : `min_chars=12`, `dedup=0.95`.

| dataset | N filtré | défauts (mcs / ms / k_umap) | HDBSCAN | Leiden (rappel) |
|---|---:|---|---|---|
| **tiktok** (FR) | 1597 | 18 / 1 / 12 | **4 clusters + 1 bruit** (~31 s) | 8 macros / 47 sous-thèmes, modul. 0.60 (~2 s) |
| **xstance** (DE/FR/IT) | 2998 | 33 / 1 / 13 | **25 clusters + 667 bruit** (~33 s) | (cf. console) |

Labels tiktok HDBSCAN (TF-IDF) : *sentiment · vidéos · contenus* ·
*réseaux · application · faire* · *corps · parfait · filles* ·
*algorithme · vidéos · triste* · **non classé**.

Override d'un knob honoré (ex. `min_cluster_size=10` sur tiktok → 34 clusters /
235 bruit). `min_chars`/`dedup`/`seed` partagés avec Leiden.

## Comparaison rapide vs Leiden

- **Granularité** : sur tiktok, Leiden produit une hiérarchie riche (8→47) là où
  HDBSCAN ne dégage que 4 thèmes plats — les témoignages FR libres sont peu
  séparables en densité après UMAP. Sur xstance (multilingue, argumenté),
  HDBSCAN trouve 25 clusters mais classe ~22 % des avis en bruit.
- **Bruit** : HDBSCAN assume un groupe « non classé » (les inclassables) ; Leiden
  affecte tout le monde à une communauté. Lecture honnête de l'incertitude vs
  couverture totale.
- **Forme** : HDBSCAN = plat (pas de macro→sous) ; Leiden = 2 niveaux.
- **Coût** : HDBSCAN ~15× plus lent (deux UMAP : 5D clustering + 2D affichage),
  ~30 s vs ~2 s pour Leiden sur ~1600 avis. Acceptable pour la comparaison live.

## Lancer / tester (NE PAS toucher la démo :8010/:5180)

```bash
# backend de test sur :8011 (la démo live reste sur :8010)
uv run --extra contender --with fastapi --with uvicorn \
    uvicorn backend.server:app --host 127.0.0.1 --port 8011

curl -s localhost:8011/params?method=hdbscan        # knobs hdbscan
curl -s -X POST localhost:8011/recluster \
    -H 'Content-Type: application/json' \
    -d '{"method":"hdbscan","dataset":"tiktok"}'     # → clusters plats + bruit

# front : tsc + build (le proxy commité cible :8010 prod)
cd frontend && npm run build
```

Non-régression : `method` absent ou `"leiden"` → comportement Leiden inchangé
(8 macros / 47 sous-thèmes sur tiktok).
