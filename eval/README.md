# Lane EVAL — l'arbitre par la mesure (`eval-as-truth`)

Banc qui **tranche Leiden vs UMAP+HDBSCAN** par les chiffres, sur une vérité
terrain : **x-stance**, dont chaque commentaire est labellisé **FAVOR / AGAINST**
par question politique. On réutilise le pipeline mergé (aucune réimplémentation
d'embeddings/clustering) :

```
x-stance brut (avec labels)        ── eval/data.py (lit data/raw/…zip, PAS ideas.jsonl)
   │  par question : embed les commentaires (e5-small, CPU)   ── pipeline/embed/
   ▼
vecteurs L2-normalisés
   │  Leiden : kNN cosine → leidenalg        ── pipeline/cluster/{knn,leiden_cluster}
   │  HDBSCAN : UMAP → HDBSCAN                ── pipeline/cluster/hdbscan_contender
   ▼
clustering prédit  ── comparé aux labels FAVOR/AGAINST → eval/metrics.py
   ▼
eval/report.md  (scorecard agrégé : NMI/ARI/pureté/silhouette + stabilité + coût)
```

## Ce que ça mesure

| Axe | Métriques |
|-----|-----------|
| **Qualité vs vérité terrain** | **NMI**, **ARI** (accord clusters↔labels, invariants au renommage), **pureté** (classe majoritaire par cluster) |
| **Qualité interne** | **silhouette** cosine (séparation dans l'espace d'embedding, sans labels) |
| **Stabilité** | **ARI bootstrap** : N sous-échantillons → accord inter-runs moyen |
| **Coût** | latence clustering, nb d'embeddings, wall-clock |

Tout est agrégé en **moyenne ± écart-type** sur les questions échantillonnées.

## Relancer

```bash
# prérequis : la source brute x-stance (idempotent ; data/ est gitignored)
uv run python -m pipeline.ingest.download --only xstance
uv sync --extra contender          # umap-learn + hdbscan (le contender)

# critère d'acceptation — écrit eval/report.md avec des nombres réels
uv run python -m eval.bench --sample-questions 8

# variantes
uv run python -m eval.bench --sample-questions 20 --bootstrap 10
uv run python -m eval.bench --no-bootstrap        # plus rapide
uv run python -m eval.bench --lang all            # de/fr/it
```

Options : `--seed` (défaut 42, reproductible), `--bootstrap N` / `--no-bootstrap`,
`--boot-frac`, `--min-comments`, `--min-per-class`, `--out`.

## Lecture (sur l'échantillon par défaut)

Sur 8 questions FR, **les deux approches recouvrent mal le clivage FAVOR/AGAINST**
(NMI ≈ 0.04–0.06, ARI ≈ 0) : l'embedding e5 capte surtout le **thème** du
commentaire, pas sa **position** — et Leiden sur-segmente (≈ 5 clusters) là où il
n'y a que 2 classes. HDBSCAN obtient une silhouette légèrement meilleure mais
rejette beaucoup de points en bruit et coûte ~40× plus cher (UMAP). À ce stade le
banc dit surtout : **détecter la polarité demande un signal dédié** (le stance
n'est pas un sous-produit gratuit du clustering thématique). Les chiffres exacts,
seedés, sont dans `eval/report.md`.

## Limites (honnêteté, Playbook §5)

- **Échantillon modeste** par défaut (8 questions) → écarts-types larges ;
  élargir avec `--sample-questions`.
- **Vérité à 2 classes** : x-stance n'a que FAVOR/AGAINST. NMI/ARI pénalisent un
  clustering qui isole des sous-thèmes d'argumentation pourtant valides — la
  silhouette nuance.
- **Domaine** : x-stance = votations suisses (FR). Le transfert vers la
  consultation TikTok (témoignages libres, **sans labels**) n'est pas validable
  ici, faute de vérité terrain — c'est la limite intrinsèque de l'exercice.
- **Params figés** (défauts pipeline), pas de sweep d'hyperparamètres.
- Le **bruit HDBSCAN** (`-1`) compte comme un cluster pour NMI/ARI/pureté
  (honnête) et est exclu de la silhouette.

## Fichiers

- `data.py` — chargeur x-stance labellisé (filtre lang, groupage/filtre par question).
- `metrics.py` — NMI/ARI/pureté/silhouette + agrégation.
- `bench.py` — orchestration : embed → 2 clusterings → métriques → bootstrap → coût.
- `report.py` — rendu `report.md`.
- `report.md` — dernière sortie committée (régénérable).
