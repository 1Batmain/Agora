# Lane EVAL — l'arbitre par la mesure (`eval-as-truth`)

Cette lane tranche les choix de clustering **par les chiffres**, pas l'intuition
(Playbook §5). Elle réutilise le pipeline mergé (aucune réimplémentation
d'embeddings/clustering) et héberge **deux bancs complémentaires** :

| Banc | Question tranchée | Entrée | Sortie |
|------|-------------------|--------|--------|
| **Qualité** (`quality_bench.py`) | **Quel modèle d'embedding ?** e5-small vs nomic-v2 vs bge-m3 — regrouper par **thème**, pas par **langue** | x-stance DE/FR/IT (thème = `topic`) | `eval/quality_report.md` |
| **Stance** (`bench.py`) | **Quel algo de clustering ?** Leiden vs UMAP+HDBSCAN | x-stance FR (labels FAVOR/AGAINST) | `eval/report.md` |

Les deux sont **indépendants** et régénérables ; ils mesurent des axes différents
(le modèle d'embedding vs l'algorithme de partition).

---

## Banc QUALITÉ — le modèle d'embedding multilingue

> Le multilingue est une **contrainte de 1er ordre** (cross-lane) : un bon
> clustering regroupe les avis par **thème**, pas par **langue**. Ce banc
> transforme « améliorer le clustering » en **nombre** et **désigne un gagnant**.

```
x-stance (DE/FR/IT, topic)   ── eval/multilingual_data.py (corpus équilibré thème×langue)
   │  POUR CHAQUE modèle (un seul chargé à la fois) :
   │    embed (e5-small | nomic-v2 | bge-m3)              ── pipeline/embed/ (registre)
   │    rang-kNN cosine → Leiden                          ── pipeline/cluster/{knn,leiden}
   ▼
clustering prédit  ── métriques multilingues → eval/{coherence,metrics}.py
   ▼
eval/quality_report.md  (scorecard + recommandation du gagnant + composite)
```

### Ce que ça mesure (par modèle)

| Axe | Métrique | Sens |
|-----|----------|:----:|
| **Cohérence de thèmes** (intrinsèque, le cœur) | **NPMI** des top-mots TF-IDF, calculé **par langue** puis moyenné | ↑ |
| **Mixité linguistique** (LE test multilingue) | **NMI(cluster, langue)** + pureté linguistique | **↓ (bas = bon)** |
| **Récupération de thème** (vérité terrain) | **NMI(cluster, topic)** + pureté thématique | ↑ |
| **Séparation interne** | silhouette (cosine) + modularité Leiden | ↑ |
| **Stabilité** | ARI bootstrap inter-runs | ↑ |
| **Coût** | chargement + latence d'encodage + dim | ↓ |

Le **gagnant** est désigné par un **score composite** transparent (normalisation
min-max inter-modèles, pondérée : mixité 30 %, cohérence 25 %, thème 20 %,
silhouette 10 %, stabilité 10 %, modularité 5 %).

**Équité inter-modèles** : graphe **rang-kNN** (k plus proches voisins, **sans
seuil de cosinus absolu**). Les modèles ont des échelles de cosinus différentes
(e5 ≈ 0.83 inter-thèmes, bge ≈ 0.48) ; un seuil fixe en avantagerait un. Le rang
est invariant à l'échelle → comparaison juste.

### Relancer

```bash
# prérequis : la source brute x-stance + l'extra einops (pour nomic-v2)
uv run python -m pipeline.ingest.download --only xstance
uv sync --extra embed-contender          # einops (nomic-embed-text-v2-moe)

# critère d'acceptation — écrit eval/quality_report.md avec des nombres réels
uv run python -m eval.quality_bench

# variantes
uv run python -m eval.quality_bench --models e5-small,bge-m3 --no-bootstrap
uv run python -m eval.quality_bench --n-topics 8 --max-per-cell 150
```

Options : `--models` (alias du registre), `--n-topics`, `--per-cell` /
`--max-per-cell`, `--min-chars`, `--k`, `--resolution`, `--bootstrap N` /
`--no-bootstrap`, `--seed` (défaut 42, reproductible), `--out`.

### Résultat (échantillon par défaut : 2 214 commentaires, DE/FR/IT, 6 thèmes)

**🏆 Gagnant : `nomic-v2`** (composite 0.85), bge-m3 second très proche (0.57),
**e5-small dernier** (0.25).

| Métrique | nomic-v2 | bge-m3 | e5-small |
|---|:--:|:--:|:--:|
| **NMI(cluster, langue)** ↓ | **0.008** | **0.004** | **0.812** |
| Pureté linguistique ↓ | 0.384 | 0.380 | 0.997 |
| NMI(cluster, thème) ↑ | 0.407 | 0.403 | 0.048 |
| Cohérence NPMI ↑ | -0.108 | -0.123 | -0.129 |
| Latence (ms/texte) ↓ | 42 | 92 | 9 |

Lecture : **e5-small (la baseline) ségrège par LANGUE** (NMI langue 0.81, pureté
linguistique 0.997 → « 1 langue = 1 cluster ») et ne retrouve pas les thèmes.
**nomic-v2 et bge-m3 mélangent les langues** (NMI langue ≈ 0) et récupèrent le
thème (NMI thème ≈ 0.40). **Piège à éviter** : e5 a la meilleure
silhouette/modularité/stabilité — parce que des clusters mono-langues sont
internes-ment nets… mais **faux**. Seules NMI(langue) et NMI(thème) le révèlent.
Chiffres exacts, seedés : `eval/quality_report.md`.

### Limites (honnêteté, Playbook §5)

- **Domaine** : x-stance = votations suisses, commentaires courts/argumentés. Le
  transfert vers TikTok (témoignages libres FR, **sans labels multilingues**)
  n'est pas validable ici.
- **IT sous-représenté** dans la source → équilibrage plafonné (quelques milliers
  de commentaires max).
- **NPMI** = co-occurrence document intra-langue (pas de fenêtre glissante
  gensim) : valeurs **comparables entre modèles** (même calcul), pas à des
  benchmarks externes ; négatives ici (textes courts, thèmes larges).
- **Params de clustering figés** (pas de sweep) ; rang-kNN choisi pour l'équité.
- **Vérité `topic`** grossière (thèmes larges) : un clustering plus fin est
  pénalisé sur NMI(thème) tout en restant cohérent.

---

## Banc STANCE — Leiden vs UMAP+HDBSCAN

Sur la même source mais via les labels **FAVOR / AGAINST** par question politique.

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

| Axe | Métriques |
|-----|-----------|
| **Qualité vs vérité terrain** | **NMI**, **ARI**, **pureté** (clusters↔labels) |
| **Qualité interne** | **silhouette** cosine |
| **Stabilité** | **ARI bootstrap** |
| **Coût** | latence clustering, nb d'embeddings, wall-clock |

```bash
uv sync --extra contender          # umap-learn + hdbscan (le contender)
uv run python -m eval.bench --sample-questions 8
uv run python -m eval.bench --sample-questions 20 --bootstrap 10
uv run python -m eval.bench --lang all            # de/fr/it
```

Lecture : sur 8 questions FR, **les deux approches recouvrent mal le clivage
FAVOR/AGAINST** (NMI ≈ 0.04–0.06) — l'embedding capte le **thème**, pas la
**position**. Détecter la polarité demande un signal dédié. Détails :
`eval/report.md`. Limites : 2 classes seulement, votations suisses, params figés,
le bruit HDBSCAN (`-1`) compte comme un cluster (sauf silhouette).

---

## Fichiers

**Banc qualité**
- `multilingual_data.py` — corpus trilingue équilibré (thème × langue) depuis x-stance.
- `coherence.py` — cohérence NPMI maison (par langue, sans gensim).
- `quality_bench.py` — orchestration : un modèle à la fois → métriques → composite → `quality_report.md`.
- `quality_report.md` — dernière sortie committée (régénérable).

**Banc stance**
- `data.py` — chargeur x-stance labellisé FAVOR/AGAINST.
- `metrics.py` — NMI/ARI/pureté/silhouette + agrégation (partagé par les deux bancs).
- `bench.py` — orchestration Leiden vs HDBSCAN.
- `report.py` / `report.md` — rendu + dernière sortie.
