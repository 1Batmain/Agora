# Banc QUALITÉ de clustering — embeddings multilingues

> 📌 **Scorecard BRUTE.** mistral-embed **ségrège par LANGUE** (pureté ling. 0.990) — le « 🏆 » interne ne le voit pas. **Verdict → [`research/bench_mistral.md`](bench_mistral.md).**

> Quel modèle d'embedding regroupe le mieux les avis par **thème** plutôt que par **langue** ? Réponse **par la mesure** : e5-small vs nomic-v2 vs bge-m3, même pipeline (rang-kNN → Leiden), seul l'embedding change.

## 🏆 Recommandation : **nomic-v2** (`nomic-ai/nomic-embed-text-v2-moe`)

Score composite **0.950** (pondéré : mixité linguistique 30 %, cohérence 25 %, récupération de thème 20 %, silhouette 10 %, stabilité 10 %, modularité 5 %).

## Corpus (honnêteté)

- **2214** commentaires x-stance, équilibrés par (thème × langue).
- Langues : {'it': 738, 'de': 738, 'fr': 738} (équilibrées → entropie ~max, NMS(cluster,langue) interprétable).
- 6 thèmes (vérité terrain `topic`) : Economy, Education, Immigration, Infrastructure & Environment, Society, Welfare.
- Filtre : commentaires ≥ 15 caractères, dédup exact. seed=42.
- Clustering : rang-kNN k=15 (sans seuil de cosinus, équité inter-modèles), Leiden resolution=1.0, seed=42.
- Bootstrap : 4 ré-échantillons (fraction 0.8). Python 3.11.15, CPU. Wall None s.

## Scorecard

| Métrique | sens | nomic-v2 | mistral-embed |
|---|:--:|:--:|:--:|
| Cohérence NPMI (intra-langue) | ↑ | -0.106 | -0.188 |
| **NMI(cluster, langue)** | ↓ | 0.008 | 0.642 |
| Pureté linguistique | ↓ | 0.384 | 0.990 |
| NMI(cluster, thème) | ↑ | 0.407 | 0.262 |
| Pureté thématique | ↑ | 0.649 | 0.461 |
| Silhouette (cosine) | ↑ | 0.069 | 0.046 |
| Modularité (Leiden) | ↑ | 0.613 | 0.711 |
| Stabilité (ARI bootstrap) | ↑ | 0.723 | 0.542 |
| Nb clusters | · | 13 | 12 |
| Dimension | · | 768 | 1024 |
| Chargement (s) | ↓ | 15.0 | 0.0 |
| Latence (ms/texte) | ↓ | 107.59 | 13.18 |
| **Score composite** | ↑ | 0.950 | 0.050 |

## Détail du score composite (normalisé min-max inter-modèles, ∈ [0,1])

| Composante (poids) | nomic-v2 | mistral-embed |
|---|:--:|:--:|
| nmi_lang (30 %) | 1.000 | 0.000 |
| coherence (25 %) | 1.000 | 0.000 |
| nmi_topic (20 %) | 1.000 | 0.000 |
| silhouette (10 %) | 1.000 | 0.000 |
| stability (10 %) | 1.000 | 0.000 |
| modularity (5 %) | 0.000 | 1.000 |

## Cohérence NPMI détaillée par langue

| Modèle | de | fr | it | moyenne |
|---|:--:|:--:|:--:|:--:|
| nomic-v2 | -0.195 | -0.050 | -0.075 | -0.106 |
| mistral-embed | -0.229 | -0.025 | -0.311 | -0.188 |

## Lecture

- **nomic-v2** gagne : NMI(cluster,langue)=0.008 (mixité — bas = les clusters ne ségrègent PAS par langue), cohérence=-0.106, NMI(cluster,thème)=0.407 (récupère le thème).
- Pire mixité : **mistral-embed** (NMI langue=0.642, pureté linguistique 0.990) — regroupe par LANGUE, pas par thème.
- La mixité linguistique est le critère central : un NMI(cluster,langue) élevé trahit un modèle qui sépare les langues au lieu des thèmes.
- **nomic-v2 vs mistral-embed** : mixité quasi identique (NMI langue 0.008 vs 0.642) ; nomic-v2 l'emporte sur la cohérence (-0.106 vs -0.188), la récupération de thème (0.407 vs 0.262) et/ou le coût (108 vs 13 ms/texte). mistral-embed reste un second proche.

## Limites (ce qui n'est PAS testé)

- **Domaine** : x-stance = votations suisses (DE/FR/IT), commentaires courts et argumentés. Le transfert vers TikTok (témoignages libres FR) n'est pas validé ici (pas de labels multilingues sur TikTok).
- **IT sous-représenté** dans la source ; l'équilibrage plafonne donc la taille par cellule. Échantillon de quelques milliers de commentaires.
- **Cohérence NPMI** = co-occurrence document intra-langue (pas de fenêtre glissante gensim) ; valeurs comparables entre modèles (même calcul), pas à des benchmarks externes.
- **Params de clustering figés** (pas de sweep) ; rang-kNN choisi pour l'équité, mais un tuning par modèle pourrait déplacer les marges.
- Le **topic x-stance** (12 thèmes larges) est une vérité terrain grossière : un clustering plus fin que les topics est pénalisé sur NMI(thème) mais peut rester cohérent.

