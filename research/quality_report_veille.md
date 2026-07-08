# Banc QUALITÉ de clustering — embeddings multilingues

> 📌 **Scorecard BRUTE (veille 2026-07-07).** Le « 🏆 » ne pondère PAS le coût de re-embed
> ni la vitesse CPU. **Verdict décisionnel (tradeoffs + reco) → [`research/bench_veille.md`](bench_veille.md).**
> Latences mesurées ici sous contention (7 modèles en série) → indicatives.

> Quel modèle d'embedding regroupe le mieux les avis par **thème** plutôt que par **langue** ? Réponse **par la mesure** : e5-small vs nomic-v2 vs bge-m3, même pipeline (rang-kNN → Leiden), seul l'embedding change.

## 🏆 Recommandation : **granite-311m-r2** (`ibm-granite/granite-embedding-311m-multilingual-r2`)

Score composite **0.834** (pondéré : mixité linguistique 30 %, cohérence 25 %, récupération de thème 20 %, silhouette 10 %, stabilité 10 %, modularité 5 %).

## Corpus (honnêteté)

- **2214** commentaires x-stance, équilibrés par (thème × langue).
- Langues : {'it': 738, 'de': 738, 'fr': 738} (équilibrées → entropie ~max, NMS(cluster,langue) interprétable).
- 6 thèmes (vérité terrain `topic`) : Economy, Education, Immigration, Infrastructure & Environment, Society, Welfare.
- Filtre : commentaires ≥ 15 caractères, dédup exact. seed=42.
- Clustering : rang-kNN k=15 (sans seuil de cosinus, équité inter-modèles), Leiden resolution=1.0, seed=42.
- Bootstrap : 4 ré-échantillons (fraction 0.8). Python 3.11.15, CPU. Wall 4888.5 s.

## Scorecard

| Métrique | sens | granite-311m-r2 | qwen3-0.6b | arctic-l | granite-97m-r2 | nomic-v2 | e5-large-instruct | e5-small |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Cohérence NPMI (intra-langue) | ↑ | -0.103 | -0.093 | -0.118 | -0.111 | -0.106 | -0.077 | -0.159 |
| **NMI(cluster, langue)** | ↓ | 0.004 | 0.005 | 0.004 | 0.004 | 0.008 | 0.504 | 0.812 |
| Pureté linguistique | ↓ | 0.376 | 0.380 | 0.374 | 0.374 | 0.384 | 0.869 | 0.997 |
| NMI(cluster, thème) | ↑ | 0.437 | 0.408 | 0.455 | 0.406 | 0.407 | 0.271 | 0.048 |
| Pureté thématique | ↑ | 0.703 | 0.673 | 0.682 | 0.684 | 0.649 | 0.482 | 0.215 |
| Silhouette (cosine) | ↑ | 0.088 | 0.072 | 0.087 | 0.077 | 0.069 | 0.055 | 0.072 |
| Modularité (Leiden) | ↑ | 0.681 | 0.652 | 0.694 | 0.643 | 0.613 | 0.604 | 0.679 |
| Stabilité (ARI bootstrap) | ↑ | 0.703 | 0.747 | 0.702 | 0.694 | 0.723 | 0.616 | 0.897 |
| Nb clusters | · | 12 | 12 | 14 | 13 | 13 | 11 | 6 |
| Dimension | · | 768 | 1024 | 1024 | 384 | 768 | 1024 | 384 |
| Chargement (s) | ↓ | 5.1 | 3.1 | 4.9 | 4.1 | 14.9 | 4.8 | 7.0 |
| Latence (ms/texte) | ↓ | 233.85 | 1071.12 | 103.28 | 88.76 | 59.09 | 612.67 | 10.07 |
| **Score composite** | ↑ | 0.834 | 0.804 | 0.802 | 0.738 | 0.722 | 0.474 | 0.193 |

## Détail du score composite (normalisé min-max inter-modèles, ∈ [0,1])

| Composante (poids) | granite-311m-r2 | qwen3-0.6b | arctic-l | granite-97m-r2 | nomic-v2 | e5-large-instruct | e5-small |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| nmi_lang (30 %) | 0.999 | 0.999 | 1.000 | 1.000 | 0.994 | 0.381 | 0.000 |
| coherence (25 %) | 0.680 | 0.811 | 0.500 | 0.581 | 0.641 | 1.000 | 0.000 |
| nmi_topic (20 %) | 0.954 | 0.883 | 1.000 | 0.878 | 0.882 | 0.547 | 0.000 |
| silhouette (10 %) | 1.000 | 0.522 | 0.960 | 0.674 | 0.437 | 0.000 | 0.515 |
| stability (10 %) | 0.310 | 0.466 | 0.306 | 0.277 | 0.380 | 0.000 | 1.000 |
| modularity (5 %) | 0.854 | 0.526 | 1.000 | 0.437 | 0.104 | 0.000 | 0.830 |

## Cohérence NPMI détaillée par langue

| Modèle | de | fr | it | moyenne |
|---|:--:|:--:|:--:|:--:|
| granite-311m-r2 | -0.172 | -0.069 | -0.069 | -0.103 |
| qwen3-0.6b | -0.185 | -0.032 | -0.061 | -0.093 |
| arctic-l | -0.179 | -0.089 | -0.086 | -0.118 |
| granite-97m-r2 | -0.179 | -0.054 | -0.101 | -0.111 |
| nomic-v2 | -0.195 | -0.050 | -0.075 | -0.106 |
| e5-large-instruct | -0.154 | 0.056 | -0.134 | -0.077 |
| e5-small | -0.019 | -0.116 | -0.343 | -0.159 |

## Lecture

- **granite-311m-r2** gagne : NMI(cluster,langue)=0.004 (mixité — bas = les clusters ne ségrègent PAS par langue), cohérence=-0.103, NMI(cluster,thème)=0.437 (récupère le thème).
- Pire mixité : **e5-small** (NMI langue=0.812, pureté linguistique 0.997) — regroupe par LANGUE, pas par thème.
- La mixité linguistique est le critère central : un NMI(cluster,langue) élevé trahit un modèle qui sépare les langues au lieu des thèmes.
- ⚠️ **Piège des métriques internes** : **e5-small** a la meilleure silhouette/modularité/stabilité — mais ses clusters sont mono-langues (pureté linguistique 0.997). Des clusters internes nets mais **faux** : silhouette, modularité et stabilité récompensent la solution dégénérée « 1 langue = 1 cluster ». D'où le rôle **décisif** de NMI(cluster,langue) et NMI(cluster,thème), qui seuls voient que le clustering n'a pas trouvé les thèmes.
- **granite-311m-r2 vs qwen3-0.6b** : mixité quasi identique (NMI langue 0.004 vs 0.005) ; granite-311m-r2 l'emporte sur la cohérence (-0.103 vs -0.093), la récupération de thème (0.437 vs 0.408) et/ou le coût (234 vs 1071 ms/texte). qwen3-0.6b reste un second proche.

## Limites (ce qui n'est PAS testé)

- **Domaine** : x-stance = votations suisses (DE/FR/IT), commentaires courts et argumentés. Le transfert vers TikTok (témoignages libres FR) n'est pas validé ici (pas de labels multilingues sur TikTok).
- **IT sous-représenté** dans la source ; l'équilibrage plafonne donc la taille par cellule. Échantillon de quelques milliers de commentaires.
- **Cohérence NPMI** = co-occurrence document intra-langue (pas de fenêtre glissante gensim) ; valeurs comparables entre modèles (même calcul), pas à des benchmarks externes.
- **Params de clustering figés** (pas de sweep) ; rang-kNN choisi pour l'équité, mais un tuning par modèle pourrait déplacer les marges.
- Le **topic x-stance** (12 thèmes larges) est une vérité terrain grossière : un clustering plus fin que les topics est pénalisé sur NMI(thème) mais peut rester cohérent.

