# Banc QUALITÉ de clustering — embeddings multilingues

> ⚠️ **Scorecard BRUTE (auto-générée) — PAS le verdict.** Le « 🏆 Recommandation :
> jina-v3 » ci-dessous ne tient compte QUE des métriques de clustering. **jina-v3 est
> CC-BY-NC-4.0 (non-commercial) → INADOPTABLE dans Agora**, quelle que soit sa qualité.
> **Verdict décisionnel (licence + coût) → [`research/bench_jina.md`](bench_jina.md).**

> Quel modèle d'embedding regroupe le mieux les avis par **thème** plutôt que par **langue** ? Réponse **par la mesure** : e5-small vs nomic-v2 vs bge-m3, même pipeline (rang-kNN → Leiden), seul l'embedding change.

## 🏆 Recommandation : **jina-v3** (`tomaarsen/jina-embeddings-v3-hf`)

Score composite **0.931** (pondéré : mixité linguistique 30 %, cohérence 25 %, récupération de thème 20 %, silhouette 10 %, stabilité 10 %, modularité 5 %).

## Corpus (honnêteté)

- **2214** commentaires x-stance, équilibrés par (thème × langue).
- Langues : {'it': 738, 'de': 738, 'fr': 738} (équilibrées → entropie ~max, NMS(cluster,langue) interprétable).
- 6 thèmes (vérité terrain `topic`) : Economy, Education, Immigration, Infrastructure & Environment, Society, Welfare.
- Filtre : commentaires ≥ 15 caractères, dédup exact. seed=42.
- Clustering : rang-kNN k=15 (sans seuil de cosinus, équité inter-modèles), Leiden resolution=1.0, seed=42.
- Bootstrap : 4 ré-échantillons (fraction 0.8). Python 3.11.15, CPU. Wall None s.

## Scorecard

| Métrique | sens | jina-v3 | nomic-v2 | e5-small |
|---|:--:|:--:|:--:|:--:|
| Cohérence NPMI (intra-langue) | ↑ | -0.056 | -0.106 | -0.159 |
| **NMI(cluster, langue)** | ↓ | 0.003 | 0.008 | 0.812 |
| Pureté linguistique | ↓ | 0.376 | 0.384 | 0.997 |
| NMI(cluster, thème) | ↑ | 0.482 | 0.407 | 0.048 |
| Pureté thématique | ↑ | 0.742 | 0.649 | 0.215 |
| Silhouette (cosine) | ↑ | 0.113 | 0.069 | 0.072 |
| Modularité (Leiden) | ↑ | 0.726 | 0.613 | 0.679 |
| Stabilité (ARI bootstrap) | ↑ | 0.776 | 0.723 | 0.897 |
| Nb clusters | · | 15 | 13 | 6 |
| Dimension | · | 1024 | 768 | 384 |
| Chargement (s) | ↓ | 4.8 | 13.2 | 6.8 |
| Latence (ms/texte) | ↓ | 210.52 | 44.57 | 8.95 |
| **Score composite** | ↑ | 0.931 | 0.592 | 0.135 |

## Détail du score composite (normalisé min-max inter-modèles, ∈ [0,1])

| Composante (poids) | jina-v3 | nomic-v2 | e5-small |
|---|:--:|:--:|:--:|
| nmi_lang (30 %) | 1.000 | 0.994 | 0.000 |
| coherence (25 %) | 1.000 | 0.512 | 0.000 |
| nmi_topic (20 %) | 1.000 | 0.827 | 0.000 |
| silhouette (10 %) | 1.000 | 0.000 | 0.059 |
| stability (10 %) | 0.307 | 0.000 | 1.000 |
| modularity (5 %) | 1.000 | 0.000 | 0.584 |

## Cohérence NPMI détaillée par langue

| Modèle | de | fr | it | moyenne |
|---|:--:|:--:|:--:|:--:|
| jina-v3 | -0.121 | 0.000 | -0.048 | -0.056 |
| nomic-v2 | -0.195 | -0.050 | -0.075 | -0.106 |
| e5-small | -0.019 | -0.116 | -0.343 | -0.159 |

## Lecture

- **jina-v3** gagne : NMI(cluster,langue)=0.003 (mixité — bas = les clusters ne ségrègent PAS par langue), cohérence=-0.056, NMI(cluster,thème)=0.482 (récupère le thème).
- Pire mixité : **e5-small** (NMI langue=0.812, pureté linguistique 0.997) — regroupe par LANGUE, pas par thème.
- La mixité linguistique est le critère central : un NMI(cluster,langue) élevé trahit un modèle qui sépare les langues au lieu des thèmes.
- ⚠️ **Piège des métriques internes** : **e5-small** a la meilleure silhouette/modularité/stabilité — mais ses clusters sont mono-langues (pureté linguistique 0.997). Des clusters internes nets mais **faux** : silhouette, modularité et stabilité récompensent la solution dégénérée « 1 langue = 1 cluster ». D'où le rôle **décisif** de NMI(cluster,langue) et NMI(cluster,thème), qui seuls voient que le clustering n'a pas trouvé les thèmes.
- **jina-v3 vs nomic-v2** : mixité quasi identique (NMI langue 0.003 vs 0.008) ; jina-v3 l'emporte sur la cohérence (-0.056 vs -0.106), la récupération de thème (0.482 vs 0.407) et/ou le coût (211 vs 45 ms/texte). nomic-v2 reste un second proche.

## Limites (ce qui n'est PAS testé)

- **Domaine** : x-stance = votations suisses (DE/FR/IT), commentaires courts et argumentés. Le transfert vers TikTok (témoignages libres FR) n'est pas validé ici (pas de labels multilingues sur TikTok).
- **IT sous-représenté** dans la source ; l'équilibrage plafonne donc la taille par cellule. Échantillon de quelques milliers de commentaires.
- **Cohérence NPMI** = co-occurrence document intra-langue (pas de fenêtre glissante gensim) ; valeurs comparables entre modèles (même calcul), pas à des benchmarks externes.
- **Params de clustering figés** (pas de sweep) ; rang-kNN choisi pour l'équité, mais un tuning par modèle pourrait déplacer les marges.
- Le **topic x-stance** (12 thèmes larges) est une vérité terrain grossière : un clustering plus fin que les topics est pénalisé sur NMI(thème) mais peut rester cohérent.

