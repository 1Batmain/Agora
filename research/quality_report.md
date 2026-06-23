# Banc QUALITÉ de clustering — embeddings multilingues

> Quel modèle d'embedding regroupe le mieux les avis par **thème** plutôt que par **langue** ? Réponse **par la mesure** : e5-small vs nomic-v2 vs bge-m3, même pipeline (rang-kNN → Leiden), seul l'embedding change.

## 🏆 Recommandation : **nomic-v2** (`nomic-ai/nomic-embed-text-v2-moe`)

Score composite **0.850** (pondéré : mixité linguistique 30 %, cohérence 25 %, récupération de thème 20 %, silhouette 10 %, stabilité 10 %, modularité 5 %).

## Corpus (honnêteté)

- **2214** commentaires x-stance, équilibrés par (thème × langue).
- Langues : {'it': 738, 'de': 738, 'fr': 738} (équilibrées → entropie ~max, NMS(cluster,langue) interprétable).
- 6 thèmes (vérité terrain `topic`) : Economy, Education, Immigration, Infrastructure & Environment, Society, Welfare.
- Filtre : commentaires ≥ 15 caractères, dédup exact. seed=42.
- Clustering : rang-kNN k=15 (sans seuil de cosinus, équité inter-modèles), Leiden resolution=1.0, seed=42.
- Bootstrap : 4 ré-échantillons (fraction 0.8). Python 3.11.15, CPU. Wall 354.6 s.

## Scorecard

| Métrique | sens | nomic-v2 | bge-m3 | e5-small |
|---|:--:|:--:|:--:|:--:|
| Cohérence NPMI (intra-langue) | ↑ | -0.108 | -0.123 | -0.129 |
| **NMI(cluster, langue)** | ↓ | 0.008 | 0.004 | 0.812 |
| Pureté linguistique | ↓ | 0.384 | 0.380 | 0.997 |
| NMI(cluster, thème) | ↑ | 0.407 | 0.403 | 0.048 |
| Pureté thématique | ↑ | 0.649 | 0.637 | 0.215 |
| Silhouette (cosine) | ↑ | 0.069 | 0.060 | 0.072 |
| Modularité (Leiden) | ↑ | 0.613 | 0.613 | 0.679 |
| Stabilité (ARI bootstrap) | ↑ | 0.723 | 0.665 | 0.907 |
| Nb clusters | · | 13 | 12 | 6 |
| Dimension | · | 768 | 1024 | 384 |
| Chargement (s) | ↓ | 10.5 | 6.1 | 10.4 |
| Latence (ms/texte) | ↓ | 42.30 | 92.15 | 9.39 |
| **Score composite** | ↑ | 0.850 | 0.567 | 0.250 |

## Détail du score composite (normalisé min-max inter-modèles, ∈ [0,1])

| Composante (poids) | nomic-v2 | bge-m3 | e5-small |
|---|:--:|:--:|:--:|
| nmi_lang (30 %) | 0.995 | 1.000 | 0.000 |
| coherence (25 %) | 1.000 | 0.277 | 0.000 |
| nmi_topic (20 %) | 1.000 | 0.988 | 0.000 |
| silhouette (10 %) | 0.777 | 0.000 | 1.000 |
| stability (10 %) | 0.238 | 0.000 | 1.000 |
| modularity (5 %) | 0.002 | 0.000 | 1.000 |

## Cohérence NPMI détaillée par langue

| Modèle | de | fr | it | moyenne |
|---|:--:|:--:|:--:|:--:|
| nomic-v2 | -0.200 | -0.042 | -0.082 | -0.108 |
| bge-m3 | -0.191 | -0.042 | -0.135 | -0.123 |
| e5-small | 0.013 | -0.097 | -0.302 | -0.129 |

## Lecture

- **nomic-v2** gagne : NMI(cluster,langue)=0.008 (mixité — bas = les clusters ne ségrègent PAS par langue), cohérence=-0.108, NMI(cluster,thème)=0.407 (récupère le thème).
- Pire mixité : **e5-small** (NMI langue=0.812, pureté linguistique 0.997) — regroupe par LANGUE, pas par thème.
- La mixité linguistique est le critère central : un NMI(cluster,langue) élevé trahit un modèle qui sépare les langues au lieu des thèmes.
- ⚠️ **Piège des métriques internes** : **e5-small** a la meilleure silhouette/modularité/stabilité — mais ses clusters sont mono-langues (pureté linguistique 0.997). Des clusters internes nets mais **faux** : silhouette, modularité et stabilité récompensent la solution dégénérée « 1 langue = 1 cluster ». D'où le rôle **décisif** de NMI(cluster,langue) et NMI(cluster,thème), qui seuls voient que le clustering n'a pas trouvé les thèmes.
- **nomic-v2 vs bge-m3** : mixité quasi identique (NMI langue 0.008 vs 0.004) ; nomic-v2 l'emporte sur la cohérence (-0.108 vs -0.123), la récupération de thème (0.407 vs 0.403) et/ou le coût (42 vs 92 ms/texte). bge-m3 reste un second proche.

## Limites (ce qui n'est PAS testé)

- **Domaine** : x-stance = votations suisses (DE/FR/IT), commentaires courts et argumentés. Le transfert vers TikTok (témoignages libres FR) n'est pas validé ici (pas de labels multilingues sur TikTok).
- **IT sous-représenté** dans la source ; l'équilibrage plafonne donc la taille par cellule. Échantillon de quelques milliers de commentaires.
- **Cohérence NPMI** = co-occurrence document intra-langue (pas de fenêtre glissante gensim) ; valeurs comparables entre modèles (même calcul), pas à des benchmarks externes.
- **Params de clustering figés** (pas de sweep) ; rang-kNN choisi pour l'équité, mais un tuning par modèle pourrait déplacer les marges.
- Le **topic x-stance** (12 thèmes larges) est une vérité terrain grossière : un clustering plus fin que les topics est pénalisé sur NMI(thème) mais peut rester cohérent.

