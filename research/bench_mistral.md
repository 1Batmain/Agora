# Verdict R&D — mistral-embed (API EU) vs nomic-v2 (le servi)

**Date** : 2026-07-07 · **Branche** : `research/bench-embedders` · **Suite de** [`bench_veille.md`](bench_veille.md)
**Réponse : NON — mistral-embed est NETTEMENT PIRE que nomic** sur le besoin d'Agora. Il **regroupe
par LANGUE**, pas par thème (le piège e5, mesuré). Protocole identique (nomic re-validé 0.008/0.407).

## Résultat chiffré (gold x-stance, n=2214 de/fr/it équilibré)

| Métrique | sens | **nomic-v2** (témoin) | **mistral-embed** |
|---|:--:|:--:|:--:|
| **NMI(cluster, langue)** | ↓ | **0.008** | **0.642** ⚠️ |
| Pureté linguistique | ↓ | 0.384 | **0.990** ⚠️ |
| **NMI(cluster, thème)** | ↑ | **0.407** | **0.262** |
| Pureté thématique | ↑ | 0.649 | 0.461 |
| Cohérence NPMI | ↑ | -0.106 | -0.188 |
| Silhouette | ↑ | 0.069 | 0.046 |
| Modularité (Leiden) | ↑ | 0.613 | **0.711** |
| Stabilité (ARI) | ↑ | 0.723 | 0.542 |
| Dimension | · | 768 | 1024 |
| Latence (ms/txt, API) | ↓ | — | ~13 (réseau, au build) |
| **Composite** | ↑ | **0.950** | 0.050 |

Scorecard brute : [`quality_report_mistral.md`](quality_report_mistral.md).

## Lecture — mistral-embed est dans la famille du piège e5

- **Pureté linguistique 0.990** ≈ « 1 langue = 1 cluster » (e5-small était à 0.997). Le NMI
  langue 0.642 confirme : mistral-embed **ségrège massivement par langue** au lieu du thème.
- Conséquence directe : **récupération de thème en chute** (NMI thème 0.262 vs 0.407 nomic ;
  −36 %). Pour un produit trilingue FR/DE/IT dont le besoin #1 est « regrouper par thème, pas
  par langue », c'est disqualifiant.
- ⚠️ **Piège des métriques internes, re-illustré** : mistral-embed a une **meilleure modularité**
  que nomic (0.711 > 0.613) — mais parce que des clusters mono-langues sont *internement* nets.
  Seuls NMI(langue)/NMI(thème) voient que le clustering est **faux**. (Même leçon qu'e5.)

**Pourquoi ?** `mistral-embed` est un embedder généraliste orienté retrieval/RAG, non optimisé
pour l'alignement cross-lingue de thèmes courts. nomic-v2 (et granite/arctic) sont entraînés
pour mixer les langues par sens — mistral-embed non.

## Contexte architectural & souveraineté (même si la qualité tranche déjà)

- **Compatible au build** : le pipeline embed en dev (clé Mistral) et la prod sert le cache ;
  mistral-embed s'utiliserait donc **au build**, pas au service → la prod « sans clé » tient.
- **MAIS** : (1) dépendance à un **service payant externe** pour la brique cœur (vs modèle
  Apache auto-hébergé) ; (2) le **corpus quitte la machine** vers l'API au build (Mistral est
  EU/RGPD — acceptable pour du public comme x-stance, à peser pour des données FR sensibles) ;
  (3) non reproductible hors-ligne / non épinglable comme un poids.
- Ces points seraient à débattre **si** la qualité était au rendez-vous. Elle ne l'est pas.

## Verdict

**NON.** mistral-embed est **mesurablement pire** que nomic pour le clustering multilingue par
thème (NMI thème 0.262 vs 0.407 ; pureté linguistique 0.990 = piège e5). Aucune raison de qualité
d'y passer, et cela ajouterait une dépendance API payante + une sortie de données au build.

**Le paysage se clarifie :** les seuls candidats qui **battent** nomic restent les **modèles
Apache auto-hébergés** de la veille (arctic-l 0.455, granite-311m 0.437). La piste API Mistral est
close. Reco inchangée : valider arctic-l / granite-311m / granite-97m sur un dataset FR servi réel.

*(Coût du bench : ~2200 embeddings mistral-embed, quelques centimes ; corpus public.)*
