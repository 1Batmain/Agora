# Expressivité de la NORME des embeddings (normalisé vs brut) — verdict

**Concern (Bob)** : on stocke des vecteurs L2-normalisés (norme≡1, vérifié) → on jette la magnitude. Perte d'expressivité ?

**Piège** : nomic-v2 a un module `Normalize` (idx 2) dans son pipeline sentence-transformers → `encode(normalize_embeddings=False)`
rend QUAND MÊME des vecteurs unitaires. La magnitude native vit dans le POOLED (modules [Transformer, Pooling]) AVANT le Normalize ;
on tronque le modèle (`[:2]`) pour la récupérer (`research/embed_norm.py`).

## La norme porte-t-elle un signal ? (pooled brut, xstance n=3000)
| mesure | valeur | lecture |
|---|--:|---|
| cv(norme) | 0.064 | la norme varie de ~6% (range 11.3–16.2) |
| corr(norme, LONGUEUR) | **−0.85** | FORTE : la norme = surtout la longueur (mean-pooling : + de tokens → moyenne plus courte) |
| eta2(norme \| topic, 12) | 0.002 | ~ZÉRO signal thématique |
| eta2(norme \| stance gold) | 0.000 | ZÉRO (FAVOR 14.00 = AGAINST 13.99) |

## Verdict
**La magnitude native de nomic est un ARTEFACT DE LONGUEUR, pas de l'expressivité sémantique.** 85% de sa variance = longueur ;
elle ne porte rien sur le thème ni la stance. → Normaliser jette du bruit de longueur, pas du sens : la normalisation est **inoffensive
(voire bénéfique)**, on ne perd rien de sémantique. Cohérent avec le verdict poids-d'arêtes ([[agora-knn-weight-verdict]]) : retoucher
les magnitudes ne crée pas de signal. **Corollaire constructif** : la norme native ne portant rien d'utile, on a le terrain libre pour
y INJECTER une masse externe (likes/engagement, cf. [[like-weighted-embeddings]] todo) à sa place.
