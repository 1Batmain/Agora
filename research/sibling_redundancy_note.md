# Verdict — la redondance entre thèmes frères est SÉMANTIQUE, pas géométrique (2026-07-15)

**Question :** sur tiktok, les feuilles sont redondantes pour un lecteur (~5 clusters
d'« addiction », ~4 de « tristesse », ~3 de « filles/enfants » exprimés autrement). La chaîne
optimise l'emboîtement, jamais la distinctivité entre frères. Peut-on DÉTECTER cette redondance
géométriquement, pour l'afficher ou fusionner — sans LLM ?

**Réponse : NON. Aucun signal géométrique fiable.**

| mesure (espace recentré) | même sujet | sujets différents | discrimination |
|---|---|---|---|
| cos des centroïdes | 0.38–0.53 | 0.38–0.53 | **aucune** |
| affinité kNN inter-cluster (k=15) | 0.032 | 0.021 | **1.51×**, faible |

Et les paires les PLUS liées par voisinage sont *cross-sujet* (dépendance ↔ tristesse 0.127 ;
harcèlement ↔ violence 0.102), pas les mêmes-sujets. L'embedding place « addiction exprimée de
5 façons » dans 5 régions réellement séparées ; le corpus entremêle les préoccupations (un
témoignage mélange addiction, temps perdu, enfants).

**Conséquences :**
- Pas de fusion par proximité (centroïde ou kNN) possible — elle soit ne fusionne rien, soit
  fusionne n'importe quoi. (Confirme a posteriori pourquoi sauce_magique / coarsening étaient
  instables.)
- La redondance est un problème de SENS. Seul un juge sémantique (LLM « même sujet ? ») peut la
  trancher — étape PAYANTE, à décider.
- Une part est irréductible : le corpus mélange vraiment les thèmes. « Thèmes » = facettes
  récurrentes, pas sujets orthogonaux.

Mesure reproductible (gratuite) : `research/sibling_redundancy.py`.
Prochaine étape possible : panel LLM sur les paires de feuilles → matrice de redondance
sémantique → afficher la confiance, voire fusionner les frères jugés identiques.
