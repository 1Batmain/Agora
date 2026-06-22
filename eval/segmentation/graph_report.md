# Segmentation par GRAPHE de mots (kNN + Leiden) — bat-elle l'attention ?

*Jeu : `gold_large.json` — N=305 (104 mono, 201 multi). Sources de vecteurs-mots : e5-base. CPU, seed=42.*

## 1. Méthode — mots = graphe, communautés = segments

- **Vecteurs-mots** [n,dim] L2-norm : `e5-base` via `word_attention(text).V` (MÊME forward que l'attention réglée → apples-to-apples) ; `nomic-v2` via `embed_word_units` (embed de prod).
- **Graphe** = arêtes de **similarité** (kNN cosinus, seuil dérivé μ−σ poolé sur tous les avis) **+** arêtes de **séquence** (mots adjacents, poids **α**). L'adjacence force des communautés ~contiguës ; **α=0 = pure similarité = le piège** (gardé comme contrôle).
- **Leiden** (`igraph`+`leidenalg`, RBConfiguration, seed fixe) → communautés. **Contiguïté IMPOSÉE** : segments = runs maximaux de même communauté le long de la séquence ; micro-runs < `min_seg` fusionnés dans le voisin le plus grand. Frontières = changements de communauté.
- **Balayage** : k∈[5, 10, 20] × α∈[0.0, 0.5, 1.0, 2.0] × résolution∈[0.5, 1.0, 1.5, 2.0, 3.0] × min_seg∈[3, 5]. La **résolution** = granularité (mono cohérent → 1 communauté = 0 frontière ; multi → 2-3).

## 2. Scorecard — graphe-Leiden vs références (même gold)

| approche | config | Pk | WindowDiff | F1_multi | P | R | mono_FP | F1_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **attention réglé-main** (e5-base) | lowmid/mean W=8 c=1.0 | 0.1493 | 0.1563 | 0.7692 | 0.8955 | 0.6742 | 0.1442 | 0.7453 |
| change-point (embeddings) | changepoint W=8 pen=3.0 | 0.2815 | 0.282 | 0.4423 | 0.4545 | 0.4307 | 0.7019 | 0.384 |
| _appris LR (réf.)_ | thr=0.1 | 0.2007 | 0.2054 | 0.7389 | 0.9027 | 0.6255 | 0.0962 | 0.7214 |
| **graphe-Leiden e5-base** | k=10 α=2.0 res=1.0 min=5 | 0.4627 | 0.5065 | 0.3097 | 0.2279 | 0.4831 | 1.0 | 0.2436 |

*(Pk/WindowDiff ↓ = mieux, sur les multi ; F1_multi = frontières tol ±1 ; mono_FP = fraction de mono sur-coupés ; F1_global = frontières mono+multi, objectif de sélection.)*

## 3. Top 15 configurations graphe-Leiden

| source | k | alpha | res | min_seg | sim_thr | Pk | WindowDiff | F1_multi | P | R | mono_FP | mono_cuts | n_clust | F1_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e5-base | 10 | 2.0 | 1.0 | 5 | 0.922 | 0.4627 | 0.5065 | 0.3097 | 0.2279 | 0.4831 | 1.0 | 2.173 | 4.15 | 0.2436 |
| e5-base | 10 | 0.0 | 1.0 | 5 | 0.922 | 0.4415 | 0.4641 | 0.2814 | 0.2227 | 0.382 | 0.9231 | 1.481 | 3.6 | 0.2321 |
| e5-base | 10 | 1.0 | 1.0 | 5 | 0.922 | 0.4675 | 0.4958 | 0.2861 | 0.2202 | 0.4082 | 1.0 | 1.788 | 3.53 | 0.23 |
| e5-base | 5 | 0.0 | 2.0 | 5 | 0.932 | 0.4823 | 0.5135 | 0.2731 | 0.2061 | 0.4045 | 0.9519 | 1.596 | 8.91 | 0.2257 |
| e5-base | 20 | 0.5 | 2.0 | 5 | 0.905 | 0.4815 | 0.5159 | 0.2688 | 0.1986 | 0.4157 | 0.9712 | 1.567 | 20.22 | 0.2245 |
| e5-base | 10 | 0.5 | 1.0 | 3 | 0.922 | 0.5224 | 0.619 | 0.2828 | 0.1885 | 0.5655 | 1.0 | 2.75 | 3.26 | 0.223 |
| e5-base | 5 | 0.5 | 1.0 | 5 | 0.932 | 0.4954 | 0.5254 | 0.2716 | 0.2026 | 0.412 | 0.9904 | 1.76 | 4.09 | 0.2216 |
| e5-base | 5 | 1.0 | 0.5 | 3 | 0.932 | 0.4543 | 0.5105 | 0.2602 | 0.1973 | 0.382 | 0.625 | 1.327 | 2.0 | 0.2213 |
| e5-base | 10 | 1.0 | 1.0 | 3 | 0.922 | 0.5284 | 0.6347 | 0.2826 | 0.1864 | 0.5843 | 1.0 | 2.952 | 3.53 | 0.2211 |
| e5-base | 10 | 2.0 | 1.0 | 3 | 0.922 | 0.548 | 0.6827 | 0.2852 | 0.1841 | 0.633 | 1.0 | 3.308 | 4.15 | 0.2211 |
| e5-base | 10 | 0.5 | 1.0 | 5 | 0.922 | 0.4668 | 0.495 | 0.2693 | 0.2091 | 0.3783 | 0.9615 | 1.606 | 3.26 | 0.2203 |
| e5-base | 10 | 0.0 | 1.0 | 3 | 0.922 | 0.5014 | 0.574 | 0.2748 | 0.1877 | 0.5131 | 1.0 | 2.538 | 3.6 | 0.2173 |
| e5-base | 5 | 0.0 | 1.5 | 5 | 0.932 | 0.4858 | 0.5155 | 0.2642 | 0.1989 | 0.3933 | 0.9327 | 1.654 | 7.11 | 0.2172 |
| e5-base | 5 | 2.0 | 0.5 | 3 | 0.932 | 0.4198 | 0.4577 | 0.2594 | 0.2067 | 0.3483 | 0.7788 | 1.346 | 2.02 | 0.217 |
| e5-base | 5 | 0.0 | 1.5 | 3 | 0.932 | 0.5437 | 0.6375 | 0.2734 | 0.1823 | 0.5468 | 1.0 | 2.788 | 7.11 | 0.215 |

## 4. Le piège de contiguïté — effet de α et de la résolution


**e5-base** — meilleure config (F1_global) par poids d'adjacence α :

| source | k | alpha | res | min_seg | sim_thr | Pk | WindowDiff | F1_multi | P | R | mono_FP | mono_cuts | n_clust | F1_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e5-base | 10 | 0.0 | 1.0 | 5 | 0.922 | 0.4415 | 0.4641 | 0.2814 | 0.2227 | 0.382 | 0.9231 | 1.481 | 3.6 | 0.2321 |
| e5-base | 20 | 0.5 | 2.0 | 5 | 0.905 | 0.4815 | 0.5159 | 0.2688 | 0.1986 | 0.4157 | 0.9712 | 1.567 | 20.22 | 0.2245 |
| e5-base | 10 | 1.0 | 1.0 | 5 | 0.922 | 0.4675 | 0.4958 | 0.2861 | 0.2202 | 0.4082 | 1.0 | 1.788 | 3.53 | 0.23 |
| e5-base | 10 | 2.0 | 1.0 | 5 | 0.922 | 0.4627 | 0.5065 | 0.3097 | 0.2279 | 0.4831 | 1.0 | 2.173 | 4.15 | 0.2436 |


**e5-base** — meilleure config par résolution (granularité) :

| source | k | alpha | res | min_seg | sim_thr | Pk | WindowDiff | F1_multi | P | R | mono_FP | mono_cuts | n_clust | F1_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e5-base | 5 | 1.0 | 0.5 | 3 | 0.932 | 0.4543 | 0.5105 | 0.2602 | 0.1973 | 0.382 | 0.625 | 1.327 | 2.0 | 0.2213 |
| e5-base | 10 | 2.0 | 1.0 | 5 | 0.922 | 0.4627 | 0.5065 | 0.3097 | 0.2279 | 0.4831 | 1.0 | 2.173 | 4.15 | 0.2436 |
| e5-base | 5 | 0.0 | 1.5 | 5 | 0.932 | 0.4858 | 0.5155 | 0.2642 | 0.1989 | 0.3933 | 0.9327 | 1.654 | 7.11 | 0.2172 |
| e5-base | 5 | 0.0 | 2.0 | 5 | 0.932 | 0.4823 | 0.5135 | 0.2731 | 0.2061 | 0.4045 | 0.9519 | 1.596 | 8.91 | 0.2257 |
| e5-base | 5 | 0.5 | 3.0 | 5 | 0.932 | 0.5306 | 0.5641 | 0.2348 | 0.1694 | 0.382 | 0.9712 | 1.798 | 12.03 | 0.1932 |

## 4b. Frontière détection ↔ abstention (le nœud)

Pour chaque résolution : la config qui **abstient le mieux** (mono_FP min) vs celle qui **détecte le mieux** (F1_multi max). Si les deux ne coïncident JAMAIS, c'est qu'aucun réglage global ne distingue « mono cohérent » de « virage de thème » au niveau MOT.


**e5-base** :

| res | abstient_monoFP | ·_F1_multi | ·_nclust | détecte_F1_multi | ·_monoFP | ·_nclust  |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 0.0 | 0.069 | 1.15 | 0.26 | 0.625 | 2.0 |
| 1.0 | 0.856 | 0.233 | 2.54 | 0.31 | 1.0 | 4.15 |
| 1.5 | 0.635 | 0.196 | 15.07 | 0.273 | 1.0 | 7.11 |
| 2.0 | 0.346 | 0.133 | 21.84 | 0.273 | 0.952 | 8.91 |
| 3.0 | 0.192 | 0.082 | 27.31 | 0.238 | 1.0 | 12.56 |


*Repère : l'attention réglé-main tient F1_multi=0.7692 ET mono_FP=0.1442 EN MÊME TEMPS. Aucune ligne ci-dessus ne s'en approche : quand le graphe abstient sur les mono, il abstient aussi sur les multi (F1_multi s'effondre) ; quand il détecte, il sur-coupe tout. Les deux colonnes ne se rejoignent jamais.*

## 5. Verdict honnête

**Meilleure config graphe : `e5-base` · k=10 · α=2.0 · res=1.0 · min_seg=5** → F1_multi=0.310 (P=0.228, R=0.483), Pk=0.463, WindowDiff=0.507, F1_global=0.244, mono_FP=1.000, 4.15 communautés/avis.

- **Bat-elle l'attention réglé-main (F1_multi=0.7692, Pk=0.1493, F1_global=0.7453, mono_FP=0.1442) ? **NON**.** ΔF1_multi=-0.459, ΔPk=+0.313 (négatif = mieux), ΔF1_global=-0.502.

- vs **change-point** (F1_multi=0.4423) : ΔF1_multi=-0.133 → le graphe ne bat pas le change-point.

- **Pourquoi ça rate — Leiden ne sait pas s'ABSTENIR.** C'est le résultat central (§4b). L'attention/le change-point calibrent un seuil GLOBAL `μ−cσ` : sur un mono cohérent, le signal ne descend jamais sous le seuil → **0 frontière**. Leiden, lui, maximise la modularité PAR document : à résolution fixe il trouve toujours une partition, même dans un graphe quasi-structureless (problème connu des communautés spurious / limite de résolution). Résultat : à res basse il collapse TOUT en 1 communauté (mono_FP→0 mais F1_multi→0.06, il rate aussi les multi) ; à res haute il coupe TOUT (F1_multi↑ mais mono_FP→1.0). **Les deux régimes ne se rejoignent jamais** au point de fonctionnement de l'attention (F1=0.7692, mono_FP=0.1442 SIMULTANÉMENT).

- **Pourquoi au niveau MOT il n'y a pas de structure** : les vecteurs-mots contextuels e5 sont quasi-colinéaires (seuil de similarité dérivé ≈0.92, μ−σ des cosinus kNN très haut → cosinus tous ~0.9+). Un mono et un multi présentent donc à peu près la même (faible) structure de communauté : il n'existe pas de granularité Leiden qui sépare « cohérent » de « virage ». La **béquille de contiguïté** (arêtes de séquence α + fusion des micro-runs) réimpose bien l'ordre, mais ne crée pas le signal de frontière qui manque.

- **La contiguïté (le piège) est gérée mais non décisive** : α=0 (pure similarité) sur-coupe à peine plus que α>0 (§4) — le mal vient de l'absence d'abstention, pas seulement de la non-contiguïté. Imposer les runs + fusionner les micro-runs évite les communautés entrelacées mais ne sauve pas le verdict.

- **Jeu** : multi = concaténation de mono-thèmes (frontières nettes par construction) → borne OPTIMISTE pour toutes les approches. mono_FP mesure l'abstention (un mono cohérent doit donner 1 seule communauté = 0 coupe).

- **Conclusion** : le graphe-Leiden de mots **NE BAT PAS** l'attention réglé-main (0.769) ni même le change-point (0.44) ; il échoue sur l'**abstention**, qui est justement ce que le seuil global `μ−cσ` de l'attention réussit. Piste si on y tenait : un graphe au niveau PHRASE/clause (moins de nœuds, structure plus nette) + un critère d'abstention explicite (ne couper que si la modularité gagnée dépasse un seuil global) — mais ce serait réinventer le seuil calibré de l'attention par un détour plus coûteux.
