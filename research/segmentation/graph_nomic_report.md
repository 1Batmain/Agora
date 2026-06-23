# Graphe de mots (kNN + Leiden) sur embeddings **NOMIC-v2** — relève-t-il le graphe ? bat-il l'attention ?

*Jeu : `gold_large.json` — N=305. Vecteurs-mots : **nomic-v2** (`embed_word_units`, embed de prod) vs **e5-base** (réf. graphe). CPU, seed=42. Balayage par `graph_seg.py` ; ce rapport lit les JSON de scores et compare (read-only).*

## 0. Réponse courte

**NON — deux fois.** nomic-v2 fait *marginalement* mieux que e5-base au niveau graphe (F1_multi 0.328 vs 0.310 ; F1_global 0.262 vs 0.244 ; Pk 0.404 vs 0.463) mais reste **à un gouffre** de l'attention réglé-main (F1_multi 0.7692). La cause d'échec est IDENTIQUE à e5 : **mono_FP=0.990** — Leiden sur-coupe quasiment tous les mono. Des embeddings de mots meilleurs (moins colinéaires : seuil dérivé ~0.634 vs ~0.922 en e5) ne créent PAS la capacité d'**abstention** qui manque.

## 1. Méthode (rappel)

Identique à `graph_seg.py`, seule la source de vecteurs-mots change. Mots = nœuds ; arêtes = **similarité** (kNN cosinus, seuil dérivé μ−σ poolé, zéro magic-number) **+ séquence** (mots adjacents, poids **α** → quasi-contiguïté). **Leiden** → communautés ; **contiguïté imposée** (runs maximaux ; micro-runs < min_seg fusionnés). Frontières = changements de communauté. Balayage : k∈[5, 10, 20] × α∈[0.0, 0.5, 1.0, 2.0] × résolution∈[0.5, 1.0, 1.5, 2.0, 3.0] × min_seg∈[3, 5] = 120 configs.

## 2. Scorecard — graphe-nomic vs graphe-e5base vs réglé-main (même gold)

| approche | config | Pk | WindowDiff | F1_multi | P | R | mono_FP | F1_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **attention réglé-main** (e5-base) | lowmid/mean W=8 c=1.0 | 0.1493 | 0.1563 | 0.7692 | 0.8955 | 0.6742 | 0.1442 | 0.7453 |
| change-point (embeddings) | changepoint W=8 pen=3.0 | 0.2815 | 0.282 | 0.4423 | 0.4545 | 0.4307 | 0.7019 | 0.384 |
| _appris LR (réf.)_ | thr=0.1 | 0.2007 | 0.2054 | 0.7389 | 0.9027 | 0.6255 | 0.0962 | 0.7214 |
| graphe-Leiden **e5-base** (réf.) | k=10 α=2.0 res=1.0 min=5 thr=0.922 | 0.4627 | 0.5065 | 0.3097 | 0.2279 | 0.4831 | 1.0 | 0.2436 |
| **graphe-Leiden nomic-v2** | k=5 α=2.0 res=0.5 min=5 thr=0.634 | 0.4041 | 0.422 | 0.3283 | 0.2586 | 0.4494 | 0.9904 | 0.262 |

*(Pk/WindowDiff ↓ = mieux, sur les multi ; F1_multi = frontières tol ±1 ; mono_FP = fraction de mono sur-coupés = mesure d'abstention ; F1_global = frontières mono+multi, objectif de sélection.)*

## 3. nomic-v2 vs e5-base au niveau graphe — qu'est-ce qui bouge ?

| métrique | e5-base | nomic-v2 | Δ (nomic−e5) |
| --- | --- | --- | --- |
| F1_multi (frontières multi) | 0.3097 | 0.3283 | 0.0186 |
| F1_global (objectif) | 0.2436 | 0.262 | 0.0184 |
| Pk (↓) | 0.4627 | 0.4041 | -0.0586 |
| WindowDiff (↓) | 0.5065 | 0.422 | -0.0845 |
| mono_FP (↓, abstention) | 1.0 | 0.9904 | -0.0096 |
| seuil-sim dérivé (μ−σ) | 0.922 | 0.634 | -0.288 |

- **Ce qui s'améliore** : Pk/WindowDiff baissent nettement (0.463→0.404) et la sur-coupe est un peu moins violente (le winner nomic met 2.89 communautés/avis, e5 en mettait 4.15). Le **seuil de similarité dérivé chute** (0.922→0.634) : les vecteurs-mots nomic sont **moins colinéaires** que les e5 (μ−σ des cosinus kNN bien plus bas → plus d'« écart » exploitable). C'est le seul vrai signe que nomic a plus de structure au niveau mot.

- **Ce qui ne bouge PAS** : mono_FP reste à **0.990** (e5 : 1.000). C'est LE point de bascule : nomic ne gagne quasi rien sur l'abstention. F1_global ne grimpe que de +0.018 et reste **3× sous** l'attention.

## 4. Top 12 configurations graphe-nomic

| k | alpha | res | min_seg | sim_thr | Pk | WindowDiff | F1_multi | P | R | mono_FP | mono_cuts | n_clust | F1_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 2.0 | 0.5 | 5 | 0.634 | 0.4041 | 0.422 | 0.3283 | 0.2586 | 0.4494 | 0.9904 | 1.779 | 2.89 | 0.262 |
| 5 | 2.0 | 0.5 | 3 | 0.634 | 0.4172 | 0.4436 | 0.3312 | 0.253 | 0.4794 | 1.0 | 2.087 | 2.89 | 0.2586 |
| 20 | 1.0 | 1.0 | 5 | 0.533 | 0.4935 | 0.5427 | 0.3118 | 0.2219 | 0.5243 | 1.0 | 2.365 | 4.06 | 0.2448 |
| 10 | 2.0 | 1.0 | 5 | 0.589 | 0.556 | 0.6161 | 0.3028 | 0.2062 | 0.5693 | 1.0 | 2.51 | 5.04 | 0.2403 |
| 20 | 1.0 | 1.0 | 3 | 0.533 | 0.5359 | 0.638 | 0.3113 | 0.2053 | 0.6442 | 1.0 | 3.183 | 4.06 | 0.2396 |
| 10 | 1.0 | 1.0 | 5 | 0.589 | 0.5236 | 0.573 | 0.3002 | 0.2109 | 0.5206 | 1.0 | 2.365 | 4.48 | 0.2372 |
| 20 | 2.0 | 1.0 | 5 | 0.533 | 0.5307 | 0.5895 | 0.3 | 0.2062 | 0.5506 | 1.0 | 2.721 | 4.75 | 0.2328 |
| 5 | 2.0 | 1.0 | 3 | 0.634 | 0.6135 | 0.7595 | 0.3036 | 0.1919 | 0.7266 | 1.0 | 3.913 | 5.45 | 0.2303 |
| 10 | 2.0 | 1.0 | 3 | 0.589 | 0.5944 | 0.7081 | 0.3021 | 0.194 | 0.6816 | 1.0 | 3.712 | 5.04 | 0.2288 |
| 5 | 2.0 | 1.0 | 5 | 0.634 | 0.5745 | 0.6387 | 0.2849 | 0.1931 | 0.5431 | 1.0 | 2.558 | 5.45 | 0.2259 |
| 20 | 2.0 | 1.0 | 3 | 0.533 | 0.5684 | 0.6664 | 0.2972 | 0.1949 | 0.6255 | 1.0 | 3.529 | 4.75 | 0.224 |
| 5 | 1.0 | 0.5 | 5 | 0.634 | 0.4244 | 0.4402 | 0.2686 | 0.2201 | 0.3446 | 0.9615 | 1.471 | 2.53 | 0.2196 |

## 5. Le nœud — abstention ↔ détection, par résolution (nomic-v2)

Pour chaque résolution : la config qui **abstient le mieux** (mono_FP min) vs celle qui **détecte le mieux** (F1_multi max). Si les deux ne coïncident JAMAIS, aucun réglage global ne sépare « mono cohérent » de « virage de thème » au niveau MOT.

| res | abstient_monoFP | ·_F1_multi | ·_nclust | détecte_F1_multi | ·_monoFP | ·_nclust  |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 0.0 | 0.007 | 1.01 | 0.331 | 1.0 | 2.89 |
| 1.0 | 0.904 | 0.214 | 3.0 | 0.312 | 1.0 | 4.06 |
| 1.5 | 0.923 | 0.192 | 14.08 | 0.277 | 1.0 | 7.7 |
| 2.0 | 0.798 | 0.164 | 20.69 | 0.252 | 1.0 | 8.85 |
| 3.0 | 0.606 | 0.121 | 26.71 | 0.231 | 1.0 | 11.68 |

*Repère : l'attention tient F1_multi=0.7692 ET mono_FP=0.1442 EN MÊME TEMPS. Aucune ligne ci-dessus ne s'en approche : à res basse nomic abstient (mono_FP→0) mais rate AUSSI les multi (F1_multi→0.007) ; dès qu'il détecte (F1_multi max ~0.33) il re-coupe TOUS les mono (mono_FP→1.0). Les deux colonnes ne se rejoignent jamais — exactement comme en e5.*

## 6. Verdict honnête

- **Bat-il l'attention (F1_multi=0.7692, Pk=0.1493, mono_FP=0.1442) ? **NON**.** ΔF1_multi=-0.441, ΔPk=+0.255, ΔF1_global=-0.483.

- **Bat-il le change-point (F1_multi=0.4423) ? **NON**.** ΔF1_multi=-0.114.

- **nomic relève-t-il le graphe ?** À peine. F1_global 0.244→0.262 (+0.018), Pk 0.463→0.404. Mieux, mais dans le même régime d'échec : le verdict graphe (NON) est inchangé.

- **Pourquoi nomic ne sauve pas le graphe — l'abstention, pas la colinéarité.** On pouvait croire que l'échec e5 venait de vecteurs-mots trop colinéaires (seuil dérivé ~0.922, cosinus ~0.9+ → graphe quasi-structureless). nomic INFIRME cette explication-là : son seuil dérivé tombe à ~0.634 (mots bien moins colinéaires, donc plus de structure de similarité disponible) — et POURTANT mono_FP reste à 0.990. Le vrai mal n'est donc pas le manque de signal de similarité, mais que **Leiden ne sait pas s'abstenir** : à résolution fixe il maximise la modularité PAR document et trouve toujours une partition (≥2 communautés) même sur un mono cohérent. C'est structurel à l'objectif Leiden, pas une affaire d'embedding.

- **Ce que l'attention réussit et que le graphe ne peut pas imiter** : un seuil GLOBAL `μ−cσ` calibré sur tout le corpus. Sur un mono, le signal ne descend jamais sous ce seuil → **0 frontière** (mono_FP=0.1442). Leiden n'a aucun équivalent global : sa « résolution » est un curseur de granularité par-document, pas un seuil d'abstention transférable (§5).

- **Conclusion** : le graphe-Leiden de mots sur **nomic-v2 NE BAT NI** l'attention réglé-main (0.7692) **NI** le change-point (0.4423) ; il bat seulement, et de peu, le graphe-e5base (0.3097). De meilleurs embeddings de mots déplacent le seuil de similarité mais ne créent pas l'abstention — qui est le verrou. Piste (déjà notée pour e5, inchangée) : graphe au niveau PHRASE/clause + critère d'abstention explicite (ne couper que si le gain de modularité dépasse un seuil global), ce qui reviendrait à réinventer le seuil calibré de l'attention par un détour plus coûteux.
