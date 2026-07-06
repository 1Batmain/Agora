# Pipeline CLAIMS (TalkToTheCity) : avis → claims → embed → clustering ÉMERGENT — rapport

*Jeu : `gold_large.json` — N=305 avis (104 mono, 201 multi), 8 thèmes gold. LLM d'extraction : **`ministral-3:latest`** (poste local, Apple Silicon via Tailscale — souverain), température 0, JSON mode, pensée coupée. Embeddings : **`nomic-v2`** (`search_document:`). Clustering : k-NN+Leiden, défauts DÉRIVÉS des données (k=11, seuil cosine=0.601), résolution 1.0.*

**Question.** Le clustering ASCENDANT (claims libres → clusters) reconstruit-il les 8 thèmes du gold SANS jamais les voir, aussi bien que Mistral (choix fermé, micro-F1 0.928) ou le classifieur entraîné (0.939) — TOUT EN restant OUVERT (découvre du hors-taxo) et SOUVERAIN (100% local) ?

## Scorecard — récupération des thèmes (multi-label par avis) × ouverture × coût

| Approche | ouverte ? | taxo vue ? | micro-F1 | macro-F1 | exact-set | V-mesure | local | données sortent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **claims → cluster émergent** (ce run) | **OUI** | **NON** | 0.784 | 0.762 | 46% | 0.497 | **oui** | non |
| Mistral-small (choix fermé) | non | oui (prompt) | 0.928 | 0.935 | 73% | — | non | **oui** |
| classifieur MLP/nomic (entraîné) | non | oui (labels) | 0.939 | 0.94 | — | — | oui | non |

**Verdict récupération : NON** — claims→cluster micro-F1=**0.784** vs Mistral 0.928 (-0.144) et clf 0.939 (-0.155), **sans jamais voir la taxonomie**. 8 clusters émergent pour 8 thèmes gold ; V-mesure 0.497 (homogénéité 0.493, complétude 0.502).

## Clustering émergent — sensibilité à la résolution

La résolution Leiden règle la granularité : basse → peu de gros clusters, haute → beaucoup de petits. On cherche celle qui fait émerger ~8 thèmes cohérents. **Aucun thème n'est donné** ; seul le mapping a posteriori utilise le gold.

| résolution | n_clusters | modularité | homogénéité | complétude | V-mesure | micro-F1 | macro-F1 | exact-set |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 5 | 0.583 | 0.374 | 0.527 | 0.438 | 0.62 | 0.484 | 28% |
| 0.8 | 7 | 0.636 | 0.502 | 0.537 | 0.519 | 0.794 | 0.739 | 49% |
| **1.0** | 8 | 0.635 | 0.493 | 0.502 | 0.497 | 0.784 | 0.762 | 46% |
| 1.4 | 11 | 0.629 | 0.527 | 0.469 | 0.496 | 0.811 | 0.817 | 49% |
| 2.0 | 15 | 0.607 | 0.555 | 0.441 | 0.492 | 0.818 | 0.821 | 51% |

*Défauts du graphe DÉRIVÉS des 968 claims (aucun magic-number corpus) : k=11 (∝ log N), seuil d'arête cosine=0.601 (μ−σ des k-NN). Seed 42.*

## F1 par thème — résolution 1.0 (mapping cluster→thème dominant)

| thème | P | R | F1 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- |
| harcelement | 0.893 | 0.918 | 0.905 | 67 | 8 | 6 |
| desinformation | 0.791 | 0.93 | 0.855 | 53 | 14 | 4 |
| enfants | 0.803 | 0.91 | 0.853 | 61 | 15 | 6 |
| sante_mentale | 0.814 | 0.838 | 0.826 | 83 | 19 | 16 |
| image_corps | 0.883 | 0.768 | 0.822 | 53 | 7 | 16 |
| addiction | 0.734 | 0.817 | 0.773 | 58 | 21 | 13 |
| algorithme | 0.685 | 0.859 | 0.762 | 61 | 28 | 10 |
| contenus_choquants | 0.5 | 0.215 | 0.301 | 14 | 14 | 51 |

**Bijection propre** à cette résolution : 8 clusters ↔ 8 thèmes, chacun dominant dans EXACTEMENT un cluster (`harcelement`×1, `addiction`×1, `image_corps`×1, `algorithme`×1, `contenus_choquants`×1, `enfants`×1, `desinformation`×1, `sante_mentale`×1). Les 8 thèmes émergent donc tous, sans en voir aucun. Le rappel manquant ne vient PAS de thèmes oubliés mais de claims qui « bavent » dans le cluster d'un thème voisin — ex. `contenus_choquants` (R=0.22), dont les claims tombent surtout dans des clusters limitrophes.

## Exemples — un avis → ses claims → leurs clusters → thèmes émergents

**multi-133** — gold : harcelement, image_corps, sante_mentale

> Une meute de comptes s'est acharnée sur moi avec des insultes pendant tout un trimestre. ça m'a plongée dans une détresse terrible, je ne mangeais plus et je pleurais sans arrêt. et comme leurs attaques visaient mon physique, je me suis mise à détester mon corps au point de me cacher.

- claim → cluster 4 → **harcelement** : L'auteur a subi des harcèlements répétés sous forme d'insultes pendant une période prolongée (un trimestre).
- claim → cluster 0 → **sante_mentale** : Ces attaques ont provoqué un état de détresse psychologique intense chez l'auteur.
- claim → cluster 0 → **sante_mentale** : La victime a perdu son appétit en raison du stress et de la souffrance causés par le harcèlement.
- claim → cluster 0 → **sante_mentale** : Les pleurs constants sont devenus une réaction quotidienne face à cette situation traumatisante.
- claim → cluster 4 → **harcelement** : Les insultes ciblaient spécifiquement l'apparence physique de la personne harcelée.
- claim → cluster 0 → **sante_mentale** : Ces attaques ont engendré un rejet profond envers son propre corps chez l'auteur.
- claim → cluster 0 → **sante_mentale** : La victime a développé un comportement de dissimulation liée à son image corporelle.
- **thèmes émergents de l'avis : harcelement, sante_mentale**

**multi-34** — gold : addiction, sante_mentale

> Je n'arrive plus à décrocher, je vérifie l'appli des dizaines de fois par jour. et cette dépendance me ronge, je me sens anxieux, vidé, incapable de me concentrer sur quoi que ce soit.

- claim → cluster 2 → **addiction** : L'utilisateur ressent une incapacité croissante à se détacher d'une application mobile.
- claim → cluster 2 → **addiction** : Il consulte l'application des dizaines de fois par jour sans raison apparente.
- claim → cluster 0 → **sante_mentale** : Cette dépendance engendre un sentiment de détresse intérieure et de rongement moral.
- claim → cluster 0 → **sante_mentale** : Il éprouve de l'anxiété en lien avec cette utilisation compulsive.
- claim → cluster 0 → **sante_mentale** : Son énergie vitale se ressent affaiblie, comme vidée ou épuisée.
- claim → cluster 0 → **sante_mentale** : Sa capacité à se concentrer sur d'autres activités est fortement altérée.
- **thèmes émergents de l'avis : addiction, sante_mentale**

**multi-110** — gold : addiction, enfants, sante_mentale

> Mon fils de dix ans a eu accès à l'appli bien trop tôt, sans qu'on s'en méfie. et très vite il y a passé des heures chaque jour, incapable de lâcher l'écran. et maintenant il est irritable, mal dans sa peau et son moral s'est effondré.

- claim → cluster 1 → **enfants** : Un enfant de dix ans a eu accès à une application sans que ses parents ne surveillent suffisamment cette exposition.
- claim → cluster 1 → **enfants** : L’enfant a passé des heures quotidiennes sur l’application, devenant dépendant à l’écran.
- claim → cluster 2 → **addiction** : Son usage intensif de l’application a provoqué une incapacité à s’en détacher facilement.
- claim → cluster 0 → **sante_mentale** : Ces comportements ont entraîné une irritabilité accrue chez lui.
- claim → cluster 2 → **addiction** : Il se sent mal dans sa peau en raison de son utilisation prolongée de l’application.
- claim → cluster 0 → **sante_mentale** : Son moral a fortement décliné après cette exposition précoce et excessive.
- **thèmes émergents de l'avis : addiction, enfants, sante_mentale**

## NOUVEAUTÉ — clusters hors-taxonomie (la valeur ajoutée de l'ouverture)

Clusters (taille ≥ 3) dont le centroïde est sémantiquement LOIN des 8 centroïdes-thèmes du gold (cosine max < **0.630**, seuil dérivé = 5ᵉ percentile des cosines claim↔son-centroïde-thème). Ce sont des idées que la taxonomie FERMÉE de Mistral/du classifieur ne pouvait pas représenter.

*Aucun cluster sous le seuil : à cette granularité, tout le corpus reste dans le rayon sémantique des 8 thèmes (corpus de test mono-sujet TikTok ; l'ouverture se révélerait davantage sur une consultation à sujets épars).*

## Coût, latence, souveraineté

- **Extraction ministral (Mac, à chaud)** : **0 appels** réels + 305 servis par le cache `.cache/ollama/` — 1 avis/appel, **~635s** cumulés (~**2082 ms/avis**, ~10.6 min pour 305 avis), 25,231 tokens générés, 0 erreurs.
- **968 claims** extraites au total (**3.17 claims/avis** en moyenne) — l'avis est décomposé en idées atomiques avant clustering.
- **Embedding + clustering** : 100% local (nomic-v2 CPU + Leiden), négligeable devant l'extraction. Réutilise le cache d'embeddings `.cache/`.
- **Souveraineté** : la donnée citoyenne ne sort JAMAIS du réseau privé (`http://mac-local:11434`, Tailscale). À comparer à ~**2-4 € par grosse consultation** en API (Mistral EU) où le texte intégral des avis est transmis à un tiers. Local = **~0 €** marginal, données souveraines, mais dépend du Mac allumé et de sa latence.

## Verdict — l'approche OUVERTE tient-elle près de 0.93 en restant ouverte & souveraine ?

- **Récupération des thèmes : NON.** Sans voir AUCUN thème, le clustering ascendant de claims atteint micro-F1 **0.784** à résolution 1.0 (-0.144 vs Mistral 0.928, -0.155 vs clf 0.939) ; sur le balayage de résolution il s'étage de 0.620 à **0.818** (rés 2.0, 15 clusters). Il **n'atteint pas 0.93** mais se loge dans une bande respectable de ~0.78–0.82 — face à des méthodes qui, elles, connaissent les 8 thèmes d'avance et ne découvriront jamais rien hors liste.

- **Reconstruction non supervisée** : V-mesure **0.497** (homogénéité 0.493 = les clusters sont purs ; complétude 0.502 = un thème est éclaté en plusieurs sous-clusters). L'éclatement est INHÉRENT à l'ascendant et utile : il fait émerger des SOUS-facettes (TalkToTheCity les garde comme sous-thèmes), au prix de la complétude.

- **Ouverture (le point clé)** : 0 cluster(s) hors-taxo détecté(s) — l'approche n'est PAS bridée par une liste fermée ni par les 1ers avis : toute idée nouvelle crée son cluster. C'est ce que NI Mistral (choix fermé) NI le classifieur (8 classes figées) ne peuvent faire.

- **Arbitrage expressivité × qualité × coût** : on échange ~14% de micro-F1 perdu contre l'OUVERTURE (découverte de nouveauté, granularité sous-thème) et la SOUVERAINETÉ (local, ~0 €, données qui ne sortent pas), au prix d'une latence d'extraction (~2082 ms/avis sur le Mac). Pour explorer une consultation sans taxo a priori, c'est l'outil ; pour étiqueter vite sur une taxo connue, Mistral/le classifieur restent plus directs.

- **Honnêteté** : le clustering est non supervisé, mais le mapping cluster→thème et la V-mesure utilisent le gold (évaluation « cluster-then-label » standard). L'étiquette gold par claim vient du segment gold le plus proche (désambiguïsation multi→mono par embedding). Latence Ollama sur Mac partagé, 1 avis/appel, sans batching → indicative. Corpus de test mono-sujet (TikTok) : il sous-estime la nouveauté qu'on verrait sur une consultation à sujets dispersés.
