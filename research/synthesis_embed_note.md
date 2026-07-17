# Verdict — ré-embedder une SYNTHÈSE canonique convertit la redondance sémantique en proximité géométrique (2026-07-17)

**Question (Bob).** La redondance entre thèmes frères est sémantique, pas géométrique (les 5
clusters d'« addiction » sont à 0.38-0.53 dans l'espace des claims, comme des sujets
différents). Peut-on la rendre géométrique en ré-embeddant une SYNTHÈSE LLM de chaque cluster
(surface → sens), pour la couche d'abstraction récursive ?

**Réponse : OUI — à condition que la synthèse soit une ÉTIQUETTE CANONIQUE, pas une phrase riche.**

## Mesure (tiktok, 4 clusters addiction + 3 clusters « filles », cosinus recentré)

| prompt de synthèse | intra-addiction | intra-filles | addiction↔filles | verdict |
|---|---|---|---|---|
| claims bruts (référence) | 0.38–0.53 | — | 0.38–0.53 | **aucune discrimination** |
| phrase abstraite riche | −0.09 | −0.13 | −0.21 | ordre correct mais séparation FAIBLE |
| **étiquette canonique 3-6 mots** | **+0.49** | −0.00 | **−0.44** | **séparation FRANCHE (écart ~0.93)** |

Étiquettes canoniques obtenues : les 4 addictions → « addiction aux réseaux sociaux » (×3) +
« addiction aux écrans ». Les 3 filles → « Risques réseaux » / « Violences en ligne » / « Normes
de beauté toxiques » (distinctes — même public, sujets différents, correctement NON fusionnés).

## Lecture

- Le LLM fait la **normalisation surface→sens** (une étiquette par cluster, bornée, cachable) ;
  la géométrie fait ensuite la **fusion** (impossible sur les claims bruts, faisable sur les
  étiquettes). C'est le levier qui manquait à la couche d'abstraction.
- **Le prompt est décisif** : une phrase riche ré-introduit la facette (« addiction comme
  évitement » vs « comme contrôle de l'attention ») → séparation faible ; une catégorie
  canonique courte collapse les facettes → séparation forte.
- Ça résout la **redondance** (5 addictions → 1 macro) ET l'**intuitivité** (macros = sujets
  canoniques, pas fourre-tout), sans souder à tort des sujets qui partagent un public.

## Reste à faire / risques

- **Calibrer la granularité de l'étiquette** : trop canonique → sur-fusion ; à mesurer.
- **Valider hors tiktok** (échantillon = 7 clusters, un corpus).
- Architecture cible : couche plate (γ + pic de modularité) → étiquette canonique par thème →
  ré-embedding → clustering des étiquettes = couche macro → récursif. Juge LLM (a) en assurance.

Repro : `research/synthesis_embed_test.py` (phrase) + variante canonique inline.
Résultats : `research/synthesis_embed_results.json`.
