# Verdict A/B — arctic-l vs nomic-v2 sur données FR servies (tiktok)

**Date** : 2026-07-08 · **Branche** : `research/bench-embedders` · **Données** : tiktok (2511 claims, FR)
**Réponse : arctic est un choix DÉFENSIBLE pour les prochains runs — marginalement meilleur, à
essayer, PAS à imposer en re-embeddant les caches.** Scorecard + macros : [`ab_embedder_tiktok.md`](ab_embedder_tiktok.md).

## Pourquoi ce test (rappel méthodo)
Sur les vraies données FR il n'y a **pas de gold** (`topic`) → on ne peut PAS mesurer NMI(thème),
l'arbitre décisif du bench x-stance. On évalue donc par (a) métriques **internes** — indicatives,
car elles peuvent récompenser des clusters « propres mais faux » (piège e5) — et surtout (b)
**inspection qualitative** des macro-thèmes. Clustering **de production** (derive_defaults → knn →
Leiden → macro-forest), embed en mémoire, **caches servis intacts**.

## Résultat chiffré (interne, indicatif)

| | nomic-v2 | arctic-l |
|---|:--:|:--:|
| Cohérence NPMI (fr) ↑ | -0.272 | **-0.255** |
| Silhouette ↑ | 0.075 | **0.090** |
| Modularité ↑ | 0.628 | **0.632** |
| Stabilité (ARI) ↑ | 0.656 | **0.669** |
| #macros / #fins | 12 / 12 | 12 / 12 |
| dim | 768 | 1024 |

arctic est **marginalement meilleur sur toutes** les métriques internes — mais les marges sont
**minimes**, et ce sont les métriques « traîtresses ». Accord des partitions NMI(nomic,arctic)=**0.608**
(structure globalement partagée, quelques thèmes recomposés).

## Ce que dit l'inspection qualitative (le vrai juge)
- **Thèmes communs** (les deux les retrouvent proprement) : addiction/dépendance, haine &
  harcèlement, vidéos/contenus choquants, désinstallation, mineurs/collège, réseaux & jeunes.
- **Différence en faveur d'arctic** : il **isole un macro « corps · comparaison · influenceurs »**
  (213 claims) — l'image de soi / comparaison sociale, un préjudice TikTok **reconnu et actionnable**
  pour un rapport civique. nomic **dilue** ce signal dans un « réseaux sociaux · influenceurs »
  plus générique.
- **Différence neutre** : nomic scinde l'affect en « addiction » (300) + « culpabilité/perte » (268) ;
  arctic les fusionne en un « sentiment · perte · addiction » (390). Les deux découpes se défendent.

## Verdict
Le signal x-stance (arctic > nomic) **se confirme, faiblement, sur données FR réelles** :
métriques internes toutes légèrement meilleures + un thème (image de soi) plus net. **Mais le gain
reste modeste** et le coût réel (embed ~2,5× plus lent, dim 1024 = caches +33 %) ne le justifie pas
pour un re-embed global.

**Recommandation opérationnelle :**
1. **Prochains runs = essayer arctic sans rien casser.** Aucun changement de code requis :
   `build_analysis(..., embedder="arctic-l")` (le paramètre existe déjà) embed le NOUVEAU dataset
   en arctic ; les caches servis (nomic) restent intacts. On juge l'arbre produit à l'œil.
2. **Ne PAS** basculer `DEFAULT_EMBEDDER` (constante cross-lane, `MODULES.md §0`) ni re-embedder les
   caches existants tant qu'un run réel n'a pas montré un gain qualitatif **franc** à l'inspection.
3. Si sur 2-3 runs arctic donne systématiquement des thèmes plus lisibles → alors seulement,
   planifier la bascule (re-embed all + re-valider). Sinon, nomic reste le défaut, très correct.

*(A/B reproductible : `research/ab_embedder_fr.py --dataset tiktok`. Aucun cache servi modifié.)*
