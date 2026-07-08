# Cible (c) « pour/contre-cadrée » — bench ENGAGEMENT — VERDICT (2026-07-08)

**Statut : RESEARCH, zéro modif pipeline.** Teste l'idée de Bob : dériver la cible sous la
contrainte « telle que chaque témoignage soit clairement POUR ou CONTRE » donnerait au LLM un
critère opérationnel → cible moins diffuse → moins d'abstention → plus d'engagement (le maillon
faible mesuré). Métrique = ENGAGEMENT (pas le fit, plat pour (b)). Bench contrôlé sur **lutte**
(22 feuilles ≥8 claims) : par feuille, dériver (a) [prompt actuel] ET (c) [pour/contre-cadré],
RE-CLASSER la stance des mêmes claims envers chacune, comparer. Harnais `cleavage_engagement.py`
(réutilise `STANCE_SYSTEM`/`run_stance`/`aggregate` servis).

## Résultat — (c) est PIRE

| | engagement moyen | %nuance | impur | clivant/consensuel |
|---|---:|---:|---:|---|
| **(a) actuel** | **0.806** | **0.194** | 0/22 | 16 / 6 |
| (c) pour/contre-cadré | 0.767 | 0.233 | 0/22 | 15 / 7 |

**Par feuille : (a) plus engagée sur 9, (c) sur 2.** (c) fait MONTER l'abstention (+4 pts de
nuance) et BAISSER l'engagement — l'inverse de l'hypothèse.

## Pourquoi — le piège de la spécificité

La consigne « binaire, tranchante, évite le consensuel » pousse le LLM vers des propositions plus
SPÉCIFIQUES (voire plus extrêmes ou déviantes). Or une cible spécifique est plus DURE à trancher
pour une contribution diffuse → elle tombe en « nuance » (pas de position sur CE point précis).
Exemples :

| feuille | (a) — central, capte large | (c) — binaire mais étroit | eng |
|---|---|---|---|
| n3 | rendre obligatoire l'éducation aux médias | interdire tout partage de fausse info non corrigée **sous 24h** | 0.82→0.55 |
| n18 | soumettre les médias à une autorité indépendante | imposer un **quota égal temps antenne candidats** (dérive) | 0.88→0.75 |
| n22 | interdire les contenus illégaux avant élections | interdire **tout envoi politique 2 semaines avant** | 0.90→0.80 |

**L'engagement est piloté par la CENTRALITÉ de la cible (capte le plus de témoignages), pas par
son caractère binaire.** Le prompt actuel (a) « résume le débat CENTRAL » est mieux calibré pour
l'engagement ; forcer le binaire réduit la couverture.

## Verdict

- **(A) Critère de dérivation « pour/contre-cadré » → NON.** Mesuré pire sur l'engagement
  (−4 pts, 9 feuilles sur 22 dégradées, mécanisme clair : spécificité → abstention). Ne pas
  adopter comme prompt de dérivation.
- **(B) Cadrage d'AFFICHAGE « Pour ou contre : \<cible\> » → OUI, gratuit.** Il ne CHANGE pas la
  cible (on garde (a)) ; il ne fait que la PRÉSENTER comme un choix binaire dans l'UI. Aucun coût,
  aucune mesure requise, gain de clarté utilisateur. C'est la moitié valide de l'idée.
- **Le vrai levier engagement reste ailleurs** (cf. `stance_large_bench.md`) : `large_noabst`
  gaté sur cible dérivée réelle + calibrer `MIN_ENGAGEMENT`. Pas la reformulation de la cible.

**Réserve** : n=22 feuilles, 1 corpus (lutte), arbre rebâti (résolution/seed par défaut, 0 impur
ici alors que le servi en avait 8 — donc le test « (c) sauve-t-il les impur ? » n'a pas de sujet
ici). L'effet sur l'engagement est directionnel mais consistant (9 vs 2) et mécaniquement
expliqué.

## Artefacts

`cleavage_engagement.py` · `cleavage_engagement_lutte-contre-les-fausses-informations.json`.
