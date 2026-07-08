# Qualité de l'engagement — l'objection de Bob, mesurée (2026-07-08)

**Statut : RESEARCH, zéro modif pipeline.** Fait suite à `cleavage_engagement_note.md` (où (c)
« pour/contre-cadré » avait MOINS d'engagement que (a) actuel). Objection de Bob : **moins
d'engagement ≠ moins bien** si les décisions restantes sont de meilleure qualité (ne pas forcer
d'opinion quand la position n'est pas claire — principe d'honnêteté d'Agora). Ce bench mesure la
QUALITÉ des décisions supplémentaires de (a). Harnais `cleavage_quality.py` (juge aveugle neutre,
réutilise les cibles déjà dérivées).

## Résultat — Bob a raison (avec un caveat)

Sur **14 claims** où **(a) tranche mais (c) s'abstient** (lutte) :

- **Confiance de (a) sur ces claims : 6 high, 8 medium, 0 low.** (a) est CONFIANT — la confiance
  auto-déclarée ne détecte PAS ces cas.
- **Juge aveugle neutre** (« position claire sur la cible, oui/non ? ») : **position claire 3/14,
  PAS claire 11/14 → taux de sur-classement de (a) = 79 %.**

Donc les décisions supplémentaires de (a) sont, en majorité, des positions **forcées** que le juge
ne retrouve pas — et (a) les stampe avec confiance. **L'engagement supplémentaire de (a) est
largement du sur-classement**, pas du signal. Sur ces claims, l'abstention de (c) est la bonne
réponse. **La quantité d'engagement n'est pas la qualité.**

## Le mécanisme — sur-classement par ASSOCIATION de sujet

La cible LARGE de (a) capte des témoignages **tangentiels** (même thème, action différente) et
la consigne anti-abstention (`STANCE_SYSTEM` : « si une lecture raisonnable permet de trancher,
TRANCHE ») les classe « favorable ». Exemples (a=high/medium, juge=pas claire) :

| cible_a (large) | témoignage (action AUTRE) | (a) |
|---|---|---|
| rendre obligatoire l'éducation aux médias | « pubs/spots sur la désinformation au cinéma » | favorable (high) |
| obliger les plateformes à modérer | « obliger les ANNONCEURS à être transparents » | favorable (med) |
| renforcer les lois existantes | « Charte de Munich qui existe déjà » | favorable (med) |

La cible ÉTROITE de (c) exige que le témoignage adresse l'action PRÉCISE → elle ne compte pas ces
tangents → moins d'engagement mais plus **on-target**. La lecture de Bob tient.

## Caveat (honnêteté dans les deux sens)

Le juge est un LLM, imparfait : au moins un cas jugé « pas clair » (« n'autoriser que les sites
labellisés en période électorale ») est en fait un soutien clair à la modération que le juge
sous-crédite. Donc **79 % est une borne HAUTE** du sur-classement de (a) ; le taux réel est un peu
plus bas. Mais la DIRECTION est nette et le mécanisme (association de sujet) est concret.

## Verdict révisé

1. **(a) actuel sur-classe par association**, confiance-aveugle — c'est un **problème du STANCE
   servi**, plus large que le débat (a)/(c). La consigne anti-abstention convertit des tangents
   en décisions confiantes.
2. **Bob a raison sur le principe** : privilégier la précision à l'engagement brut. MAIS le bon
   LEVIER n'est pas la reformulation de cible (c) — (c) gagne en précision seulement par EFFET DE
   BORD de sa NARROWNESS (et perd la couverture de vraies positions larges légitimes). Le levier
   propre, cible-agnostique :
   - **un pré-check « le témoignage adresse-t-il CETTE action ? »** avant de classer la stance, ou
   - **assouplir la consigne anti-abstention** / **gate de confiance** (mais la confiance est mal
     calibrée ICI : 0 low sur des sur-classements — à revalider).
3. **Tension ENGAGEMENT ↔ HONNÊTETÉ, désormais sur la table avec des chiffres.** Le verdict
   `stance_large_bench.md` valorisait `large_noabst` POUR son engagement (+8 pts) ; ce résultat
   suggère qu'une part de cet engagement est du sur-classement par association. → **re-mesurer
   large_noabst sur la QUALITÉ (juge aveugle), pas que sur le rendement**, avant de l'adopter.

## Ce qui change pour la lane

Ni (b) ni (c) ne s'adoptent comme prompt de cible. Mais l'objection de Bob a débusqué un vrai
défaut du **STANCE servi** (sur-classement par association, non capté par la confiance) — c'est
le prochain sujet stance à instruire, indépendamment de la cible. Prototype de correctif possible :
un **pré-filtre de pertinence** (le témoignage parle-t-il de l'action ?) avant la classification.

## Artefacts

`cleavage_quality.py` · `cleavage_quality_lutte-contre-les-fausses-informations.json` ·
(amende le verdict de `cleavage_engagement_note.md`).
