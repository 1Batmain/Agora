# Pré-filtre de pertinence avant stance — VERDICT (2026-07-08)

**Statut : RESEARCH, zéro modif pipeline.** Correctif du sur-classement par association mesuré
dans `cleavage_quality_note.md` (le STANCE servi stampe « favorable » des témoignages tangentiels,
confiant). Correctif CIBLE-AGNOSTIQUE : avant la stance, un pré-filtre « ce témoignage PORTE-t-il
sur CETTE action ? » (pertinence, indépendante du pour/contre) ; non-pertinent → abstention. On
GARDE la cible large (a) actuelle. Harnais `relevance_prefilter.py`, corpus lutte (cible (a) déjà
dérivée), juge aveugle neutre robuste (backoff).

## Résultat — le filtre marche

| | valeur |
|---|---|
| claims classés | 268 |
| décidés BASELINE | 224 |
| décidés APRÈS filtre | **148** (−76, −34 %) |
| retraits corrects (juge « pas de position claire ») | **56 / 76 = 76 %** |
| retraits erronés (vraie position jetée) | 18 / 76 = 24 % |
| **confiance des retirés** | **43 high · 33 medium · 0 low** |
| précision des GARDÉS-décidés (échantillon 20) | **16/20 = 0.80** |

### Lecture — le compromis précision/rappel

En estimant les vraies positions (gardés clairs ≈ 0.80×148 = 118, + 18 retirés-clairs) ≈ **136** :
- **précision baseline ≈ 136/224 = 0.61 → filtrée 0.80** (**+19 pts**) ;
- **rappel des vraies positions ≈ 118/136 = 0.87** (on perd 13 %).

**Gros gain de précision pour une perte de rappel modérée.** Le filtre retire majoritairement des
sur-classements (76 %), et — point clé — il attaque des décisions **CONFIANTES** (43 high) que la
confiance auto-déclarée ne signalait pas. C'est exactement le défaut que l'objection de Bob a
débusqué, corrigé sans toucher à la cible.

## Verdict

- **Le pré-filtre de pertinence est un correctif VALIDE et adoptable du STANCE.** Précision
  0.61 → 0.80, rappel ~0.87, cible-agnostique. Il enlève le sur-classement par association que ni
  la cible ni la confiance ne réglaient.
- **À calibrer** : 24 % de retraits erronés = le filtre est un peu strict (le prompt « porte sur
  cette action » peut être assoupli pour récupérer des positions indirectes légitimes). Knob.
- **Intégration prod** : en une seule passe — ajouter un champ `porte_sur_action` à `STANCE_SYSTEM`
  et ne classer fav/def QUE si vrai (sinon nuance). Pas de passe LLM en plus. (Ici séparé pour
  isoler l'effet.)

## Réserves

- 1 corpus (lutte), n=268 claims, 76 retraits jugés, échantillon gardés = 20 (bruité). À rejouer
  sur un 2ᵉ corpus (granddebat/tiktok) avant rollout.
- Juge = LLM (même famille) — circularité douce, atténuée par un cadrage DIFFÉRENT du filtre
  (« porte sur l'action » ≠ « position claire ») et confirmée par le 24 % de désaccord filtre↔juge
  (ils ne sont pas identiques). Un panel humain serait le gold.
- **Impact sur le verdict `large_noabst`** : son +8 pts d'engagement est à re-mesurer AVEC ce
  pré-filtre (une part est probablement du sur-classement par association). Prochaine étape stance.

## Artefacts

`relevance_prefilter.py` · `relevance_prefilter_lutte-contre-les-fausses-informations.json`.
Chaîne : `cleavage_engagement_note.md` → `cleavage_quality_note.md` → ce note.
