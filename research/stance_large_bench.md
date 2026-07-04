# Bench STANCE — mistral-small vs mistral-large (gold x-stance)

**Verdict : servir `small` par défaut ; adopter `large_noabst` (large + consigne
anti-abstention) SI le budget le permet ET après validation sur corpus réel.
`large` seul → NON.**

Validé par Bob. Reproduit le protocole servi (`research/run_stance_validation.py`) :
`stance_batch` / `STANCE_SYSTEM` / T=0, batché par question (BATCH=10), cible = la
question fermée x-stance de chaque contribution, mapping favorable→FAVOR,
defavorable→AGAINST, nuance→ABSTAIN, comparé au gold `props.label`.

- Harnais : `research/stance_large_bench.py` · brut : `research/stance_large_bench_raw.jsonl`
  · métriques détaillées : `research/stance_large_bench_metrics.md`.
- Échantillon **seedé** (SEED=42, ~170 contributions/langue) = **527 contributions**
  (de 179 / fr 178 / it 170 ; FAVOR 292 / AGAINST 235), **le même** pour les 3 configs.
- Coût : 97 batches/config → **~291 appels** Mistral au total (dans la borne 200-400).

## Résultats (mesure propre)

| config | modèle | %nuance | acc décidés | **rendement** (correctes/total) | couverture |
|---|---|---|---|---|---|
| **small** (servi) | mistral-small-latest | 13.5 % | 0.796 | 69 % | 87 % |
| **large** | mistral-large-latest | **25.4 %** | **0.885** | 66 % | 75 % |
| **large_noabst** | mistral-large-latest + anti-abst | **11.0 %** | 0.861 | **77 %** | **89 %** |

« rendement » = part de contributions correctement **décidées** sur le total (abstention
comptée comme non-rendue) — la métrique qui compte pour l'engagement affiché.

Calibration monotone et propre pour les trois (high < medium < low en abstention ;
accuracy des décidés décroissante) — cf. `stance_large_bench_metrics.md`. Homogène
sur les 3 langues (large_noabst : 10-13 % nuance, acc décidés 0.858-0.869 partout).

**Tête-à-tête** (McNemar, sur les items que les DEUX décident) :
- `small` vs `large_noabst` : **33 – 7** en faveur de large_noabst (n=430) → gain d'accuracy
  net et significatif (≈ p<0.001), pas du bruit.
- `large` vs `large_noabst` : **2 – 1** → nul. La consigne anti-abstention **ne coûte RIEN**
  en accuracy sur les décidés ; elle ne fait que convertir des abstentions (majoritairement
  correctes) en décisions.

## Lecture

1. **`large` seul → NON.** Il abstient **2× plus** que small (25 % vs 13 %) sans
   contrepartie d'engagement : son gain d'accuracy (0.885) reste enfermé derrière la
   « nuance ». Rendement 66 % < small 69 %. C'est la signature de l'incident du 4/07
   (sur-classement « nuance »), **confirmée** — même si l'ampleur sur ce gold propre
   (25 %) est loin de l'effondrement observé en prod (voir caveat).

2. **`large_noabst` → le meilleur des trois.** L'ajout d'une consigne prioritaire
   « ne réserve “nuance” qu'aux contributions VRAIMENT sans position, sinon tranche »
   ramène l'abstention **sous celle de small** (11 % vs 13.5 %) tout en gardant l'essentiel
   de l'avantage d'accuracy de large (0.861 décidés vs 0.796 pour small). Rendement **77 %**,
   +8 pts vs small. C'est le levier qui neutralise l'effondrement d'engagement.

3. **`small` (servi) reste un défaut sûr.** Calibration propre, engagement correct,
   coût bas (RPM élevé, pas de backoff requis), accuracy 0.796 décidés — cohérent avec le
   verdict de validation existant (0.79). Aucune régression : on ne le retire que si
   large_noabst est validé sur corpus réel et budgété.

## Piège méthodologique majeur (à retenir)

**Sans backoff, `large` FABRIQUE de la fausse « nuance ».** Premier run (6 workers,
sans retry) : `large` affichait **80 % de nuance** — dont l'écrasante majorité étaient en
réalité des **échecs d'appel** (429 RPM bas de large) tombant sur le repli
`nuance/(échec LLM)`, PAS des décisions du modèle (fr et it à 100 % d'« échec LLM »).
Après ajout d'un **backoff exponentiel** (repris de `pipeline/claims/backend.py`, retries
sur 429/5xx/réseau) + concurrence réduite à 3 workers pour large : **0 échec**, vraie
nuance = 25 %. → **Toute passe stance sur large DOIT avoir un backoff/RPM-headroom**,
sinon elle se dégrade silencieusement en abstention massive. Ceci explique probablement
une partie de l'incident du 4/07 si la passe prod large manquait de marge RPM.

## Caveat — ce gold n'est pas le corpus servi

x-stance = réponses de sondage **mono-sujet**, avec une **cible = question fermée explicite**
(oui/non). C'est BEAUCOUP plus facile à trancher que les contributions libres et diffuses
de la prod (granddebat/tiktok), dont la cible est un **objet de clivage DÉRIVÉ**. Donc :
- ce bench **sous-estime** l'abstention de large sur corpus réel — l'effondrement du 4/07
  était sur du réel, pas sur ce gold ;
- le gain d'accuracy de large_noabst y est **certain** ; son niveau d'engagement réel ne
  l'est pas tant qu'on ne l'a pas mesuré sur granddebat/tiktok.

## Recommandation opérationnelle

- **Maintenant** : garder `small` en prod (rien à changer, sûr et peu coûteux).
- **Si l'on veut la qualité-max** : basculer sur **`large_noabst`** — c.-à-d. ajouter la
  consigne anti-abstention à `STANCE_SYSTEM` **et** exiger un backoff (RPM). Gate avant
  rollout : rejouer sur un échantillon **granddebat/tiktok** pour confirmer que
  l'engagement tient hors du cadre « question explicite ».
- **Ne jamais** servir `large` **sans** la consigne anti-abstention **ni** sans backoff.
- **Seuils** : inutile d'ajouter un filtre de confiance dur — les garde-fous existants
  (`MIN_ENGAGEMENT=0.35`, pureté) suffisent, et large_noabst relève naturellement
  l'engagement. La confiance auto-déclarée reste bien calibrée (high ≫ low) pour un
  affichage prudent si besoin.
