# Validation de la passe de stance contre le gold xstance (FAVOR/AGAINST)

**Branche** `work/stance-validation` · R&D · 2026-06-30
**Question (Bob)** : quelle est la PRÉCISION RÉELLE de notre classifieur de stance face à une vérité terrain ?

## Protocole

- **Données** : `backend/cache/xstance/ideas.jsonl`, 3000 commentaires servis (FAVOR 1565 / AGAINST 1435,
  équilibré ; de/fr/it 1000 chacun). xstance est nativement *question fermée → prise de position* :
  chaque avis porte un gold `props.label ∈ {FAVOR, AGAINST}` et `props.question`.
- **Notre passe, telle quelle** : `backend/build_opinion.py` (`STANCE_SYSTEM` + `stance_batch`,
  `mistral-small-latest`, `temperature=0`, batch=10), **cible = la propre `question` du commentaire**.
  Le modèle prédit `favorable` / `defavorable` / `nuance` + une **confiance auto-déclarée**
  (`high`/`medium`/`low`).
- **Mapping** : `favorable → FAVOR`, `defavorable → AGAINST`, `nuance → abstention`. L'abstention
  est, par construction du gold (toujours net), comptée comme **erreur** dans l'accuracy principale.
- Scripts : `research/run_stance_validation.py` (passe brute → `stance_validation_raw.jsonl`),
  `research/analyze_stance_validation.py` (métriques → `stance_validation_metrics.md`).

## Résultats globaux (n=3000)

| Métrique | Valeur |
|---|---|
| **Accuracy** (abstention = erreur) | **0.672** (2017/3000) |
| Accuracy sur les décidés (abstention exclue) | 0.790 (2017/2554) |
| Taux d'abstention sur gold net | **14.9 %** (446/3000) |

| classe | précision | rappel | F1 | support |
|---|---|---|---|---|
| FAVOR | 0.805 | 0.672 | 0.732 | 1565 |
| AGAINST | 0.774 | 0.673 | 0.720 | 1435 |

**Matrice de confusion**

| gold ↓ / pred → | FAVOR | AGAINST | ABSTENTION | total |
|---|---|---|---|---|
| **FAVOR** | 1051 | 282 | 232 | 1565 |
| **AGAINST** | 255 | 966 | 214 | 1435 |

Lecture : **pas de biais de classe systématique**. Les deux rappels sont quasi identiques (0.672 / 0.673),
l'abstention se répartit à peu près également (232 FAVOR / 214 AGAINST), et les confusions franches
(FAVOR↔AGAINST) sont symétriques (282 vs 255). Quand le modèle tranche, il a raison ~79 % du temps ;
le reste de l'écart vient surtout de son refus de trancher (l'abstention), pas d'une inversion polaire.

## Par langue

| langue | n | accuracy (abst=err) | accuracy décidés | abstention | F1 FAVOR | F1 AGAINST |
|---|---|---|---|---|---|---|
| de | 1000 | 0.683 | 0.795 | 14.1 % | 0.724 | 0.745 |
| fr | 1000 | 0.679 | 0.802 | 15.3 % | 0.739 | 0.732 |
| it | 1000 | 0.655 | 0.772 | 15.2 % | 0.733 | 0.679 |

Les trois langues sont **homogènes** (écart d'accuracy-décidés ≤ 3 pts). L'italien est légèrement
en retrait, concentré sur la classe AGAINST (F1 0.679, rappel 0.642) — le modèle y rate un peu plus
les oppositions. Pas de langue « cassée ».

## Par confiance auto-déclarée — le résultat le plus important

| confiance | n (part) | % abstention | accuracy sur décidés |
|---|---|---|---|
| **high** | 2200 (73 %) | **2.4 %** | **0.808** |
| medium | 700 (23 %) | 42.4 % | 0.695 |
| low | 100 (3 %) | 97.0 % | 0.667 |

Le **score de confiance qu'on vient d'ajouter est bien calibré** et c'est le vrai enseignement :

- Sur les **73 % de prédictions `high`**, le modèle tranche presque toujours (2.4 % d'abstention)
  et atteint **0.808** d'accuracy — son meilleur régime.
- Quand il hésite (`medium`/`low`), il **s'abstient massivement** plutôt que de deviner :
  42 % puis 97 % d'abstention. Autrement dit, l'essentiel de nos 446 abstentions globales tombe
  pile sur les cas qu'il signale lui-même comme incertains.
- La confiance est donc une **bande de fiabilité utilisable** : filtrer sur `high` donne un
  sous-ensemble propre à ~81 %, et les cas écartés sont ceux que le modèle savait douteux.

## Bonus (cible=question vs cible=objet-de-clivage dérivé)

**Non exécuté.** L'objet de clivage dérivé (`derive_cleavage`) opère sur une FEUILLE de thème
(mots-clés + échantillon de contributions d'un cluster), pas sur un commentaire isolé répondant
à une question fermée. xstance fournit déjà la cible idéale et native (la question) : c'est
précisément ce que cette validation mesure. Comparer aux cibles dérivées demanderait de reclusteriser
xstance puis de dériver un clivage par feuille — hors périmètre de cette passe, et déjà tranché
ailleurs ([[agora-opinion-target-verdict]], [[agora-stance-cluster-subject-verdict]] : la cible
per-claim n'est pas agrégeable, l'objet de clivage dérivé à la feuille est la seule cible cohérente
pour MESURER une répartition d'opinion).

## Verdict honnête sur la fiabilité

**xstance est un benchmark adverse** : commentaires politiques courts (votations suisses), stance
souvent implicite, ironique ou conditionnelle. C'est un plancher dur, pas un cas favorable.

1. **Comme classifieur binaire forcé contre gold : honnête mais modeste — 0.672.** Le coût principal
   n'est pas l'inversion polaire (rare et symétrique) mais l'**abstention** (14.9 % de `nuance` sur
   un gold pourtant net). Notre `STANCE_SYSTEM` a une porte de sortie `nuance` légitime et conservatrice ;
   xstance la pénalise par construction.

2. **Comme classifieur calibré qui sait quand se taire : fiable — 0.808 sur les 73 % qu'il assume.**
   La confiance n'est pas cosmétique : elle sépare proprement le « tranchable » du « douteux ».
   C'est exactement le comportement voulu pour une répartition d'opinion honnête, où l'on préfère
   un `nuance` à une fausse certitude.

3. **Pas de pathologie** : pas de biais de classe, pas de langue effondrée, confusions polaires
   symétriques. La passe est saine, juste prudente.

**Conclusion** : sur la tâche binaire dure de xstance, la passe est **correcte sans être brillante**
(0.79 quand elle tranche), et surtout **honnête sur sa propre incertitude** (confiance calibrée :
0.808 en `high`, abstention concentrée sur `medium`/`low`). Pour notre usage réel — agréger une
répartition d'opinion par thème, pas étiqueter chaque commentaire — c'est le bon compromis : on
peut filtrer sur `high` pour une bande propre et traiter l'abstention comme un signal, pas comme
un échec. Le `0.672` brut est un plancher adverse, pas la fiabilité opérationnelle.
