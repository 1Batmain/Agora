# A/B extraction : prompt ACTUEL (A) vs prompt RELÂCHÉ (B) — qualité des claims

**R&D pur** (`research/` only, aucun fichier produit modifié). Branche `work/extract-ab`.
Modèle d'extraction **et** juge : `mistral-large-latest`, température 0, gate verbatim
`align_spans` appliqué aux DEUX bras (seuls les claims ancrés mot-pour-mot sont comptés).
Échantillon **200 avis**, grand débat prioritaire (110 longs/multi-thèmes + 35 tiktok +
30 xstance + 25 république numérique). Reproductible : `python -m research.extract_ab --judge`.

## Hypothèse (Bob)
La cible étant **rétrogradée** (n'est plus le sujet de stance, cf. `agora-stance-subject-verdict`),
on peut **relâcher l'obligation dure de cible** — qui faisait *dropper* des claims (mode
paresseux, 88→61 % dans le verdict cible-repli) — et viser une **segmentation COMPLÈTE des
avis multi-thèmes**, sans sacrifier verbatim/sélectivité/regroupement.

## Les deux bras
- **A** = `CLAIM_SYS` de prod (`pipeline.claims.extract`), tel quel. Cible **obligatoire** :
  « si tu n'arrives pas à pointer l'objet… NE L'EXTRAIS PAS ».
- **B** = prompt relâché (`CLAIM_SYS_B`, dans `research/extract_ab.py`). **GARDE** sélectivité,
  verbatim strict, regroupement. **CHANGE** : (a) `target` = **indice OPTIONNEL** (`null` OK,
  jamais dropper un claim de fond faute de cible) ; (b) règle n°1 = **COMPLÉTUDE** : sur un avis
  multi-thèmes, capturer CHAQUE prise de position, séparer les thèmes distincts, n'en oublier aucun.

## Quantitatif (200 avis, gate verbatim aux deux)

| Métrique | A (actuel) | B (relâché) | Δ |
|---|---|---|---|
| **claims / avis — TOUS** | 3.28 | **3.77** | **+15 %** |
| **claims / avis — GRAND DÉBAT** | 4.65 | **5.39** | **+16 %** |
| taux de passage **verbatim** | 0.982 | **0.988** | +0.6 pt |
| % claims à **cible** exploitable | 0.66 | 0.56 | −10 pt *(attendu : cible relâchée)* |
| longueur moyenne des spans (car.) | 121 | 119 | ≈ |
| % avis **vides** (0 claim) | 4.0 % | 2.0 % | −2 pt |
| n claims valides | 656 | 753 | +97 |

Par dataset (claims/avis) : **grand débat +16 %**, tiktok +23 %, xstance +4 %, repnum +3 %.

**Lecture.** B capture **+15 % de claims**, concentré là où c'était l'enjeu (avis longs/argumentés
du grand débat, +16 %), **sans dégrader le verbatim** (98.8 % ≥ 98.2 %) et **sans raccourcir les
spans** (119 ≈ 121 car. → le gain ne vient pas d'un découpage en miettes, mais de positions
distinctes en plus). Moins d'avis « vides ». Le % de cible baisse (66→56 %), conséquence **voulue**
du relâchement — ce n'est pas une régression mais le mécanisme même de B (ne plus dropper).

## LLM-juge neutre (40 avis grand débat multi-thèmes, lots anonymisés, ordre alterné)

Le juge reçoit l'avis + les deux lots de claims (« lot 1 / lot 2 », ordre alterné pour annuler le
biais de position) et tranche, par dimension indépendante, quel lot est le meilleur.

| Dimension | A gagne | B gagne | nul | Verdict |
|---|---|---|---|---|
| **Complétude** (capture le plus de positions réelles) | 6 | **15** | 19 | **B** (2.5×) |
| **Segmentation** (1 claim = 1 idée, ni fusion ni coupe) | 9 | **21** | 10 | **B** (2.3×) |
| **Bruit / propreté** (le moins de narratif/redite/sur-frag.) | **20** | 13 | 7 | **A** |

**Net** des victoires de lot (B − A) : complétude **+9**, segmentation **+12**, bruit **−7** →
**+14 net en faveur de B**. B gagne nettement les deux dimensions visées (rappel, segmentation) ;
A reste plus **propre** — B paie sa complétude par un surcroît de bruit modéré (narratif/redite),
mais ce coût est plus que compensé par le gain de rappel+segmentation.

## Exemples côte à côte

### Ex. 1 — grand débat `1-9230` (multi-thèmes : démocratie + moralisation) — A 7 claims, B 12
Juge : *« le lot 2 capture plus de prises de position (morale en politique, crise de crédibilité,
détails des propositions) et segmente mieux, mais le lot 1 est plus épuré ».*

- **A** sort les propositions **nues** : « interdiction de casier non vierge », « rendre public le
  patrimoine », « avoir travaillé dans la société civile 5 ans », « pas plus de deux mandats »,
  « interdiction de cumul »… mais **laisse tomber** le diagnostic qui les motive.
- **B** garde chaque proposition **avec sa justification regroupée** (« Rendre public le détail de
  son patrimoine… *L'élu est au service du peuple, pas là pour s'enrichir* ») **et** capture les
  prises de position de fond qu'A a omises : « la morale en politique est un principe inatteignable »,
  « La démocratie souffre d'une crise de crédibilité, donc de confiance ».
- ⚠️ Coût : B ajoute 1-2 fragments plus narratifs (« ces mêmes citoyens savent penser… ») → le
  surcroît de bruit que le juge pointe.

### Ex. 2 — grand débat `1-19459` (mille-feuille territorial) — A 4 claims, B 7
- **A** fusionne en un claim : « moins d'élus, trop de strates au niveau des collectivités… ».
- **B** capture en plus le thème **« d'où le mille feuille entre état et les différents niveaux des
  collectivités »** (qu'A laisse tomber) — vrai gain de complétude.
- ⚠️ Mais B segmente aussi « moins d'élus » / « trop de strates » / « trop de responsabilités
  dispersées » en claims séparés là où ce sont des facettes d'**une** idée → illustre le **risque de
  sur-fragmentation** quand la complétude est poussée. Net : le juge tranche quand même pour B
  (gain du thème mille-feuille > coût de la coupe).

## Verdict — **OUI, adopter B** (chiffré)

B **améliore la qualité** : **+15 % de rappel** global (**+16 %** sur le grand débat, la cible),
**segmentation nettement meilleure** (juge 21-9) et **complétude nettement meilleure** (15-6),
le tout **à verbatim égal/supérieur** (98.8 %) et **sans allongement/raccourcissement** des spans.
La baisse du % de cible (66→56) est l'effet **recherché** du relâchement, pas un défaut.

**Nuance honnête à la question « sans ajouter de bruit ? » : non, B ajoute un bruit modéré**
(juge propreté A 20 / B 13). Ce n'est donc pas un gain gratuit mais un **arbitrage rappel↔précision**
clairement favorable : le net du juge est **+14 pour B**, et le bruit ajouté reste du **verbatim
ancré** (donc clusterisable, non hallucinatoire), tandis que les positions oubliées par A sont, elles,
définitivement perdues pour la couverture.

### Recommandation
1. **Adopter B et relancer une re-extraction complète** (tous datasets servis). Le gain de
   couverture sert directement la métrique *couverture* du contrat de métriques.
2. **Mitigation optionnelle du bruit** si la console révèle des micro-claims narratifs : renforcer
   légèrement la règle n°2 (sélectivité) de B — *sans* retoucher la complétude (n°1) ni ré-imposer
   la cible dure. À A/B-tester de la même façon avant de figer.
3. La cible reste un **indice secondaire** (clustering + traçabilité), conforme à
   `agora-stance-subject-verdict` — ne pas la re-promouvoir en filtre.

---
*Artefacts : `research/extract_ab.py` (extraction + B + métriques), `research/extract_ab_judge.py`
(juge), `research/extract_ab_cache/{raw_A,raw_B,metrics,judge}.json` (réponses brutes mises en cache).*
