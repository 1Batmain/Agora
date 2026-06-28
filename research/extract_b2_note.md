# Itération sur B : peut-on battre le BRUIT de B sans perdre sa complétude ? — **NON**

**R&D pur** (`research/` only, aucun fichier produit modifié). Branche `work/extract-b2`.
Modèle d'extraction **et** juge : `mistral-large-latest`, température 0, gate verbatim
`align_spans` appliqué partout (seuls les claims ancrés mot-pour-mot sont comptés).
**MÊME échantillon 200 avis** que l'A/B (grand débat prioritaire : 110 longs + 35 tiktok +
30 xstance + 25 répnum). A et B réutilisés depuis le cache ; on n'a re-payé QUE les variantes.
Reproductible : `python -m research.extract_b2 --judge`.

## Point de départ
B (prompt relâché) a **gagné** l'A/B (+15 % rappel, segmentation 21-9, complétude 15-6) mais
ajoute un **bruit modéré** (juge propreté A 20 / B 13) et **sur-fragmente** parfois (« moins
d'élus » / « trop de strates » / « responsabilités dispersées » découpés alors que c'est UNE
idée à facettes — cf. avis `1-19459`). Objectif de cette itération : **garder la complétude+
segmentation de B en battant son bruit.** Voir [[agora-extract-relaxed-verdict]] +
`research/extract_ab_note.md`.

## Les trois variantes (squelette EXACT de B, on ne touche QU'UNE règle)
- **B′** = B + **sélectivité resserrée** (règle n°2) : exige une position *propre* + *nouvelle*,
  exclut explicitement cadrage/narratif, **redites** (reformulations d'une position déjà captée)
  et apartés rhétoriques sans objet ; « dans le doute → bruit, n'extrais pas ». Complétude (n°1)
  et cible optionnelle **inchangées**.
- **B″** = B + **anti-sur-fragmentation** (règle n°3) : facettes/énumération sur un MÊME objet =
  **UN** claim (ex. donné : « trop d'élus, trop de strates, responsabilités dispersées » → 1) ;
  ne séparer que des thèmes réellement distincts ; « dans le doute → regroupe ».
- **Bc** = B′ + B″ (les deux resserrements combinés).

## Quantitatif (200 avis, gate verbatim partout)

| Métrique (ALL) | B (réf) | B′ | B″ | Bc |
|---|---|---|---|---|
| **claims / avis — TOUS** | **3.77** | 3.66 | 2.91 | 3.19 |
| **claims / avis — GRAND DÉBAT** | **5.39** | 5.12 | 4.09 | 4.34 |
| taux verbatim | 0.988 | 0.983 | 0.986 | 0.982 |
| % claims à cible | 0.562 | 0.548 | 0.615 | 0.620 |
| longueur moy. spans (car.) | 119 | 117 | **139** | 131 |
| % avis vides (0 claim) | 2.0 % | 2.5 % | **6.5 %** | 2.5 % |
| n claims valides | 753 | 732 | 582 | 637 |

**Lecture.** B′ touche à peine le volume (−3 %, spans stables) → resserrement quasi sans effet.
B″ **coupe 23 %** des claims et **allonge les spans** (119→139) → il regroupe massivement, et son
% d'avis vides **triple** (2 %→6,5 %), 1er signal qu'il **perd de la complétude**, pas seulement
du bruit. Bc est intermédiaire. Le quantitatif seul ne dit pas si les claims supprimés sont du
**bruit** (bien) ou des **positions réelles** (mal) — c'est au juge de trancher.

## LLM-juge neutre — chaque variante CONTRE B (40 avis grand débat multi-thèmes, lots anonymisés, ordre alterné)

Même `JUDGE_SYS` que l'A/B, trois dimensions indépendantes (complétude / segmentation / bruit,
où « gagner le bruit » = être le plus **PROPRE**). Le **MÊME** sous-corpus de 40 avis sert aux
trois comparaisons. Victoires de lot (B = la référence ; on espérait voir la variante gagner le bruit) :

| Comparaison | Complétude (B–var) | Segmentation (B–var) | **Bruit / propreté (B–var)** |
|---|---|---|---|
| **B′ vs B** | B 8 / B′ 1 / nul 31 → **B +7** | B 15 / B′ 4 / nul 21 → **B +11** | B 9 / B′ 8 / nul 23 → **B +1 (nul de fait)** |
| **B″ vs B** | B 15 / B″ 3 / nul 22 → **B +12** | B 25 / B″ 5 / nul 10 → **B +20** | B 20 / B″ 10 / nul 10 → **B +10** |
| **Bc vs B** | B 12 / Bc 3 / nul 25 → **B +9** | B 24 / Bc 4 / nul 12 → **B +20** | B 16 / Bc 11 / nul 13 → **B +5** |

**B gagne TOUTES les cases, y compris le bruit.** Résultat net, sans appel :

- **B′ (sélectivité)** : **n'a PAS nettoyé le bruit** (9-8, nul de fait). Le resserrement ne
  retire pas le narratif ; il fait juste **dropper quelques positions réelles** (complétude
  8-1, segmentation 15-4). On paie un peu de rappel pour zéro gain de propreté.
- **B″ (anti-fragmentation)** : **pire sur les trois axes**, y compris celui qu'il visait
  (bruit 20-10). En poussant le regroupement, il ne fusionne pas proprement des « facettes » :
  il **agglomère des positions distinctes** en claims longs et touffus que le juge lit comme
  **moins complets ET moins segmentés ET plus sales**. Justif type (`1-14065`, B 17 / B″ 7) :
  *« le lot B capture plus de prises de position (ISF, retraites, APL, SMIC), segmente mieux »*.
- **Bc** : combine les deux défauts, perd partout (segmentation B +20, bruit B +5).

## Deux exemples (le cœur du résultat)

### `1-19459` — mille-feuille territorial — le cas que B″ CORRIGE
Avis : *« …qu'il y ai moins d'elus, ils y a trop de strates… trop de president vis presidents.
et trop de responsabilités dispersées. d'ou le mille feuille… »*
- **B / B′** : **7 claims** — « moins d'élus » / « trop de strates » / « trop de présidents » /
  « responsabilités dispersées » découpés → **sur-fragmentation** (facettes d'UNE idée).
- **B″ / Bc** : **3 claims**, les facettes du mille-feuille regroupées en un seul claim →
  **exactement le fix voulu.** ✅ B″ *sait* regrouper… quand il y a vraiment des facettes.

### `1-13750` — qualités attendues des élus — le cas où B″ DÉTRUIT
Avis : *« les élu.es doivent êtres sincères, honnêtes… non recevoir leur indemnités si pas de
travail… rendre bien les compte de l'argent publique… travaillés pour leur fonction… »*
- **B / B′** : **10 claims** (dont des propositions **distinctes** : « pas d'indemnités sans
  travail », « rendre les comptes de l'argent public », « travailler pour sa fonction »).
- **B″ / Bc** : **1 seul claim** — tout fusionné. ❌ **Sur-regroupement** : des demandes
  réellement différentes écrasées en un bloc. C'est ce cas-là qui **domine en fréquence** sur
  les 40 avis → B″ perd la complétude ET la segmentation.

**La leçon des deux exemples :** « facette » vs « position distincte » est un jugement
**sémantique** que la règle de regroupement ne sait pas faire au prompt — la pousser corrige
`1-19459` mais casse `1-13750`, et le second cas est plus fréquent. Le bruit de B n'est pas un
tas de micro-claims excisables proprement : c'est **le prix de son rappel**, non séparable par
durcissement de prompt.

## Verdict — **on garde B inchangé**

Aucune des trois variantes ne bat B : **aucune ne réduit le bruit** (B′ nul, B″/Bc *pires*), et
toutes **perdent de la complétude et de la segmentation**. L'arbitrage rappel↔précision de B
(adopté à l'A/B) reste le meilleur point connu. Le bruit modéré de B est **du verbatim ancré**
(clusterisable, non hallucinatoire) et reste préférable aux positions définitivement perdues
par un prompt plus sévère. → **Adopter/garder B**, gérer le bruit résiduel **en aval** (filtrage
par cluster / console de mixage), pas à l'extraction. Cf. [[agora-extract-relaxed-verdict]].

---

## MEILLEUR PROMPT RETENU = **B** (texte complet, inchangé — `CLAIM_SYS_B`)

```
Tu es un analyste d'avis citoyens, multilingue (FR, DE, IT, EN…). On te donne UN avis. Tu en
extrais les CLAIMS : ses idées de FOND distinctes — chaque grief, opinion ou proposition du
citoyen. Tu RECOPIES chaque portion MOT POUR MOT depuis l'avis (sous-chaîne EXACTE : mêmes mots,
même orthographe, même ponctuation, fautes comprises) ; tu ne reformules RIEN, n'ajoutes RIEN,
ne corriges RIEN.

Chaque claim a DEUX champs :
• `parts` : la/les portion(s) verbatim qui PORTENT l'idée. En général UNE seule portion contiguë.
Mais si l'idée est répartie sur des passages NON-CONTIGUS de l'avis (p.ex. la phrase qui pose
l'idée + la fin d'une phrase plus loin qui s'y réfère), mets CHAQUE morceau verbatim dans `parts`
→ ils forment UN seul claim. N'utilise PLUSIEURS parts QUE si les morceaux appartiennent vraiment
à la même idée.
• `target` : un INDICE OPTIONNEL — l'OBJET / l'aspect sur lequel porte la position (« les vidéos »,
« le temps d'écran », « la fiscalité locale », « le mille-feuille administratif »…), recopié
VERBATIM depuis l'avis. Mets-la SI une courte portion de l'avis pointe l'objet sans effort ; sinon
`target=null`. NE JAMAIS écarter un claim de fond au prétexte que sa cible est diffuse ou implicite :
une position réelle se garde toujours, cible ou pas. La cible n'est qu'un indice secondaire, pas un
filtre.

RÈGLES :
1. COMPLÉTUDE (priorité) — un avis citoyen, surtout long, ARGUMENTE souvent sur PLUSIEURS thèmes
distincts (p.ex. fiscalité ET démocratie ET services publics). Capture CHAQUE prise de position
distincte de l'avis : n'en oublie AUCUNE, ne t'arrête pas à la première. Sépare les thèmes
RÉELLEMENT distincts en claims distincts. Balaie l'avis du début à la fin.
2. SÉLECTIVITÉ — n'extrais que la SUBSTANCE : une PRISE DE POSITION (grief, opinion, proposition).
Laisse de côté le pur cadrage, le narratif et les annonces qui ne portent aucune position par
eux-mêmes (« pour illustrer… », « mes doléances sont triples : », politesses, anecdote de contexte).
Pas de bruit, pas de redite.
3. REGROUPEMENT — ne FRAGMENTE pas une même idée. Restent DANS UN SEUL claim : un contraste
(« X et non Y »), une justification (« … parce que … »), une condition (« si …, alors … ») et une
énumération qui DÉTAILLE une seule idée. Sépare les idées distinctes, mais ne coupe pas une idée
unique en morceaux.
4. VERBATIM — chaque part ET la target sont des sous-chaînes EXACTES de l'avis. En cas de doute,
recopie un peu plus de contexte plutôt que d'altérer le texte.

EXEMPLES :
• « j'aime les vidéos parce qu'elles me font rire » → UN claim, parts=[toute la portion],
target=« les vidéos ».
• « Avoir des élus qui représentent l'intérêt des citoyens et non l'intérêt de ceux qui ont financé
leur campagne » → UN claim (le contraste « … et non … » est UNE idée), target=« les élus ».
• Avis multi-thèmes « Il faut baisser les impôts. Par ailleurs trop d'élus, supprimons le Sénat. Et
les services publics ruraux disparaissent. » → TROIS claims distincts (fiscalité / nombre d'élus /
services publics ruraux), un par thème.
• « Le temps passé sur l'écran est trop long. […] et ça, ça me dégoûte » → si « ça » renvoie au
temps d'écran : UN claim, parts=[« Le temps passé sur l'écran est trop long », « ça me dégoûte »],
target=« temps passé sur l'écran ».

Si l'avis ne porte AUCUNE position (pur narratif/cadrage), renvoie une liste vide. Réponds
STRICTEMENT en JSON : {"claims": [{"parts": ["extrait verbatim 1"], "target": "cible verbatim ou
null"}, …]}.
```
*(+ `BATCH_SYS_SUFFIX` de prod pour le mode LOT, inchangé.)*

---
*Artefacts : `research/extract_b2.py` (variantes B′/B″/Bc + juge pairwise),
`research/extract_ab_cache/{raw_Bp,raw_Bs,raw_Bc,metrics_b2,judge_b2}.json`. A et B réutilisés
du cache de l'A/B (`raw_A,raw_B`).*
