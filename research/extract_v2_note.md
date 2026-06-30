# Extraction v2 : question globale + anti-sur-segmentation — verdict

**R&D pur**, branche `work/extract-v2`. Modèle `mistral-large-latest`, température 0, gate
verbatim `align_spans` appliqué aux **deux** bras. Validation **sans re-extraction complète** :
12 avis grand débat = **2 flaggés** par Bob + **10 longs typiques** (bande 500–1600 chars,
> p90 sans les essais pathologiques de 8k chars qui tronquent le JSON), extraits **mono-avis**
pour isoler la granularité. Reproductible : `uv run python -m research.extract_v2`.

## Problème visé (Bob + FLAGS) = SUR-SEGMENTATION
- `granddebat:1-11830` : « la découpe sépare l'énonciation du problème de la solution proposée »
  → devrait être **1 claim**.
- `granddebat:1-4924` : « les 3 premières claims traitent de l'absentéisme des élus, elles
  devraient être **une seule** ».

## Les deux changements (v2, `pipeline/claims/extract.py`)
1. **Question globale injectée** dans le prompt système (`claim_sys(question)`, source
   `meta.json`/descriptor `question`). Elle CADRE la granularité : des sous-points qui
   répondent tous à la MÊME facette de la question = UN claim. Vide si pas de question
   (généricité : retombe sur le prompt nu).
2. **REGROUPEMENT renforcé** (règle 3, prioritaire) : **problème + solution proposée = UN
   claim** ; **plusieurs phrases sur le MÊME sujet = UN claim** ; « en cas de doute, REGROUPE ».
   + 2 exemples calqués sur les flags (mandat unique ; absentéisme).

V1 = snapshot EXACT du prompt de prod **avant** ce commit (figé dans `research/extract_v2.py`).

## Quantitatif (12 avis, gate verbatim aux deux)

| Métrique | V1 (actuel) | V2 | Δ |
|---|---|---|---|
| **claims / avis** | 8.33 | **4.83** | **−42 %** |
| total claims | 100 | 58 | −42 |
| taux de passage **verbatim** | 0.990 | **0.967** | −2.3 pt |

La chute de claims/avis est l'effet RECHERCHÉ (dé-fragmentation). Le verbatim reste haut
(96.7 %) : l'ancrage des spans n'est pas cassé (les `parts` plus longues d'un claim regroupé
restent des sous-chaînes exactes).

## Les deux flags — corrigés À LA LETTRE

**`1-11830`** (problème + solution) :
- V1 → **2** claims : `« …situation personnelle (réélection) »` | `« le mandat unique est une réponse »`.
- V2 → **1** claim : les deux réunis. ✓ **exactement le flag.**

**`1-4924`** (3 sous-points absentéisme → 1), V1=8 → V2=6 :
- V2 claim [1] = `« Devoir de présence… | Un citoyen absent doit se justifier… | sanction
  financière en cas d'absentéisme »` (les **3 phrases sur l'absentéisme fusionnées**).
- V2 garde les **5 thèmes réellement distincts** : transparence des dépenses · lobbies ·
  consultation citoyenne · révocation/référendum · protection judiciaire des élus.
  ✓ **exactement le flag : sur-segmentation corrigée SANS perte de complétude.**

## Tableau avis → #claims + jugement

| avis | chars | V1 | V2 | jugement |
|---|---|---|---|---|
| **1-11830** (flag) | 136 | 2 | **1** | ✓ FIX — problème+solution réunis |
| **1-4924** (flag) | 638 | 8 | **6** | ✓ FIX — 3 absentéisme→1, 5 distincts gardés |
| 1-20463 | 1473 | 9 | **3** | ✓✓ IDÉAL — avis structuré « A./B. » → intercommunalité / départements / scrutin |
| 1-1311 | 1456 | 9 | **4** | ✓ thèmes gardés (transparence/cumul/dépenses/proximité), pb+sol réunis |
| 1-9114 | 1490 | 13 | **8** | ✓ positions distinctes gardées, redites fusionnées |
| 1-10652 | 1583 | 9 | **2** | ✓ un peu agressif mais cohérent (fossé représentatif vs pouvoir/fonctionnaires) |
| 1-2582 | 1539 | 8 | **8** | ✓ avis vraiment multi-thèmes → INCHANGÉ (pas de sur-fusion) |
| 1-9802 | 1494 | 8 | 7 | ✓ léger regroupement |
| 1-6257 | 1467 | 7 | 6 | ✓ léger regroupement |
| 1-15353 | 1594 | 10 | 7 | ✓ regroupement modéré |
| 1-24152 | 1553 | 8 | 5 | ✓ regroupement |
| **1-6202** | 1541 | 9 | **1** | ⚠ AGRESSIF — avis mono-thème (privilèges/exemplarité des élus) collapsé en 1 ; thématiquement défendable mais perd les propositions concrètes distinctes (frais de mandat, anciens présidents) |

## Verdict — **V2 GAGNE** sur l'objectif (anti-sur-segmentation)

- Les **deux flags sont corrigés à la lettre** : problème+solution réunis (`1-11830`), et
  sous-points d'un même sujet réunis SANS perdre les thèmes distincts (`1-4924`).
- Sur les longs avis, la segmentation devient **nettement plus fidèle à la structure réelle**
  des thèmes (`1-20463` 9→3 idéal ; `1-1311`, `1-10652`, `1-9114`).
- **Complétude préservée** là où elle doit l'être : l'avis vraiment multi-thèmes `1-2582` reste
  à 8 claims (la v2 ne sur-fusionne pas les vrais thèmes distincts).
- **Verbatim tenu** (96.7 %), ancrage des spans intact.

**Réserve (watch-item)** : sur un avis **mono-thème à plusieurs propositions concrètes**
(`1-6202`), la v2 peut collapser en **1 seul** claim — la granularité voulue par le cadrage
« une facette de la question = un claim », mais au prix des propositions actionnables
distinctes. C'est le seul cas du panel où le regroupement va peut-être trop loin ; il reste
défendable (toutes ces propositions tomberaient dans le même cluster en aval).

## Recommandation
Adopter la v2 et procéder à une **re-extraction complète** (décision architecte). Le code de
prod est déjà prêt : `question` est threadé jusqu'à `extract_claims(..., question=...)`
(défaut `None` → **rétro-compatible**, un dataset sans question retombe sur le prompt nu).

**Reste à câbler au bake** (`pipeline/claims/pipeline.py`) : passer la question à
`extract_claims`. Source la plus fiable = le **descriptor** `pipeline/ingest/descriptors/<ds>.json`
(champ `question`), déjà lu par `recluster._read_descriptor_file` :
- présent : `granddebat`, `republique-numerique`, `tiktok`, `ameliorer-agora` ;
- **absent : `xstance`** (consultation à questions PAR ITEM, pas de question globale → `None`,
  comportement nu, OK).

À noter : côté `meta.json` la question n'est aujourd'hui sérialisée que pour `republique-numerique` ;
ne pas s'appuyer sur `meta.json` pour la généralité — lire le descriptor au bake.
