# x-stance multi-questions — fondation mère→enfants (Phase 1 DATA)

> Branche `work/multiq-data`. x-stance (ZurichNLP) est le **patron de référence** d'une
> consultation **multi-questions** : ≠ granddebat/tiktok (UNE question ouverte unique),
> ≠ repnum (multi-articles d'un même projet de loi). Ici, N questions FERMÉES (oui/non,
> FAVOR/AGAINST) regroupées en topics. C'est le cas qui fonde le modèle **mère → enfants**.

## Ce qui a changé (re-ingestion)
- **Ingester rendu générique** (`pipeline/ingest/sources.py` + `build.py`) : nouveau knob
  déclaratif `props` dans le descripteur → `{nom_prop: réf_colonne}`. Les métadonnées de
  source listées sont **préservées dans `props` de chaque idée**, sans jamais écraser une
  prop canonique (`text`, `lang`, `author_hash`…). **Zéro hardcoding** : aucun champ
  corpus dans le code, tout vient du descripteur (cf. `[[agora-genericity-no-hardcoding]]`).
- **Descripteur xstance** (`descriptors/xstance.json`) déclare :
  `props = {question, question_id, topic, label}`.
- **Cache ré-ingéré** : `backend/cache/xstance/ideas.jsonl` porte maintenant
  `props.question / question_id / topic / label` pour **100 % des 3000 idées servies**.
  Réécriture **alignée** (même ordre, même `author_hash`) → `embeddings.npy` (3000×768) et
  `translations.json` **inchangés**, pas de ré-embedding (Phase 1 = data only).

Exemple d'idée enrichie :
```json
{"id":"xstance:49454", "props":{ "text":"…", "lang":"de", "author_hash":"33a87fb98a0a1699",
  "question":"Befürworten Sie eine Erhöhung des Rentenalters (z.B. auf 67 Jahre)?",
  "question_id":"3412", "topic":"Welfare", "label":"FAVOR"}}
```

## Investigation chiffrée

### Corpus complet (train+valid+test = 67 271 commentaires)
- **Questions distinctes : 194** (par `question_id`).
  ⚠️ 517 *textes* de question distincts = **artefact multilingue** : un même `question_id`
  a jusqu'à 3 variantes de texte (DE/FR/IT). **L'unité « question » est `question_id` (194)**,
  pas le texte (517), sinon on triple les enfants.
- **Topics distincts : 12.**
- **Langues** : DE 48 612 (72 %), FR 17 213 (26 %), IT 1 446 (2 %).
- **Labels** : FAVOR 34 001 (51 %), AGAINST 33 270 (49 %) — équilibré par construction.
- **Par question** : moyenne 346.8, médiane 302, min 9, max 1231 commentaires.

Distribution **par topic** (corpus complet) :

| Topic | n | % |
|---|---:|---:|
| Infrastructure & Environment | 9 590 | 14.3 |
| Welfare | 8 508 | 12.6 |
| Education | 7 639 | 11.4 |
| Economy | 6 899 | 10.3 |
| Society | 6 275 | 9.3 |
| Immigration | 6 270 | 9.3 |
| Security | 5 193 | 7.7 |
| Healthcare | 4 711 | 7.0 |
| Foreign Policy | 4 393 | 6.5 |
| Finances | 3 980 | 5.9 |
| Political System | 2 645 | 3.9 |
| Digitisation | 1 168 | 1.7 |

### Sous-ensemble SERVI (cache `backend/cache/xstance`, 3000, équilibré par langue)
- Échantillon `balance=lang, cap=3000, seed=42` → **1000 DE / 1000 FR / 1000 IT**.
- **191 questions** (`question_id`), 505 textes, **12 topics**.
- Labels : FAVOR 1565 / AGAINST 1435.
- **Par question** : moyenne **15.7**, médiane **14**, **min 1**, max 66 → très clairsemé,
  beaucoup de singletons.
- **Par topic** (ce que la carte analyse réellement) :

| Topic | n (sur 3000) |
|---|---:|
| Infrastructure & Environment | 442 |
| Welfare | 397 |
| Education | 326 |
| Economy | 324 |
| Immigration | 282 |
| Society | 265 |
| Security | 258 |
| Healthcare | 218 |
| Foreign Policy | 178 |
| Finances | 164 |
| Political System | 110 |
| Digitisation | 36 |

## Recommandation — granularité ENFANT = **TOPIC (12)**

Le seuil du brief (*question si ~≤30, sinon topic ~12*) tranche **nettement** :

1. **194 questions ≫ 30** → la granularité « question » est exclue : ni navigable
   (194 enfants), ni dérivable proprement (il faudrait dédupliquer les 517 textes via
   `question_id`, et gérer le multilinguisme par question).
2. **Densité « question » trop faible sur le servi** : moyenne 15.7, médiane 14, **min 1**
   commentaire/question → enfants vides ou quasi (singletons), non analysables.
3. **12 topics = sweet-spot** : nombre exactement dans la cible (~12), distribution
   **équilibrée** (servi 36–442 ; complet 1168–9590), couvre 100 % des commentaires,
   et correspond **à la taxonomie officielle x-stance** (vérité terrain, cf.
   `[[agora-official-syntheses]]`) → benchmarkable.
4. Seul **Digitisation** est mince (36 servi / 1168 complet) mais reste un enfant viable ;
   les 11 autres sont confortables.

**Structure mère→enfants proposée (Phase 2, hors scope ici)**
- **Mère** : « Prises de position sur des votations/questions politiques suisses (x-stance) ».
- **12 enfants = topics**, chacun portant ses questions (`question_id`) et l'axe de clivage
  FAVOR/AGAINST agrégeable (cohérent avec `[[agora-stance-cluster-subject-verdict]]` :
  la stance s'agrège sur le sujet du cluster). `question`/`question_id` restent un **signal
  fin** (sous-regroupement / traçabilité), pas la granularité d'affichage.

> Parallèle **repnum** (cf. `descriptors/republique-numerique.json` → `_multiq`) : même
> patron mère→enfants, mais enfants = **TITRE (3)** ou **Section (18)** du projet de loi.
> Les deux corpus multi-thèmes valident le besoin d'un niveau « mère » au-dessus des enfants.

## NE PAS faire (Phase 2, après revue)
Modèle parent/enfant servi, sous-analyses par enfant, front. Cette note s'arrête à la
data + la reco chiffrée.
