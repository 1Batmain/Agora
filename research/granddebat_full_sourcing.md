# Repérage — Grand Débat National COMPLET (les 4 thèmes)

Source officielle : **data.gouv.fr — « Données ouvertes du Grand Débat National »**, Licence
Ouverte 2.0 (`lov2`). 68 ressources (snapshots à dates multiples + réunions + comptes-rendus).
On veut les 4 CSV **« Contributions »** (texte libre) au **snapshot final 21/03/2019**.

## Les 4 fichiers (snapshot final 21/03/2019)

| thème | taille | URL |
|---|---|---|
| Démocratie et citoyenneté | 405 Mo | `.../20200611-224804/democratie-et-citoyennete.csv` |
| La fiscalité et les dépenses publiques | 303 Mo | `.../20200611-214446/la-fiscalite-et-les-depenses-publiques.csv` |
| La transition écologique | 305 Mo | `.../20200611-182240/la-transition-ecologique.csv` |
| Organisation de l'État et des services publics | 219 Mo | `.../20200611-233643/organisation-de-letat-et-des-services-publics.csv` |

Préfixe : `https://static.data.gouv.fr/resources/donnees-ouvertes-du-grand-debat-national/`
**Total ≈ 1,23 Go.** (Les « questionnaire-*.csv » sont les questions FERMÉES rapides — NON.)

## Structure (identique aux 4 thèmes)

- **Cols 0-10 : métadonnées** — `id, reference, title, createdAt, publishedAt, updatedAt,`
  `trashed, trashedStatus, authorId, authorType, authorZipCode`.
  - `trashed`/`trashedStatus` → **filtrer les contributions supprimées** à l'ingestion.
  - `authorId` → dédup par répondant possible (1 contribution = 1 répondant sinon).
  - `authorType, authorZipCode` → **démographie** (Agora la gère déjà).
- **Cols 11+ : les QUESTIONS** — chacune `QUXVlc3Rpb246NNN - <intitulé>`. Le préfixe base64
  encode un **ID de question STABLE** (`Question:110`…). Mélange OUVERTES (opinions riches) et
  FERMÉES (oui/non, « Si oui… », « Pourquoi ? » → réponses courtes).

## ⚠️ Gotcha : les positions de colonnes BOUGENT entre snapshots

Démocratie, « Que faudrait-il faire pour renouer le lien… » :
- descripteur actuel (snapshot ancien, 46 col) → **col 13**
- snapshot final (48 col) → **col 14**

→ **Mapper par ID de question (en-tête), jamais par index positionnel.** C'est un point de
conception d'ingestion : le descripteur devrait sélectionner la colonne texte par match sur
l'ID `Question:NNN` (ou sur l'intitulé), robuste au snapshot.

## Questions OUVERTES riches par thème (candidates à ingérer, cadre = leur intitulé)

- **Démocratie** : renouer le lien élus/citoyens · mieux représenter les sensibilités ·
  associer les citoyens aux décisions · consulter directement · rôle des assemblées/Sénat ·
  laïcité · discriminations · immigration (clivant) · autres points.
- **Fiscalité** : baisser la dépense publique · rendre la fiscalité plus juste · impôts à
  baisser · domaines prioritaires de protection sociale · politiques à réduire · autres points.
- **Écologie** : problème concret le plus important · apporter des réponses · inciter à changer
  ses comportements · solutions les plus simples · autres points.
- **Organisation** : que pensez-vous de l'organisation de l'État · nouveaux services souhaités ·
  améliorations préconisées · rôle État/collectivités · autres points.

(Les « Si oui… / Pourquoi ? / oui-non » sont fermées → à exclure ou traiter comme
justifications, cf. proposition extraction P4.)

## Plan d'ingestion proposé

1. **Mapper par ID de question** (adapter le lecteur d'ingestion pour sélectionner la/les
   colonne(s) texte par `Question:NNN`, pas par index) — corrige la gotcha snapshot.
2. **Un cadre (question) par sous-corpus** — chaque question ouverte devient un sous-dataset
   avec SON intitulé comme `question` (align avec la proposition extraction P4 : cadre par
   groupe d'avis, pas par dataset ; `build_children --by question_id` existe déjà).
3. **Filtrer `trashed`** + `--min-chars` (réponses vides) + langue FR détectée en aval.
4. **DÉCISION DE VOLUME** (le point à trancher) : le snapshot complet, c'est ~2 M de
   contributions, 1,23 Go, coût LLM d'extraction non trivial. Options :
   - échantillon par question (`--cap` / `--balance`) pour la R&D — recommandé pour démarrer ;
   - un thème complet d'abord (témoin multi-question), puis étendre ;
   - les 4 thèmes complets = le vrai test macro (les 4 thèmes = macro-vérité), mais lourd.

## Reproduire l'énumération
```
curl -s "https://www.data.gouv.fr/api/1/datasets/donnees-ouvertes-du-grand-debat-national/" \
  | python3 -c "import json,sys; [print(r['title'], r['url']) for r in json.load(sys.stdin)['resources']]"
```
Récupérer un en-tête sans tout télécharger : `curl -s -r 0-20000 <url> | head -1`.
