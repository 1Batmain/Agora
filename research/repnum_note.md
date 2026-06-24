# Benchmark — Consultation « République numérique » (loi Lemaire, 2015)

> Branche `work/dataset-repnum`. **But de validation** : Agora retrouve-t-il, en
> non-supervisé (claims → cible → clustering → hiérarchie), les **grands thèmes
> officiels** de cette consultation ? On compare ici la **synthèse/structure
> officielle** (gouvernement / projet de loi) aux **macros** que produit notre
> pipeline. C'est un corpus *gold* : la consultation portait sur un projet de loi
> dont le plan (Titres/Chapitres/Sections) **est** la taxonomie de référence.

## La source

- Consultation **« Pour une République numérique »** (projet de loi porté par
  Axelle Lemaire, secrétaire d'État au Numérique), plateforme
  `republique-numerique.fr`, **du 26 sept. au 18 oct. 2015**.
- Open data **Etalab / data.gouv.fr** (Licence Ouverte) — fichier
  `projet-de-loi-numerique-consultation-anonyme.csv` (30,9 Mo, 21 colonnes).
- Descripteur : `pipeline/ingest/descriptors/republique-numerique.json`.

### Chiffres officiels de la participation
- **21 330 contributeurs**.
- **~150 000 votes**.
- **> 8 500** arguments, amendements et propositions d'articles nouveaux.

(Source : présentation officielle de la consultation / restitution gouvernementale ;
cf. liens en bas.)

## Ce qu'Agora a ingéré

Le CSV mélange 5 types de contenu (colonne `Type.de.contenu`). On ne garde que
le **texte libre argumentatif** via le knob déclaratif `keep_where` :

| `Type.de.contenu` | n (brut) | gardé ? | pourquoi |
|---|---:|:--:|---|
| **Vote** | 147 637 | ✗ | contenu = « Pour »/« Contre »/« Mitigé » (3 car.), pas du texte |
| **Argument** | 5 990 | ✓ | argument pour/contre rédigé |
| **Modification** | 1 387 | ✓ | modification d'article + justification |
| **Proposition** | 692 | ✓ | proposition citoyenne nouvelle |
| **Source** | 415 | ✗ | simple lien / URL, non argumentatif |

- **Argumentatif retenu : 8 069** contributions (cohérent avec le « > 8 500 »
  officiel, qui inclut aussi les sources/liens).
- **Cache servi : 3 000** avis (cap aligné sur granddebat/xstance, `min_chars≥30`,
  dédup exacte, `seed=42`), dont **2 971 fr**. → `backend/cache/republique-numerique/`.

## Taxonomie OFFICIELLE de référence (plan du projet de loi)

Le projet de loi s'organise en **3 axes** (= les 3 Titres ; un Titre IV technique
« Outre-mer » mis à part). C'est notre **gold** thématique :

### Axe 1 — La circulation des données et du savoir *(Titre Ier, art. 1–18)*
- Ouverture des données publiques (**open data** par défaut)
- **Données d'intérêt général** (ouverture de données privées d'utilité publique)
- Service public de la donnée
- **Économie du savoir** : open access aux publications scientifiques, fouille
  de textes et de données (TDM)
- (Débat citoyen fort) **domaine commun informationnel**

### Axe 2 — La protection des droits dans la société numérique *(Titre II, art. 19–34)*
- **Neutralité du net**
- **Portabilité / récupération** des données par l'utilisateur
- **Loyauté des plateformes** (transparence, information du consommateur, avis en ligne)
- **Protection de la vie privée / données personnelles** (pouvoirs CNIL, sanctions)
- **Droit à l'oubli** (notamment des mineurs)
- **Mort numérique** (sort des données après le décès)
- Confidentialité des correspondances privées

### Axe 3 — L'accès au numérique pour tous *(Titre III, art. 35–45)*
- **Couverture / connectivité** du territoire (réseaux, fibre, mobile)
- **Inclusion numérique** des publics fragiles (maintien de la connexion internet)
- **Accessibilité** des services aux **personnes handicapées**

> Remarque : la colonne `Catégorie` du CSV porte, pour les
> propositions/modifications, le **code de section** du projet de loi
> (`TITRE Ier - Chapitre Ier - Section 1`, …) — soit une étiquette gold *par
> contribution* exploitable pour mesurer plus finement la correspondance
> macro↔section (piste d'éval supervisée ultérieure).

## Macros obtenues par Agora *(à compléter post-build)*

<!-- BENCH_MACROS_START -->
⏳ `build_analysis --dataset republique-numerique --model mistral-large-latest
--enrich-model mistral-small-latest` en cours (extraction claims+cible verbatim
sur 3 000 avis). Les macros (titres + tailles) et la correspondance avec les
3 axes officiels seront renseignées ici à la fin du build.
<!-- BENCH_MACROS_END -->

## Liens
- Jeu de données open data : <https://www.data.gouv.fr/datasets/consultation-sur-le-projet-de-loi-republique-numerique/>
- Plateforme de la consultation : <https://www.republique-numerique.fr/>
- Rapport Sénat (structure du projet de loi) : <https://www.senat.fr/rap/a15-528/a15-528_mono.html>
- Loi n° 2016-1321 du 7 octobre 2016 pour une République numérique (Legifrance).
