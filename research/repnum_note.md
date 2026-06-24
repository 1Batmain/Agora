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

## Macros obtenues par Agora

<!-- BENCH_MACROS_START -->
Build : `mistral-large-latest` (extraction claims+cible) + `mistral-small-latest`
(enrichissement). **337 thèmes**, **8 macros**, 273 feuilles, profondeur 4.
Qualité d'extraction (`backend.verify_claims_cache`) :

- **2 724 / 3 000** avis portent ≥ 1 claim (90,8 % — l'extracteur sélectif laisse
  le narratif sans opinion ciblée) ; **5 842 claims**.
- **Verbatim : 5 842 / 5 842 = 100,00 %** (gate dur respecté). ✅
- **Cible** : couverture 70,1 % des claims ; **cibles verbatim 100,0 %**. ✅

### Les 8 macros (niveau 0)

| n_avis | macro |
|---:|---|
| **2 858** | Droit et services numériques libres *(macro fourre-tout)* |
| 81 | Accès et diffusion des documents administratifs |
| 80 | Vote électronique et accès |
| 40 | Sécurité et abus des paiements par SMS |
| 32 | Droit de libre accès scientifique |
| 11 | Accessibilité téléphonique pour personnes sourdes |
| 11 | Données de référence publiques |
| 1 | Renault Clio courroie refaite *(contribution hors-sujet — fidèle à la donnée brute)* |

> **Constat clé — sur-concentration au niveau macro.** Une seule macro capte
> **92 %** des voix (`concentration=0.92`, `effective_themes≈1.5`). À la résolution
> par défaut, le niveau 0 n'isole PAS proprement les 3 axes officiels : presque
> tout est « du droit du numérique », donc tout fusionne. C'est l'effet
> « granularité instable » déjà noté (corpus mono-domaine, fortement convergent :
> `convergence_cumulée=0.79`). Le signal thématique vit **un cran plus bas**.

### Recouvrement des axes officiels au niveau 1 (sous-thèmes de la macro dominante)

C'est là que la taxonomie officielle **réapparaît nettement** (11 sous-thèmes) :

| n_avis | sous-thème Agora | ↔ axe officiel |
|---:|---|---|
| 541 | Accès internet **neutralité** et débit | Axe 2 (neutralité du net) + Axe 3 (couverture/débit) |
| 522 | **Protection données personnelles** publiques | Axe 2 (données personnelles) |
| 480 | **Transparence** des associations et collectivités | Axe 1 (open data / transparence) |
| 432 | Amélioration de la qualité des lois | *(méta-débat sur la consultation)* |
| 429 | **Logiciels libres** et licences | Axe 1 (économie du savoir / logiciel libre) |
| 370 | Numérique éducatif et sécurité | Axe 3 (inclusion / éducation) |
| 281 | Droit propriété intellectuelle / **domaine public** | Axe 1 (domaine commun informationnel) |
| 209 | **Accès ouvert aux publications scientifiques** | Axe 1 (open access) |
| 170 | **Sanctions CNIL** renforcées | Axe 2 (vie privée / CNIL) |
| 158 | **Régulation des communications** électroniques | Axe 3 (régulation / couverture) |
| 734 | Amélioration des propositions citoyennes | *(méta — modalités de contribution)* |

### Verdict benchmark

**Agora retrouve les grands thèmes officiels** (open data/transparence, open
access scientifique, logiciel libre, domaine public, neutralité du net, données
personnelles & CNIL, accès/couverture) — **mais au niveau 1**, pas au niveau
macro. Le corpus est mono-domaine et très convergent : à résolution par défaut le
niveau 0 sur-fusionne (1 macro = 92 %). **Piste** : pour ce type de consultation
mono-loi, servir le niveau 1 comme « macros » (ou monter la résolution / le
`τ coarsen` dans la console `/sandbox`) donne la carte la plus lisible. Deux
contributions hors-sujet (Clio, paiement SMS) ressortent isolées — comportement
correct (fidélité à la donnée brute, pas de filtrage thématique arbitraire).
<!-- BENCH_MACROS_END -->

## Liens
- Jeu de données open data : <https://www.data.gouv.fr/datasets/consultation-sur-le-projet-de-loi-republique-numerique/>
- Plateforme de la consultation : <https://www.republique-numerique.fr/>
- Rapport Sénat (structure du projet de loi) : <https://www.senat.fr/rap/a15-528/a15-528_mono.html>
- Loi n° 2016-1321 du 7 octobre 2016 pour une République numérique (Legifrance).
