# Témoin — synthèses Agora vs synthèse OFFICIELLE du Grand Débat National (2019)

> Worker **granddebat-witness** · branche `work/granddebat-witness` · 2026-06-29
> But : se servir de la synthèse OFFICIELLE (OpinionWay / granddebat.fr) comme
> **témoin/cible** pour juger la qualité des thèmes + synthèses qu'Agora fait
> émerger sur `granddebat`. Reproduire : `research/granddebat_witness.py`
> (juge `mistral-large-latest`) → `research/granddebat_witness_results.json`.

## Cadrage du corpus (à lire avant les chiffres)

Ce dataset n'est **pas** le Grand Débat entier : c'est l'espace **« Démocratie et
citoyenneté »**, et précisément la **colonne 13 = une question OUVERTE** : *« Que
faudrait-il faire pour renouer le lien entre les citoyens et les élus qui les
représentent ? »* (cf. `pipeline/ingest/descriptors/granddebat.json`).

Donc des **4 axes officiels** du Grand Débat (Fiscalité & dépenses · Organisation
de l'État · Transition écologique · Démocratie & citoyenneté), **un seul** est
échantillonné. Tester « Agora retrouve-t-il les 4 axes ? » n'a pas de sens ici
(3 axes ne sont pas dans le corpus). Le **vrai test de couverture** porte sur les
**sous-thèmes OFFICIELS de l'axe Démocratie & citoyenneté**, tels que listés dans
la synthèse OpinionWay (cf. [[agora-official-syntheses]]).

## Vue Agora (cache `analysis.json` + `insights/`, build mergé)

18 macro-thèmes · 34 feuilles · 5 611 avis · 6 991 claims (extraction relâchée B,
cf. [[agora-extract-relaxed-verdict]]). Indices globaux dérivés :

| Indice | Valeur | Lecture |
|---|---|---|
| Effusion (équité des voix) | **0.90** | voix très réparties, `effective_themes` ≈ **13.6** |
| Concentration (part du top) | **0.15** | aucune domination (Gini 0.37) |
| Consensus global | 0.38 | accord interne modéré (question clivante, normal) |

→ **Contraste net avec repnum** (cf. [[agora-repnum-benchmark-verdict]]) où Agora
sur-concentrait au macro (top-share **0.92**, `effective_themes` ≈ 1.5). Sur la
**question OUVERTE et contrastée** du Grand Débat, Agora **ne s'effondre pas** : il
étale le débat en ~14 sujets effectifs — exactement ce que le descripteur
prédisait (« opinions VARIÉES et CONTRASTÉES »).

## Table de mapping — macro Agora → sous-thème officiel

Juge Mistral-large : `alignment` 0–5, `verdict` (faithful / finer_split / partial /
mismatch). Trié par poids social.

| Macro Agora | avis | → Sous-thème officiel | align | verdict |
|---|---:|---|:---:|---|
| n0 Comprendre les réalités des Français | 856 | lien / proximité / confiance | 5 | faithful |
| n1 Lien élu-électeurs & démocratie locale | 749 | lien / proximité / confiance | 5 | faithful |
| n2 Participation citoyenne aux décisions | 464 | démocratie participative / consultation | 5 | faithful |
| n3 Réforme du fonctionnement parlementaire | 389 | réduction du nombre d'élus / rôle du Parlement | 4 | finer_split |
| n4 Représentation politique & partis | 387 | renouvellement de la classe politique / lobbies | 5 | faithful |
| n5 Avantages & salaires des élus | 354 | privilèges / rémunérations (moralisation) | 5 | faithful |
| n27 Débats publics citoyens organisés | 341 | réunions / concertation de proximité | 5 | faithful |
| n19 Proportionnelle aux législatives | 313 | mode de scrutin / proportionnelle / vote blanc | 5 | faithful |
| n20 Réforme de l'organisation territoriale | 299 | décentralisation / millefeuille | 5 | faithful |
| n29 Transparence des dépenses publiques | 295 | transparence & contrôle de l'argent public | 5 | faithful |
| n30 Responsabilité des élus & transparence | 295 | reddition de comptes / respect des promesses | 5 | faithful |
| n28 Casier judiciaire des élus | 285 | probité / casier / inéligibilité | 5 | faithful |
| n31 Limitation des mandats | 244 | cumul / limitation des mandats | 5 | faithful |
| n32 Référendums d'initiative citoyenne | 230 | RIC / référendum | 5 | faithful |
| n33 Assemblée citoyenne tirée au sort | 93 | tirage au sort / assemblée citoyenne | 5 | faithful |
| **n34 Limitation de vitesse (80 km/h)** | 15 | — | — | **AJOUT hors-axe** |
| **n35 Techniques de communication PNL** | 1 | — | — | **AJOUT (bruit)** |
| **n36 Parcours éclectiques des énarques** | 1 | — | — | **AJOUT (bruit)** |

## Couverture des sous-thèmes officiels (in-scope)

**14 / 14 sous-thèmes officiels couverts = 100 %.** Aucun manque.
**Alignement moyen = 4.93 / 5** ; **14 faithful · 1 finer_split · 0 partial · 0 mismatch.**

- Le seul `finer_split` (n3, align 4) : Agora regroupe « réduire le nombre d'élus »
  avec le fonctionnement parlementaire (quorum, rôle Sénat) et déborde sur la
  proportionnelle — le juge le note plus large que le seul item officiel, pas faux.
- Deux macros (**n0** et **n1**) tombent sur le même sous-thème officiel
  *lien / proximité / confiance* : doublon de découpe (candidat à fusion), mais les
  deux sont jugés faithful (n0 = posture/écoute, n1 = mécanismes concrets).

### Manques apparents — TOUS hors-scope du corpus, pas des ratés d'Agora
Immigration · laïcité · civisme / service national : sous-thèmes de l'axe Démocratie
& citoyenneté **mais portés par d'autres colonnes** de la source (pas la question 13).
Leur absence est **correcte** — le corpus ne les contient pas.

### Ajouts d'Agora (hors sous-thèmes officiels)
- **n34 — 80 km/h (15 avis)** : débordement *gilets jaunes* réel dans les
  contributions, hors axe démocratie. Présence **légitime** (Agora ne censure pas),
  mais à reléguer comme thème mineur. Le juge n'avait pas de cible → non noté.
- **n35 PNL · n36 énarques (1 avis chacun)** : **bruit singleton**. C'est le prix du
  rappel de l'extraction relâchée B (cf. [[agora-extract-b2-iteration-verdict]]) ;
  à filtrer en aval (seuil de taille), pas au prompt.

## Verdict

✅ **Agora PASSE le témoin sur granddebat.** Sur la question ouverte « lien
citoyens-élus », il **retrouve 100 % des sous-thèmes officiels** de l'axe Démocratie
& citoyenneté, avec un **alignement de 4.93/5** jugé par mistral-large, **zéro
mismatch**. Les seuls « manques » sont hors-corpus (autres colonnes), les seuls
« ajouts » sont un débordement thématique légitime + 2 singletons de bruit.

**Plus fin que l'officiel.** Là où la synthèse OpinionWay agrège, Agora **désagrège**
en sous-thèmes distincts et actionnables : la *moralisation* officielle éclate en 4
macros (privilèges n5 · transparence des dépenses n29 · reddition de comptes n30 ·
casier n28) ; la *participation* en 4 (consultation n2 · débats n27 · RIC n32 ·
tirage au sort n33). C'est une **granularité supérieure**, pas une dérive : chaque
macro est apparié 1:1 (ou n:1) à l'officiel et jugé fidèle.

**Le bon contre-exemple de repnum.** Le mode d'échec « sur-concentration macro »
([[agora-repnum-benchmark-verdict]]) **ne se produit pas** ici (top-share 0.15 vs
0.92) : il était l'artefact d'un corpus mono-domaine convergent, pas une faiblesse
de la méthode. Sur un corpus ouvert et contrasté, Agora produit une carte riche,
équilibrée et alignée sur la vérité terrain.

**Actions suivantes (hors-scope de ce témoin)** : (1) envisager la fusion n0+n1
(même sous-thème) ; (2) filtre de taille en aval pour évacuer les singletons
(n35/n36) ; (3) reléguer n34 (hors-axe) visuellement.
