# Grand Débat National — synthèse officielle vs Agora, côte à côte

**La même question, les mêmes réponses citoyennes, deux méthodes.** En 2019, la Mission du
Grand Débat a fait traiter les contributions par OpinionWay (grille de lecture + analyste).
En 2026, Agora fait émerger les thèmes automatiquement, sans grille. Ce document met les
deux résultats côte à côte — honnêtement, sources vérifiées.

> **Question analysée** (thème « La démocratie et la citoyenneté ») :
> *« Que faudrait-il faire pour renouer le lien entre les citoyens et les élus qui les
> représentent ? »* — question ouverte, réponses libres non suggérées.

---

## Les deux dispositifs

| | **Synthèse officielle (2019)** | **Agora (2026)** |
|---|---|---|
| Opérateur | OpinionWay (+ QWAM pour les verbatims), pour la Mission du Grand Débat | pipeline open-source (ce dépôt) |
| Méthode | grille de catégories construite par l'analyste, codage des contributions | **thèmes émergents** (extraction verbatim → embeddings → graphe → Leiden), zéro grille a priori |
| Volume traité | **118 356 contributions** à cette question (plateforme + papier numérisé BnF) | **3 000 réponses** échantillonnées de l'export open data (28 384 lignes) — **11 % de l'extrait** |
| Norme / cadre | ISO 20252, collège des garants | code + prompts publics, métriques et limites affichées dans l'outil |
| Délai | consultation close le 18 mars 2019 → synthèse actualisée **juin 2019** (~3 mois) | **~1 h de calcul** sur l'échantillon, coût LLM total **≈ 2,6 $** (mesuré/estimé, servi par `/cost`) |
| Livrable | rapport PDF ~200 p., catégories + % | carte navigable : 19 grands thèmes → 355 thèmes hiérarchisés → chaque **phrase citoyenne verbatim** |

---

## Résultats côte à côte

**Colonne officielle** : % de contributions classées dans la catégorie (multi-réponses,
base 118 356). **Colonne Agora** : le(s) thème(s) émergent(s) correspondant(s), établis par
un protocole de témoin (mapping + juge) — voir « Le témoin » plus bas.

| Sous-thème officiel (OpinionWay, juin 2019) | % off. | Thème émergent Agora (correspondance) |
|---|---:|---|
| Renforcer l'exemplarité des élus (promesses, dignité des débats, inéligibilité…) | **30,0 %** | Responsabilité des élus & transparence · Casier judiciaire des élus · Comprendre les réalités des citoyens |
| Modifier les règles électorales (↓ nombre de députés, proportionnelle, vote blanc…) | **16,3 %** | Présence et rôle des députés · Représentation proportionnelle & tirage au sort · Reconnaissance du vote blanc |
| Écouter les citoyens (référendums, RIC, échanges, société civile) | **8,7 %** | Participation citoyenne et organisation des débats · Référendums citoyens et constitution |
| Renforcer la transparence de la vie publique (argent public) | **6,5 %** | Transparence totale des élus · Transparence des dépenses publiques |
| Réduire revenus & avantages des élus (cumul, alignement fiscal…) | **4,5 %** | Réduction des avantages et privilèges des élus · Durée et cumul des mandats |
| Réformer les institutions (Sénat, services de l'État) + niveau local | 3,7 % + 2,1 % | Réorganisation territoriale et pouvoirs publics |
| Renforcer le civisme par l'éducation | 2,0 % | *(présent en sous-thèmes, pas en macro)* |
| Corps intermédiaires (renforcer / diminuer) | 1,6 % / 0,8 % | *(présent en sous-thèmes)* |
| Autres contributions classées | 2,4 % | — |
| **« Trop peu citées ou inclassables »** | **29,0 %** | **structurées : longue traîne de 355 thèmes hiérarchisés, traçables** |
| Non-réponses | 22,9 % | *(hors périmètre : Agora n'analyse que les réponses)* |

⚠️ **Les % ne sont pas directement comparables** : l'officiel compte des mentions
multi-réponses sur 118 356 contributions ; Agora compte des appartenances multi-thèmes sur
3 000 réponses échantillonnées. La comparaison porte sur la **structure retrouvée**, pas
sur les pourcentages.

### Le chiffre qui résume tout

> Dans le traitement officiel, **29,0 % des contributions** à cette question finissent en
> **« trop peu citées ou inclassables »** — presque une contribution exprimée sur trois
> sort de la grille. Agora n'a pas de grille : cette longue traîne est **structurée**
> (hiérarchie de 355 thèmes) et chaque idée reste **traçable jusqu'à la phrase exacte** du
> citoyen. *(Honnêteté : une partie de ces 29 % est simplement rare — Agora la range aussi
> en petits thèmes de faible poids, dont quelques singletons de bruit assumés.)*

---

## Le témoin : Agora retrouve-t-il la synthèse officielle ?

Protocole (reproductible : `research/granddebat_witness.py`) : les 14 sous-thèmes officiels
de l'axe pertinents pour cette question servent de **vérité terrain** ; chaque grand thème
Agora est mappé vers le sous-thème officiel le plus proche puis noté par un juge LLM
(alignement 0–5, verdict faithful / finer_split / partial / mismatch).

| Résultat | Valeur |
|---|---|
| **Couverture des sous-thèmes officiels** | **14 / 14 = 100 %** (aucun manque) |
| Alignement moyen (juge) | **4,93 / 5** (v1) · **4,57 / 5** après ré-extraction v2 (14/14 maintenu, recadrages de titres) |
| Mismatchs | **0** |
| Verdicts | 14 faithful · 1 finer_split · 0 partial · 0 mismatch |

**Ce qu'Agora trouve en plus** (hors grille officielle) : un thème *limitation à 80 km/h*
(débordement gilets jaunes réel dans les contributions — l'outil ne censure pas le
hors-sujet, il le montre) et quelques singletons de bruit (1 avis), prix assumé du rappel
d'extraction.

---

## Honnêteté : ce que chaque approche a que l'autre n'a pas

**Pour l'officiel** — l'échelle complète (118 356 contributions, papier inclus), une norme
(ISO 20252), des garants, un codage supervisé par des humains. C'est la référence, et c'est
précisément pour ça qu'on l'utilise comme témoin.

**Pour Agora** — l'émergence sans grille a priori (la structure vient des données), la
traçabilité verbatim jusqu'à la phrase, la longue traîne structurée au lieu d'« inclassables »,
l'exploration interactive (thème → sous-thème → témoignages surlignés → sentiment envers
l'objet de clivage), et un coût de ~2,6 $ / ~1 h là où le dispositif officiel a demandé un
marché public et des mois.

**Limites franches d'Agora sur ce corpus** :
- **Échantillon** : 3 000 réponses analysées sur 28 384 de l'export open data (11 %), soit
  ~2,5 % des 118 356 contributions officielles. Le témoin valide l'échantillon, pas le corpus
  entier ; aucune analyse de stabilité au re-tirage n'a encore été faite.
- **Le juge d'alignement est un LLM**, pas un panel humain (le protocole et les notes sont
  publics et rejouables).
- La grille officielle étant **imposée a priori**, ce témoin prouve qu'Agora *retrouve* la
  structure officielle — c'est l'objet du benchmark — pas que l'une des deux lectures est
  « la vraie ».
- Les % d'Agora décrivent des contributions volontaires classées automatiquement — **ce
  n'est pas un sondage** (mention affichée dans l'outil).

---

## Sources (vérifiées le 2026-07-02)

1. **Synthèse officielle** : *Traitement des données issues du grand débat national — « La
   démocratie et la citoyenneté »*, OpinionWay pour la Mission du Grand Débat National,
   version actualisée, **juin 2019**. Téléchargée depuis la page officielle des synthèses :
   <https://granddebat.fr/pages/syntheses-du-grand-debat> (PDF :
   `granddebat.fr/media/default/0001/01/7967bf7a5ea62fe6c284469196d9c829e26ac14a.pdf`).
   Question 3, pp. 34–36 : catégories et %, dont « Autres contributions trop peu citées ou
   inclassables : 29,0 % » et « Non réponses : 22,9 % » (p. 36), base 118 356 contributions.
2. **Méthodologie officielle** : pages « La méthodologie » du même PDF (questions rédigées
   par la mission interministérielle ; plateforme Cap Collectif ; papier numérisé BnF–Numen ;
   traitement OpinionWay selon ISO 20252 ; verbatims avec QWAM) ; Q&A OpinionWay du
   15 février 2019 : <https://www.opinion-way.com/images/blogs/OpinionWay_-_QA_sur_lanalyse_des_contributions_Grand_D%C3%A9bat_National_-_15_f%C3%A9vrier_2019.pdf>.
3. **Données analysées par Agora** : export open data du Grand Débat (granddebat.fr /
   data.gouv.fr, Licence Ouverte 2.0), réponses à la question ouverte du thème Démocratie
   & citoyenneté (28 384 lignes dans notre extrait ; échantillon analysé : 3 000, seed 42).
4. **Protocole de témoin + résultats** : `research/granddebat_witness_note.md`,
   `research/granddebat_witness.py`, `research/granddebat_witness_v2_results.json` (ce dépôt).
5. **Coût & durées Agora** : endpoint `/cost?dataset=granddebat` (phases mesurées à l'appel ;
   extraction estimée a posteriori et marquée comme telle).
