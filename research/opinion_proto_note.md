# PROTO OPINION — quelle CIBLE mesure le mieux la répartition pour/contre par thème ?

> Worker **opinion-proto** · branche `work/opinion-proto` · 2026-06-29 · R&D pur (`research/`,
> aucun cache produit touché — lecture seule du dépôt principal pour `claims.json` + clé).
> Scripts : `research/opinion_proto.py` (clustering + 3 cibles × stance), `research/opinion_annotate.py`
> (gold manuel + taux d'erreur). Chiffres : `opinion_proto_results.json`, `opinion_proto_annot.json`.

Le **livrable** Agora : mesurer la **RÉPARTITION** d'opinion par thème (favorable / défavorable /
nuancé), pas un seul côté. Dataset = **GRAND DÉBAT** « démocratie et citoyenneté » : **3000**
contributions à **UNE** question ouverte
> *« Que faudrait-il faire pour renouer le lien entre les citoyens et les élus qui les représentent ? »*

S'appuie sur [[agora-stance-subject-verdict]] (sujet=cluster agrège net) et [[agora-official-syntheses]]
(axes officiels Grand Débat). Clustering = chemin de prod `build_live_tree` (k dérivé) → **13 macros**,
**10 analysées** (n ≥ 100). Stance : `mistral-small-latest`, prompt **identique** pour les 3 cibles,
**mêmes** 60 claims verbatim échantillonnés par cluster → seule la chaîne-cible change (design propre).

---

## TL;DR — verdict sur la cible

| cible | engagement¹ | opposition² | sépare-t-elle un vrai pour/contre ? |
|---|--:|--:|---|
| **T1 titre du cluster** | 0.76 | 0.21 | **NON — ambiguïté de signe.** Une charge contre les élus se classe tantôt « favorable » (au contrôle), tantôt « défavorable » (négatif) → faux 50/50 qui *ressemble* à un clivage mais est du bruit. |
| **T2 objet de clivage dérivé** | 0.48 | 0.14 | **OUI — seul axe cohérent.** « pour ou contre [instaurer le RIC] » est non-ambigu. Sa « faible » couverture est en fait **correcte** (il s'abstient hors-mesure) et **diagnostique** : les macros agrègent PLUSIEURS mesures. |
| **T3 question de consultation** | 0.88 | 0.26 | **NON — mesure le TON, pas la stance.** « Un casier vierge doit être exigé » (proposition constructive) est classé *défavorable* à la question car le ton est critique envers les élus. Engagement le plus haut = **factice**. |

¹ engagement = (fav+def)/n = 1 − part de « nuance ». ² opposition = min(fav,def)/(fav+def) = part du camp minoritaire.

**Cible retenue pour la prod : T2, l'objet de clivage dérivé, formulé en PROPOSITION POLAIRE**
(« instaurer le RIC », « tirer au sort des citoyens », « réduire le nombre d'élus »). C'est la **seule**
cible dont « favorable / défavorable » désigne une chose **constante et interprétable**. Sa contrepartie :
elle doit être dérivée à la **bonne granularité** (un thème ≈ une proposition), sinon elle s'abstient
beaucoup (cf. §granularité). **Taux d'erreur stance mesuré : 11.1 %** (§erreur).

---

## 1. Le nœud dur, en chiffres — pourquoi T1 et T3 trompent

### T1 (titre) — ambiguïté de signe
Le titre d'un thème de réforme (« Responsabilité et contrôle des élus », « Réformer le lien… ») est
**bivalent** : un même grief peut se lire « favorable » (je veux ce contrôle) OU « défavorable » (je suis
négatif sur les élus). Le LLM oscille → des splits 29/28, 39/19 qui **imitent** un clivage sans en être un.

| claim (cluster n14 « Responsabilité et contrôle des élus ») | stance T1 (titre) | stance T2 (cleavage = « exiger casier vierge ») |
|---|---|---|
| « quand je les entends critiquer notre justice… ils ne pensent qu'à eux » | **defavorable** | nuance |
| « Un casier judiciaire vierge doit être exigé » | favorable | **favorable** (net) |

→ La 1ʳᵉ ligne : grief anti-élus → T1 le compte « défavorable » alors que l'auteur **réclame** plus de
contrôle. Signe inversé. T2 s'abstient proprement (le grief ne parle pas du casier).

### T3 (question) — confond grief et stance
Être « défavorable » à *« que faudrait-il faire pour renouer le lien ? »* n'a pas de sens : le LLM y
projette la **polarité de ton**. Résultat, des **propositions constructives** sont étiquetées *défavorable* :

| claim | stance T3 (question) | réalité |
|---|---|---|
| « Un casier judiciaire vierge doit être exigé » | **defavorable** | proposition POUR une mesure |
| « rendre les recommandations de la Cour des Comptes exécutoires » | **defavorable** | proposition POUR une mesure |

→ T3 maximise l'« engagement » (0.88) en mesurant **cynisme vs constructif**, pas pour/contre une mesure.
Confirme et **précise** le rejet de la question comme cible ([[agora-target-question-fallback-verdict]]).

### T2 (cleavage) — le seul axe cohérent
« pour ou contre *[instaurer le RIC]* » est non-ambigu. Quand un claim parle d'une AUTRE mesure
(proportionnelle, vote blanc… sous la cible RIC), T2 répond **nuance** — ce qui est **correct**, pas un
défaut : cela révèle que le macro `n11` agrège plusieurs propositions distinctes.

---

## 2. Granularité — la « faible couverture » de T2 mesure l'impureté des macros

L'engagement de T2 par cluster **trace** la pureté du thème : élevé quand le macro ≈ une proposition,
faible quand il en empile plusieurs.

| macro | objet de clivage dérivé (T2) | engagement T2 | lecture |
|---|---|--:|---|
| n15 | instaurer une transparence totale des comptes des élus | **0.75** | thème ≈ 1 mesure → couvre bien |
| n16 | organiser des débats publics réguliers sur le terrain | 0.73 | idem |
| n13 | limiter le cumul des mandats | 0.65 | idem |
| n2 | parler vrai / arrêter la langue de bois | 0.62 | idem |
| n1 | réunions locales trimestrielles avec les députés | 0.52 | mixte |
| n14 | exiger un casier judiciaire vierge | 0.33 | **macro large** (responsabilité ⊋ casier) → s'abstient |
| n0 | réduire le cumul des mandats | 0.32 | **fourre-tout** (« réformer le lien ») → cleavage mal ajusté |
| n11 | instaurer le RIC | 0.25 | **bundle** RIC + proportionnelle + vote blanc + 7 ans… |
| n3 | réduire le nombre de députés/sénateurs | 0.22 | macro = « statut des élus » ⊋ nombre |

→ **Reco granularité** : dériver la proposition-cible au niveau **feuille / sous-thème** (un objet de
clivage par nœud fin), ou émettre **plusieurs** objets par macro et router chaque claim vers le plus
proche. Au niveau macro, T2 reste **juste mais partiel**.

---

## 3. Agrégats d'opinion par thème (cible T2) — clivants vs consensuels

Répartition sur les claims **engagés** (favorable+défavorable ; « nuance » = hors-mesure, exclu du ratio) :

| macro | objet de clivage (T2) | fav | def | % favorable | opposition | profil |
|---|---|--:|--:|--:|--:|---|
| n16 | débats publics sur le terrain | 42 | 2 | **95 %** | 0.05 | **consensuel** |
| n11 | instaurer le RIC | 14 | 1 | 93 % | 0.07 | consensuel (couv. faible) |
| n2 | parler vrai / langue de bois | 34 | 3 | 92 % | 0.08 | consensuel |
| n13 | limiter le cumul des mandats | 36 | 3 | 92 % | 0.08 | consensuel |
| n1 | réunions locales avec les députés | 27 | 4 | 87 % | 0.13 | consensuel |
| n14 | casier judiciaire vierge | 17 | 3 | 85 % | 0.15 | consensuel |
| **n12** | **tirer au sort des citoyens** | 21 | 4 | **84 %** | **0.16** | **clivant léger** |
| **n15** | **transparence totale des élus** | 36 | 9 | **80 %** | **0.20** | **clivant léger** |
| n3 | réduire le nombre de députés | 10 | 3 | 77 % | 0.23 | couv. faible |
| n0 | réduire le cumul (macro fourre-tout) | 5 | 14 | 26 % | 0.26 | **artefact** (cleavage mal ajusté) |

**Lecture clé.** Dans une consultation OUVERTE « que faudrait-il faire ? », les thèmes sont
**consensuels par construction** : les gens se regroupent autour des mesures qu'ils **proposent** (donc
soutiennent). Le vrai clivage pour/contre vit **ENTRE** thèmes (quelle mesure domine) ; **DANS** un thème
il se réduit à une **minorité de sceptiques** (4–20 %). T2 fait remonter cette minorité **proprement** là
où elle existe (sceptiques du **tirage au sort** n12, cyniques de la **transparence** n15). Le cas n0 est
un **garde-fou** : sur un macro fourre-tout, l'objet de clivage est mal ajusté et le « def » se gonfle de
griefs mal signés (même mécanisme que T1) → **n'agréger la stance que sur des thèmes assez purs**.

---

## 4. Témoin — confrontation à la synthèse OFFICIELLE Grand Débat

Axes officiels « démocratie & citoyenneté » ([[agora-official-syntheses]]) : RIC, proportionnelle, vote
blanc, non-cumul des mandats, réduction/contrôle des élus — **demandes majoritaires** ; le tirage au sort
est l'innovation la plus **discutée**.

| attendu officiel | mesure Agora (T2) | accord |
|---|---|---|
| RIC très demandé | n11 RIC = **93 % favorable** | ✓ |
| Non-cumul / limitation des mandats demandé | n13 = **92 % favorable** | ✓ |
| Réduction du nombre d'élus (demande populaire) | n3 = **77 % favorable** | ✓ |
| Transparence / contrôle des élus demandé | n15 = 80 %, n14 = 85 % favorable | ✓ |
| Tirage au sort = l'idée la plus **clivante** | n12 = opposition **la + forte (hors artefact)** | ✓ |

→ Les agrégats de stance Agora **vont dans le sens** de la synthèse officielle : mesures de réforme
démocratique **majoritairement réclamées**, et le **scepticisme se concentre** sur le tirage au sort (et,
sous forme de cynisme, sur la transparence). Pas de contradiction relevée.

---

## 5. Taux d'erreur stance (échantillon annoté à la main)

45 claims tirés au hasard (cible = T2), **gold posé à la main** vs label LLM
(`research/opinion_annotate.py`, gold figé dans `opinion_proto_annot.json`) :

```
N = 45   accord = 40/45 = 88.9 %   erreur = 11.1 %
              tp/fp/fn   prec  rec
  favorable   13/ 3/ 0   0.81  1.00
  defavorable  0/ 2/ 0   0.00   —
  nuance      27/ 0/ 5   1.00  0.84
```

**Les 5 erreurs sont toutes des sur-attributions** (jamais une stance ratée) :
- **3 « favorable » trop zélés** sur une mesure *adjacente mais distincte* : « co-préparation des lois »,
  « élus accessibles », « consulter le peuple » ≠ la cible précise → devraient être *nuance*.
- **2 « défavorable » parasites** issus du **ton de grief** : « suppression du Sénat », « honteux de voir
  nos députés dormir » → mal signés (même piège que T1/T3).

Propriétés rassurantes pour la prod : **rappel favorable = 1.00** (aucune vraie adhésion manquée),
**précision nuance = 1.00** (le modèle s'abstient à bon escient, jamais à tort). Le risque = **gonfler
légèrement la mesure dominante** et **émettre de rares « défavorable » fantômes** sur les macros
impurs/chargés de griefs. La classe « défavorable » est **fragile** (0/2 ici) → à fiabiliser.

---

## 6. Recommandation d'architecture — feature « répartition d'opinion » de prod

1. **Cible = objet de clivage dérivé**, formulé en **proposition polaire** (1 passe LLM/thème,
   `CLEAVAGE_SYSTEM` du proto), **pas** le titre ni la question. C'est le seul axe où favorable/défavorable
   est constant et interprétable.
2. **Dériver la cible à la granularité FEUILLE** (un thème fin ≈ une proposition). Au niveau macro,
   émettre N objets et router chaque claim vers le plus proche, OU descendre d'un cran dans l'arbre. La
   couverture de T2 (engagement) sert de **jauge de pureté** : engagement faible ⇒ thème à re-subdiviser.
3. **Servir la répartition comme [N favorables / M défavorables / K nuancés] sur *[proposition]***, avec
   le **% favorable parmi les engagés** + un **badge clivant/consensuel** (seuil opposition ≈ 0.15).
   Sur cette consultation, l'info-clé est *« thème consensuel à 90 %+ »* vs *« sceptiques notables (16–20 %) »*.
4. **Garde-fous anti-bruit** (le taux d'erreur est surtout de la sur-attribution) :
   - durcir le prompt « ne juge QUE si le claim porte sur CETTE mesure ; sinon nuance » (réduit les
     favorables adjacents) ;
   - **n'afficher la stance que sur des thèmes assez purs** (engagement T2 ≥ seuil) — sinon afficher le
     thème sans répartition (le cas n0 montre qu'un macro fourre-tout produit des « def » fantômes) ;
   - **double-lecture des « défavorable »** (classe fragile) avant de les servir comme opposition.
5. **Traçabilité** : ancrer chaque stance sur le claim verbatim ([[agora-spans-anchor-textclean]]) +
   justif courte ; la cible per-claim de `claims.json` reste un **signal de clustering**, pas la cible de
   stance ([[agora-stance-subject-verdict]]).
6. **Coût** : batch 10 claims/appel, mistral-small. Ici 10 thèmes × 60 claims × 3 cibles ≈ 180 appels ;
   en prod (1 seule cible, par thème, au build) c'est ~1 appel/10 claims → négligeable et cachable.

**Reco honnête** : la valeur d'Agora sur une consultation ouverte n'est PAS « X % pour / Y % contre » DANS
chaque thème (ils sont consensuels par construction), mais (a) **quelles propositions dominent** (taille
relative des thèmes) et (b) **où se cache une minorité d'opposition réelle** (tirage au sort, transparence)
— c'est exactement ce que T2 fait remonter, et ce sur quoi cadrer l'UI.

---

## 7. Reproduire

```
export MISTRAL_API_KEY=$(cat var/mistral.key)         # ou repli lecture seule dépôt principal
PYTHONPATH=. uv run --extra contender --extra embed-contender --extra faiss --with fastapi \
  python research/opinion_proto.py                    # clustering + 3 cibles × stance → results.json + annot.json
python research/opinion_annotate.py                   # gold manuel + taux d'erreur → annot.json
```
