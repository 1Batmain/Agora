# Sonde — extraction de claims sur de vrais tours de parole AN

**Fait le 2026-08-04.** Question : *une réplique de débat, sortie de son fil, garde-t-elle son
sens ?* C'était la dernière inconnue avant de figer la conception du pipeline live.

Protocole : séance **055 du 28 nov. 2024** (« Abrogation de la retraite à 64 ans »), 269 tours
substantiels, **échantillon de 20 tours étalés** sur toute la séance, passés au **vrai**
`extract_claims` (batching + repli mono-avis + ancrage `align_spans`), modèle réel
`mistral-large-latest`, question de cadrage « Faut-il abroger la retraite à 64 ans ? ».
Script jetable : `scratchpad/probe_an.py`. Coût : quelques centimes.

> ⚠️ Échantillon de 20 tours, une seule séance. Suffisant pour **détecter** des modes de panne
> (une panne vue est une panne réelle), **insuffisant** pour en chiffrer la fréquence. Aucun
> ratio ci-dessous ne doit être cité comme un taux — cf. la règle de robustesse des bancs.

---

## 1. Le résultat rassurant : l'extraction verbatim tient

| Mesure | Débat AN | Corpus citoyen (référence) |
|---|--:|--:|
| claims par tour | **1,55** | 1,45 |
| replis d'ancrage (tour entier = 1 claim) | 15 % | — |

Les claims sortent **propres, ancrés, bien découpés**. La machinerie verbatim ne se dégrade pas
sur du discours parlementaire. **Et la déixis n'est PAS le problème** que je redoutais : 16 % des
claims portent un déictique, presque tous bénins (« ce texte », « cette proposition de loi »
restent clairs dès lors que le sujet de la séance est connu).

**Mon hypothèse de départ était la mauvaise.** Les vrais problèmes sont ailleurs — et l'un est
sérieux.

## 2. ⚠️ Risque n°1 : le discours rapporté inverse la position

Un orateur cite l'argument adverse **pour le réfuter**. Extrait comme claim, il est attribué à
son auteur — à l'envers.

> **Mme la ministre** (opposée à l'abrogation) :
> *« En repoussant de deux années l'âge légal, nous serions des voleurs de vie. »*
> …suivi, dans le même tour, de : « L'expression est faite pour frapper les esprits. Elle y
> réussit **mais elle nous conduit sur une fausse route**. »

Le claim retenu est la **thèse adverse**, citée pour être démolie. Une stance calculée dessus
classerait la ministre **favorable à l'abrogation** — l'exact contraire de sa position.

C'est plus grave que la déixis : ça ne produit pas de l'illisible (visible), ça produit du
**faux plausible** (invisible). Et ça frappe précisément ce qui fait l'intérêt du cas AN :
l'attribution de position par orateur.

**Pas de réponse existante dans le dépôt.** Deux pistes, à trancher par mesure :
consigne d'extraction explicite (« n'extrais pas ce que l'orateur cite pour le réfuter »), ou
contrôle en aval au moment de la stance. La rhétorique parlementaire est *construite* sur ce
procédé — ce n'est pas un cas marginal.

## 3. ⚠️ Risque n°2 : une grande part du débat ne porte pas sur le fond

Sur les 31 claims extraits, à ma lecture, **environ la moitié** ne sont pas des positions sur
la question posée mais :

- des **attaques personnelles** — « Faire l'influenceur sur les réseaux sociaux, c'est une
  chose. Être député en est une autre ! », « Dans le fond, vous êtes des populistes », « nous
  n'allons pas directement à l'Ehpad, ni au cercueil ! » ;
- de la **procédure** — recevabilité au titre de l'article 40, avis de la commission, « le
  rapporteur refuse cet amendement au motif que… » ;
- du **hors-sujet politique** — « il est temps d'abroger le 49.3 », « un président qui a perdu
  deux élections… démissionne ».

Ce n'est pas un défaut d'extraction : **c'est ce qu'est un débat parlementaire.** Une
consultation citoyenne répond à une question ; une séance est aussi un théâtre d'affrontement.

Conséquence : le **pré-filtre de pertinence** cesse d'être un raffinement pour devenir une
pièce centrale. Bonne nouvelle — il **existe déjà** et il est validé
(`research/relevance_calibration_note.md`, intégré au prompt de stance de `build_opinion`) ;
il faudra le remonter en amont et le recalibrer sur ce registre.

## 4. Deux défauts structurels — mesurés sur les 94 709 tours du corpus

| Constat | Ampleur | Conséquence |
|---|--:|---|
| **Tours du même orateur qui se suivent** | **36,9 %** | une intervention est **coupée en plusieurs paragraphes** par les interruptions → il faut **recoller** avant d'extraire, sinon on hache des raisonnements |
| Tours commençant par « … » | 6,9 % | même cause, cas le plus visible (phrase reprise en cours) |
| Libellé « M./Mme le·a président·e » **passant** le filtre `roledebat` | **1,7 %** (1 629 tours) | l'attribut n'est pas toujours posé → fuite de parole procédurale du perchoir. **2 des 3 replis d'ancrage** de l'échantillon venaient de là |

Le recollage à 36,9 % est le plus important : il change l'unité de traitement. **Un tour de
parole n'est pas un paragraphe** — c'est une suite de paragraphes du même orateur.

## 5. Verdict

**Le pipeline d'extraction transfère au débat parlementaire** — le verbatim tient, le
découpage est bon, la déixis est un faux problème. Ce qui ne transfère pas, c'est l'hypothèse
implicite du produit actuel : *« un texte = une position sincère sur la question posée »*.

Trois ajouts deviennent **obligatoires** (et non « souhaitables ») dans le pipeline live :

1. **recollage** des paragraphes consécutifs d'un même orateur (avant extraction) ;
2. **filtre du perchoir durci** — sur le libellé d'orateur, pas seulement `roledebat` ;
3. **pré-filtre de pertinence** en amont, pour écarter attaque personnelle et procédure.

Et une **inconnue de recherche ouverte**, à traiter avant toute promesse sur les positions :
**le discours rapporté**. Tant qu'il n'est pas traité, une position par orateur peut être
publiée à l'envers.

## 6. Suite

Ces trois ajouts sont exactement des **couches** au sens de `PIPELINE_PROFILE.md` — ce qui
confirme l'ordre : la classe de profil d'abord, ces couches ensuite. Le risque « discours
rapporté » rejoint la table des risques de `LIVE_SCENARIO.md`, en **haute** gravité, à côté de
« positions ≠ embeddings ».

Artefacts : `scratchpad/probe_an_dump.txt` (les 20 tours et leurs claims, lisibles),
`scratchpad/probe_an_results.json`.
