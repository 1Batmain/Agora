# Témoin v2 — synthèses Agora (granddebat RE-EXTRAIT v2) vs synthèse OFFICIELLE

> Worker **witness-v2** · branche `work/witness-v2` · 2026-06-30
> But : refaire la vérif témoin ([[agora-granddebat-witness]]) sur le granddebat
> **RE-EXTRAIT v2** (extraction v2, cf. [[agora-extract-v2-verdict]]) et comparer
> à la baseline v1. Harness : `research/granddebat_witness_v2.py`
> (juge `mistral-large-latest`, TÉMOIN officiel IDENTIQUE à v1) →
> `research/granddebat_witness_v2_results.json`.

## Cadrage (inchangé)

Même corpus, même TÉMOIN qu'en v1 : espace **« Démocratie & citoyenneté »**,
question OUVERTE *« renouer le lien citoyens ↔ élus »*. Couverture mesurée sur les
**14 sous-thèmes OFFICIELS** de l'axe (synthèse OpinionWay, cf.
[[agora-official-syntheses]]). Immigration / laïcité / civisme restent hors-scope
(autres colonnes de la source) — leur absence n'est pas un raté.

## Ce qui change en v2 : consolidation

L'extraction v2 (moins sur-segmentée) produit une carte **plus consolidée** :
**19 macros** (vs 18 en v1), dont **13 in-scope**, **1 hors-axe** (80 km/h) et
**5 singletons** (1 avis) ignorés par consigne. Surtout, **une seule macro v2
couvre désormais DEUX sous-thèmes officiels** là où v1 les éclatait :

| Macro v2 | avis | couvre (officiel) |
|---|---:|---|
| n32 Participation citoyenne & organisation des débats | 702 | participation **+** débats/concertation |
| n182 Transparence totale des élus | 293 | transparence dépenses **+** reddition de comptes |
| n254 Représentation proportionnelle & tirage au sort | 260 | scrutin proportionnel **+** tirage au sort |

→ Le mapping autorise donc une **liste** de sous-thèmes par macro ; chaque paire
(macro, officiel) est jugée séparément, et la couverture compte les sous-thèmes
officiels distincts atteints (méthode comparable à v1).

## Résultats v2 vs v1

| Métrique | v1 (baseline) | **v2 (re-extrait)** | Δ |
|---|:---:|:---:|:---:|
| Sous-thèmes officiels couverts | 14 / 14 | **14 / 14** | **=** |
| **Couverture** | **100 %** | **100 %** | **=** |
| Mismatch | 0 | **0** | **=** |
| Alignement moyen (par sous-thème couvert) | 4.93 / 5 | **4.57 / 5** | **−0.36** |
| Alignement moyen (toutes paires jugées) | 4.93 / 5¹ | **4.50 / 5** (16 paires) | −0.43 |
| faithful / finer_split / partial / mismatch | 14 / 1 / 0 / 0 | **10 / 3 / 3 / 0** | — |
| Macros notées (in-scope) | 15 | 13 | −2 |
| Ajout hors-axe | n34 80 km/h (15) | n347 80 km/h (14) | = |
| Singletons de bruit | 2 (n35, n36) | 5 (n350–n354) | +3 |

¹ v1 jugeait 1 sous-thème par macro, 15 macros notées → moyenne 4.93 sur 15.

### Macros v2 → témoin (détail)

| Macro v2 | avis | → officiel | align | verdict |
|---|---:|---|:---:|---|
| n32 Participation & débats | 702 | participation_consultation | 5 | faithful |
| n32 | 702 | debats_concertation | 5 | faithful |
| n0 Réformer le lien élus-citoyens | 699 | lien_proximite_confiance | 5 | faithful |
| n83 Comprendre les réalités | 564 | lien_proximite_confiance | 5 | faithful |
| n136 Avantages & privilèges | 367 | privileges_remunerations | 5 | faithful |
| n182 Transparence totale | 293 | transparence_depenses | 5 | faithful |
| n182 | 293 | reddition_comptes | 4 | finer_split |
| n232 Présence & rôle des députés | 285 | nombre_elus | **3** | **partial** |
| n199 Bienveillance politique & médias | 269 | renouvellement_classe_politique | **4** | **partial** |
| n254 Proportionnelle & tirage au sort | 260 | scrutin_proportionnelle | 4 | finer_split |
| n254 | 260 | tirage_au_sort | 4 | finer_split |
| n270 Élus condamnés & inéligibilité | 231 | casier_probite | 5 | faithful |
| n293 Durée & cumul des mandats | 225 | cumul_limitation_mandats | 5 | faithful |
| n313 Référendums citoyens | 202 | ric_referendum | 5 | faithful |
| n322 Réorganisation territoriale | 163 | decentralisation_territoriale | 5 | faithful |
| n346 Reconnaissance du vote blanc | 98 | scrutin_proportionnelle | **3** | **partial** |
| **n347 Limitation 80 km/h** | 14 | — | — | **AJOUT hors-axe** |
| n350–n354 (1 avis chacun) | 1 | — | — | **singletons ignorés** |

## D'où vient la baisse d'alignement (−0.36)

100 % de couverture et 0 mismatch sont **maintenus**. La baisse vient de **3
appariements `partial`** (vs 0 en v1), tous explicables par un **recadrage de
titre**, pas par une perte de thème :

1. **n232 « Présence et rôle des députés » → nombre_elus (3, partial).** v2 cadre
   ce macro sur la **présence en circonscription** (quotas Paris/circo) plutôt que
   sur la **réduction du nombre** d'élus. Le contenu officiel est partiellement
   couvert (rôle du Parlement) mais l'angle « moins d'élus » est dilué. En v1, le
   macro homologue (n3 « fonctionnement parlementaire ») marquait 4 finer_split.
2. **n199 « Bienveillance politique et médias » → renouvellement_classe_politique
   (4, partial).** Macro le plus **bruité** de la carte v2 : il agrège lobbies +
   carrière politique (bon) **mais aussi** « bienveillance manipulatrice » et
   médias (hors sujet officiel). Le juge note l'élargissement excessif. v1 avait
   ici n4 « Représentation politique & partis » = 5 faithful, plus net.
3. **n346 « Reconnaissance du vote blanc » → scrutin_proportionnelle (3,
   partial).** N'est `partial` que parce qu'il est **redondant** : le sous-thème
   « mode de scrutin » est déjà capté à 4 par n254. Le vote blanc est une facette
   légitime et **isolée** par v2 — la couverture du sous-thème reste assurée (4).

Les 3 `finer_split` (n182→reddition, n254→proportionnelle, n254→tirage) sont des
**gains de granularité**, pas des défauts : le juge note que le thème émergent
**dépasse** le sous-thème officiel (plus fin / plus large), exactement comme le
`finer_split` unique de v1.

## Verdict chiffré

**v2 MAINTIENT le passage du témoin, avec un alignement légèrement abaissé.**

- **Couverture : 100 % (14/14), identique à v1. Zéro manque, zéro mismatch** —
  comme v1. La consolidation v2 (3 macros couvrant 2 sous-thèmes chacune) **ne
  perd aucun sous-thème officiel** : c'est le résultat clé, cohérent avec le
  verdict aveugle « zéro perte de thème » de [[agora-extract-v2-verdict]].
- **Alignement : 4.57/5 (par sous-thème couvert) vs 4.93/5 en v1 → −0.36.**
  Baisse réelle mais modeste, **entièrement portée par 3 `partial`** issus de
  **recadrages de titre** (n232 présence vs nombre ; n199 macro bruité ; n346
  redondant), non d'erreurs d'appariement.
- **Bruit singleton en hausse** : 5 macros à 1 avis (vs 2 en v1) — prix du rappel
  de l'extraction, à filtrer en aval par seuil de taille (déjà recommandé en v1),
  sans impact sur le témoin (ignorés par consigne).

**Lecture pour l'architecte.** Le rebuild v2 **ne dégrade pas la fidélité au
témoin sur l'essentiel** (couverture + zéro mismatch tenus). Le léger recul
d'alignement est diagnostique et actionnable : (1) **n199 « Bienveillance
politique et médias »** est un titre faible/bruité — candidat à re-titrage ou
re-clustering ; (2) **n232** gagnerait un cadrage incluant « nombre d'élus » ;
(3) **n346 vote blanc** pourrait être rattaché à n254 (mode de scrutin) pour
lever la redondance. Aucun de ces points ne remet en cause l'extraction v2 ;
ce sont des réglages de titres/granularité macro en aval.
