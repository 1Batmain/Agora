# Verdict — k est un ROBINET DE ZOOM ; la hiérarchie se LIT dans l'emboîtement (2026-07-14)

Aboutissement de la réflexion sur la hiérarchie (`HIERARCHY_TAU`, `EMBEDDING_SPACE`,
`HIERARCHY_KMOD`). Point de départ : « peut-être que tiktok et x-stance sont un seul layer,
et en créer d'autres par-dessus nous force à mentir » (Bob).

- **OUI** — la chaîne d'emboîtement remplace `derive_k(N)` à la racine. *Câblé, défaut.*
- **NON** — `sauce_magique` (re-coupe) est RETIRÉE. *Code supprimé.*

## Le problème

`derive_k(N) = 3.8·log₁₀N` devinait le nombre de thèmes à partir de la **taille** du corpus.
Une formule qui ne regarde jamais le contenu : deux corpus de même taille et de structure
opposée recevaient le même k.

## La méthode

On ne DÉRIVE pas k, on le **balaie** (k = rayon de voisinage = zoom), on clusterise à chaque
palier, et on lit la hiérarchie dans l'**emboîtement** des partitions les unes dans les autres
— poupées russes. La **propreté** d'un saut est l'emboîtement normalisé : `0` = hasard
(étiquettes grossières mélangées = modèle nul), `1` = parfait. Aucun seuil arbitraire.

Le 1er étage de la chaîne donne les thèmes **fins**, le 2e la couche **macro**. Un cluster fin
hérite du macro où tombe la MAJORITÉ de ses claims (l'emboîtement n'étant jamais parfait, c'est
un vote, jamais un appariement d'identifiants).

Code : `pipeline/cluster/layers.py` · mesure : `research/k_layers.py`.

## Les mesures (espace recentré, claims du cache, seed 42)

| corpus | chaîne | lecture |
|---|---|---|
| tiktok | 16 → 9 (0.70) → 4 (0.82) | cascade |
| x-stance | 24 → 14 (0.77) → 7 (0.78) | cascade — **14 ≈ les 12 topics du gold humain** |
| république-numérique | 31 → 17 (0.69) → 9 (0.79) → 5 (0.65) | cascade |
| **témoin** tiktok+x-stance | 21 → **2 (0.94)** | 2 domaines collés exprès : saut net |

## Ce que ça dit

**Aucun corpus RÉEL n'a de frontière macro nette.** Tout tient entre 0.65 et 0.82, alors que le
témoin artificiel monte à 0.94 — la méthode détecte donc la hiérarchie *quand elle existe*, et
la platitude de nos corpus est réelle, pas un défaut de mesure. La couche grossière est une
**commodité de navigation**, pas une structure que le corpus imposerait.

**Décision produit (Bob, 2026-07-14) : l'affichage reste UNE coupe.** La chaîne sert à
*choisir* la coupe, pas à la faire naviguer. Le front n'expose ni curseur de k ni jauge.

Le **niveau fin**, lui, est validé de l'extérieur : sur x-stance — seul corpus annoté par des
humains — la chaîne s'arrête à 14 thèmes là où les annotateurs en avaient défini 12.

La propreté reste une **jauge continue**, jamais un verdict binaire : tracer une coupe
« plat / feuilleté » exigerait un magic number. `CLEAN_FLOOR` n'annote que les sorties de
mesure ; le code ne s'en sert jamais pour trancher.

## Pourquoi sauce_magique est retirée

Elle minimisait `α(1−cohésion) + β|log(N_eff/N_cible)| + γ·poussière + δ·top1` pour re-couper
la façade macro. Elle **fragmentait des thèmes cohérents**, et sa raison d'être — le macro
géant à 99.9 % de granddebat — est un **artefact d'ANISOTROPIE** : il s'évapore une fois
l'espace recentré (top1 : 0.999 → 0.185, cf. `EMBEDDING_SPACE.md`).

Elle entrait de surcroît en **conflit** avec la chaîne : sur tiktok, la mesure dit 9 macros,
sauce_magique en refaisait 13. Une seule autorité sur la hiérarchie à la fois.
Cliquet de non-régression : `test_sauce_magique_a_disparu`.

## Ce que ça règle

« Addiction corporelle » : le titre absurde venait de FORCER une fusion (k=120 codé en dur)
que la structure ne porte pas. La chaîne sépare « comparaison des corps féminins » de
« addiction au scroll » — deux macros distincts, plus de titre soudé à inventer.

## Limite honnête

Le **plafond mémoire** du balayage (`MAX_EDGES` ; le graphe kNN pèse n·k arêtes) fait sauter
les paliers les plus grossiers sur les très gros corpus. Les paliers sautés sont **signalés**,
jamais tronqués en silence. **granddebat (36 k claims) n'est pas encore mesuré** pour cette
raison — c'est la dette ouverte de ce verdict.

## Reproduire

```
uv run --extra embed-contender --extra faiss python -u research/k_layers.py tiktok xstance mix
```
Mesures gratuites : claims et embeddings servis du cache, aucun appel LLM. Lancer dans tmux —
le balayage est long et un gros corpus tient la mémoire.
