# Calibration des poids `sauce_magique` — VERDICT

**VERDICT : NON. On garde les poids v1 `W = (α=1, β=0,5, γ=1, δ=1)`.** La calibration
multi-golds ne produit AUCUN jeu de poids qui batte v1 de façon crédible ; le seul
« gain » mesuré est l'artefact trivial « plus de clusters = meilleur gold », qui est
exactement ce que la fonction objectif est faite pour RÉSISTER (lisibilité, anti-géant).
Aucune modification de `backend/recut.py`.

Harness : `research/sauce_calibration.py` · sortie brute :
`research/sauce_calibration_results.json`. Zéro appel LLM (embeddings nomic locaux).

## Protocole

Pour chaque dataset (granddebat 22k dev, xstance, republique-numerique) : on génère des
COUPES candidates de l'arbre EXISTANT (antichaînes couvrant les feuilles — coupes de
niveau fixe + `best_cut` glouton sous ~33 rayons de poids + coupes aléatoires,
dédupliquées ; ~40 par dataset, 4 seulement pour granddebat qui est déjà quasi plat, avec
reconstruction du géant `n0` comme racine virtuelle). On score chaque coupe par un GOLD
indépendant de la fonction objectif :

- **xstance** : NMI(affectation des claims à la coupe ↔ topic officiel x-stance porté par
  l'avis) — sans LLM ;
- **granddebat / repnum** : F1 d'appariement par embeddings nomic entre les nœuds de la
  coupe (titre+mots-clés) et les sous-thèmes OFFICIELS (14 OpinionWay / plan du projet de
  loi Titres I-III).

Calibration = grille `α,β,γ,δ ∈ {0,25 ; 0,5 ; 1 ; 2}` (175 rayons uniques, le classement
étant invariant d'échelle). Critère = meilleure corrélation de rang (Spearman) MOYENNE,
sur les 3 datasets, entre le classement `sauce_magique` des coupes et le classement gold.

## Résultat : aucun poids ne bat v1, tous anti-corrèlent

| rayon | mean | granddebat | xstance | repnum |
|---|---|---|---|---|
| **best grille** α=2 β=0,25 γ=0,25 δ=2 | **-0,194** | +1,000 | -0,913 | -0,670 |
| α=2 β=0,25 γ=0,5 δ=2 | -0,206 | +1,000 | -0,902 | -0,715 |
| … | … | | | |
| **v1** (1 ; 0,5 ; 1 ; 1) | **-0,257** | +1,000 | -0,905 | -0,866 |

Le Spearman moyen est NÉGATIF partout — y compris pour le « meilleur » rayon (-0,194).
Un Spearman négatif signifie que la fonction objectif préfère (score bas) les coupes que
le gold juge MOINS bonnes. L'écart best↔v1 (0,06) est entièrement porté par repnum et
provient d'un seul mécanisme : `β=0,25` relâche la pénalité de nombre de clusters et
`δ=2` durcit l'anti-géant, ce qui pousse vers des coupes plus FINES.

## Diagnostic : les golds ne mesurent que la granularité

C'est le point décisif. Corrélation de rang **taille de coupe ↔ gold** :

- granddebat **+0,200** (dégénéré : 4 coupes, le géant illisible domine)
- xstance **+0,867**
- repnum **+0,851**

Sur xstance et repnum, le gold est quasi une fonction MONOTONE du nombre de clusters :
NMI et F1-embeddings montent mécaniquement quand on subdivise (plus de nœuds → meilleure
couverture des topics/officiels). Or `sauce_magique` ne CHERCHE PAS la granularité : elle
équilibre la lisibilité (N_eff proche de N_cible ≈ 11–14) et pénalise la poussière et le
géant. Minimiser cet objectif anti-corrèle donc, par construction, avec un gold piloté par
la taille. « Battre v1 » sur ces golds = simplement fragmenter davantage — ce qui trahit
la propriété même que l'objectif existe pour garantir.

## Confirmation par les façades servies

`best_cut` avec v1 vs avec le meilleur rayon calibré produit la MÊME coupe sur 2 des 3
datasets :

| dataset | façade v1 | façade calibrée | gold v1 → calibré |
|---|---|---|---|
| granddebat | 37 clusters, top1 0,141 | **37, identique** | 0,767 = 0,767 |
| xstance | 17 clusters, top1 0,172 | **17, identique** | 0,317 = 0,317 |
| repnum | 23 clusters, N_eff 11,4 ≈ N_cible 11,5 | 38 clusters, **N_eff 27,4 ≫ N_cible 11,5** | 0,691 → 0,717 |

Le seul écart (repnum) voit le rayon calibré produire 38 macros à N_eff 27,4 pour une
cible de 11,5 : de la SUR-FRAGMENTATION caractérisée. Son gold n'est « meilleur » (+0,026)
que parce que le F1-embeddings récompense le surcroît de nœuds — le même artefact. Sur les
deux autres datasets, calibré et v1 sont indiscernables.

## Robustesse du proxy

Validation du proxy embeddings (utilisé pour granddebat/repnum) contre le NMI sur
xstance : **Spearman(F1-embed, NMI) = 0,465** — corrélation seulement modérée. Le gold
embeddings sur lequel repose l'unique « gain » (repnum) n'est donc que faiblement adossé
au gold dur (NMI). Raison de plus de ne pas rebâtir les poids dessus.

## Conclusion

1. **On garde v1** `(1 ; 0,5 ; 1 ; 1)`. Aucun code touché.
2. Le vrai résultat de cette calibration est NÉGATIF et instructif : **les golds
   disponibles (NMI, F1-embeddings vs officiels) sont dominés par une monotonie triviale
   avec la granularité** (Spearman taille↔gold 0,85–0,87), orthogonale voire antagoniste à
   l'objectif de `sauce_magique` (lisibilité, anti-géant). Ils ne constituent pas une cible
   de calibration valide pour cette fonction : la corrélation de rang sur toutes les
   antichaînes récompense « couper plus fin », pas « couper mieux ».
3. Pour calibrer honnêtement les poids il faudrait un gold qui pénalise la
   sur-fragmentation (p. ex. un F1 apparié pénalisant les nœuds officiels sur-comptés /
   la dispersion d'un même officiel sur plusieurs nœuds), ou un jugement humain/LLM à
   l'AVEUGLE sur la lisibilité des façades (cf. l'exigence de validation aveugle des autres
   verdicts du projet). Chantier séparé, hors périmètre (protocole « sans LLM »).
4. Les poids v1 restent des poids POSÉS À LA MAIN, non validés positivement — leur seule
   garantie reste le témoin granddebat 22k re-coupé (14/14, 0 mismatch, cf.
   `sauce_magique_note.md`). Cette limite (docstring de `recut.py`, point 1) demeure ouverte.
