# sauce_magique v1 — re-coupe de l'arbre : VERDICT

**VERDICT : OUI.** La re-coupe sauce_magique répare la façade macro effondrée du corpus
complet granddebat (22 174 avis) sans toucher au clustering : **21 → 37 macros, top1
99,9 % → 14,1 %**, et la nouvelle façade tient le témoin officiel : **couverture 14/14,
0 mismatch, alignement 4,71/5** (couverture) / 4,61 (toutes paires). Câblée au pipeline
(`backend/recut.py`, appelée par `build_analysis` après `build_theme_tree`).

## Contexte : l'effondrement macro à l'échelle

Sur granddebat 3k, la façade macro était saine (19 macros, témoin v2 : 14/14, 0 mismatch).
Sur le corpus complet 22k, le coarsening racine (fusion μ+σ des centroïdes) a
transitivement aspiré presque tout dans UN macro : `n0` = 22 172 avis = **99,9 % des
voix** (+ 20 racines singleton de bruit). La structure FINE de l'arbre restait bonne
(les enfants/petits-enfants de n0 sont les vrais thèmes) — c'est la COUPE servie qui
était mauvaise, pas le clustering. Idée (Bob) : chercher a posteriori le « niveau de
récursivité » le plus lisible, en équilibrant équi-répartition des voix et nombre de
clusters.

## La fonction objectif v1 (à MINIMISER)

```
sauce_magique = α·(1 − cohésion)            # qualité sémantique, pondérée voix
              + β·|log(N_eff / N_cible)|    # lisibilité : nb EFFECTIF de clusters
              + γ·poussière                 # part des voix en clusters < 0,5 %
              + δ·top1                      # dominance du plus gros (anti-géant)
```

- **N_eff** = exp(entropie de Shannon des parts de voix) : le géant à 99 % → ~1 ;
  quatorze clusters équilibrés → ~14. C'est lui qui rend le terme β insensible aux
  singletons de bruit (ils ne pèsent rien dans l'entropie).
- **N_cible** = max(6, ln(voix)·1,4) — dérivé des données (~14 à 22k, ~11 à 3k),
  aucune constante de corpus.
- **Poids v1** : α=1, β=0,5, γ=1, δ=1 — posés à la main, **NON calibrés** (cf. limites).
- **Recherche** : descente gloutonne sur l'arbre existant — on part des racines et on
  éclate le nœud dont l'explosion améliore le plus le score, jusqu'à stabilité.
  L'antichaîne obtenue devient la façade macro (re-racinage, ancêtres dissous, ids
  de nœuds INCHANGÉS, couleurs palette réassignées).

## Résultat de coupe (granddebat 22k, dev → prod `granddebat-complet`)

| | avant (racines) | après (coupe optimisée) |
|---|---|---|
| clusters | 21 | 37 (17 réels + 20 singletons de bruit) |
| N_eff | 1,0 | **14,0** (N_cible 14,6) |
| top1 | **99,9 %** | **14,1 %** (n1, 4 621 avis) |
| cohésion pondérée | 0,606 | 0,655 |
| poussière | 0,001 | 0,001 |
| score v1 | 2,709 | **0,506** |

Un seul nœud dissous (`n0`) : la coupe = les 17 enfants de n0 + les 20 racines
existantes. Application au cache SANS re-extraction ni appel LLM
(`research/apply_recut.py` : arbre reproduit à l'identique depuis les caches,
sanity-check, champs LLM recopiés par id, `analysis.json`/`avis.json` réécrits par les
fonctions du build ; claim_stance recouvrement 100 %). Vérifié en prod :
`POST /api/analysis {"dataset":"granddebat-complet"}` → 37 macros, top1 14,1 %,
`params.recut` tracé ; le `granddebat` 3k validé est intact.

## Témoin officiel (protocole granddebat_witness_v2, juge mistral-large)

`research/granddebat_witness_recut.py` (mapping expert macro → sous-thème(s), 20
singletons ignorés comme en v2) → `granddebat_witness_recut_results.json` :

- **Couverture : 14/14 sous-thèmes officiels (100 %), 0 manque, 0 mismatch.**
- Alignement moyen : **4,71/5** en couverture (meilleure paire par sous-thème),
  4,61 sur les 18 paires jugées — 13 faithful, 2 finer_split, 3 partial.
- Repères : v1 (3k) 4,93 · v2 (3k re-extrait) 4,57 — le 22k re-coupé est **entre les
  deux**, avec la même couverture parfaite.
- Paires « partial » (pas des erreurs de thème, des recouvrements partiels) :
  n24 « Comportements politiques » ↔ renouvellement_classe_politique (3/5),
  n34 « Lobbying » ↔ renouvellement_classe_politique (3/5),
  n31 « Vote blanc » ↔ scrutin_proportionnelle (4/5, facette).
- Consolidations légitimes : n2 (4 590 avis) couvre participation + débats/concertation
  + tirage_au_sort (403 claims « tirage/tirés au sort » dans ses citations).
- Addition hors-axe attendue : n32 « Éducation civique » (527 avis — colonnes civisme
  de la source, hors question « lien citoyens-élus »).

## Limites (et suite)

1. **Poids non calibrés.** α/β/γ/δ v1 sont posés à la main ; le score ne compare que
   des coupes d'un MÊME arbre. Étape suivante : **calibration multi-golds** (témoin
   granddebat, x-stance 12 topics, repnum) par corrélation de rang entre le score et
   la qualité jugée, avant de faire confiance au score inter-corpus.
2. **Coupe gloutonne, non exhaustive.** On n'explore que les explosions successives
   depuis les racines (jamais de fusion, pas de retour arrière) : l'optimum global de
   la fonction sur toutes les antichaînes n'est pas garanti. Suffisant ici (le géant
   est le cas pathologique visé), à revisiter si des arbres plus profonds coincent.
3. **Les singletons de bruit restent en façade** (20 macros à 1 avis). N_eff les rend
   invisibles au score, mais l'UI les montre ; leur absorption/écrémage est un chantier
   séparé (filtrage aval, cf. verdict extraction B « le bruit est le prix du rappel »).
4. **La re-coupe ne répare pas le coarsening.** Elle corrige la façade a posteriori ;
   la cause (fusion transitive μ+σ à l'échelle) reste — un rebuild futur passe
   désormais par `recut_tree` au build, donc l'effet est neutralisé en sortie.

## Écart rencontré (traçabilité)

Premier run d'`apply_recut` : `build_theme_tree(ds)` sans `model=` résolvait le modèle
d'extraction PAR DÉFAUT (`ministral-3b-latest`) ≠ clé du cache claims
(`mistral-large-latest`) → une re-extraction complète a DÉMARRÉ (tuée après ~12 min,
`claims.json` intact — l'écriture n'a lieu qu'en fin ; quelques centaines d'appels
ministral-3b perdus). Correctif dans le script : modèle d'extraction du build passé
explicitement + vérification fail-closed de la couverture du cache AVANT tout appel
pipeline. Leçon générique : **tout rejeu hors `build_analysis` doit épingler le modèle
d'extraction** (le cache claims est clé par modèle).
