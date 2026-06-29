# VERDICT — transformation du poids des arêtes k-NN avant Leiden

**Question.** Aujourd'hui une arête du graphe k-NN porte le **cosinus brut** et Leiden
l'optimise tel quel (`leiden_cluster.run_leiden`, `weights="weight"`). **Accentuer les
liens forts** par une transformation du poids donne-t-il une MEILLEURE partition —
sur la **modularité** ET (repnum) l'**alignement à la taxo officielle** ?

**Réponse : NON.** Aucun transform ne bat le cosinus brut sur les deux critères.
- `cos²` / `cos³` : effet **niveau-bruit** sur la modularité juste (|ΔQ_raw| ≤ 0.0035) et
  le **signe change selon le dataset** → aucun gagnant cohérent ; alignement gold
  **légèrement pire** que brut sur les 3 indices.
- noyau **gaussien** : **nuisible** — il sur-fragmente (10→32, 13→43, 18→124 clusters),
  effondre Q_raw (−0.14 à −0.23) et dégrade l'alignement gold (ARI 0.022 vs 0.054).

**Reco prod : garder le cosinus brut.** Ne pas appliquer de transform du poids.

---

## Méthode (R&D pur, zéro modif prod)

Harness `research/knn_weight_sweep.py`. Réutilise telles quelles les briques prod
(`load_cache`, `knn_search`, `derive_defaults`, `build_knn_graph`, `run_leiden`,
`build_live_tree`) — aucun re-embed, aucun LLM, aucun fichier produit touché. Le
transform vit en **monkeypatch local** (le poids cosinus brut `w∈[seuil,1]` de chaque
arête est transformé AVANT Leiden ; le graphe — arêtes/seuil/k — est par ailleurs
identique, donc on isole l'effet du **poids** seul). `k` = défaut dérivé `derive_k(N)`
(12-13, le réglage prod ; cf. verdict k-sweep), pour isoler le poids du nombre de voisins.

**Transforms** (σ = écart-type des distances 1−cos des arêtes du graphe) :
`raw` = `w` · `cos2` = `w²` · `cos3` = `w³` · `gauss` = `exp(-((1−w)²)/(2σ²))`.

**Métriques.** Le piège : la modularité dépend des poids, donc `cos²` gonfle
*mécaniquement* la modularité mesurée sur SES propres poids (`Q_self`) sans mieux
découper. Yardstick **juste et commun** = `Q_raw` : modularité de la partition produite
par le transform, **mesurée sur les poids cosinus BRUTS** (« est-ce une meilleure coupe
de la structure de similarité originale ? »). Plus, repnum : **alignement gold** (ARI/NMI/V
vs Titre I/II/III du projet de loi) via `build_live_tree` (macros), `run_leiden`
monkeypatché pour appliquer le transform partout (racine + sous-arbres).

## Résultats chiffrés

Partition RACINE — `Q_raw` = yardstick comparable ; `Q_self` = ce que Leiden a optimisé
(non comparable, montré pour exposer le piège).

| dataset | transform | n_clusters | **Q_raw** | Q_self | ΔQ_raw vs brut |
|---|---|---:|---:|---:|---:|
| **tiktok** (N=1621) | raw | 10 | **0.6024** | 0.6024 | — |
| | cos2 | 8 | 0.6049 | 0.6092 | +0.0025 |
| | cos3 | 9 | **0.6059** | 0.6135 | +0.0035 |
| | gauss | 32 | 0.4660 | 0.8291 | **−0.1364** |
| **granddebat** (N=3000) | raw | 13 | **0.5340** | 0.5340 | — |
| | cos2 | 12 | 0.5322 | 0.5360 | −0.0018 |
| | cos3 | 11 | 0.5338 | 0.5424 | −0.0002 |
| | gauss | 43 | 0.3977 | 0.8295 | **−0.1363** |
| **xstance** (N=3000) | raw | 18 | **0.6093** | 0.6093 | — |
| | cos2 | 18 | **0.6107** | 0.6135 | +0.0014 |
| | cos3 | 18 | 0.6104 | 0.6161 | +0.0011 |
| | gauss | 124 | 0.3782 | 0.9250 | **−0.2311** |
| **republique-numerique** (N=3000) | raw | 16 | **0.6284** | 0.6284 | — |
| | cos2 | 16 | 0.6277 | 0.6419 | −0.0007 |
| | cos3 | 17 | 0.6264 | 0.6575 | −0.0020 |
| | gauss | 72 | 0.4899 | 0.9032 | **−0.1385** |

Alignement gold **repnum** (axes officiels Titre I/II/III) :

| transform | n_macros | ARI | NMI | V |
|---|---:|---:|---:|---:|
| **raw** | 16 | **0.054** | **0.168** | **0.168** |
| cos2 | 16 | 0.049 | 0.159 | 0.159 |
| cos3 | 17 | 0.046 | 0.164 | 0.164 |
| gauss | 15 | 0.022 | 0.060 | 0.060 |

(JSON brut : `/tmp/claude-1000/-home-bat-agora-worktrees-knn-weight/knn_weight_results.json`.)

## Lecture

- **`Q_self` ment.** `cos²`/`cos³`/`gauss` affichent un `Q_self` plus haut (jusqu'à 0.93)
  parce que concentrer la masse de poids sur les paires fortes gonfle la modularité
  *du graphe transformé* — pas la qualité de la coupe. À yardstick commun (`Q_raw`),
  le gain s'évapore : `cos²`/`cos³` oscillent dans ±0.0035 (bruit, signe instable),
  `gauss` s'effondre.
- **`cos²`/`cos³` : neutres au mieux.** Aucun gain robuste : positif minuscule sur
  tiktok/xstance, négatif sur granddebat/repnum, et alignement gold systématiquement
  un cran sous le brut. Pas de raison de l'adopter.
- **`gauss` : à proscrire.** Écraser les liens faibles **shatter** le graphe (×3 à ×7
  clusters) ; Leiden isole des cliques quasi-dupliquées (Q_self élevé) au prix de la
  structure réelle (Q_raw −0.14/−0.23, gold ARI ÷2.5). C'est l'anti-objectif d'Agora
  (carte de thèmes lisible).
- **Cohérent avec les verdicts voisins** ([[agora-k-sweep-verdict]], [[agora-knn-k-verdict]],
  [[agora-graph-seg-verdict]]) : la structure de communauté du graphe k-NN cosinus est déjà
  près de son plafond ; retoucher les poids (comme retoucher `k`) ne crée pas de signal,
  il en détruit. Le cosinus brut reste le bon défaut.

**Décision : statu quo (cosinus brut). Aucune action prod.**
