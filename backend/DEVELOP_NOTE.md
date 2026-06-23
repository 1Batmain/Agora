# D1 — Représentants & citations par CENTRALITÉ × DÉVELOPPEMENT

But : ne plus surfacer la reformulation générique courte (le médoïde, plus proche du
centroïde) mais l'**argument étoffé** on-topic. Cœur : `backend/develop.py`. Appliqué à
`_representatives` (analysis.py / state.py) ET `citations_for_theme` (citations.py).

## Mesure préalable — longueur ↔ distance au centroïde (intuition de Bob confirmée)

Corrélation de Pearson, mesurée par feuille puis poolée (dist z-scorée par feuille) :

| dataset    | feuilles | corr intra-feuille (moy. pondérée) | corr poolée |
|------------|----------|------------------------------------|-------------|
| tiktok     | 236      | **+0.46**                          | +0.26       |
| granddebat | 25       | **+0.38**                          | +0.17       |

Distance médiane (z) par quartile de longueur — monotone croissante :

| quartile longueur | tiktok dist_z | granddebat dist_z |
|-------------------|---------------|-------------------|
| Q0 (≈20–23 car.)  | −0.46         | −0.50             |
| Q3 (≈140–170 car.)| **+0.42**     | **+0.58**         |

→ Les claims **courts** sont **près du centroïde** (génériques) ; les **développés**
sont **plus loin**. Le pur médoïde surface donc le générique court — d'où le re-ranking.

## Score

`score = garde-fou(centralité) × développement`, avec :

- **développement** ∈ [0,1] = `0.50·longueur(rang relatif) + 0.35·spécificité(idf moyen)
  + 0.15·raisonnement(connecteurs argumentatifs multilingues + chiffres)`. Tout dérivé
  des données (idf corpus calculé une fois au build, partagé) ; **zéro mot de domaine codé**.
- **garde-fou centralité** = gate multiplicatif : 1 dans le gros du nuage (claims on-topic,
  sim ≥ 20ᵉ centile), décroît proportionnellement pour les outliers → jamais de hors-sujet
  en tête. Les claims développés modérément distants (ce qu'on VEUT) restent à plein poids.

## Effet mesuré (auto-test :8011, build worktree)

Longueur **médiane** des représentants top-1 (médoïde brut → D1) :
tiktok **40 → 117**, granddebat **38 → 148**. Citations top-5 (centroïde-pur → D1) :
tiktok 32 → 76, granddebat 20 → 93. Garde-fou : les reps de plus faible centralité du nœud
restent à cosinus absolu élevé (0.76+) et on-topic (inspection manuelle). Vues existantes
intactes : `dist_to_centroid`/`weight` toujours renvoyés, `development` ajouté en bonus.

Harnais de mesure rejouable : `backend/measure_develop.py` (corr longueur↔distance par dataset).
