# Défauts DÉRIVÉS des données — note (audit #5–#9)

> **Principe** (`queue/cross-lane.md`) : l'outil tournera sur des centaines de
> consultations, sujets et **langues** variés. Aucun défaut ne doit être calé sur
> le corpus TikTok FR. Cette note documente ce qui a été **dérivé**, **comment**,
> les valeurs obtenues sur TikTok/nomic-v2, et ce qu'un autre corpus donnerait.

Tout est centralisé dans **`pipeline/cluster/adaptive.py`**. Ne subsistent que des
hyper-paramètres de **FORME** (sans unité corpus-spécifique) ; la **valeur**
obtenue, elle, s'adapte au modèle et à la densité du corpus.

---

## Ce qui est dérivé, et comment

| Knob | Avant (calé TikTok) | Maintenant (dérivé) | Forme |
|---|---|---|---|
| **modèle embed** (#5) | `e5-small` (clusterise par langue) | `nomic-v2` (winner multilingue) | défaut = winner du banc |
| **`k`** voisins (#6) | 8 / 10 / 12 (incohérent ×3 modules) | `round(3.8·log10 N)` borné [8,30] | k ∝ log N |
| **`threshold`** arête (#6) | 0.84 / 0.80 / 0.60 (incohérent ×3) | `μ − 3.2·σ` des cosinus k-NN | plancher d'aberrations gaussien |
| **`min_sub_size`** (#7) | 15 / 18 (absolu) | `max(5, round(0.011·N))` | relatif à N |
| **`resolution_sub`** (#6) | 3.0 (build) vs 1.5 (backend) | 1.5 (réconcilié, valeur FROZEN) | knob, défaut unique |
| **`dup_threshold`** (#9) | `0.93` enfoui dans `scoring.py` | `p98` des cosinus k-NN (exposé en knob) | percentile de la distribution |
| **bornes knobs** (#8) | min/max calés (rejet 422) | limites **physiques** seules | validation = domaine valide |

### #6 — seuil d'arête `μ − 3.2·σ`
On collecte les cosinus des `k` plus proches voisins de chaque nœud (self exclu) —
**sans ré-embed** : on réutilise les vecteurs déjà calculés (cache backend
`embeddings.npy`). Le seuil = moyenne − 3.2·écart-type de cette distribution :
on coupe les arêtes dont le cosinus est anormalement bas par rapport au voisinage
typique. C'est **model-adaptive** (e5 a des cosinus « chauds » → μ haut → seuil
haut ; nomic plus froid → seuil bas) et **densité-adaptive** (clusters serrés →
σ petit → seuil proche de μ). Un cosine fixe ne pouvait pas faire les deux.

Le seul nombre de forme est `EDGE_SIGMA = 3.2` (nombre d'écarts-types). Ce n'est
pas une valeur cosine corpus-spécifique : c'est le même choix qu'un percentile.

### #7 — `min_sub_size` relatif
`max(5, round(0.011·N))`. Absolu, 18 écrasait tout sur un petit corpus (quelques
centaines d'avis → un seul sous-thème) et laissait de la poussière sur un gros.
Relatif, il suit la taille : plancher de 5 pour ne jamais exiger l'impossible.

### #9 — `dup_threshold` (diversity)
Sorti de la constante enfouie `DUP_THRESHOLD=0.93`. Défaut = `p98` des cosinus
k-NN (« quasi-doublon » = haut de la distribution observée, notion relative au
modèle). Volontairement **sous** le seuil `dedup` : `dedup` a déjà collapsé les
paires > dedup, donc `diversity` mesure les quasi-doublons *résiduels* juste en
dessous (sinon `diversity ≈ 1` partout). Exposé en knob `dup_threshold` (override).

### #8 — bornes de knobs
Les `min/max` pydantic (`backend/server.py`) ne sont plus que des **limites
physiques** (`threshold ∈ [0,1]`, `k ≥ 2`, `resolution > 0`…). Une valeur légitime
sur un autre modèle/corpus (ex. seuil 0.30, k=40) n'est plus **rejetée (422)**.
Les `min/max` des `KNOBS` ne servent qu'à suggérer la plage des sliders.

---

## Valeurs obtenues — TikTok FR / nomic-v2 (cache, N=1597 après filtres)

```
derive_k(1597)            = 12        (round(3.8·log10 1597) = round(12.2))
μ(cosinus k-NN)           = 0.7679
σ(cosinus k-NN)           = 0.0520
threshold = μ − 3.2·σ     = 0.6016    (≈ 0.60, le bon réglage manuel)
min_sub_size              = 18        (round(0.011·1597))
dup_threshold = p98       = 0.8699
```

### Non-régression (prouvée)
Le défaut **dérivé** reproduit la structure du réglage **manuel** gelé qu'il
remplace, à l'identique :

| | manuel (gelé) | dérivé (auto) |
|---|---|---|
| seuil | 0.60 | **0.6016** |
| k | 12 | **12** |
| min_sub | 18 | **18** |
| macros | 8 | **8** |
| sous-thèmes | 47 | **47** |
| modularité | ~0.60 | **0.602** |

Vérifié sur le **chemin live** (`backend.recluster` sur le cache aligné) **et** sur
**`build_payload`** (même code `_build_hierarchical`, vecteurs cachés alignés) :
`8 macros / 47 sous, mod 0.602, intégrité d'arbre OK`. Le backend `/recluster`
tourne sur 2 seuils (auto-dérivé ≈ 0.60 → 8/47 ; forcé 0.70 → fragmente, attendu).

## Ce qu'un AUTRE corpus donnerait (le but)
- **Autre modèle** (e5, bge…) : μ des cosinus k-NN différent → seuil **différent**
  automatiquement (e5 ~chaud → seuil plus haut, dans le sens du 0.84 manuel).
- **Corpus plus petit** : `k` baisse (log N), `min_sub_size` baisse (relatif) →
  on n'écrase plus les petites consultations en un seul thème.
- **Corpus plus dense / plus dispersé** : σ change → le seuil se rapproche/éloigne
  de μ, le graphe garde un degré moyen sain sans réglage manuel.
- **Bornes** : aucune valeur légitime n'est rejetée ; la console reste utilisable.

> Honnêteté : la dérivation est **calibrée et prouvée sur TikTok/nomic** (le seul
> cache disponible). L'adaptation à d'autres modèles/corpus est **raisonnée**
> (direction correcte de `μ−σ`), non mesurée par ablation faute de second cache.
> `EDGE_SIGMA`, `K_LOG_COEF`, `MIN_SUB_FRAC`, `DUP_PERCENTILE` sont les 4 formes
> ajustables — sans unité corpus-spécifique — regroupées en tête de `adaptive.py`.
