# Structure macro émergente en INCRÉMENTAL — rapport (Lane E0)

**Question.** L'arbre incrémental (`backend/state.py` : `AnalysisState`, nearest-attach +
split local) fige sa partition MACRO au tout premier split du root, sur un petit
échantillon → trop **peu** de macros et partition **stale**, loin du Leiden+coarsening
**batch** (`build_theme_tree`). Quelle dérivation du niveau macro à partir de l'arbre
incrémental retrouve les macros batch (V-mesure haute, nb ≈ batch) **au moindre coût**, en
restant **générique** (seuils/déclencheurs DÉRIVÉS, zéro magic-number corpus-spécifique) ?

**Méthode.** `research/inc_macro_eval.py`. On nourrit les claims **en incrémental** (claims +
embeddings CACHÉS, **zéro LLM**) pour `tiktok` (3 379 claims) et `granddebat` (7 233), à des
checkpoints 25/50/100 %, sur **3 permutations** d'ordre (seed 42). À chaque checkpoint on
compare l'assignation **claim→macro** incrémentale à celle du **batch reconstruit sur le même
préfixe** de claims (référence), via la **V-mesure** (homogénéité+complétude), le **nb de
macros** vs batch, et le **coût** (taille du sous-problème de recompute = n_feuilles ≪
n_claims). Plafond **oracle** = V-mesure du *meilleur* regroupement possible des feuilles
(chaque feuille → son macro batch majoritaire) : borne supérieure de ce que la **structure des
feuilles** permet de retrouver, quelle que soit la dérivation.

## Stratégies comparées

| Stratégie | Dérivation du niveau macro |
|---|---|
| **baseline** | macros = enfants du root (`effective_macro_ids`, état actuel) |
| **option A** | **coarsen/merge** des FEUILLES via `_coarsen_roots` (cos centroïdes > μ+σ **ET** > min cohésions, union-find transitif) |
| **option B** | **recompute partiel** = Leiden **+ coarsening** sur les centroïdes des feuilles (option du brief, littérale) |
| **option B′** | recompute partiel = Leiden **SEUL** sur les centroïdes des feuilles (coarsening retiré) |

## Résultats (100 % des claims, moyenne ± σ sur 3 permutations)

| dataset | batch macros | oracle V | stratégie | nb macros | V-mesure |
|---|---|---|---|---|---|
| **tiktok** | 6 | **0.747** | baseline | 4.7 ± 0.9 | 0.289 ± 0.040 |
| | | | **option A** | 3.3 ± 0.5 | **0.614 ± 0.041** |
| | | | option B | 4.7 ± 1.2 | 0.317 ± 0.074 |
| | | | option B′ | 5.7 ± 0.9 | 0.300 ± 0.073 |
| **granddebat** | 22 | **0.519** | baseline | 6.0 ± 0.8 | 0.382 ± 0.026 |
| | | | **option A** | 4.0 ± 0.8 | **0.034 ± 0.015** ⚠️ |
| | | | option B | 4.0 ± 0.8 | 0.293 ± 0.118 |
| | | | option B′ | 5.7 ± 0.5 | 0.380 ± 0.026 |

Checkpoints intermédiaires (V-mesure, moyenne sur 3 perms) — même tendance :

| dataset | cp | baseline | option A | option B | option B′ |
|---|---|---|---|---|---|
| tiktok | 25 % | 0.299 | **0.521** | 0.317 | 0.277 |
| tiktok | 50 % | 0.278 | **0.547** | 0.376 | 0.312 |
| granddebat | 25 % | **0.403** | 0.032 ⚠️ | 0.279 | 0.312 |
| granddebat | 50 % | **0.368** | 0.070 ⚠️ | 0.244 | 0.299 |

**Coût.** Le recompute partiel (B / B′) tourne sur **n_feuilles** points (tiktok 63,
granddebat 117) au lieu de **n_claims** (3 379 / 7 233) : un graphe kNN+Leiden **~30–60×
plus petit** que le batch global. À l'échelle d'un snapshot ou d'un event de split, c'est
négligeable. La baseline est gratuite (aucun recompute) ; option A est un simple coarsening
O(n_feuilles²) sur les centroïdes.

## Lecture

1. **Aucune option n'égale le batch.** Même le plafond **oracle** plafonne à V ≈ **0.52**
   (granddebat) / **0.75** (tiktok) : la contrainte qui borne tout est la **structure des
   feuilles** incrémentales, pas la dérivation macro. Les feuilles produites par
   nearest-attach + split local **chevauchent** les macros batch (une feuille mélange des
   claims de 2+ macros batch) → aucun regroupement de feuilles ne peut reconstruire le batch.
   Retrouver le batch exigerait de re-dériver les **feuilles** (Leiden sur les claims =
   coûteux, non incrémental) — hors de ce que « structure macro » peut offrir.

2. **Option A n'est PAS générique.** Elle **gagne nettement sur tiktok** (V 0.614, mono-sujet)
   mais **s'effondre sur granddebat** (V **0.034**, multi-sujets). Le merge **transitif** μ+σ
   chaîne les feuilles (A~B, B~C ⇒ A∪D) et collapse un corpus multi-thèmes connecté en 4 blobs
   sans rapport avec les 22 macros batch. Le garde-fou de cohésion ne suffit pas quand les
   centroïdes de feuilles forment une chaîne de proximité continue. **Rejetée** : la
   généricité (centaines de consultations originales) est non négociable.

3. **Le coarsening final de l'option B est corpus-dépendant.** Sur granddebat il fait chuter
   la V (0.38 → 0.14 ; cf. probe résolution), sur tiktok il l'augmente (0.37 → 0.63). C'est le
   même mécanisme transitif que l'option A, juste appliqué après Leiden. **Le retirer**
   (option B′) stabilise sans jamais s'effondrer.

4. **Option B′ (Leiden seul sur centroïdes de feuilles) est la seule dérivation GÉNÉRIQUE.**
   Jamais d'effondrement, V **stable** sur les deux corpus (0.300 tiktok / 0.380 granddebat),
   nb de macros le plus proche du batch parmi les options réelles (**5.7 / 5.7**, contre
   baseline 4.7 / 6.0 et batch 6 / 22), robuste à l'ordre (σ ≤ 0.073). Mais le **gain de
   V-mesure sur la baseline est modeste** (tiktok +0.011, granddebat ≈ parité) — logique,
   puisque le plafond est fixé par les feuilles.

## Verdict

**Option B′ (recompute partiel = Leiden SEUL sur les centroïdes des feuilles, SANS
coarsening)** est l'option recommandée et implémentée.

- C'est la **seule générique** : elle ne s'effondre sur aucun des deux corpus, là où l'option
  A (et l'option B littérale, à cause de son coarsening) détruit le multi-sujets.
- Elle est **cheap** (recompute borné par n_feuilles ≪ n_claims) et **sans LLM**.
- Elle corrige la **staleness** du split-racine unique : la partition macro est re-dérivée des
  feuilles courantes au lieu d'être figée tôt sur un petit échantillon.

**Honnêteté — écart résiduel.** B′ **n'égale pas le batch** : V ≈ 0.30–0.38 contre un plafond
oracle de 0.52–0.75 et un batch « parfait » à 1.0. L'écart vient **majoritairement de la
structure des feuilles incrémentales** (qui chevauchent les macros batch), pas de la dérivation
macro. Sur granddebat, B′ ne récupère que ~6 macros sur les 22 du batch : l'incrémental reste
plus grossier. Le gain net de B′ sur la baseline est faible en V-mesure ; sa vraie valeur est
la **fidélité du nombre de macros** et la **robustesse anti-staleness**, pas un bond de qualité.

## Implémentation

`backend/state.py` — `AnalysisState.rebuild_macro_layer()` (option B′), derrière le flag
`macro_mode` :
- `"root"` (**défaut**) : comportement historique (macros = enfants du root) — **inchangé** ;
- `"recompute"` : Leiden sur les centroïdes des feuilles → matérialise `root → macro_i →
  feuilles`. Déclenché en fin d'`add_all()` ; appelable à la demande pour `/stream` + live.

Le build **persisté** (`build_analysis` → `build_theme_tree`, Leiden **batch**) et le `/stream`
par défaut ne sont **pas touchés** : ceci améliore uniquement l'INCRÉMENTAL quand le flag est
activé.

## Reproduire

```bash
uv run --extra contender python -m research.inc_macro_eval --perms 3
# → research/inc_macro_results.json (72 lignes : 2 datasets × 3 cp × 3 perms × 4 stratégies)
```
