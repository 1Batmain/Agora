# CLEANUP généricité — audit #4, #14, #15

Trois derniers items de hardcoding langue/domaine retirés. Tout est désormais
dérivé des données ou généré, conforme à la directive « zéro hardcoding ».

## #4 🔴 — `eval/coherence.py` langue-agnostique (banc qualité)
Ce module mesure la cohérence NPMI des thèmes et **sert à choisir le modèle de
prod**. Une métrique faussée hors DE/FR/IT invalide ce choix.

| | Avant | Après |
|---|---|---|
| Tokenizer | `[a-zàâä…]+` (latin only) | `re.compile(r"[^\W\d_]+", re.UNICODE)` — toute écriture (cyrillique, grec, CJK…) |
| Stopwords | set DE/FR/IT figé (~200 mots) | `derive_corpus_stopwords` (réutilise `pipeline.cluster.naming`) par sous-corpus de langue |
| API | `per_language_coherence(...)` | inchangée — `eval/quality_bench.py` importe toujours |

Conséquence : sur une consultation non-latine, `findall` renvoie des tokens non
vides et la cohérence ne dégénère plus (test cyrillique : `overall ≈ 0.58`).

## #15 🟢 — `pipeline/cluster/palette.py` générative
| | Avant | Après |
|---|---|---|
| Couleurs | `PALETTE` figée de 20 + `id % 20` → collisions ≥ 20 communautés | `palette(n)` : N teintes HSV équiréparties (S/V fixes, lisibles fond sombre) |
| API | `color_for(id)` | `color_for(id, n)` (rétro-compatible ; repli nombre d'or sans `n`) |

`build.py` passe `n_clusters` / `len(macro_members)` → zéro collision quel que
soit le nombre de thèmes. Vérifié : 50 couleurs distinctes / 50.

## #14 🟢 — `frontend/index.html`
`<html lang="fr">` → `<html>` (neutre) : la langue n'est plus présumée française.
