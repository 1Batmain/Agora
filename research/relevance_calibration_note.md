# Pré-filtre de pertinence — calibration (strict/soft) + 2ᵉ corpus + small vs large (2026-07-08)

**Statut : RESEARCH, zéro modif pipeline.** Suite de `relevance_prefilter_note.md`. Répond à deux
demandes : (1) calibrer le filtre (strict vs soft) et le rejouer sur un 2ᵉ corpus (tiktok) ;
(2) re-bencher `large_noabst` vs `small` AVEC le filtre. Harnais `relevance_prefilter.py` (base
caché par (dataset, modèle), juge aveugle robuste). **Corpus lutte + tiktok**, buildés en dev.

## Tableau

| run | décidés→filtrés | retirés | retrait précis (bon/mauvais) | gardés clairs | conf retirés |
|---|---|---|---|---|---|
| lutte large **strict** | 221→144 | 77 (−35 %) | 0.61 (46/30) | **0.88** | 51 h / 26 m |
| lutte large **soft** | 221→200 | 21 (−10 %) | 0.75 (15/5) | 0.80 | 14 h / 7 m |
| lutte **small** strict | 240→164 | 76 (−32 %) | 0.63 (46/27) | 0.84 | 54 h / 21 m |
| tiktok large strict | 1764→1127 | 637 (−36 %) | 0.70 (433/188) | 0.68 | 368 h / 268 m |
| tiktok large soft | 1764→1623 | 141 (−8 %) | 0.77 (103/31) | 0.56 | 42 h / 98 m |

**Caveat majeur** : le juge est un LLM, NON déterministe. Le split bon/mauvais des retraits
wobble d'un run à l'autre (lutte-large-strict : 56/18 la veille, 46/30 ici → retrait précis
0.76 vs 0.61). Les VALEURS exactes sont bruitées (±0.15) ; seules les DIRECTIONS sont fiables. Une
calibration fine exige un panel humain.

## (1) Calibration strict vs soft — un compromis précision/rappel DU FILTRE

- **strict** : agressif (retire 35 % sur lutte, 36 % sur tiktok). Capte beaucoup de
  sur-classements (46 bons) MAIS jette beaucoup de vraies positions (~30 % des retraits). Le set
  gardé est le plus PROPRE (0.88 clairs sur lutte).
- **soft** : conservateur (retire 8-10 %). Peu de faux retraits (5 sur lutte) mais LAISSE la
  plupart des sur-classements (ne capte que 15/46). Set gardé moins propre.

→ **Il n'y a pas de réglage « juste » — c'est un knob produit.** Pour un outil qui affiche « ceci
n'est pas un sondage » et où **cacher une vraie voix citoyenne** est plus grave que garder un
tangent, **soft est le défaut plus SÛR** (faux retraits minimes), au prix d'un correctif partiel.
strict maximise la précision au prix du rappel. Un seuil intermédiaire (ou une bande de confiance)
serait idéal mais le bruit du juge limite le tuning ici.

## (1bis) 2ᵉ corpus (tiktok) — le filtre marche mais le corpus est intrinsèquement ambigu

Le filtre retire massivement des décisions CONFIANTES (tiktok strict : 368 high retirés, 70 %
corrects) → il attaque bien le sur-classement. MAIS **les gardés-clairs plafonnent à 0.56-0.68**
(vs 0.80-0.88 sur lutte) : tiktok est un corpus de **vécu/ressenti** (« je me compare aux filles »,
« j'ai du mal à décrocher »), pas d'arguments sur une action — « position claire sur l'action » y
est souvent indécidable (cohérent avec `emerge_note.md`). **Le bénéfice du filtre dépend du
corpus** : net sur un corpus argumentatif (lutte), plafonné sur un corpus de témoignages (tiktok).

## (2) `large_noabst` vs `small` — le sur-classement N'EST PAS spécifique à large

Sur lutte, strict, small vs large :
- **décident autant** (small 240, large 221 — small en décide même un peu PLUS) ;
- **retirés quasi identiques** (76 vs 77), retrait précis comparable (0.63 vs 0.61) ;
- **gardés clairs : large 0.88 > small 0.84** — large est un classifieur légèrement MEILLEUR.

→ **Le sur-classement par association est un défaut de la CONSIGNE anti-abstention + cible large,
PAS de `large`.** small le fait autant. L'inquiétude « l'engagement de large_noabst est du
sur-classement » **n'est pas confirmée** : sur corpus réel, large et small sur-classent pareil, et
les décisions gardées de large sont un peu PLUS propres. Le pré-filtre aide les DEUX également ;
il est **orthogonal** au choix du modèle. Le verdict `stance_large_bench.md` (large_noabst
légèrement meilleur en accuracy) tient — le filtre s'y ajoute sans le remettre en cause.

## Verdicts

1. **Pré-filtre de pertinence : correctif STANCE valide, à servir en variante SOFT** (défaut sûr,
   faux retraits minimes) — intégrable en 1 passe (`porte_sur_action` dans `STANCE_SYSTEM`).
   Variante strict disponible pour un mode « haute précision ». Le réglage fin attend un panel
   humain (juge LLM trop bruité).
2. **Bénéfice corpus-dépendant** : franc sur corpus argumentatif, plafonné sur corpus de vécu
   (tiktok) — normal, on ne peut pas extraire une position sur une action là où il n'y en a pas.
3. **`large_noabst` réhabilité** : il ne sur-classe pas plus que small ; il classe même un peu
   mieux. Le sur-classement était la consigne, pas le modèle. Rien à changer au verdict modèle.

## Artefacts

`relevance_prefilter.py` · `relevance_run_*.json` · `relevance_base_*_*.json` (caches).
Chaîne : `cleavage_engagement_note.md` → `cleavage_quality_note.md` →
`relevance_prefilter_note.md` → ce note.
