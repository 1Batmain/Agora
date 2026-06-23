# Décodage ADAPTATIF par document — le point de fonctionnement qui transfère

*La tête apprise classe bien les frontières mais un **seuil absolu** sur `P(frontière|p)` ne transfère pas cross-domaine (cf. `learned_report.md` §0). Ici on remplace le seuil fixe par un **point de fonctionnement adaptatif PAR DOCUMENT**, toutes params réglées sur la **SOURCE uniquement** (WikiSection EN/DE + négatifs mono EN/DE/FR) = vrai zéro-shot, évalué sur notre **gold témoignages FR** (201 multi + 104 mono). Encodeur `intfloat/multilingual-e5-base` GELÉ. Source = 3050 docs (1500 multi positifs + 1550 mono négatifs). Seed=0, CPU. Tailles réduites vs `learned_seg` (la variable testée est le **décodage**, pas la taille de train).*

**Variantes de décodage** (toutes réglées source, objectif F1_global = détecter sans sur-couper les mono) :
- `fixed` : seuil absolu sur P (la baseline qui ne transfère pas — pour contraste).
- `calib+fixed` : isotonic (source) → P calibrée, puis seuil absolu (V4).
- `rel` : coupe aux **maxima locaux de P** au-dessus de `μ_P + k·σ_P` du DOC (V1, k réglé source).
- `rel+floor` : `rel` ET `P > floor` (floor = percentile de P source) (V2).
- `rel+gate:STAT` : `rel` + **abstention** si la distribution de P du doc est plate (stat de pic < τ) ; STAT ∈ {σ_P, max P, (max−μ)/σ, kurtosis}, τ réglé source (V3).
- `rel+gate:best+floor` : meilleur gate (source) + plancher (V2+V3).

## Références (gold)

| approche | F1_multi | Pk | mono_FP | F1_global |
| --- | --- | --- | --- | --- |
| attention RÉGLÉE-main (réf 0.769) | 0.7692 | 0.1493 | 0.1442 | 0.7453 |
| change-point (réf 0.44) | 0.4423 | 0.2815 | 0.7019 | 0.384 |

## Grille zéro-shot — {LR,GBM} × {avec,sans négatifs} × décodage

*`src gf1` / `src mFP` = F1_global / mono_FP sur la SOURCE (où le décodeur est réglé). Colonnes gold = **transfert zéro-shot** (params jamais vues le gold). Objectif : `F1_multi` HAUT (> 0.769) ET `mono_FP` BAS (≈/< 0.14).*

| config | décodage | src gf1 | src mFP | gold F1_multi | gold Pk | gold mono_FP | gold P | gold R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR +nég | fixed | 0.683 | 0.092 | 0.044 | 0.347 | 0.0 | 1.0 | 0.022 |
| LR +nég | calib+fixed | 0.683 | 0.085 | 0.044 | 0.347 | 0.0 | 1.0 | 0.022 |
| LR +nég | rel | 0.496 | 0.855 | 0.8633 | 0.137 | 1.0 | 0.902 | 0.828 |
| LR +nég | rel+floor | 0.652 | 0.136 | 0.0652 | 0.344 | 0.01 | 1.0 | 0.034 |
| LR +nég | rel+gate:sigma | 0.556 | 0.091 | 0.0368 | 0.349 | 0.0 | 1.0 | 0.019 |
| LR +nég | rel+gate:maxp | 0.566 | 0.078 | 0.0295 | 0.35 | 0.0 | 1.0 | 0.015 |
| LR +nég | rel+gate:peak_z | 0.502 | 0.77 | 0.8633 | 0.137 | 1.0 | 0.902 | 0.828 |
| LR +nég | rel+gate:kurt | 0.516 | 0.47 | 0.8571 | 0.142 | 0.529 | 0.911 | 0.809 |
| LR +nég | rel+gate:maxp+floor | 0.651 | 0.077 | 0.0295 | 0.35 | 0.0 | 1.0 | 0.015 |
| LR sans nég | fixed | 0.638 | 0.335 | 0.5585 | 0.24 | 0.173 | 0.963 | 0.393 |
| LR sans nég | calib+fixed | 0.635 | 0.335 | 0.5464 | 0.244 | 0.173 | 0.936 | 0.386 |
| LR sans nég | rel | 0.497 | 0.801 | 0.8889 | 0.123 | 0.981 | 0.879 | 0.899 |
| LR sans nég | rel+floor | 0.638 | 0.308 | 0.6276 | 0.223 | 0.202 | 0.984 | 0.461 |
| LR sans nég | rel+gate:sigma | 0.535 | 0.359 | 0.6541 | 0.214 | 0.26 | 0.88 | 0.521 |
| LR sans nég | rel+gate:maxp | 0.545 | 0.231 | 0.4831 | 0.263 | 0.125 | 0.966 | 0.322 |
| LR sans nég | rel+gate:peak_z | 0.501 | 0.754 | 0.8848 | 0.125 | 0.981 | 0.878 | 0.891 |
| LR sans nég | rel+gate:kurt | 0.507 | 0.647 | 0.8868 | 0.124 | 0.962 | 0.879 | 0.895 |
| LR sans nég | rel+gate:maxp+floor | 0.635 | 0.231 | 0.4553 | 0.269 | 0.125 | 0.988 | 0.296 |
| GBM +nég | fixed | 0.718 | 0.023 | 0.0 | 0.355 | 0.0 | 1.0 | 0.0 |
| GBM +nég | calib+fixed | 0.719 | 0.023 | 0.0 | 0.355 | 0.0 | 1.0 | 0.0 |
| GBM +nég | rel | 0.533 | 0.763 | 0.888 | 0.122 | 1.0 | 0.916 | 0.861 |
| GBM +nég | rel+floor | 0.683 | 0.055 | 0.0 | 0.355 | 0.0 | 1.0 | 0.0 |
| GBM +nég | rel+gate:sigma | 0.609 | 0.048 | 0.0 | 0.355 | 0.0 | 1.0 | 0.0 |
| GBM +nég | rel+gate:maxp | 0.611 | 0.043 | 0.0 | 0.355 | 0.0 | 1.0 | 0.0 |
| GBM +nég | rel+gate:peak_z | 0.54 | 0.628 | 0.888 | 0.122 | 0.962 | 0.916 | 0.861 |
| GBM +nég | rel+gate:kurt | 0.548 | 0.597 | 0.888 | 0.122 | 0.952 | 0.916 | 0.861 |
| GBM +nég | rel+gate:maxp+floor | 0.684 | 0.043 | 0.0 | 0.355 | 0.0 | 1.0 | 0.0 |
| GBM sans nég | fixed | 0.699 | 0.161 | 0.5084 | 0.247 | 0.135 | 1.0 | 0.341 |
| GBM sans nég | calib+fixed | 0.7 | 0.141 | 0.4699 | 0.257 | 0.106 | 1.0 | 0.307 |
| GBM sans nég | rel | 0.539 | 0.792 | 0.9283 | 0.104 | 1.0 | 0.935 | 0.921 |
| GBM sans nég | rel+floor | 0.661 | 0.288 | 0.7884 | 0.16 | 0.346 | 0.973 | 0.663 |
| GBM sans nég | rel+gate:sigma | 0.58 | 0.28 | 0.8162 | 0.152 | 0.394 | 0.95 | 0.715 |
| GBM sans nég | rel+gate:maxp | 0.609 | 0.115 | 0.4223 | 0.272 | 0.067 | 0.973 | 0.27 |
| GBM sans nég | rel+gate:peak_z | 0.554 | 0.499 | 0.8112 | 0.169 | 0.74 | 0.95 | 0.708 |
| GBM sans nég | rel+gate:kurt | 0.554 | 0.513 | 0.8753 | 0.135 | 0.721 | 0.964 | 0.801 |
| GBM sans nég | rel+gate:maxp+floor | 0.672 | 0.115 | 0.394 | 0.276 | 0.067 | 0.971 | 0.247 |

## Verdict — une variante adaptative bat-elle le réglé-main 0.769 en zéro-shot ?

*Le point de fonctionnement utile doit **détecter ET s'abstenir** : on retient la variante de **F1_multi max SOUS contrainte mono_FP ≤ 0.174** (réglé-main 0.1442 + marge). Un gros F1 avec mono_FP élevé = sur-coupe, pas une victoire.*

- **Plafond de détection** (mono ignoré) : `rel` sur GBM sans nég → F1_multi=0.928 mais mono_FP=1.000 → **sur-coupe tout** : les variantes purement relatives (`rel`, `peak_z`, `kurt`) montent à ~0.9 de F1 en coupant CHAQUE avis, mono compris. Inutilisable.

- **Meilleur zéro-shot UTILE** (mono_FP ≤ 0.174) : `fixed` sur **LR sans nég** → F1_multi=**0.5585** (P=0.963, R=0.393), Pk=0.240, **mono_FP=0.173** — c'est le **seuil fixe, PAS une variante adaptative**, qui gagne sous contrainte.

- vs **réglé-main** (F1_multi=0.7692, mono_FP=0.1442) : ΔF1_multi=**-0.2107**, Δmono_FP=+0.029.

- **Apport propre du décodage adaptatif** : meilleure variante de la famille `rel` sous contrainte = `rel+gate:maxp` (LR sans nég) → F1_multi=0.483, mono_FP=0.125. vs son seuil fixe (même tête) F1_multi=0.559 → Δ=-0.075. Le décodage par-document **n’améliore pas** le seuil fixe à ce point de fonctionnement.

- **❌ **NON** — sous contrainte d'abstention, le meilleur zéro-shot (F1_multi=0.559) reste loin du réglé-main (0.769). Détecter ET s'abstenir en zéro-shot n'est pas atteint.**

- **Diagnostic (oracle) — c'est le RANKING, pas seulement le point de coupe** : même réglé DIRECTEMENT sur le gold (triche) sous la même contrainte, le meilleur plafonne à F1_multi=**0.729** (mono_FP=0.087, `fixed`, GBM +nég) — **encore sous 0.769**. Donc le décodage adaptatif ne laisse pas d'argent sur la table : à mono_FP comparable au réglé-main, le **score de la tête apprise lui-même plafonne** sous l'attention réglée-main. Le verrou n'est pas (que) le transfert du seuil — c'est la séparabilité multi/mono dans le score appris, que le par-document ne crée pas s'il n'y est pas.

- vs **change-point** (F1_multi=0.4423) : le meilleur zéro-shot utile (0.559) fait Δ=+0.116.

## Plafond (oracle) — décodage réglé sur le gold (triche, pour le headroom)

*Mêmes variantes mais réglées DIRECTEMENT sur le gold (meilleure SOUS la même contrainte mono_FP ≤ 0.174 ; « cap non tenu » = aucune ne tient, on montre la moins sur-coupante). Si l'oracle dépasse 0.769 mais pas le zéro-shot → c'est le **réglage** qui ne transfère pas ; sinon → le **ranking** lui-même plafonne.*

| config | meilleur décodage (oracle) | F1_multi | Pk | mono_FP | P | R |
| --- | --- | --- | --- | --- | --- | --- |
| LR +nég | fixed | 0.7248 | 0.205 | 0.115 | 0.9 | 0.607 |
| LR sans nég | fixed (cap non tenu) | 0.8376 | 0.144 | 0.51 | 0.825 | 0.85 |
| GBM +nég | fixed | 0.7294 | 0.197 | 0.087 | 0.941 | 0.596 |
| GBM sans nég | calib+fixed (cap non tenu) | 0.8863 | 0.12 | 0.51 | 0.93 | 0.846 |

## Honnêteté & généricité

- **Discipline zéro-shot** : k, τ (gate), floor, map isotonic — TOUT réglé sur la source (WikiSection-CV OOF + mono), jamais sur le gold. Le seul chiffre qui compte est la colonne `gold` de la grille. L'oracle est explicitement étiqueté triche (headroom).

- **Pourquoi le gate est crucial** : un seuil purement relatif (`rel`) coupe toujours le maximum local le moins pire → il NE PEUT PAS s'abstenir sur un mono cohérent (voir sa colonne `gold mono_FP`). Le gate de platitude/pic est ce qui rend l'abstention possible SANS seuil absolu transféré.

- **Généricité** : zéro lexique, zéro constante magique non dérivée — μ_P/σ_P par doc, τ/floor = percentiles de la source, k sans dimension. Calculable sur n'importe quelle consultation, n'importe quelle langue (transfert EN/DE→FR).
