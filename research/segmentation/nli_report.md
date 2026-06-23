# Segmentation par CROSS-ENCODEUR « même sujet ? » (NLI) — bat-elle l'attention ?

*Jeu : `gold_large.json` — N=305 (104 mono, 201 multi). Balayage : `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli`. CPU, seed=0.*

## 0. Faisabilité & coût (un forward NLI par jointure)

- **minilm** (`MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli`) : **OUI.** `AutoModelForSequenceClassification` 3 classes ['entailment', 'neutral', 'contradiction'], indices dérivés `{'entail': 0, 'neutral': 1, 'contra': 2}`. ~**3.2 ms/paire** CPU (indicatif, sensible à la charge). Coût plein-`gold_large` (3 formulations gratuites = 2 forwards/jointure × 3 W = 60852 paires) ≈ **0.05 h**. → **balayé en entier** (ce rapport).
- **mdeberta** (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`) : **OUI.** `AutoModelForSequenceClassification` 3 classes ['entailment', 'neutral', 'contradiction'], indices dérivés `{'entail': 0, 'neutral': 1, 'contra': 2}`. ~**187.4 ms/paire** CPU (indicatif, sensible à la charge). Coût plein-`gold_large` (3 formulations gratuites = 2 forwards/jointure × 3 W = 60852 paires) ≈ **3.17 h**. → **non balayé en entier ici** (≈ 59× plus lent que `minilm`), confirmation sur sous-échantillon (§4).

## 1. Méthode — signal de frontière par jugement NLI

- **Unité = mot** (suite de non-espaces, identique au banc embeddings/attention). À chaque jointure candidate p, **bloc gauche** = W mots avant p, **bloc droit** = W mots après p (chaînes de texte).
- **Cross-encodeur NLI** : `P(entail | gauche, droite)` = « le bloc droit est-il une *suite* du gauche ? » = score **« même sujet »**. On calcule les **deux sens** (gauche→droite et droite→gauche) → 3 formulations, toutes BAS = frontière :
  - `entail_lr` : `P(entail)` gauche→droite seul (directionnel) ;
  - `entail_sym` : moyenne des deux sens (symétrique) ;
  - `entail_minus_neutral` : marge `P(entail) − P(neutral)`, symétrisée.
- **Pourquoi entailment et pas contradiction** : deux thèmes *distincts* (sommeil vs harcèlement) donnent **« neutral »** (≈0.99), pas « contradiction ». Le signal de rupture est donc `1 − P(entail)`, jamais la contradiction (≈0 partout ici).
- **Frontières** = minima locaux du signal sous `μ − c·σ`, μ/σ **poolés GLOBALEMENT** sur tous les avis (un seuil par-avis ne peut jamais s'abstenir sur un mono cohérent). `min_seg=3` mots, zéro magic-number absolu.
- **Balayage** : formulation × W∈[4, 8, 12] × seuil c∈[0.5, 1.0, 1.5, 2.0].

## 2. NLI vs attention (0.769) vs change-point

| approche | config | Pk | WindowDiff | F1_multi | P | R | mono_FP | F1_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| change-point (embeddings) | W=8 pen=3.0 | 0.2815 | 0.282 | 0.4423 | 0.4545 | 0.4307 | 0.7019 | 0.384 |
| **attention** (e5-base) | lowmid/mean W=8 c=1.0 | 0.1493 | 0.1563 | 0.7692 | 0.8955 | 0.6742 | 0.1442 | 0.7453 |
| **NLI** (minilm) | entail_sym W=12 c=0.5 | 0.5449 | 0.7001 | 0.2528 | 0.1604 | 0.5955 | 0.9519 | 0.2014 |

*(Pk/WindowDiff ↓ = mieux, sur multi ; F1_multi = frontières tol ±1 ; mono_FP = fraction de mono sur-coupés ; F1_global = mono+multi, objectif.)*

## 3. Top 12 configurations NLI

| model | formulation | W | c | Pk | WindowDiff | F1_multi | P | R | mono_FP | mono_cuts | F1_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minilm | entail_sym | 12 | 0.5 | 0.5449 | 0.7001 | 0.2528 | 0.1604 | 0.5955 | 0.9519 | 3.087 | 0.2014 |
| minilm | entail_sym | 8 | 0.5 | 0.5761 | 0.7289 | 0.2444 | 0.154 | 0.5918 | 0.9904 | 3.356 | 0.1924 |
| minilm | entail_minus_neutral | 8 | 1.0 | 0.4794 | 0.5397 | 0.2203 | 0.1663 | 0.3258 | 0.7885 | 1.76 | 0.1788 |
| minilm | entail_minus_neutral | 8 | 0.5 | 0.5311 | 0.6333 | 0.2207 | 0.1484 | 0.4307 | 0.9231 | 2.654 | 0.1745 |
| minilm | entail_minus_neutral | 8 | 1.5 | 0.4154 | 0.4375 | 0.2004 | 0.2063 | 0.1948 | 0.5 | 0.923 | 0.1691 |
| minilm | entail_minus_neutral | 12 | 0.5 | 0.5165 | 0.6186 | 0.2086 | 0.141 | 0.4007 | 0.8173 | 2.385 | 0.168 |
| minilm | entail_minus_neutral | 12 | 1.0 | 0.461 | 0.5267 | 0.203 | 0.1525 | 0.3034 | 0.7019 | 1.75 | 0.1653 |
| minilm | entail_sym | 4 | 0.5 | 0.5834 | 0.7173 | 0.2152 | 0.1356 | 0.5206 | 0.9904 | 3.75 | 0.1653 |
| minilm | entail_sym | 4 | 1.0 | 0.4993 | 0.5609 | 0.1995 | 0.1438 | 0.3258 | 0.9423 | 2.221 | 0.1578 |
| minilm | entail_minus_neutral | 4 | 0.5 | 0.5656 | 0.672 | 0.2027 | 0.1309 | 0.4494 | 1.0 | 3.288 | 0.1573 |
| minilm | entail_lr | 8 | 0.5 | 0.5755 | 0.6935 | 0.1985 | 0.1274 | 0.4494 | 0.9904 | 3.154 | 0.1561 |
| minilm | entail_minus_neutral | 12 | 1.5 | 0.3927 | 0.4188 | 0.1774 | 0.1921 | 0.1648 | 0.4519 | 0.74 | 0.1536 |

### Meilleure config par formulation

| model | formulation | W | c | Pk | WindowDiff | F1_multi | P | R | mono_FP | mono_cuts | F1_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minilm | entail_lr | 8 | 0.5 | 0.5755 | 0.6935 | 0.1985 | 0.1274 | 0.4494 | 0.9904 | 3.154 | 0.1561 |
| minilm | entail_sym | 12 | 0.5 | 0.5449 | 0.7001 | 0.2528 | 0.1604 | 0.5955 | 0.9519 | 3.087 | 0.2014 |
| minilm | entail_minus_neutral | 8 | 1.0 | 0.4794 | 0.5397 | 0.2203 | 0.1663 | 0.3258 | 0.7885 | 1.76 | 0.1788 |

## 4. Confirmation `mdeberta` (sous-échantillon N=24)

*Le modèle lourd est ~59× plus lent ⇒ plein-gold infaisable. On le compare à `minilm` sur le MÊME sous-échantillon (12 mono, 12 multi) pour voir si un cross-encodeur plus fort changerait le verdict.*

| model | formulation | W | c | Pk | WindowDiff | F1_multi | P | R | mono_FP | mono_cuts | F1_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minilm | entail_sym | 12 | 0.5 | 0.5694 | 0.6841 | 0.209 | 0.1321 | 0.5 | 0.9167 | 3.75 | 0.125 |
| mdeberta | entail_sym | 12 | 0.5 | 0.3398 | 0.3398 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |

- Sur ce sous-échantillon : `minilm` F1_multi=0.209 / F1_global=0.125 ; `mdeberta` F1_multi=0.000 / F1_global=0.000 (à ce seuil global, `mdeberta` **s'abstient totalement** — R=0, aucune coupe : ses probas d'entailment sont trop uniformes pour faire saillir une frontière). **Le modèle lourd N'AMÉLIORE PAS nettement** le score → le plafond ne vient pas de la taille du cross-encodeur, mais de l'inadéquation du signal NLI à la cohésion thématique sur blocs courts.

## 5. Verdict honnête

**Meilleure config NLI : `minilm` · entail_sym · W=12 · c=0.5** → F1_multi=0.253 (P=0.160, R=0.596), Pk=0.545, WindowDiff=0.700, F1_global=0.201, mono_FP=0.952.

- **Le NLI bat-il l'attention (F1_multi 0.7692, Pk 0.1493, F1_global 0.7453) ? NON.** ΔF1_multi=-0.516, ΔPk=+0.396 (négatif = mieux), ΔF1_global=-0.544.

- **vs change-point** (F1_multi 0.4423, F1_global 0.384) : ΔF1_multi=-0.190, ΔF1_global=-0.183.

- **Coût** : un (en fait deux) forward(s) de cross-encodeur PAR JOINTURE — bien plus cher que l'attention (un seul forward d'encodeur par avis donne TOUT le signal) ou le change-point (embeddings + PELT). `minilm` rend le balayage faisable ; `mdeberta` (precision ↑) coûte ~59× plus → confirmé sur sous-échantillon, pas balayé en entier sur CPU.

- **Honnêteté NLI** : un modèle MNLI/XNLI juge l'*entailment logique*, pas le *même-sujet* directement ; on détourne `P(entail)` comme proxy de cohésion. Sur des blocs de quelques mots (peu de contexte), le jugement est bruité — d'où le plafond. Le jeu (multi = concaténation de mono-thèmes) est une borne OPTIMISTE.
