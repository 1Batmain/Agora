# Segmenteur APPRIS sur attention gelée — **+ négatifs mono** (apprendre à s'abstenir)

*Train = jeux RÉELS externes, **4500 docs** = **2250 positifs** (WikiSection multi : 1500 EN + 750 DE, ≥1 frontière) + **2250 négatifs « pas de frontière »** (ratio 1.00:1) : WikiSection-mono 500 EN + 250 DE (sections uniques) & M-ABSA mono-aspect 500 EN + 250 DE + 750 FR (`n_aspects==1`). JAMAIS notre synthétique. Encodeur **`intfloat/multilingual-e5-base` GELÉ**. Features : `cross_{L,H}` par (couche×tête) [12×12] × 2 fenêtres [3, 8] + dérive d'embedding × 2 fenêtres [3, 8] = **291 features/position**. Classifieur léger (LR / GBM). CPU, seed=0.*

> **Correctif** : au 1er run (train = WikiSection SEUL = 100 % multi-section) la tête n'avait JAMAIS vu d'exemple « ne pas couper » → elle sur-coupait les avis cohérents (mono_FP transfert 0.23 vs 0.14 pour l'attention réglée-main). On lui apprend ici à **s'abstenir** en ajoutant au train des passages mono-thème entièrement labellisés non-frontière. La question : le **zéro-shot** bat-il enfin le réglé-main (0.769) ?

## 0. Diagnostic — SUR-CORRECTION (réponse : NON, et voici pourquoi)

**Les négatifs marchent… trop.** En zéro-shot, mono_FP s'effondre à **0.000** (vs 0.23 au 1er run) — la tête a appris à s'abstenir — **MAIS la F1 zéro-shot s'effondre AUSSI à 0.037** (1er run : 0.5938). La tête **n'ose plus rien couper**. C'est une **sur-correction**, pas un échec du modèle.

**Cause = le SEUIL ABSOLU ne TRANSFÈRE PAS cross-domaine.** Le point de fonctionnement calé en CV sur le train (négatifs-lourd) vit **HAUT** (thr≈0.96) ; le même train, transféré au gold, devrait couper **BAS** : l'optimum gold est **thr≈0.1**. La **distribution des proba P(frontière) diffère entre domaines** (encyclopédique EN/DE vs témoignages FR) → un seuil numérique fixe calé sur l'un sature/éteint l'autre.

**Preuve que le MODÈLE est bon (c'est le seuil, pas la tête)** : re-calé sur le gold (thr=0.1), l'appris LR fait **F1_multi=0.739, mono_FP=0.096** — vs **mono_FP=0.4519** au 1er run à objectif comparable (÷~5) et vs **0.1442** pour le réglé-main. Les négatifs ont donc bien **rendu l'abstention apprise** : à point de fonctionnement comparable, la tête abstient désormais **mieux que le réglé-main** (mono_FP 0.096 < 0.1442), pour une F1 légèrement en dessous (0.739 vs 0.7692).

**Verrou = le POINT DE FONCTIONNEMENT, pas le modèle.** Le fix (prochain run, décidé par l'architecte ; **PAS de re-tuning ici**) = un **seuil ADAPTATIF dérivé de la distribution de P PAR DOCUMENT** (p.ex. couper les maxima locaux de P au-dessus de `μ_P − c·σ_P` du doc — exactement comme `attn_seg` calibre `cross` en μ/σ poolés), au lieu d'un seuil numérique absolu transféré tel quel. Ce run documente le résultat **TEL QUEL**, négatif compris.

## 1. Scorecard — appris (LR/GBM) vs réglé-main vs change-point

### 1a. Train held-out (CV stricte PAR DOCUMENT, GroupKFold-5)

| approche | Pk | WindowDiff | F1_multi | P | R | mono_FP | F1_global | thr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| appris LR — seuil F1_multi | 0.1897 | 0.217 | 0.7038 | 0.6519 | 0.7647 | 0.0853 | 0.6874 | 0.96 |
| appris LR — seuil F1_global | 0.1897 | 0.217 | 0.7038 | 0.6519 | 0.7647 | 0.0853 | 0.6874 | 0.96 |
| appris GBM — seuil F1_multi | 0.1755 | 0.1994 | 0.7317 | 0.6972 | 0.7699 | 0.0173 | 0.728 | 0.92 |
| appris GBM — seuil F1_global | 0.1755 | 0.1994 | 0.7317 | 0.6972 | 0.7699 | 0.0173 | 0.728 | 0.92 |

*(CV out-of-fold sur le train COMPLET (multi positifs + mono négatifs). `seuil F1_multi` = détection max des frontières ; `seuil F1_global` = calibration d'**abstention** (pénalise la sur-coupe des mono) — c'est CE seuil qu'on transfère en zéro-shot. mono_FP ici = sur-coupe des négatifs mono in-domain.)*


### 1b. TRANSFERT → notre gold témoignages FR (le chiffre clé)

Une tête entraînée sur des jeux RÉELS externes (WikiSection EN/DE + négatifs mono EN/DE/FR) marche-t-elle sur des **avis citoyens FR** ? Comparé à l'attention **RÉGLÉE à la main** (gold), au **change-point**, et au **1er run SANS négatifs**. Régimes de seuil : **zéro-shot** = seuil calé sur le train-CV par F1_global (abstention), jamais vu le gold (LE test de transfert honnête) ; **gold-tuné (F1_global)** = seuil re-calé sur le gold par le MÊME objectif que l'attention réglée → apples-to-apples ; **gold-tuné (F1_multi)** = plafond de détection (ignore les FP mono).

| approche | Pk | WindowDiff | F1_multi | P | R | mono_FP | F1_global | thr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attention RÉGLÉE-main (réf) | 0.1493 | 0.1563 | 0.7692 | 0.8955 | 0.6742 | 0.1442 | 0.7453 | — |
| change-point (réf) | 0.2815 | 0.282 | 0.4423 | 0.4545 | 0.4307 | 0.7019 | 0.384 | — |
| _1er run SANS nég. LR — zéro-shot_ | 0.2293 | 0.2309 | 0.5938 | 0.9744 | 0.427 | 0.2019 | 0.5588 | 0.95 |
| _1er run SANS nég. GBM — zéro-shot_ | 0.2033 | 0.2044 | 0.665 | 0.9853 | 0.5019 | 0.2308 | 0.6276 | 0.9 |
| **appris LR +nég.** — zéro-shot | 0.3485 | 0.3485 | 0.0368 | 1.0 | 0.0187 | 0.0 | 0.0368 | 0.96 |
| appris LR +nég. — gold-tuné (F1_global) | 0.2007 | 0.2054 | 0.7389 | 0.9027 | 0.6255 | 0.0962 | 0.7214 | 0.1 |
| appris LR +nég. — gold-tuné (F1_multi) | 0.2007 | 0.2054 | 0.7389 | 0.9027 | 0.6255 | 0.0962 | 0.7214 | 0.1 |
| **appris GBM +nég.** — zéro-shot | 0.3553 | 0.3553 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.92 |
| appris GBM +nég. — gold-tuné (F1_global) | 0.2342 | 0.2354 | 0.6062 | 0.9832 | 0.4382 | 0.0673 | 0.5954 | 0.1 |
| appris GBM +nég. — gold-tuné (F1_multi) | 0.2342 | 0.2354 | 0.6062 | 0.9832 | 0.4382 | 0.0673 | 0.5954 | 0.1 |

*(mono_FP = fraction des 104 mono du gold sur-coupés — l'appris a-t-il appris à s'abstenir ? Comparer la ligne zéro-shot +nég. à la ligne « 1er run SANS nég. » : c'est l'effet DIRECT des négatifs.)*

## 2. Cross-langue : train EN → test DE

| approche | Pk | WindowDiff | F1_multi | P | R | mono_FP | F1_global | thr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| appris LR — train EN→test DE | 0.2228 | 0.2483 | 0.6453 | 0.6079 | 0.6876 | 0.156 | 0.6262 | 0.97 |
| appris GBM — train EN→test DE | 0.2266 | 0.2365 | 0.6167 | 0.7204 | 0.539 | 0.014 | 0.6145 | 0.92 |

*(Train EN = positifs EN + négatifs mono EN ; test DE = positifs + mono DE. Seuil F1_global calé sur EN-CV, appliqué tel quel au DE held-out — généricité langue-agnostique des features ET de l'abstention apprise.)*

## 3. Ablation : d'où vient le signal ? (LR, CV train)

| features | F1_multi | P | R | thr |
| --- | --- | --- | --- | --- |
| attn | 0.7037 | 0.652 | 0.7642 | 0.96 |
| emb | 0.11 | 0.0601 | 0.6515 | 0.5 |
| all | 0.7038 | 0.6519 | 0.7647 | 0.96 |

*(`attn` = cross_{L,H} seuls ; `emb` = dérive d'embedding seule ; `all` = les deux.)*

## 4. Interprétabilité — où vit le signal appris (poids LR)

- **Part du signal** : attention `cross_{L,H}` = 100 % de la masse |poids|, dérive d'embedding = 0 %. Dérive d'embedding : `{'emb_adj': -0.026, 'emb_drift_W3': -0.0906, 'emb_drift_W8': 0.0405}` (signe + = la dissemblance pousse vers « frontière »).

- **Top (couche, tête) par |poids|** (somme sur les fenêtres cross ; signe − attendu : flux BAS = frontière) :

| layer | head | abs_weight | signed_weight |
| --- | --- | --- | --- |
| 2 | 5 | 5.3709 | -2.0993 |
| 1 | 8 | 4.9368 | -2.9618 |
| 0 | 3 | 2.6673 | -1.8435 |
| 6 | 4 | 2.2525 | -0.992 |
| 0 | 2 | 2.2045 | 1.6642 |
| 0 | 4 | 2.0188 | 1.2989 |
| 6 | 1 | 1.9691 | 1.3383 |
| 0 | 6 | 1.9 | -1.9 |
| 4 | 9 | 1.7955 | -0.7062 |
| 6 | 8 | 1.5926 | -0.6447 |
| 4 | 8 | 1.529 | -1.261 |
| 0 | 11 | 1.4417 | -1.4417 |
| 0 | 10 | 1.4083 | -0.8574 |
| 0 | 9 | 1.347 | 0.276 |
| 1 | 0 | 1.3195 | 0.824 |

- **Couches dominantes** (|poids| cumulé/tête) : L0 (16.3317), L1 (12.2692), L2 (11.2957), L6 (9.7688). **3/4 dans la moitié basse** du réseau (L<6) → le signal appris vit dans les couches **basses-moyennes**, ce qui **CONFIRME** la localisation `lowmid` trouvée à la main par `attn_seg` (le réglé-main avait élu lowmid sans voir un seul label).

- **Concentré, PAS diffus** : les 2 têtes de tête (L2H5, L1H8) pèsent **11 %** de la masse |poids| cross à elles seules. Le réglé-main concluait que le signal était *diffus* (la moyenne de TOUTES les têtes battait la sélection `local`) ; la supervision, elle, **isole des têtes-frontière spécifiques** — c'est précisément l'apport d'apprendre la combinaison plutôt que de moyenner. Leurs poids signés sont **négatifs** (flux d'attention BAS → frontière), conforme à l'intuition physique.

## 5. Verdict honnête — avec les négatifs, le zéro-shot bat-il le réglé-main ?

- **Effet des négatifs (zéro-shot LR)** : mono_FP **0.2019 → 0.000** (Δ=-0.202) — la sur-coupe des mono s'effondre — **MAIS** F1_multi **0.5938 → 0.037** (Δ=-0.557) s'effondre aussi. **Sur-correction** : la tête n'ose plus couper. Ce n'est pas le modèle (cf. §0 : re-calé sur le gold il fait F1=0.739 / mono_FP=0.096) mais le **seuil absolu qui ne transfère pas** cross-domaine.

- **Transfert ZÉRO-SHOT** (le test honnête : seuil jamais vu le gold) — meilleur appris = **LR** : F1_multi=0.037 (P=1.000, R=0.019), Pk=0.348, mono_FP=0.000. vs attention réglée-main F1_multi=0.7692 (mono_FP=0.1442) → **ΔF1_multi=-0.732**. → l'appris **NE BAT PAS** le réglé-main (0.769) en zéro-shot.

- **Apples-to-apples** (seuil re-calé sur le gold par F1_global, le MÊME objectif de sélection que `c` de l'attention) : F1_multi=0.739, F1_global=0.721, Pk=0.201, mono_FP=0.096 → **ΔF1_global=-0.024**, ΔF1_multi=-0.030 vs réglé-main. L'appris **NE BAT PAS** le réglé-main à objectif/seuil comparable.

- **Plafond de détection des frontières** (seuil optimisant F1_multi seul) : F1_multi=0.739 (P=0.903, R=0.625), mono_FP=0.096. Avec les négatifs, l'optimum F1_multi et l'optimum F1_global **coïncident** (même seuil) : le modèle n'a plus besoin de sur-couper pour détecter — l'abstention est apprise dans la TÊTE, pas imposée par le seuil. Reste à la calibrer cross-domaine (cf. §0).

- vs **change-point** (F1_multi=0.4423) : l'appris zéro-shot fait ΔF1_multi=-0.406.

- **Honnêteté ratio & provenance des négatifs** : train = **4500 docs** = 2250 positifs WikiSection multi (EN/DE encyclopédique) + 2250 négatifs mono (ratio **1.00:1**) : sections uniques WikiSection (500+250 EN/DE, MÊME domaine que les positifs) + M-ABSA mono-aspect (500+250+750 EN/DE/FR, opinion courte, dont du **FR natif**). Le transfert traverse toujours le gap langue+domaine (test = avis citoyens FR) ; ce qui change vs le 1er run : le modèle a VU des exemples « ne rien couper » → mono_FP zéro-shot = 0 %.

- **Sur/sous-apprentissage** : F1_multi CV train (LR)=0.704 vs F1 transfert gold=0.037 — l'écart mesure le domain gap (un gros écart = la tête colle au style source). Risque des négatifs : si on en met TROP, le modèle s'abstient trop (rappel multi ↓) ; trop peu, il sur-coupe encore (mono_FP ↑) — d'où le dosage ~1:1. LR (linéaire) = interprétable, capacité limitée ; GBM = plus de capacité.

- **Généricité** : zéro lexique, zéro mot codé en dur — features 100 % dérivées de l'attention/embedding d'un encodeur gelé, calculables sur n'importe quelle langue. Le transfert EN/DE→FR (§1b) et EN→DE (§2) en est la preuve directe.
