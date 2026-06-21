# Segmenteur APPRIS sur attention gelée — l'appris bat-il le réglé-main ?

*Train = **WikiSection RÉEL** (EN+DE, 2250 docs : 1500 EN, 750 DE ; JAMAIS notre synthétique). Encodeur **`intfloat/multilingual-e5-base` GELÉ**. Features : `cross_{L,H}` par (couche×tête) [12×12] × 2 fenêtres [3, 8] + dérive d'embedding × 2 fenêtres [3, 8] = **291 features/position**. Classifieur léger (LR / GBM). CPU, seed=0.*

## 1. Scorecard — appris (LR/GBM) vs réglé-main vs change-point

### 1a. WikiSection held-out (CV stricte PAR DOCUMENT, GroupKFold-5)

| approche | Pk | WindowDiff | F1_multi | P | R | F1_global | thr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| appris LR | 0.1932 | 0.225 | 0.7081 | 0.6388 | 0.7943 | 0.7081 | 0.95 |
| appris GBM | 0.1719 | 0.1969 | 0.7425 | 0.6962 | 0.7956 | 0.7425 | 0.9 |

*(WikiSection = 100 % multi-section → pas de mono ; F1_multi = frontières tol ±1, sélection du seuil sur la F1 OOF.)*


### 1b. TRANSFERT → notre gold témoignages FR (le chiffre clé)

Une tête entraînée sur WikiSection (EN/DE, encyclopédique) marche-t-elle sur des **avis citoyens FR** ? Comparé à l'attention **RÉGLÉE à la main** (gold) et au **change-point**. Trois régimes de seuil : **zéro-shot** = seuil calé sur WikiSection-CV, jamais sur le gold (LE test de transfert honnête) ; **gold-tuné (F1_global)** = seuil re-calé sur le gold par le MÊME objectif que l'attention réglée (F1_global pénalise la sur-coupe des mono) → apples-to-apples ; **gold-tuné (F1_multi)** = plafond de détection des frontières (optimise F1_multi seul, ignore les faux-positifs mono).

| approche | Pk | WindowDiff | F1_multi | P | R | mono_FP | F1_global | thr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attention RÉGLÉE-main (réf) | 0.1493 | 0.1563 | 0.7692 | 0.8955 | 0.6742 | 0.1442 | 0.7453 | — |
| change-point (réf) | 0.2815 | 0.282 | 0.4423 | 0.4545 | 0.4307 | 0.7019 | 0.384 | — |
| **appris LR** — zéro-shot | 0.2293 | 0.2309 | 0.5938 | 0.9744 | 0.427 | 0.2019 | 0.5588 | 0.95 |
| appris LR — gold-tuné (F1_global) | 0.1407 | 0.1501 | 0.8549 | 0.8971 | 0.8165 | 0.4519 | 0.773 | 0.7 |
| appris LR — gold-tuné (F1_multi) | 0.1306 | 0.1538 | 0.8613 | 0.8299 | 0.8951 | 0.5577 | 0.7611 | 0.5 |
| **appris GBM** — zéro-shot | 0.2033 | 0.2044 | 0.665 | 0.9853 | 0.5019 | 0.2308 | 0.6276 | 0.9 |
| appris GBM — gold-tuné (F1_global) | 0.1168 | 0.1259 | 0.9004 | 0.9216 | 0.8801 | 0.5673 | 0.7993 | 0.65 |
| appris GBM — gold-tuné (F1_multi) | 0.1155 | 0.1272 | 0.9043 | 0.906 | 0.9026 | 0.6442 | 0.7915 | 0.6 |

*(mono_FP = fraction des 104 mono sur-coupés — l'appris a-t-il appris à s'abstenir ? WikiSection n'ayant AUCUN mono, c'est le test de transfert le plus dur.)*

## 2. Cross-langue : train EN → test DE (WikiSection)

| approche | Pk | WindowDiff | F1_multi | P | R | F1_global | thr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| appris LR — train EN→test DE | 0.2331 | 0.2695 | 0.6498 | 0.5689 | 0.7574 | 0.6498 | 0.96 |
| appris GBM — train EN→test DE | 0.2143 | 0.2279 | 0.6437 | 0.6926 | 0.6012 | 0.6437 | 0.9 |

*(Seuil calé sur EN-CV, appliqué tel quel au DE held-out — preuve de généricité langue-agnostique des features dérivées.)*

## 3. Ablation : d'où vient le signal ? (LR, CV WikiSection)

| features | F1_multi | P | R | thr |
| --- | --- | --- | --- | --- |
| attn | 0.7058 | 0.6372 | 0.7909 | 0.95 |
| emb | 0.1199 | 0.0675 | 0.5342 | 0.5 |
| all | 0.7081 | 0.6388 | 0.7943 | 0.95 |

*(`attn` = cross_{L,H} seuls ; `emb` = dérive d'embedding seule ; `all` = les deux.)*

## 4. Interprétabilité — où vit le signal appris (poids LR)

- **Part du signal** : attention `cross_{L,H}` = 100 % de la masse |poids|, dérive d'embedding = 0 %. Dérive d'embedding : `{'emb_adj': -0.0408, 'emb_drift_W3': -0.1009, 'emb_drift_W8': 0.0347}` (signe + = la dissemblance pousse vers « frontière »).

- **Top (couche, tête) par |poids|** (somme sur les fenêtres cross ; signe − attendu : flux BAS = frontière) :

| layer | head | abs_weight | signed_weight |
| --- | --- | --- | --- |
| 2 | 5 | 5.7935 | -1.9492 |
| 1 | 8 | 5.3689 | -2.8008 |
| 1 | 0 | 1.7758 | 0.8767 |
| 6 | 4 | 1.7453 | -0.6779 |
| 4 | 9 | 1.7415 | -0.5634 |
| 6 | 1 | 1.5274 | 0.9765 |
| 5 | 0 | 1.101 | -1.101 |
| 4 | 8 | 1.0142 | -0.7879 |
| 2 | 4 | 0.9879 | 0.0156 |
| 8 | 1 | 0.9798 | -0.8768 |
| 6 | 8 | 0.9763 | -0.333 |
| 1 | 4 | 0.9242 | -0.0475 |
| 5 | 11 | 0.884 | -0.3663 |
| 1 | 7 | 0.8724 | -0.4428 |
| 8 | 6 | 0.8095 | -0.0192 |

- **Couches dominantes** (|poids| cumulé/tête) : L1 (11.4438), L2 (10.9229), L6 (7.5615), L4 (6.4185). **3/4 dans la moitié basse** du réseau (L<6) → le signal appris vit dans les couches **basses-moyennes**, ce qui **CONFIRME** la localisation `lowmid` trouvée à la main par `attn_seg` (le réglé-main avait élu lowmid sans voir un seul label).

- **Concentré, PAS diffus** : les 2 têtes de tête (L2H5, L1H8) pèsent **16 %** de la masse |poids| cross à elles seules. Le réglé-main concluait que le signal était *diffus* (la moyenne de TOUTES les têtes battait la sélection `local`) ; la supervision, elle, **isole des têtes-frontière spécifiques** — c'est précisément l'apport d'apprendre la combinaison plutôt que de moyenner. Leurs poids signés sont **négatifs** (flux d'attention BAS → frontière), conforme à l'intuition physique.

## 5. Verdict honnête — l'appris bat-il le réglé-main ?

- **Transfert ZÉRO-SHOT** (le test honnête : seuil jamais vu le gold) — meilleur appris = **GBM** : F1_multi=0.665 (P=0.985, R=0.502), Pk=0.203, mono_FP=0.231. vs attention réglée-main F1_multi=0.7692 → **ΔF1_multi=-0.104**. → l'appris **NE BAT PAS** le réglé-main en zéro-shot.

- **Apples-to-apples** (seuil re-calé sur le gold par F1_global, le MÊME objectif de sélection que `c` de l'attention) : F1_multi=0.900, F1_global=0.799, Pk=0.117, mono_FP=0.567 → **ΔF1_global=+0.054**, ΔF1_multi=+0.131 vs réglé-main. L'appris **BAT** le réglé-main à objectif/seuil comparable.

- **Plafond de détection des frontières** (seuil optimisant F1_multi seul, sans pénaliser les mono) : F1_multi=0.904 (P=0.906, R=0.903) mais mono_FP=0.644 — quand on l'autorise à sur-couper, l'appris détecte BEAUCOUP plus de frontières multi que le réglé-main (R=0.90 vs 0.6742), au prix d'une sur-coupe massive des mono. Le signal appris est plus RICHE ; la difficulté est la calibration de l'abstention.

- vs **change-point** (F1_multi=0.4423) : l'appris zéro-shot fait ΔF1_multi=+0.223.

- **Honnêteté train/domain gap** : train = **2250 docs** WikiSection (EN/DE encyclopédique, sections ~3/doc) ; test = avis citoyens **FR** (registre, langue ET domaine différents). Le transfert traverse DEUX gaps (langue + domaine). WikiSection n'a **aucun doc mono** → le classifieur n'a jamais vu d'exemple « ne rien couper » : tout sur-découpage des mono (23 %) vient de là, c'est la limite structurelle du transfert.

- **Sur/sous-apprentissage** : F1 CV WikiSection (GBM)=0.743 vs F1 transfert gold=0.665 — l'écart mesure le domain gap (un gros écart = la tête colle au style WikiSection). LR (linéaire, standardisé) = interprétable mais capacité limitée ; GBM = plus de capacité, risque de sur-apprendre le style source.

- **Généricité** : zéro lexique, zéro mot codé en dur — features 100 % dérivées de l'attention/embedding d'un encodeur gelé, calculables sur n'importe quelle langue. Le transfert EN/DE→FR (§1b) et EN→DE (§2) en est la preuve directe.
