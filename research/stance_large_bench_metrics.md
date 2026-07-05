# Bench stance small vs large — métriques (calculées)

Échantillon gold x-stance seedé (SEED=42, ~170/langue). Même échantillon pour toutes les configs.

## Synthèse cross-config

| config | modèle | n | %nuance | acc décidés | acc brute |
|---|---|---|---|---|---|
| small | mistral-small-latest | 527 | 13.5% | 0.796 | 0.689 |
| large | mistral-large-latest | 527 | 25.4% | 0.885 | 0.660 |
| large_noabst | mistral-large-latest | 527 | 11.0% | 0.861 | 0.767 |


## Config : small

### small — GLOBAL  (n=527, abstention=71 = 13.5%)
- **Accuracy sur décidés (abstention exclue)** : **0.796**  (363/456)
- Accuracy brute (abstention=erreur) : 0.689  (363/527)

| classe | précision | rappel | F1 | support |
|---|---|---|---|---|
| FAVOR | 0.763 | 0.784 | 0.774 | 292 |
| AGAINST | 0.859 | 0.570 | 0.685 | 235 |

#### Calibration confiance — small
| confiance | n | %abstention | accuracy décidés |
|---|---|---|---|
| high | 417 | 4.3% | 0.820 |
| medium | 99 | 43.4% | 0.625 |
| low | 11 | 90.9% | 1.000 |


**Par langue :**

### small — de  (n=179, abstention=28 = 15.6%)
- **Accuracy sur décidés (abstention exclue)** : **0.768**  (116/151)
- Accuracy brute (abstention=erreur) : 0.648  (116/179)

| classe | précision | rappel | F1 | support |
|---|---|---|---|---|
| FAVOR | 0.730 | 0.699 | 0.714 | 93 |
| AGAINST | 0.823 | 0.593 | 0.689 | 86 |

### small — fr  (n=178, abstention=24 = 13.5%)
- **Accuracy sur décidés (abstention exclue)** : **0.812**  (125/154)
- Accuracy brute (abstention=erreur) : 0.702  (125/178)

| classe | précision | rappel | F1 | support |
|---|---|---|---|---|
| FAVOR | 0.769 | 0.816 | 0.792 | 98 |
| AGAINST | 0.900 | 0.562 | 0.692 | 80 |

### small — it  (n=170, abstention=19 = 11.2%)
- **Accuracy sur décidés (abstention exclue)** : **0.808**  (122/151)
- Accuracy brute (abstention=erreur) : 0.718  (122/170)

| classe | précision | rappel | F1 | support |
|---|---|---|---|---|
| FAVOR | 0.785 | 0.832 | 0.808 | 101 |
| AGAINST | 0.864 | 0.551 | 0.673 | 69 |


## Config : large

### large — GLOBAL  (n=527, abstention=134 = 25.4%)
- **Accuracy sur décidés (abstention exclue)** : **0.885**  (348/393)
- Accuracy brute (abstention=erreur) : 0.660  (348/527)

| classe | précision | rappel | F1 | support |
|---|---|---|---|---|
| FAVOR | 0.872 | 0.654 | 0.748 | 292 |
| AGAINST | 0.902 | 0.668 | 0.768 | 235 |

#### Calibration confiance — large
| confiance | n | %abstention | accuracy décidés |
|---|---|---|---|
| high | 356 | 9.0% | 0.901 |
| medium | 157 | 56.1% | 0.812 |
| low | 14 | 100.0% | 0.000 |


**Par langue :**

### large — de  (n=179, abstention=45 = 25.1%)
- **Accuracy sur décidés (abstention exclue)** : **0.881**  (118/134)
- Accuracy brute (abstention=erreur) : 0.659  (118/179)

| classe | précision | rappel | F1 | support |
|---|---|---|---|---|
| FAVOR | 0.851 | 0.613 | 0.713 | 93 |
| AGAINST | 0.910 | 0.709 | 0.797 | 86 |

### large — fr  (n=178, abstention=48 = 27.0%)
- **Accuracy sur décidés (abstention exclue)** : **0.892**  (116/130)
- Accuracy brute (abstention=erreur) : 0.652  (116/178)

| classe | précision | rappel | F1 | support |
|---|---|---|---|---|
| FAVOR | 0.890 | 0.663 | 0.760 | 98 |
| AGAINST | 0.895 | 0.637 | 0.745 | 80 |

### large — it  (n=170, abstention=41 = 24.1%)
- **Accuracy sur décidés (abstention exclue)** : **0.884**  (114/129)
- Accuracy brute (abstention=erreur) : 0.671  (114/170)

| classe | précision | rappel | F1 | support |
|---|---|---|---|---|
| FAVOR | 0.873 | 0.683 | 0.767 | 101 |
| AGAINST | 0.900 | 0.652 | 0.756 | 69 |


## Config : large_noabst

### large_noabst — GLOBAL  (n=527, abstention=58 = 11.0%)
- **Accuracy sur décidés (abstention exclue)** : **0.861**  (404/469)
- Accuracy brute (abstention=erreur) : 0.767  (404/527)

| classe | précision | rappel | F1 | support |
|---|---|---|---|---|
| FAVOR | 0.847 | 0.798 | 0.822 | 292 |
| AGAINST | 0.881 | 0.728 | 0.797 | 235 |

#### Calibration confiance — large_noabst
| confiance | n | %abstention | accuracy décidés |
|---|---|---|---|
| high | 350 | 0.3% | 0.883 |
| medium | 165 | 27.3% | 0.800 |
| low | 12 | 100.0% | 0.000 |


**Par langue :**

### large_noabst — de  (n=179, abstention=24 = 13.4%)
- **Accuracy sur décidés (abstention exclue)** : **0.858**  (133/155)
- Accuracy brute (abstention=erreur) : 0.743  (133/179)

| classe | précision | rappel | F1 | support |
|---|---|---|---|---|
| FAVOR | 0.843 | 0.753 | 0.795 | 93 |
| AGAINST | 0.875 | 0.733 | 0.797 | 86 |

### large_noabst — fr  (n=178, abstention=17 = 9.6%)
- **Accuracy sur décidés (abstention exclue)** : **0.857**  (138/161)
- Accuracy brute (abstention=erreur) : 0.775  (138/178)

| classe | précision | rappel | F1 | support |
|---|---|---|---|---|
| FAVOR | 0.840 | 0.806 | 0.823 | 98 |
| AGAINST | 0.881 | 0.738 | 0.803 | 80 |

### large_noabst — it  (n=170, abstention=17 = 10.0%)
- **Accuracy sur décidés (abstention exclue)** : **0.869**  (133/153)
- Accuracy brute (abstention=erreur) : 0.782  (133/170)

| classe | précision | rappel | F1 | support |
|---|---|---|---|---|
| FAVOR | 0.857 | 0.832 | 0.844 | 101 |
| AGAINST | 0.891 | 0.710 | 0.790 | 69 |
