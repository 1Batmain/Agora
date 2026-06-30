# Métriques de validation stance (calculées)

### GLOBAL  (n=3000, abstention=446 = 14.9%)
- **Accuracy** (abstention=erreur) : **0.672**  (2017/3000)
- Accuracy sur décidés (abstention exclue) : 0.790  (2017/2554)

| classe | précision | rappel | F1 | support |
|---|---|---|---|---|
| FAVOR | 0.805 | 0.672 | 0.732 | 1565 |
| AGAINST | 0.774 | 0.673 | 0.720 | 1435 |

### Matrice de confusion — GLOBAL

gold ↓ / pred → | FAVOR | AGAINST | ABSTAIN | total |
|---|---|---|---|---|
| **FAVOR** | 1051 | 282 | 232 | 1565 |
| **AGAINST** | 255 | 966 | 214 | 1435 |


## Par langue

### langue=de  (n=1000, abstention=141 = 14.1%)
- **Accuracy** (abstention=erreur) : **0.683**  (683/1000)
- Accuracy sur décidés (abstention exclue) : 0.795  (683/859)

| classe | précision | rappel | F1 | support |
|---|---|---|---|---|
| FAVOR | 0.794 | 0.666 | 0.724 | 485 |
| AGAINST | 0.796 | 0.699 | 0.745 | 515 |

### langue=fr  (n=1000, abstention=153 = 15.3%)
- **Accuracy** (abstention=erreur) : **0.679**  (679/1000)
- Accuracy sur décidés (abstention exclue) : 0.802  (679/847)

| classe | précision | rappel | F1 | support |
|---|---|---|---|---|
| FAVOR | 0.804 | 0.683 | 0.739 | 521 |
| AGAINST | 0.800 | 0.674 | 0.732 | 479 |

### langue=it  (n=1000, abstention=152 = 15.2%)
- **Accuracy** (abstention=erreur) : **0.655**  (655/1000)
- Accuracy sur décidés (abstention exclue) : 0.772  (655/848)

| classe | précision | rappel | F1 | support |
|---|---|---|---|---|
| FAVOR | 0.816 | 0.665 | 0.733 | 559 |
| AGAINST | 0.722 | 0.642 | 0.679 | 441 |


## Par confiance auto-déclarée

### confidence=high  (n=2200, abstention=52 = 2.4%)
- **Accuracy** (abstention=erreur) : **0.789**  (1735/2200)
- Accuracy sur décidés (abstention exclue) : 0.808  (1735/2148)

| classe | précision | rappel | F1 | support |
|---|---|---|---|---|
| FAVOR | 0.812 | 0.780 | 0.795 | 1089 |
| AGAINST | 0.804 | 0.797 | 0.801 | 1111 |

### confidence=medium  (n=700, abstention=297 = 42.4%)
- **Accuracy** (abstention=erreur) : **0.400**  (280/700)
- Accuracy sur décidés (abstention exclue) : 0.695  (280/403)

| classe | précision | rappel | F1 | support |
|---|---|---|---|---|
| FAVOR | 0.775 | 0.481 | 0.593 | 416 |
| AGAINST | 0.552 | 0.282 | 0.373 | 284 |

### confidence=low  (n=100, abstention=97 = 97.0%)
- **Accuracy** (abstention=erreur) : **0.020**  (2/100)
- Accuracy sur décidés (abstention exclue) : 0.667  (2/3)

| classe | précision | rappel | F1 | support |
|---|---|---|---|---|
| FAVOR | 1.000 | 0.033 | 0.065 | 60 |
| AGAINST | 0.000 | 0.000 | 0.000 | 40 |


## Synthèse confiance (les low se trompent-ils plus ?)

| confiance | n | %abstention | accuracy décidés |
|---|---|---|---|
| high | 2200 | 2.4% | 0.808 |
| medium | 700 | 42.4% | 0.695 |
| low | 100 | 97.0% | 0.667 |
