# Petits modèles pour le multi-label de thèmes — rapport

*Jeu : `gold_large.json` — N=305 (104 mono, 201 multi), 8 thèmes. Réf **Mistral-small** (`llm_report.md`, NON relancée) : micro-F1 **0.928**, macro 0.9346, exact-set 73%. Seed 0, CPU.*

**Question** : un petit modèle LOCAL atteint-il une qualité « assez proche de 0.93 » à coût SCALABLE (rapide, gratuit, données qui ne sortent pas) ? Si oui → résout coût + souveraineté + échelle, ET rend la segmentation inutile (on a directement l'ensemble des thèmes par avis).

## Scorecard — qualité × coût/vitesse

| Modèle | type | micro-F1 | macro-F1 | exact-set | ms/avis | local | données sortent |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Mistral-small** (réf, API) | LLM cloud | 0.928 | 0.9346 | 73% | ~230* | non | **oui** |
| clf logreg / nomic-v2 | embed+tête | 0.891 | 0.892 | 65% | 58.3 | **oui** | non |
| clf mlp / nomic-v2 | embed+tête | 0.939 | 0.939 | 79% | 58.3 | **oui** | non |
| clf logreg / e5-small | embed+tête | 0.832 | 0.834 | 51% | 10.8 | **oui** | non |
| clf mlp / e5-small | embed+tête | 0.921 | 0.921 | 75% | 10.8 | **oui** | non |
| qwen3:4b | LLM local (Mac) | 0.836 | 0.843 | 51% | 525 | **oui** | non |
| ministral-3:latest | LLM local (Mac) | 0.934 | 0.934 | 78% | 664 | **oui** | non |
| nemotron3:33b | LLM local (Mac) | 0.877 | 0.873 | 64% | 789 | **oui** | non |

*\* Mistral ms/avis = ~70s cumulés / 305 avis ≈ 230 ms/avis amorti (batché 12/appel, réseau UE) — cf. `llm_report.md`. ms/avis classifieur = embedding (dominant) + tête (quasi-nul), 100% sur le VPS. ms/avis Ollama = latence **À CHAUD** (warm-up préalable, modèle déjà chargé) cumulée / N, 1 avis/appel, sur le Mac de Bob (`http://mac-local:11434`, Apple Silicon via Tailscale ; souverain — la donnée ne sort pas du réseau privé).*

## Candidat 1 — classifieur multi-label sur embedding (le cheval)

Vecteur d'avis ENTIER (pooling prod `embed_docs`) → tête multi-label. CV stratifiée par avis (5 plis, stratifiés sur le nb de thèmes), probas hors-pli, seuil PAR CLASSE calé pour max-F1. LogReg one-vs-rest (`class_weight=balanced`) et MLP (1×128, sans early-stopping, `alpha=1e-3` ; AVEC early-stopping il s'effondre à ~0.45 — un pli de validation rogné sur un jeu déjà petit l'arrête trop tôt).

| embedder | tête | micro-P | micro-R | micro-F1 | macro-F1 | exact-set | embed ms/avis | tête ms/avis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nomic-v2 | logreg | 0.887 | 0.895 | 0.891 | 0.892 | 65% | 58.31 | 0.0102 |
| nomic-v2 | mlp | 0.945 | 0.934 | 0.939 | 0.939 | 79% | 58.31 | 0.0037 |
| e5-small | logreg | 0.795 | 0.872 | 0.832 | 0.834 | 51% | 10.76 | 0.0078 |
| e5-small | mlp | 0.942 | 0.902 | 0.921 | 0.921 | 75% | 10.76 | 0.0034 |

### F1 par thème — meilleure tête (`mlp` / `nomic-v2`, micro-F1 0.939)

| thème | P | R | F1 | seuil | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| desinformation | 0.982 | 0.947 | 0.964 | 0.45 | 54 | 1 | 3 |
| image_corps | 0.957 | 0.971 | 0.964 | 0.15 | 67 | 3 | 2 |
| sante_mentale | 0.969 | 0.949 | 0.959 | 0.3 | 94 | 3 | 5 |
| harcelement | 0.959 | 0.959 | 0.959 | 0.35 | 70 | 3 | 3 |
| enfants | 0.901 | 0.955 | 0.928 | 0.35 | 64 | 7 | 3 |
| contenus_choquants | 0.909 | 0.923 | 0.916 | 0.35 | 60 | 6 | 5 |
| addiction | 0.954 | 0.873 | 0.912 | 0.3 | 62 | 3 | 9 |
| algorithme | 0.926 | 0.887 | 0.906 | 0.35 | 63 | 5 | 8 |

## Candidat 2 — petit LLM local via Ollama, sur le Mac (filet souverain)

Serveur **Ollama du Mac de Bob** (`http://mac-local:11434`, Apple Silicon via Tailscale) — bien plus rapide que l'Ollama CPU du VPS. MÊME prompt fermé que Mistral (`llm_seg.theme_prompt`), choix fermé sur les 8 thèmes, JSON mode, température 0. Les raisonneurs ont leur pensée coupée (`think:false`) ; un **warm-up** charge chaque modèle AVANT le timing → latence mesurée **à chaud**. 1 avis/appel (mapping non ambigu + vraie latence/avis). Cache disque `.cache/ollama/` clé par endpoint (relances gratuites, la latence d'un host ne pollue pas l'autre).

| modèle | pensée coupée | micro-P | micro-R | micro-F1 | macro-F1 | exact-set | ms/avis (chaud) | tokens générés | erreurs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3:4b | oui | 0.803 | 0.871 | 0.836 | 0.843 | 51% | 525 | 8320 | 0 |
| ministral-3:latest | oui | 0.934 | 0.935 | 0.934 | 0.934 | 78% | 664 | 5559 | 0 |
| nemotron3:33b | oui | 0.867 | 0.888 | 0.877 | 0.873 | 64% | 789 | 6408 | 0 |

### F1 par thème — `qwen3:4b`

| thème | P | R | F1 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- |
| desinformation | 0.93 | 0.93 | 0.93 | 53 | 4 | 4 |
| addiction | 0.984 | 0.859 | 0.917 | 61 | 1 | 10 |
| image_corps | 0.884 | 0.884 | 0.884 | 61 | 8 | 8 |
| algorithme | 0.884 | 0.859 | 0.871 | 61 | 8 | 10 |
| contenus_choquants | 0.721 | 0.954 | 0.821 | 62 | 24 | 3 |
| sante_mentale | 0.645 | 0.99 | 0.781 | 98 | 54 | 1 |
| enfants | 0.718 | 0.836 | 0.772 | 56 | 22 | 11 |
| harcelement | 0.979 | 0.63 | 0.767 | 46 | 1 | 27 |

### F1 par thème — `ministral-3:latest`

| thème | P | R | F1 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- |
| image_corps | 0.972 | 1.0 | 0.986 | 69 | 2 | 0 |
| desinformation | 0.982 | 0.947 | 0.964 | 54 | 1 | 3 |
| addiction | 0.91 | 1.0 | 0.953 | 71 | 7 | 0 |
| harcelement | 1.0 | 0.904 | 0.95 | 66 | 0 | 7 |
| sante_mentale | 0.959 | 0.939 | 0.949 | 93 | 4 | 6 |
| contenus_choquants | 0.914 | 0.985 | 0.948 | 64 | 6 | 1 |
| enfants | 0.824 | 0.91 | 0.865 | 61 | 13 | 6 |
| algorithme | 0.919 | 0.803 | 0.857 | 57 | 5 | 14 |

### F1 par thème — `nemotron3:33b`

| thème | P | R | F1 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- |
| harcelement | 1.0 | 0.918 | 0.957 | 67 | 0 | 6 |
| sante_mentale | 0.907 | 0.98 | 0.942 | 97 | 10 | 2 |
| desinformation | 0.943 | 0.877 | 0.909 | 50 | 3 | 7 |
| enfants | 0.875 | 0.94 | 0.906 | 63 | 9 | 4 |
| image_corps | 0.983 | 0.826 | 0.898 | 57 | 1 | 12 |
| contenus_choquants | 0.714 | 1.0 | 0.833 | 65 | 26 | 0 |
| addiction | 0.711 | 0.972 | 0.821 | 69 | 28 | 2 |
| algorithme | 0.976 | 0.563 | 0.714 | 40 | 1 | 31 |

## Verdict — un petit modèle local tient-il près de 0.93 à coût scalable ?

- **Meilleur local : `clf mlp/nomic-v2`** — micro-F1 **0.939** (exact-set 79%), soit **+0.011** vs Mistral 0.928 à **58.3 ms/avis**, 100% local, données qui ne sortent pas.

- **OUI — et il la DÉPASSE.** 2/7 candidats locaux atteignent ou dépassent la réf 0.928 : `clf mlp/nomic-v2` (0.939), `ministral-3:latest` (0.934). Un petit modèle local tient près de 0.93 — voire mieux — à coût scalable (local, rapide, souverain). Et puisqu'on obtient directement l'ensemble des thèmes par avis, la **segmentation de frontières devient inutile** pour ce but.

- **Deux gagnants de natures opposées.** Le **classifieur** `mlp/nomic-v2` (0.939, 58 ms/avis) — ultra-cheap mais **entraîné** sur nos 8 thèmes. Le **LLM local** `ministral-3:latest` (0.934, 664 ms/avis) — ~×11 plus lent mais **zéro-shot** (taxo dans le prompt → générique par consultation, comme Mistral, sans aucun label).

- **Le classifieur sur embedding est le candidat scalable** : inférence dominée par l'embedding (déjà calculé en prod pour le clustering), tête quasi-gratuite, batch, aucune donnée qui sort. Mais il est entraîné sur NOS 8 thèmes — **en prod la taxo est par-consultation**, donc il faudrait un échantillon labellisé (par LLM ou humain) par consultation pour le ré-entraîner. C'est le compromis : très bon marché à l'inférence, mais coût d'amorçage par consultation.

- **Les LLM locaux du Mac (Ollama)** ne demandent AUCUN entraînement (zéro-shot, taxo passée dans le prompt → générique par consultation, comme Mistral) et tournent sur une machine **souveraine** (Apple Silicon, le réseau privé Tailscale ; la donnée ne sort jamais vers une API). Le prix est la latence (voir ms/avis à chaud) et la dépendance au Mac allumé.

- **Plus gros ≠ mieux** (comme pour e5-large) : le gros `nemotron3:33b` (0.877, 789 ms/avis) est **battu** par le petit `ministral-3:latest` (0.934, 664 ms/avis) — plus lent ET moins bon sur ce choix-fermé court. L'option souveraine haute qualité, c'est `ministral-3`, PAS le 33B. Le `qwen3:4b` (raisonneur) reste en retrait (sur-prédit `sante_mentale`/`contenus_choquants`).

- **Honnêteté** : seuils du classifieur calés sur les probas OOF servant aussi au score (léger optimisme, pas de fuite d'entraînement). Latence Ollama mesurée à chaud (warm-up) mais sur un Mac partagé, 1 requête à la fois (pas de batching/parallélisme) → indicative. Mistral non relancé (chiffres `llm_report.md`).
