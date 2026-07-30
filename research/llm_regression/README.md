# Régression du pipeline vers de plus petits modèles — ledger R&D

**But** : porter chaque rôle LLM du pipeline sur le plus petit modèle possible et **mesurer** la
perte, verdict à l'appui (« mesurer avant d'adopter »). Le coût € du pipeline est dominé par les
appels `mistral-large`.

## Carte des rôles LLM (au 2026-07-24)
| Rôle | Modèle servi | Statut régression |
|---|---|---|
| Extraction des claims | `mistral-large` (gel hackathon ; défaut *code* = ministral-3b) | **VERDICT : → ministral-3b** (ci-dessous) |
| Enrichissement (titres, accroches, descriptions, insights) | `mistral-large` | pilote titres OK, à bencher robuste |
| Opinion / arguments | `mistral-large` / `mistral-small` | à bencher |
| Abstraction (profils) | `mistral-small` | ok par défaut |

## Progression (chaque script = une étape)
1. `titles_pilot.py` — pilote **titres** (validation du harnais juge-en-aveugle). N=10, non décisionnel.
2. `extraction_cascade_bench.py` — **cascade** d'escalade sur le plancher verbatim (idée de Bob) :
   le petit modèle sur un gros N, seuls les échecs remontent. Objectif, gratuit (pas de juge).
3. `extraction_headtohead_bench.py` — **face-à-face** couverture + qualité sur les MÊMES avis,
   juge = `mistral-medium` en aveugle (medium exclu des candidats). Verdict décisionnel apparié.
4. `extraction_judge_me.py` — re-extraction 3b vs large **avec textes sauvés** + métriques
   objectives de surdécoupage + dump lisible pour **jugement par l'agent** (Claude).
5. `extraction_prompt_tune.py` — **prompt d'extraction adapté PAR MODÈLE** (corrige le défaut de
   chacun sans changer de modèle).

## VERDICT — Extraction : `ministral-3b` (≈ large, ~150× moins cher)
Face-à-face 191 avis atomisables, 3 datasets, juge medium en aveugle :
- ministral-3b **4.06/5**, large **4.04/5** → Δ +0.02 (dans le bruit) = **aucune différence mesurable**.
- 3b extrait *plus* de claims (2.48 vs 1.91) pour une couverture ≈ égale (0.87× vs 0.90×).

**Jugement par l'agent (échantillon des cas les plus divergents)** : le motif dominant n'est PAS
le surdécoupage de 3b mais le **sous-découpage de large** — large empile des sujets distincts en
méga-claims (ex. une liste de 10 réformes = 1 seul claim), mauvais pour le clustering. 3b atomise
plus fin, généralement mieux adapté. **Défaut réel mais minoritaire de 3b** : il fragmente parfois
une phrase courte (avis tiktok émotionnels) ou casse aux guillemets (« paye », « patron » seuls).

**Enseignement** : le vrai levier de granularité n'est pas la taille du modèle mais le **prompt**.
D'où l'étape 5 (prompt par modèle). L'extraction étant EXTRACTIVE (copier des portions verbatim),
elle est intrinsèquement favorable aux petits modèles → le verdict est robuste.

## Étape 5 — Prompt par modèle (`extraction_prompt_tune.py`, N=30)
Nudge ajouté au prompt système, ciblé sur le défaut de chaque modèle. Baseline vs adapté :
- **3b + anti-fragmentation** : GAIN net sur avis courts (n 3.63→2.80, fragments ≤3 mots 4→2, arrête
  de couper contraste/problème-solution/énumération). **Défaut** : sur-corrige parfois jusqu'à
  LÂCHER un claim (avis très court → 0). À resserrer (« un claim entier, mais JAMAIS zéro »).
- **large + anti-sous-découpage** : gain modeste (n 2.63→3.10) ; faible sur avis à PUCES (large y
  atomise déjà) — à re-tester sur les listes en PROSE (son vrai point faible, sous-représenté ici).

**Verdict (non final, N=30)** : le mécanisme du prompt-par-modèle MARCHE (les deux convergent vers
la bonne granularité), c'est un levier qualité gratuit. La combinaison prometteuse = **ministral-3b
\+ prompt anti-fragmentation** (fin, propre, ~150× moins cher). À solidifier : resserrer le nudge 3b
(anti-zéro) + échantillon plus grand avec listes en prose.

## Conventions
Lancer depuis la racine du dépôt :
```
MISTRAL_API_KEY=$(cat var/mistral.key) uv run --extra contender --extra embed-contender \
  --extra faiss python -u research/llm_regression/<script>.py [args]
```
Sorties (`*_results.json`, `*.txt`) écrites dans ce dossier. Prix relatifs (output $/M) :
large 6 · medium 2 · small 0.3 · ministral-8b 0.10 · ministral-3b 0.04.
