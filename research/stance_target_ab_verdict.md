# Cible de stance a/b/c — VERDICT du juge aveugle

**Statut : RESEARCH, zéro modif pipeline.** Ce fichier DÉPOUILLE le panel aveugle lancé sur
les 26 paires anonymisées de `stance_target_ab_panel.jsonl`. Il tranche l'impression
mécanique de `stance_target_ab_note.md` (métriques : (b) dé-duplique sans coût, (c) à double
tranchant). Le panel confirme (b) et écarte (c).

## Protocole (identique aux panels extraction-v2)

- **Juge** : `mistral-large-latest`, température 0, aligné sur la décision prod
  « mistral-large par défaut pour toutes les tâches LLM ».
- **Panel de 3 passes indépendantes** par paire, prompts de juge LÉGÈREMENT variés (neutre /
  accent « central > saillant » / accent « polaire + éviter le passe-partout ») pour ne pas
  jouer 3× le même biais de phrasé. Tous exigent les mêmes critères : capter le débat
  **CENTRAL** (pas la facette saillante/bruyante) **en restant polaire et débattable**.
- **Aveugle** : chaque paire montre deux cibles X/Y (ordre aléatoire seedé, provenance a/b/c
  masquée) + titre + 3 contributions de contexte NEUTRE. Les votes ont été jetés AVANT toute
  lecture de la clé `stance_target_ab_panel_key.json` (dépouillé après).
- **Décision de paire = majorité** des 3 votes ; les égalités sont comptées.
- Harnais : `research/stance_panel_judge.py` (vote aveugle) → `..._panel_votes.jsonl` ;
  dé-anonymisation `research/stance_panel_tally.py`. Repro :

      MISTRAL_API_KEY=… python3 research/stance_panel_judge.py
      python3 research/stance_panel_tally.py

## Fiabilité du panel

26 paires, 78 appels, **0 erreur**. **0 égalité** : chaque paire a dégagé une majorité stricte
X ou Y. 21/26 paires **unanimes (3-0)**, 5/26 à **2-1** (n13-ab, n163-ac, n163-bc, n267-bc,
n29-bc). Signal net, peu de bruit inter-juges.

## Résultats — duels de majorité

| duel | gagne | perd | nul | lecture |
|---|---:|---:|---:|---|
| **(b) vs (a) actuel** | **b : 7** | a : 2 | 0 | (b) domine largement |
| (c) vs (a) actuel | c : 4 | a : 4 | 0 | **match nul** |
| (b) vs (c) | **b : 6** | c : 3 | 0 | (b) domine (c) |

**Score net (victoires − défaites, tous duels) : (b) = +8 · (c) = −3 · (a) = −5.**
Ordre du panel sans ambiguïté : **b ≻ c ≻ a**.

## Verdict par variante vs actuel (a)

- **(b) CENTRAL + mots-clés distinctifs en contexte → OUI.** 7-2 contre l'actuel, 6-3 contre
  (c), meilleur score net. Le panel confirme point par point l'impression mécanique : (b)
  re-mobilise le vocabulaire propre du thème (INTER-CIBLES 0.759 → 0.720) **sans décrocher du
  sujet** (fit_titre plat 0.801 → 0.800, médiane en hausse). Cas d'école n33 « Perte de temps
  ET culpabilité » : les 3 juges préfèrent (b) « limiter le temps sur TikTok **pour réduire la
  culpabilité** » à (a) « limiter le temps sur les réseaux » (générique) — (b) récupère les
  deux facettes du titre. Les 2 seules feuilles où (a) tient : n164 « Perte de notion du
  temps » et n29 « mal-être » (formulations (b) jugées un peu moins nettes), sans renverser la
  tendance.

- **(c) CLAIMS DISTINCTIFS → NON.** 4-4 contre l'actuel (n'améliore rien en net) et battu 3-6
  par (b). C'est très exactement le **double tranchant** annoncé : quand (c) gagne il est plus
  informatif (n163 « réguler » plutôt que « limiter », n273 « design addictif »), mais quand
  il perd il **sort du sujet** — la dérive vers le SAILLANT que cleavage-v2 a écartée
  (cf. décrochages fit_titre n29 0.85→0.67, n13 dans la note). Le juge aveugle sanctionne
  cette instabilité : gain nul vs actuel. À NE PAS adopter en l'état.

## Décision recommandée pour le rebuild

**Adopter (b) : cadrer les mots-clés c-TF-IDF comme « ce qui SÉPARE ce thème des voisins » +
consigne de contraste dans `cleavage_system` (`backend/build_opinion.py`).** C'est un
changement de **prompt uniquement** — zéro modif d'input/échantillonnage, zéro nouveau helper,
compatible avec le garde-fou « central > saillant » de cleavage-v2 (la représentativité ne
baisse pas). Levier le plus sûr et le mieux noté.

**Ne pas adopter (c)** (sélection par distinctivité pure de l'input) : gain nul vs actuel,
risque saillant matérialisé.

**Réserves (à mesurer avant de généraliser) :**
- n=1 corpus (tiktok, FR, mono-sujet). Rien ne se transpose sur granddebat sans re-mesure.
- La généricité résiduelle est en AMONT (feuilles quasi-synonymes « Addiction »/« Dépendance »/
  « Temps perdu »). (b) les écarte à la marge ; le vrai levier de dé-duplication reste le
  clustering / nommage contrastif en amont (cf. `cluster_merge_note.md`).
- **Réconciliation (c) restée DÉBLOQUÉE mais non-décisive** : le sélecteur canonique
  `select_distinctive_claims` a atterri sur `feat/titles-ancres` (même logique c-TF-IDF que le
  sélecteur local) ; comme (c) est écarté ici, ré-exécuter (c) avec le canonique n'est plus une
  priorité — à faire seulement si on veut vérifier que le canonique ne renverse pas le nul.

## Artefacts

- `stance_target_ab_panel_votes.jsonl` — 26 paires, votes bruts des 3 juges + décision (aveugle).
- `stance_panel_judge.py` / `stance_panel_tally.py` — vote aveugle / dépouillement.
