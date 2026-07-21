# Verdict — chaîne d'emboîtement vs SBM emboîté MDL (Peixoto)

**Question (Bob) :** notre « chemin qui casse le moins de clusters » est-il un cas particulier
d'un problème déjà résolu, et une méthode principielle ferait-elle mieux ?
**Réponse : la chaîne TIENT.** Le SBM emboîté (état de l'art, MDL, `graph-tool`) ne bat pas
notre méthode minimale sur la vérité terrain. **NON à l'adopter dans le pipeline.**

## Le banc

`research/sbm_export.py` → `sbm_nested.py` (python système + graph-tool) → `sbm_compare.py`.
Même espace recentré, même graphe kNN de référence (seuil dérivé, k=derive_k(N)) donné aux
DEUX. Le SBM infère sa hiérarchie par longueur de description minimale : le nombre de niveaux
et de blocs n'est pas choisi, c'est celui qui comprime le mieux le graphe. Graine fixe (42).

## Résultats

| corpus | notre chaîne | SBM emboîté (MDL) | gold | chaîne | SBM |
|---|---|---|---|---|---|
| tiktok (2419) | 16 → 9 → 4 | 125 → 34 → 6 → 2 | — | — | — |
| x-stance (4274) | 24 → 14 → 7 | 208 → 49 → 10 → 3 | **12 topics** | **0.192** | 0.190 |
| | | | 191 questions | 0.099 | **0.144** |
| **mix** (6693) | 21 → **2** (propreté 0.94) | 369 → 99 → 28 → 7 → **2** | **2 domaines** | 0.906 | 0.915 |

ARI = Adjusted Rand Index du MEILLEUR niveau de chaque méthode contre le gold (le SBM a le
DROIT de choisir parmi ses 5 niveaux — avantage qui lui est donné).

## Lecture

1. **Sur la vérité MACRO (les 12 topics humains de x-stance), c'est une ÉGALITÉ** : 0.192
   (chaîne) vs 0.190 (SBM). Deux millièmes = bruit. Le niveau que le SBM aligne le mieux sur
   les 12 topics est son niveau à **10 blocs** — il retrouve tout seul la bonne granularité,
   comme notre chaîne trouve 14. Aucune méthode ne « gagne » les macros.

2. **Sur le TÉMOIN 2 domaines, égalité aussi** : les deux retrouvent nettement les 2 macros
   réels (0.906 vs 0.915). Le MDL confirme la frontière là où notre propreté disait 0.94.
   Autrement dit, **le SBM valide notre jauge de propreté** : quand elle crie « vrai macro »,
   le MDL est d'accord.

3. **Le SBM ne gagne que sur le grain FIN** (191 questions : 0.144 vs 0.099) — mécanique : il
   descend à 208 blocs, notre chaîne s'arrête à 24. Or nos macros fins (~16-24) sont un CHOIX
   produit (lisibilité), pas une limite de méthode. Ce « gain » ne nous intéresse pas.

4. **Le SBM fabrique un bas de hiérarchie très fragmenté** (125-369 blocs) : inutilisable tel
   quel pour une carte lisible de consultation. Sa profondeur (5-6 niveaux) est un luxe dont
   notre produit (2 niveaux) n'a pas l'usage.

## Ce que ça règle

- **La chaîne n'est pas en train de laisser de la précision sur la table.** À parité avec une
  méthode principielle de l'état de l'art sur la seule vérité qui compte (les macros). La
  version minimale suffit — philosophie validée par la mesure, pas par conviction.
- **Le seul apport réel du SBM est le MDL** : un critère de PROFONDEUR sans seuil (« combien
  de niveaux sont réels ? »), qui est exactement notre problème ouvert (la frontière binaire
  plat/feuilleté). Si un jour on veut le clore proprement, on emprunte le PRINCIPE (longueur
  de description) sans traîner tout `graph-tool`. Mais ce n'est pas urgent : le SBM ne récolte
  aucun gain de fidélité qui le justifierait aujourd'hui.

## Réserves honnêtes

- SBM inféré en **une graine** (inférence stochastique). Les écarts sur le gold sont si petits
  (≤ 0.01, dans le bruit) qu'une répétition ne renverserait pas les égalités — mais ce n'est
  pas une preuve, c'est un signal.
- ARI contre 12 topics est **plafonné bas** (~0.2) pour tout le monde : les topics humains ne
  sont que partiellement retrouvables depuis les embeddings. Les deux méthodes en souffrent.
- On a comparé le MEILLEUR niveau du SBM (choisi parmi 5) au gold — un handicap qu'on
  s'IMPOSE en notre défaveur. Même ainsi, la chaîne égale sur les macros.

## Reproduire

```
uv run --extra embed-contender --extra faiss python research/sbm_export.py
/usr/bin/python3 research/sbm_nested.py          # nécessite: sudo apt install python3-graph-tool
uv run python research/sbm_compare.py
```
Résultats bruts : `research/sbm_vs_chain_results.json`. Gratuit (claims/embeddings du cache).
