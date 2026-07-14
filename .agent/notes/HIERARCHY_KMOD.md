# Décision — hiérarchie par modulation de k, assumée comme CONTINUUM (2026-07-12)

Aboutissement de la refonte du clustering (verdicts `HIERARCHY_TAU.md`, `EMBEDDING_SPACE.md`).
Validé par Bob : « on assume le continuum ».

## La décision, en une phrase
Le **niveau fin** (Leiden r=1.0, espace recentré) est la VÉRITÉ servie ; les **niveaux
grossiers** de navigation sont générés par le **rayon de voisinage k**, et l'on ACCEPTE
qu'ils ne soient pas emboîtés — un thème fin peut relever de deux thèmes larges. La
hiérarchie est un CONTINUUM, pas un arbre strict.

## Comment on en est arrivé là (6 méthodes, 1 conclusion)
Toutes cherchaient à regrouper les feuilles sous des macros. Toutes échouent, pour la même
raison ou en révélant le même fait :

| méthode | résultat |
|---|---|
| coarsening cosinus (μ+σ) | fusionne « harcèlement » avec « corps » — anisotropie |
| sauce_magique (α,β,γ,δ) | FRAGMENTE au lieu de regrouper (dissout les parents) |
| modularité comme critère de fusion | ΔQ ≤ 0 partout — Leiden a déjà maximisé Q |
| pression / densité d'interface | 0 fusion — c'est la modularité mal posée |
| agrégation en super-graphe | 3 macros (biais du petit graphe) ou 19 = les feuilles |
| **balayage de k** | **produit les couches grossières, mais NON emboîtées** |

Fait mesuré sur gold externe (x-stance, 12 topics annotés) : quelle que soit la méthode,
la couche la plus fidèle est TOUJOURS la plus FINE. ARI k=8 : 0.237 → k=250 : 0.146.
**Le corpus n'a pas de niveau macro « plus vrai ».** Les gros regroupements sont plus
LISIBLES, pas plus JUSTES. C'est une commodité de navigation, pas une structure cachée.

## Pourquoi k, et pas la résolution
k change le GRAPHE (quelles arêtes existent) ; la résolution repondère un graphe fixe.
Balayer k coalesce les thèmes organiquement (tiktok : k8→15, k30→9, k120→6, k250→5) — c'est
le meilleur générateur de couches grossières testé, et un seul robinet.

## Le prix, assumé : NON-EMBOÎTEMENT
Pureté d'emboîtement mesurée (tiktok) : k8→k30 = 0.85, k30→k120 = 0.79, k120→k250 = 0.91.
Un thème fin sur cinq (au pire palier) se répartit entre DEUX parents grossiers. Rendu
visuel : artefact « Modulation par k » (diagramme alluvial, forks cerclés).

Bob assume : ce n'est pas un rangement en tiroirs, c'est un continuum où « harcèlement » et
« comparaison des corps » se recoupent RÉELLEMENT. Plus fidèle au réel qu'un arbre forcé.

## ⚠️ CONSÉQUENCE CONTRAT (bloquant, à scoper)
Le contrat front↔back (`frontend/src/redesign/contract.ts`) porte un `parent_id` UNIQUE, et
le front rend un arbre (d3-pack). Le continuum est un **DAG** : un nœud a 1..N parents.
C'est une re-architecture du modèle de données ET du rendu, pas un réglage. À NE PAS
bricoler — à spécifier (T-N11).

## Ce qui reste ouvert (à trancher AVANT de câbler)
1. Quels k définissent les niveaux de navigation ? Dérivés de N, ou échelle fixe ? (les
   plateaux de stabilité donnent des candidats, pas LE choix.)
2. Représentation front d'un nœud multi-parent : appartenance pondérée ? affichage sous
   chaque parent avec un poids ? un seul « parent principal » + tags secondaires ?
3. Détermininisme : k est stable, mais Leiden a une graine — figer seed, mesurer la
   stabilité de la façade multi-k sous retrait de 1 % (harnais `tau_robustness.py`).
4. Le nommage des thèmes LARGES reste un problème ouvert (c-TF-IDF dilué → titres vagues ;
   nommer depuis les enfants → slogans hallucinés). Non résolu.

## Ce qui est DÉJÀ acquis et indépendant de tout ça
- Recentrage des embeddings : OUI (verdict `EMBEDDING_SPACE.md`), gratuit, +19 % ARI gold.
- Suppression de tau / RES_LADDER / min_sub_size local : PR #24 (intégrité + déterminisme).
- Ces deux-là ne dépendent PAS du continuum et peuvent atterrir avant.
