# Lane viz — frontend animé · panneaux thèmes · audit

Owns: `frontend/`. Port `:5180`.
**Base = fork de la viz de `dummy`** (`~/forge/dummy/frontend`, lecture seule) :
React + `@react-three/fiber` + `d3-force-3d` dans un **web worker** (`scene/{GraphScene,
Nodes,Links}`, `app/GraphCanvas`, `workers/forceLayout.*`). Le protocole worker est
STABLE : `init` (batch) puis `addNodes` (live, positions préservées). Nœud coloré par
`cluster_id` Leiden.

## T-V1 · Scaffold front (fork base dummy)
- Goal : porter la base R3F+worker de dummy sur `:5180`, alimentée par un
  `GraphPayload` statique (Phase 1 batch). Adapter palette = couleur par `cluster_id`.
- Accept : essaim batch TikTok rendu ; port libre ; positions stables.
- Deps : contrat figé, nlp (graphe + clusters batch).

## T-V2 · Animation de l'essaim (le clou du show)
- Goal : Phase 1 — apparition + force-layout des thèmes colorés. Phase 2 — bascule
  streaming : `addNodes` à chaque vague d'avis → agrégation live, recoloration,
  fusion/scission visibles.
- Accept : montée fluide ; transitions douces ; Phase 2 sans réécriture du scaffold.
- Deps : T-V1, stream (Phase 2).

## T-V3 · Panneaux thèmes
- Goal : liste des thèmes triés par poids ; label, taille, weight, diversity, consensus.
- Accept : se met à jour en live ; classement lisible.
- Deps : T-V1.

## T-V4 · Drill-down auditable (transparence = critère jury)
- Goal : clic thème → réponses sources + keywords + scores ; traçabilité du cluster.
- Accept : chaque thème ouvre ses contributions réelles.
- Deps : T-V3.

## T-V5 · Contrôles
- Goal : fusion/split manuels, scrub temporel (évolution des thèmes dans le temps).
- Accept : actions reflétées dans l'état ; timeline navigable.
- Deps : T-V2, T-V3.
