# Agora — Lane VIZ (frontend)

Essaim 3D des consultations citoyennes : un nuage de nœuds (avis) qui s'auto-organise
en **communautés Leiden = thèmes**, coloré par thème, avec un panneau latéral
auditable (drill-down thème → avis sources).

Port : **5180 uniquement**.

## D'où vient la base

Le **moteur de layout** est forké (lecture seule) de `~/forge/dummy/frontend` :
React + `@react-three/fiber` + `d3-force-3d` exécuté dans un **web worker**.

Repris **verbatim** (le contrat figé, cf. `queue/cross-lane.md`) :

- `src/workers/forceLayout.protocol.ts` — protocole worker STABLE
  (`init` batch · `addNodes` live · `focus` · `setParams` · `pin`/`unpin`).
- `src/workers/forceLayout.worker.ts` — la simulation d3-force-3d (forces +
  gravité par nœud + collision), émet les positions à ~30 Hz (SharedArrayBuffer
  si dispo, sinon buffers transférables).
- `src/workers/forceLayout.client.ts` — client mince autour du worker.
- `src/types/d3-force-3d.d.ts` — typings du module.

**Réécrit / nettoyé** (le HUD métier de dummy — exercices/anatomie/nutrition/chat/
agent — est jeté) :

- `src/scene/{GraphScene,Nodes,Links}.tsx` — rendu R3F autonome (Canvas +
  `OrbitControls` drei), spheres instanciées colorées par `node.color`, arêtes
  k-NN en `LineSegments` (luminosité ∝ `props.weight`).
- `src/hud/ThemesPanel.tsx` — panneau latéral générique (le seul HUD conservé).
- `src/state/useGraphStore.ts` — état UI minimal (thème sélectionné, hover).
- `src/lib/graphData.ts` — types canoniques + indexation.

> dummy n'est **jamais** modifié ni lancé. Le moteur est copié, pas importé.

## Données (Phase 1 — batch, pas de backend)

`public/graph.sample.json` = copie de
`pipeline/cluster/fixtures/graph.sample.json`, un `GraphPayload { meta, nodes,
links, themes }` (36 avis, 188 arêtes k-NN, 6 thèmes Leiden).

⚠️ **Contrat** : sur chaque nœud, `cluster_id` (int) et `color` (hex) sont au
**TOP-LEVEL** (pas dans `props`). L'essaim est coloré par `node.color`.

## Lancer

```bash
cd frontend
npm install          # une seule fois (box partagée — sobre)
npm run dev          # → http://localhost:5180
```

`npm run dev` force déjà `--port 5180` (et `vite.config.ts` met `strictPort`).
Ports interdits (autres projets / Ollama) : `8000 5173 8765 11434`.

Les en-têtes COOP/COEP sont activés (`vite.config.ts`) pour autoriser
`SharedArrayBuffer` (transport zéro-copie des positions du worker).

## À livrer — Phase 1 (fait)

1. Essaim 3D force-directed des 36 avis, coloré par thème (`node.color`).
2. Liens = arêtes k-NN (`links`, `props.weight`).
3. Panneau thèmes trié par poids : `label`, `keywords`, `size`, `weight_sum`,
   `diversity`, `consensus`.
4. Drill-down auditable : clic sur un thème (ou un nœud) → liste des avis membres
   (`node.props.text`) du cluster — la transparence est un critère jury.

## Phase 2 (live) — ce qui reste

Le protocole worker expose déjà `addNodes` (ajout incrémental, positions
préservées). Pour passer au live, **sans réécrire le scaffold** :

1. Dans `src/scene/GraphScene.tsx`, à l'endroit marqué `// Phase 2 hook`, ouvrir
   un WebSocket (cf. `queue/cross-lane.md` → backend lane `:8010`).
2. Sur `snapshot` : reconstruire le `GraphPayload` initial (comme le fetch actuel).
3. Sur `idea_added` / `edges_added` : appeler
   `clientRef.current.addNodes(nodes, links)` à chaque vague d'avis → l'essaim
   grandit, le layout se ré-équilibre localement.
4. Sur `cluster_updated` / `cluster_merged` / `cluster_split` : mettre à jour
   `themes` + recolorer (recopier `node.color` depuis le nouvel état du cluster).

Le rendu (`Nodes`/`Links`) lit déjà les positions par index depuis le buffer du
worker — il suivra l'essaim qui grandit sans modification.
