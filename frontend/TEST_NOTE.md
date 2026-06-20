# Note de test — Phase 1 (2026-06-20)

Pas de navigateur headless disponible sur la box partagée (et pas d'install lourde
imposée). Vérification faite par build + transform + contrat de données.

## Vérifié automatiquement

- `npm install` : OK (148 paquets, une seule fois).
- `npx tsc --noEmit` : **0 erreur**.
- `npm run build` (`tsc -b && vite build`) : **OK** — le worker se bundle en chunk
  séparé (`forceLayout.worker-*.js`, 30 kB).
- `npm run dev` : serveur **UP sur :5180** (strictPort). Aucun port interdit utilisé
  (8000/5173/8765/11434 sont pris par d'autres projets et restent intacts).
- En-têtes HTTP `Cross-Origin-Opener-Policy: same-origin` +
  `Cross-Origin-Embedder-Policy: require-corp` présents → `SharedArrayBuffer` actif.
- `GET /graph.sample.json` → HTTP 200, 91088 octets.
- Chaque module (`App`, `GraphScene`, `Nodes`, `Links`, `ThemesPanel`,
  `useGraphStore`, `graphData`, `forceLayout.client`) se transforme via vite en
  HTTP 200 (aucune erreur d'import/syntaxe).

## Contrat de données validé (script node sur le fixture)

- **36 nœuds**, 188 liens, **6 thèmes**.
- 0 nœud sans `cluster_id`/`color` au top-level (contrat respecté).
- **6 couleurs distinctes** : `#4e79a7 #e15759 #f28e2b #76b7b2 #59a14f #b07aa1`.
- Thèmes triables par `weight_sum` ; pour chaque thème, 100 % des `member_ids`
  résolvent vers un `props.text` (drill-down auditable garanti).

## À vérifier à l'œil (ouvrir http://localhost:5180)

1. L'essaim des 36 sphères apparaît et se déploie (force-layout dans le worker),
   colorées en 6 thèmes.
2. Le panneau de droite liste les 6 thèmes triés par poids, avec
   label/keywords/size/weight/diversity/consensus.
3. Clic sur un thème → la carte s'ouvre et liste les avis sources du cluster ;
   l'essaim met en valeur ce thème (les autres nœuds/arêtes s'estompent).
4. Clic sur un nœud → ouvre le thème correspondant dans le panneau.
5. Orbit / zoom / pan à la souris (OrbitControls).
