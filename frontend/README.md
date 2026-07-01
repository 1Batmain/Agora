# Agora — Lane CONSOLE (frontend)

Console d'exploration du pipeline de consultation citoyenne. Tous les knobs de
clustering sont réglables **en live** → re-clustering serveur → **viz 2D circle
packing zoomable** sur la hiérarchie `macro → sous-thème → avis`.

Port : **5180 uniquement**.

> Pivot UX (2026-06-20) : l'ancien essaim 3D (R3F / three / d3-force-3d, forké de
> `dummy`) a été **retiré**. Plus de web worker, plus de three.js — la console est
> du React + SVG + `d3-hierarchy` (pack layout). `package.json` est allégé d'autant.

## Ce que fait la console

1. **Circle packing zoomable** (`d3-hierarchy` `pack`) — `src/CirclePack.tsx` :
   - macro-thèmes = grandes bulles (taille ∝ `weight_sum`, couleur = `color` du
     thème, label) ;
   - **clic sur un macro = zoom** dans ses sous-thèmes ;
   - **clic sur un sous-thème = zoom + avis sources** (`node.props.text`) listés
     dans le panneau de droite ;
   - **clic sur un avis** = l'avis en entier dans le panneau ;
   - **clic sur le fond = dézoom** d'un niveau. Transitions douces (tween rAF du
     `view` [centre, diamètre] → labels à taille d'écran constante, pas de jank).

2. **Panneau KNOBS** (`src/KnobsPanel.tsx`) construit depuis `GET /api/params` :
   un slider par knob (`dedup`, `min_chars`, `k`, `threshold`, `resolution_macro`,
   `resolution_sub`, `min_sub_size`). Chaque changement est **debouncé ~300 ms**
   puis envoyé en `POST /api/recluster {knobs}` → la viz et les stats se mettent à
   jour.

3. **Stats live** (`src/StatsBar.tsx`) : `meta.stats` du payload — `n_macros`,
   `n_subs`, `n_nodes`, `modularité`, `took_ms` — pour bâtir l'intuition « ce knob
   fait ça ». Indicateur `● live :8010` / `○ statique`.

## Backend & proxy

Le backend de re-clustering (lane *stream*) vit sur **:8010**. `vite.config.ts`
configure un **proxy** : `server.proxy['/api'] → http://localhost:8010` avec
`rewrite` qui retire le préfixe `/api`. Le front appelle donc `/api/params` et
`/api/recluster` (→ `:8010/params`, `:8010/recluster`), sans souci CORS/host.

Knobs (défauts + bornes — contrat figé, gagnant `nomic-v2`) :

| knob | défaut | borne | effet |
|---|---|---|---|
| `dedup` (cosine) | 0.95 | 0.90–0.99 | fusion near-dups |
| `min_chars` | 12 | 0–40 | filtre avis courts |
| `k` (voisins) | 12 | 5–30 | densité k-NN |
| `threshold` (cosine) | 0.60 | 0.40–0.85 | coupe les arêtes |
| `resolution_macro` | 1.0 | 0.3–3.0 | granularité macros |
| `resolution_sub` | 1.5 | 0.5–4.0 | granularité sous-thèmes |
| `min_sub_size` | 18 | 5–40 | fusion des miettes |

`GET /api/params` peut surcharger ces défauts/bornes/pas ; sinon ce sont les
valeurs ci-dessus (`src/api.ts` → `DEFAULT_KNOBS`).

## Repli (backend down)

Si `:8010` ne répond pas, la console charge le `graph.json` statique de `public/`
(à défaut `graph.sample.json`) en **lecture seule** : les sliders sont **grisés**
et l'indicateur passe à `○ statique`. Les stats sont alors reconstruites depuis
`themes`/`nodes`/`meta.clustering`. C'est l'état par défaut tant que la lane
stream n'a pas démarré son serveur.

## Données

`GraphPayload { meta, nodes, links, themes }` — même shape que `graph.json`.

- `themes` = hiérarchie à 2 niveaux : `level=0` (macros) → `level=1`
  (sous-thèmes, `parent_id`/`children[]`).
- chaque nœud porte `cluster_id` (sous-thème) et `macro_id` (macro) au
  **TOP-LEVEL**, plus `color` ; `props.text` = l'avis source, `props.weight` = poids.

## Lancer

```bash
cd frontend
npm install          # une seule fois (box partagée — sobre)
npm run dev          # → http://localhost:5180
npm run build        # tsc -b && vite build (type-check + bundle)
```

`npm run dev` force `--port 5180` (`vite.config.ts` met `strictPort`).
Ports interdits (autres projets / Ollama) : `8000 5173 8765 11434`.

## Vérification d'acceptation

- `npm run build` (tsc) passe ✅
- `:5180` sert l'app ; `/graph.json` statique = 200 ; `/api/params` proxifie vers
  `:8010` (500/ECONNREFUSED quand le backend est down → repli statique) ✅
- Hiérarchie réelle vérifiée (1 root, 8 macros, 47 sous-thèmes, 1597 avis ;
  rayons pack tous positifs, 0 avis orphelin) ✅
- Flux live vérifié via backend mock éphémère : `/api/params` lu, `/api/recluster`
  POST → `meta.stats` rendues ✅

Checklist visuelle (pas de navigateur headless dans cet env — à confirmer à l'œil
sur `http://localhost:5180`) :

- [ ] 8 grandes bulles macro colorées + labels visibles à l'ouverture
- [ ] clic sur une macro → zoom doux, labels des sous-thèmes apparaissent
- [ ] clic sur un sous-thème → zoom + liste des avis sources à droite
- [ ] clic sur un avis → texte complet à droite
- [ ] clic sur le fond → dézoom d'un niveau
- [ ] (backend up) bouger un slider → après ~300 ms la viz + les stats changent
- [ ] (backend down) sliders grisés, bandeau « lecture seule », viz depuis graph.json
