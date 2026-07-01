# Refonte front — « Agora » pour députés (groom)

> Statut : **GROOMED, pas lancé.** Attendre « go » pour dispatcher.
> Décidé avec Bob (2026-06) : public = **députés** ; 2 onglets ; identité **DSFR recoloré orange** ;
> layout 3 colonnes ; canvas = **carte spatiale UMAP** ; granularité = **split adaptatif à la variance** ;
> source = **claims émergents** (pipeline ouvert déjà câblé).

## Vision
App 3 colonnes, 2 onglets :
- **Onglet Députés** : épuré, lecture. **Onglet Analystes** : + réglages clustering, stats poussées.
- **Gauche** : panneau d'outils (dataset, recherche, filtres ; analystes : knobs clustering).
- **Centre** : **carte 2D spatiale** des thèmes (UMAP des centroïdes, distance = proximité sémantique),
  bulles = thèmes + stats (taille=nb avis, poids, consensus). **Aucun avis affiché** à ce niveau.
  Zoom/drill **adaptatif** : un thème ne se subdivise que si sa **dispersion interne** dépasse un seuil
  **dérivé** ; sinon feuille → citations. Au niveau le plus fin : citations **triées par proximité au centroïde**.
- **Droite** : **insights Markdown générés par LLM**, **liés au niveau de zoom** :
  - vue globale → **synthèse globale** + données saillantes ;
  - thème sélectionné → **synthèse du thème** ;
  - citations → navigation (pas de LLM, juste tri/lecture).
- **Identité** : codes de l'État (DSFR : grille, typo, composants) avec **primaire bleu → orange Agora**.
  NB : DSFR officiel impose le bleu → on s'en **inspire** (override des variables couleur), pas de conformité gouv stricte.

## Contrat de données (cross-lane backend ↔ front) — À FIGER avant de coder
```
POST /analysis {dataset, backend?(api|mac|auto)} ->
  {
    themes: [{ id, label, x, y,                # x,y = position UMAP 2D
               n_avis, n_claims, weight, consensus, dispersion,
               parent_id|null, has_children:bool }],
    edges:  [{ a, b, weight }],                # co-occurrence inter-thèmes
    params, backend_used
  }
GET /insights {dataset, level: "global"|"theme", id?} -> { markdown }   # LLM, CACHÉ, API par défaut
GET /citations {dataset, theme_id} -> [{ text, dist_to_centroid, weight }]  # triées centroïde
```
Hiérarchie adaptative : l'arbre est calculé côté backend (variance-driven), le front ne fait que naviguer.

## Tâches (lanes)

### Lane BACKEND
- **B1 — Projection + relations** : UMAP 2D des centroïdes de thèmes (positions x,y stables, seed) +
  arêtes de co-occurrence. Réutilise `pipeline/claims` + embeddings. Acceptance : `/analysis` renvoie themes(x,y)+edges.
- **B2 — Hiérarchie adaptative à la variance** : pour chaque thème, mesurer la **dispersion interne**
  (ex. distance cosinus moyenne au centroïde, ou variance des embeddings claims). Si > seuil **DÉRIVÉ**
  (relatif à la dispersion globale / gap), sous-clusteriser (Leiden résolution +). Sinon feuille.
  Arbre profondeur variable. Acceptance : thèmes homogènes = feuilles, thèmes hétérogènes = subdivisés ; zéro magic-number.
- **B3 — Insights LLM par niveau** : `/insights` global & par thème, Markdown, **API Mistral par défaut**
  (réutilise le multi-backend `pipeline/claims`), **caché par (dataset,level,id)**. Réutilise `backend/synthesize.py`.
- **B4 — Citations triées centroïde** : `/citations` par thème, tri par proximité au centroïde + poids.

### Lane FRONT
- **F1 — Shell DSFR recoloré** : adopter DSFR (react-dsfr ou CSS DSFR), override primaire → orange Agora,
  logo Agora, 2 onglets Députés/Analystes. Header style État.
- **F2 — Layout 3 colonnes** (outils | canvas | insights), responsive raisonnable.
- **F3 — Canvas carte spatiale** : D3/canvas, thèmes positionnés (x,y UMAP), bulles taille=poids, arêtes co-occurrence,
  **zoom/drill adaptatif** (descend si has_children), **aucun avis** jusqu'à la feuille. Transitions fluides.
- **F4 — Panneau insights Markdown** : rend `/insights` selon le niveau courant (global/thème) ; spinner pendant génération.
- **F5 — Panneau outils gauche** : dataset, recherche, filtres ; onglet Analystes : knobs clustering (résolution, backend, méthode).
- **F6 — Navigation citations** : au niveau feuille, liste des citations triées centroïde (réutilise `AvisPanel`).

### Lane CONTRAT (cross-lane)
- **C0 — Figer le contrat /analysis,/insights,/citations** ci-dessus AVANT B/F (sinon back & front divergent).

## Ordre de build conseillé
C0 (contrat) → B1+B2 (données canvas) en // de F1+F2 (shell) → B3+B4 // F3 → F4+F5+F6 → polish.

## Contraintes transverses
Ports :8010/:5180. Généricité (tout dérivé, zéro thème/seuil hardcodé). Secrets var/ jamais loggés.
LLM = API par défaut (Mac opt-in). Ne pas casser les endpoints existants (/recluster, /datasets, /synthesize, /claims).
Onglet Analystes peut garder l'ancienne vue clustering en plus.
