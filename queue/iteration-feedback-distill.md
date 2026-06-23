# Itération — feedback flags · esthétique · distillation découpe · représentants (GROOM)

> Statut : **GROOMED, pas lancé.** Attendre « go »/« run ». Base = après merge du cleanup (repo propre).
> Décidé avec Bob (2026-06). Front (flag + esthétique) et Cœur (distillation + représentants) **parallélisables**.

## Décisions / contexte
- Le cœur extraction est passé en **extractif verbatim** ; le prompt actuel sur-découpe (énumérations « que…que »,
  contrastes) et ramasse du **narratif/méta** (non-claims). On veut un **témoin mistral-large** comme cible.
- Représentants = médoïde (centroïde) → favorise le générique court ; on veut **valoriser les arguments développés**.

---

## LANE FRONT (esthétique + feedback) — `frontend/`

### F1 — Système de FLAG sur les avis — ❌ ABANDONNÉ (Bob, pas besoin pour l'instant)

### F2 — Contexte de collecte DANS la synthèse globale (Markdown)
- **But** : l'intro contexte séparée fait « bordélique ». **L'intégrer dans la synthèse globale** (vue Markdown de
  la vue globale). Retirer le bloc intro séparé ; le contexte devient le début de la synthèse globale. *(esthétique)*
- **Dépend de** : B2 (le contexte intégré côté backend dans l'insight global) — repli front si absent.
- **Acceptance** : vue globale = une seule synthèse Markdown qui commence par le contexte, plus de bloc intro à part.

### F3 — Hover d'un nœud : alléger + déclutter le graphe + position stable
- **Contenu hover** = **TITRE du cluster + phrase d'accroche (hook)** (+ voix/convergence). **Retirer la description** (3ᵉ partie, de trop).
- **Déclutter le graphe** : **RETIRER le nom/label du cluster SOUS le nœud** (ex. « Temps passé sur les vidéos » ne doit PAS
  apparaître dans le graphe — trop fouillis). Les bulles n'ont **plus de label texte** ; le nom n'apparaît **qu'au hover**.
- **Position STABLE du hover** : le panneau d'info hover s'affiche **TOUJOURS en HAUT à DROITE de la fenêtre** (position fixe,
  consistante) — **PAS** un tooltip flottant au-dessus du nœud (rendu aléatoire et peu clair).
- **Acceptance** : graphe = bulles sans label ; au survol d'une bulle, le **panneau haut-droite** affiche titre + accroche (+ stats),
  toujours au même endroit ; plus aucune description.

---

## LANE BACKEND (flags + contexte) — `backend/`

### B1 — Endpoint FLAGS — ❌ ABANDONNÉ (Bob, pas besoin pour l'instant)

### B2 — Contexte dans la synthèse globale
- **But** : intégrer `dataset_context` **dans l'insight GLOBAL** (le markdown de la synthèse globale) au lieu d'un
  champ séparé — backend compose une synthèse qui s'ouvre sur le contexte. Garder `dataset_context` aussi exposé
  (repli) mais le front F2 n'affiche plus de bloc séparé. **Acceptance** : `/insights global` commence par le contexte.

---

## LANE CŒUR A — Distillation de la découpe — `pipeline/claims/` + un harnais d'éval

### C1 — TÉMOIN mistral-large (à juger visuellement AVANT de distiller)
- **But** : produire une découpe **mistral-large** avec le **meilleur prompt** (sélectivité + regroupement + few-shot
  tirés des cas de Bob), et la rendre **VISIBLE dans l'interface** pour jugement visuel.
- **Comment** : ré-extraire un dataset (commencer **petit/échantillon** pour le coût — ex. tiktok ou un sous-ensemble
  granddebat) avec `mistral-large-latest` + le prompt témoin, rebuild, **visualiser dans l'UI** (surlignages/claims).
- **Gate** : Bob juge **visuellement** la découpe. Si satisfaisante → C2. Sinon → itérer le prompt témoin.
- **Acceptance** : un dataset (ou échantillon) découpé par mistral-large, explorable sur `:5180`, jugé OK par Bob.

### C2 — Régression + distillation vers un petit modèle (gated sur C1 validé)
- **But** : utiliser le témoin mistral-large comme **gold** → optimiser le prompt d'un **plus petit modèle** pour
  s'en approcher.
- **CRITÈRE DE CLAIM (raffiné par Bob)** : un claim doit **porter explicitement (a) sa THÉMATIQUE et (b) la POSITION
  du citoyen** dessus, **verbatim** (sans paraphrase). Rationale : le claim est l'unité de DEUX clustering (thématique
  + stance) → un fragment qui ampute la position est inutilisable pour la stance. Renforce l'anti-fragmentation.
  Le **témoin mistral-large (C1)** ET le petit modèle doivent suivre ce critère ; on peut l'ajouter au scoring de l'éval.
- **MODÈLE DE CLAIM MULTI-SPANS (design, Bob)** : un claim peut prendre **plusieurs portions NON-CONTIGUËS** d'un avis
  (ex. phrase 1 + fin de la phrase 3 qui y réfère + dernière phrase = 1 claim ; le reste = un autre). À implémenter
  **avec** C2/C3 (on retouche déjà prompt + modèle) :
  - Données : `Span=(start,end)`, `Claim={spans:[Span,...], text}` ; mono-span = liste de 1 → **rétro-compatible**.
  - Extraction : le LLM renvoie `{"claims":[{"parts":["verbatim A","verbatim B"]}, ...]}` ; **chaque part validée
    sous-chaîne exacte** (`align_spans` par part) → zéro hallucination ; part non ancrée = rejetée.
  - Embedding : vecteur = embed du **texte joint** (le claim reste 1 unité).
  - Highlight : surligner **toutes** les portions du claim à la couleur du cluster (`avis.json` : autoriser N spans/claim).
  - Métrique IoU de l'éval : généraliser au **set de spans** (IoU sur l'union des plages).
- **Métrique** : **recouvrement de spans** (IoU des plages de caractères) entre découpe-petit-modèle et gold →
  F1 de segmentation, + écart du nombre de claims (anti sur/sous-découpage). Verbatim-rate (sous-chaîne exacte).
- **Boucle** : itérer `CLAIM_SYS` (+ few-shots) sur le petit modèle pour maximiser la F1-span vs gold. **Stratégie
  modèle** : commencer par **mistral-small** ; s'il n'atteint pas nos standards de qualité, monter au modèle juste
  au-dessus, et ainsi de suite jusqu'à atteindre le standard (en espérant que small suffise). **Verser les exemples
  étiquetés de Bob** comme ancres (ses labels priment).
- **Sortie** : `CLAIM_SYS` figé + modèle d'extraction retenu + **mini-jeu de régression versionné** (avis→spans
  attendus) rejouable à chaque future modif de prompt. **Acceptance** : le petit modèle retenu atteint une F1-span
  ≥ seuil convenu vs le témoin, à coût raisonnable ; régression committée.

### C3 — Bascule prod
- Figer `CLAIM_SYS` + modèle d'extraction retenu, **ré-extraire** (claims cachés périmés) + rebuild les datasets.

## LANE CŒUR B — Valoriser les commentaires DÉVELOPPÉS — `backend/analysis.py`

### D1 — Re-ranking des représentants & citations par centralité × développement
- **But** : ne plus surfacer la reformulation générique courte, mais les **arguments étoffés** et pertinents.
- **Mesure d'abord** : corrélation **longueur du claim ↔ distance au centroïde** (confirmer l'intuition de Bob),
  pour calibrer le poids.
- **Score** : `centralité × développement` où développement = longueur normalisée + marqueurs de raisonnement
  (« parce que », « car », « afin de », chiffres, exemples) + spécificité (mots rares/idf). Appliqué à
  `_representatives` ET `citations_for_theme`. Option : exposer DEUX registres (résumé central + argument développé).
- **Acceptance** : à granularité égale, les représentants/citations affichés sont plus argumentés (vérif visuelle +
  longueur médiane ↑) sans tomber dans le hors-sujet (centralité gardée en garde-fou).

---

## CONTRAT cross-lane (à figer avant de coder)
- **/flag** (B1) : shape ci-dessus, consommée par F1.
- **dataset_context dans /insights global** (B2) : consommé par F2 (repli si absent).
- C2/C3 changent `CLAIM_SYS` (cœur) → re-extraction nécessaire (build), prévoir le coût.

## ORDRE conseillé (au « run »)
1. Front F2/F3 (esthétique, rapide) ∥ Backend B1 + Front F1 (flags) ∥ D1 (représentants, mesure puis code).
2. C1 (témoin mistral-large) → **gate visuel Bob** → C2 (distillation) → C3 (bascule + ré-extraction + rebuild).
3. Merge par lane après gate ; rebuild final ; revue `:5180`.

## DÉCISIONS (tranchées par Bob)
- F3 : hover = **titre + accroche**, description retirée.
- C1 : témoin sur **tiktok**.
- C2 : commencer **mistral-small**, monter d'un cran seulement si le standard n'est pas atteint, jusqu'à l'atteindre.

---

## LANE E — Architecture LIVE / streaming + soumission citoyenne (la grande évolution)

> Vision Bob : **une place où les citoyens envoient un avis et voient les tendances émerger EN LIVE**. Et le bon
> prétexte pour **factoriser le code proprement** (sortir du build monolithique). **Lane la plus lourde** —
> probablement sa propre PHASE/run après le batch ci-dessus ; on la groome maintenant.

### E0 — Design (TRANCHÉ avec Bob) : incrémental local par split-sur-divergence
- **Pas d'UMAP** : le front est déjà en d3-pack (position non-sémantique). → **retirer `_project_2d`/UMAP du build**
  (calcul mort, gain d'efficience) ; plus de problème de stabilité de positions.
- **Algo incrémental** (réutilise la logique variance-adaptative, en LOCAL) :
  - nouveau claim → **rattaché au cluster feuille le plus proche** (cos centroïde max), **toujours** (pas de buffer) ;
  - maj **O(1)** du centroïde + dispersion (identités `‖S‖`) ;
  - **SI** dispersion du cluster > τ (dérivé) **ET** split en ≥2 sous-thèmes viables (`_subdivide` local) → **on split CE cluster** ;
    les nouveaux thèmes **émergent du split** (pas de germe séparé).
- **Recalcul LOCAL uniquement** (le split d'un cluster), jamais de re-cluster global dans le flux normal → efficient/rapide.
- Objectif perf : attach O(n_clusters·d), maj O(d), split O(taille) rare. Garder la généricité (τ et seuils dérivés).

### E1 — Refactor pipeline INCRÉMENTAL
- Séparer proprement : **ingestion** / **traitement par-avis** (extract+embed, déjà isolé/caché) / **agrégation/état**
  (l'arbre + indices) / **vue**. État courant mutable + recompute périodique. C'est la « bonne factorisation ».

### E2 — Visualisation LIVE du build
- Pendant un build, **streamer les résultats partiels** au front (SSE/WebSocket) → la carte se **peuple en live**
  (thèmes qui apparaissent/grossissent au fil des avis traités). Réutilise `status.json` + pousse des payloads partiels.

### E3 — Soumission citoyenne
- `POST /submit {dataset, text}` → l'avis est traité (extract→embed→assigné/germé) → contribue à la vue live + au cache.
- Front : champ de soumission d'avis + carte qui se met à jour. (Modération/garde-fous = à prévoir si public.)

### E4 — Front LIVE
- Le front **s'abonne** aux updates (SSE/WebSocket) et **anime** la carte/dashboard en continu (nouveaux thèmes, poids
  évoluant). Transitions douces (déjà du d3-force → animer l'entrée/sortie/croissance des bulles).

### Notes / risques
- Le **clustering incrémental** est le vrai défi (E0) ; ne pas sous-estimer. Faire un design pass + un prototype avant
  de s'engager. Le reste (SSE, submit, anim front) est standard.
- Probable **phase dédiée** : finir le batch (flags/esthétique/distillation/représentants) puis attaquer Lane E.

### Acquis R&D E0 (`research/inc_macro_report.md`)
- Le **split-sur-dispersion** (écart d'embedding > τ dérivé) est notre core et marche **en batch**. Le goulot de
  l'incrémental est **le rattachement glouton** (feuilles qui chevauchent les macros batch), PAS le critère de split.
  Oracle plafonné à V 0.52-0.75 → l'incrémental pur ne peut égaler le batch.
- **Option B′** (Leiden seul sur centroïdes de feuilles) mergée derrière flag `macro_mode` (défaut inchangé) : seule
  dérivation macro **générique** (A et B s'effondrent sur multi-sujets). Cap retenu pour le live : **incrémental B′
  + recompute BATCH périodique** (ancre de qualité, borné).

### Recompute ADAPTATIF piloté par la STABILITÉ (idée Bob — futur, pas maintenant)
- Mesurer la **stabilité de la structure** pendant l'agrégation (ex. ARI / V-mesure entre snapshots de membership
  successifs, ou % de claims qui changent de cluster, ou dérive des centroïdes).
- **Structure instable → recompute fréquent ; structure stable/convergée → recompute rare.** Cadence **dérivée**,
  pas fixe → beaucoup de recomputes au début (churn), de moins en moins à mesure que les thèmes se figent. À groomer
  plus finement quand on attaquera le live sérieusement. **Pour l'instant : on reste simple** (trigger basique).

---

## LANE STANCE — analyse pour/contre DANS un thème (futur ; note de grooming Bob)

> Idée : la même unité (claim) sert à DEUX clustering — **thématique** (déjà) puis **stance**. Dans un thème, on
> représente **qui est pour / contre** et on donne un **visuel d'opposition** des avis.

- **Prérequis** : le critère de claim « sujet + position, verbatim » (cf. C2) — sans la position, pas de stance.
- **Signal de stance** : PAS les embeddings (qui encodent le sujet, pas la polarité — cf. réserve sur « convergence
  cumulée »). → signal DÉDIÉ : **LLM** (classer le claim pour/contre/neutre vis-à-vis de la proposition du thème) ou
  **NLI** (le claim soutient-il / contredit-il l'assertion du thème ?). À benchmarker (cf. l'éval NLI déjà faite en research/).
- **Dans le thème** : positionner les claims sur un **axe d'opposition** (pour ↔ contre), ou 2 sous-groupes. Visuel :
  barre/diverging, ou split de la bulle en deux camps.
- **Indice** : un vrai indice de **polarisation** par thème (remplacerait/complèterait la « convergence cumulée »
  qui ne capte que le thématique).
- **Statut** : à groomer/concevoir plus finement quand on y arrivera ; **note ici** pour orienter dès maintenant le
  critère de claim (C1/C2). Phase après le batch + Lane E.

---

## LANE LOAD-ANIM — l'animation de chargement = rejeu du batch (décidé, après ux-batch)
> Pas une fonction séparée : c'est **comment le graphe se charge par défaut**. À faire **après le merge d'ux-batch**
> (touche le front principal, conflit sinon).
- **Backend** : `/stream` **rejoue l'analyse BATCH cachée** (PAS l'incrémental) — pour chaque claim dans l'ordre,
  `claim_added` vers son cluster BATCH ; `theme_born`/`theme_split` à l'apparition. Lecture du cache, cheap, déterministe.
  L'incrémental `AnalysisState` reste dans le code (**parké**) pour quand on craquera sa qualité.
- **Front** : au **chargement de la page**, le graphe principal (`RedesignApp`+`SpatialMap`) joue la **construction
  animée** (via `/stream`) puis **se fige** en interactif. **RETIRER** `LiveView` + boutons « Rejouer live/démo » séparés.
- **Acceptance** : ouvrir la page → le graphe se construit en animé (vraies données, structure batch propre, pas de
  doublons aléatoires) puis devient interactif. Plus de mode séparé.
